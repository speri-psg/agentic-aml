"""
sar_scorer.py — SAR Propensity Model

Trains a RandomForestClassifier on DF_SAR at startup.
Optionally joins per-customer network graph features from network_features.csv.

Uses cross-validated out-of-fold (OOF) predictions for the training population
so that each customer's score comes from a model that never saw them during
training — producing differentiated probabilities rather than memorised ones.

The full-data model is retained for scoring future/unseen customers.

Exposes score_alerts(df_sar) → df with sar_prob column (0–1).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict

BASE_FEATURE_COLS = [
    "alert_count", "rule_count", "max_trigger", "max_z_score",
    "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
    "total_trxn_amt", "cashout_count",
]

NETWORK_FEATURE_COLS = [
    "out_degree", "in_degree", "unique_banks_sent_to", "unique_banks_rcvd_from",
    "structuring_score", "fan_out_score", "txn_velocity", "cross_bank_ratio",
    "total_out_txns", "total_out_amt", "max_single_txn", "avg_txn_amt",
]

_model               = None
_roc_auc             = None
_feature_importances = None
_network_df          = None   # cached network features DataFrame
_oof_lookup          = None   # dict: customer_id -> OOF probability


def _load_network_features(network_csv: str | None = None) -> pd.DataFrame | None:
    """Load network_features.csv if it exists, indexed by customer_id."""
    global _network_df
    if _network_df is not None:
        return _network_df

    if network_csv is None:
        here = os.path.dirname(os.path.abspath(__file__))
        network_csv = os.path.join(here, "docs", "network_features.csv")

    if not os.path.exists(network_csv):
        return None

    df = pd.read_csv(network_csv)
    if "customer_id" not in df.columns:
        return None

    _network_df = df.set_index("customer_id")
    print(f"[sar_scorer] Loaded network features for {len(_network_df):,} customers")
    return _network_df


def _prepare_X(df: pd.DataFrame, net_df: pd.DataFrame | None = None) -> pd.DataFrame:
    available = [c for c in BASE_FEATURE_COLS if c in df.columns]
    X = df[available].copy().fillna(0)

    if "customer_type" in df.columns:
        X["is_business"] = (df["customer_type"] == "BUSINESS").astype(int)

    if net_df is not None and "customer_id" in df.columns:
        net_available = [c for c in NETWORK_FEATURE_COLS if c in net_df.columns]
        joined = df["customer_id"].map(
            net_df[net_available].to_dict(orient="index")
        ).apply(pd.Series)
        for col in net_available:
            if col in joined.columns:
                X[col] = joined[col].fillna(0).values

    return X


def train(df_sar: pd.DataFrame, network_csv: str | None = None):
    """
    Train the SAR propensity model and compute out-of-fold predictions.

    OOF predictions use StratifiedKFold(5) so each customer's score is from
    a model that never saw them — giving differentiated probabilities in the
    worklist rather than memorised near-identical scores.

    Returns (model, roc_auc).
    """
    global _model, _roc_auc, _feature_importances, _oof_lookup

    net_df = _load_network_features(network_csv)

    df = df_sar.dropna(subset=["is_sar"]).copy()
    X  = _prepare_X(df, net_df)
    y  = df["is_sar"].astype(int)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # ── Out-of-fold predictions ───────────────────────────────────────────────
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    oof_probs = np.round(oof[:, 1], 4)

    # Store as customer_id -> prob lookup for fast matching at score time
    if "customer_id" in df.columns:
        _oof_lookup = dict(zip(df["customer_id"].values, oof_probs))
    else:
        _oof_lookup = None

    # ── Full-data model (for scoring unseen customers) ────────────────────────
    clf.fit(X, y)
    _model = clf

    # ROC-AUC on OOF predictions (honest estimate)
    from sklearn.metrics import roc_auc_score
    _roc_auc = round(float(roc_auc_score(y, oof_probs)), 3)

    _feature_importances = pd.Series(
        _model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    n_sar   = int(y.sum())
    n_total = len(y)
    net_tag = " + network" if net_df is not None else ""
    print(
        f"[sar_scorer] Trained on {n_total:,} alerts | SARs={n_sar:,} "
        f"({100*n_sar/n_total:.1f}%) | features={len(X.columns)}{net_tag} | "
        f"OOF ROC-AUC={_roc_auc}"
    )
    return _model, _roc_auc


def score_alerts(df_sar: pd.DataFrame) -> pd.DataFrame:
    """
    Return df_sar with sar_prob column (0–1), sorted descending.

    For customers that were in the training set, uses out-of-fold predictions
    (honest, generalised scores). For any unseen customers, falls back to the
    full-data model.
    """
    if _model is None:
        raise RuntimeError("Model not trained — call train() first.")

    net_df = _load_network_features()
    out    = df_sar.copy()

    # Start with full-model predictions for everyone
    X    = _prepare_X(out, net_df)
    prob = _model.predict_proba(X)[:, 1]
    out["sar_prob"] = np.round(prob, 4)

    # Overwrite training-population rows with OOF predictions (honest scores)
    if _oof_lookup is not None and "customer_id" in out.columns:
        out["sar_prob"] = out.apply(
            lambda r: _oof_lookup.get(r["customer_id"], r["sar_prob"]), axis=1
        )

    return out.sort_values("sar_prob", ascending=False).reset_index(drop=True)


def get_roc_auc() -> float | None:
    return _roc_auc


def get_feature_importances() -> pd.Series | None:
    return _feature_importances
