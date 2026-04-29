"""
prepare_cluster_labels.py — Pre-compute K-Means cluster labels for all customers.

Uses identical parameters to perform_clustering() in lambda_ds_performance.py
(n_clusters=4, random_state=42) so that cluster numbers match what the app displays
when a user runs ds_cluster_analysis.

Output: docs/customer_cluster_labels.csv
  customer_id | dynamic_segment | cluster (1-based, matches app display)

Run once after ds_data_prep.py:
    python prepare_cluster_labels.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import DS_CSV

CLUSTER_LABELS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "customer_cluster_labels.csv")

N_CLUSTERS   = 4
RANDOM_STATE = 42
N_INIT       = 10

NUMERIC_COLS = [
    "avg_num_trxns", "avg_weekly_trxn_amt", "trxn_amt_monthly",
    "INCOME", "CURRENT_BALANCE", "ACCT_AGE_YEARS", "AGE",
]
CAT_COLS = [
    "ACCOUNT_TYPE", "GENDER", "AGE_CATEGORY",
    "ACCT_OPEN_CHANNEL", "NNM", "OFAC", "314b",
    "CITIZENSHIP", "RESIDENCY_COUNTRY",
]


def _cluster_segment(df_seg, seg_name):
    """Run K-Means on active accounts in one segment. Returns df with 'cluster' column (1-based)."""
    df_active = df_seg[df_seg["avg_num_trxns"].fillna(0) > 0].copy()
    print(f"  {seg_name}: {len(df_active):,} active accounts")

    num_cols = [c for c in NUMERIC_COLS if c in df_active.columns]
    cat_cols = [c for c in CAT_COLS if c in df_active.columns]

    X_num = df_active[num_cols].fillna(df_active[num_cols].median())
    if cat_cols:
        df_enc = pd.get_dummies(df_active[cat_cols], drop_first=True).fillna(0)
        X = pd.concat([X_num.reset_index(drop=True), df_enc.reset_index(drop=True)], axis=1)
    else:
        X = X_num.reset_index(drop=True)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))

    km     = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(X_scaled)       # 0-based
    df_active = df_active.copy()
    df_active["cluster"] = labels + 1       # 1-based to match app display

    print(f"  Cluster sizes: { {c: int((df_active['cluster']==c).sum()) for c in range(1, N_CLUSTERS+1)} }")
    return df_active[["customer_id", "dynamic_segment", "cluster"]]


def main():
    print(f"Loading {DS_CSV} ...")
    df = pd.read_csv(DS_CSV, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    results = []
    for seg_id, seg_name in [(0, "Business"), (1, "Individual")]:
        df_seg = df[df["dynamic_segment"] == seg_id]
        results.append(_cluster_segment(df_seg, seg_name))

    out = pd.concat(results, ignore_index=True)
    out.to_csv(CLUSTER_LABELS_CSV, index=False)
    print(f"\nWrote {len(out):,} rows to {CLUSTER_LABELS_CSV}")

    # Quick overlap check with rule sweep data
    rule_sweep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "rule_sweep_data.csv")
    if os.path.exists(rule_sweep_path):
        rs = pd.read_csv(rule_sweep_path)
        overlap = rs["customer_id"].isin(out["customer_id"]).sum()
        print(f"Rule sweep overlap: {overlap:,} / {len(rs):,} alerted customers have cluster labels ({100*overlap/len(rs):.1f}%)")


if __name__ == "__main__":
    main()
