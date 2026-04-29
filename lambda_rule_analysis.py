"""
lambda_rule_analysis.py — Rule-level FP/FN sweep using actual condition parameters

Depends on docs/rule_sweep_data.csv produced by prepare_rule_sweep_data.py.

Each rule has condition parameters that were set per-customer from their
transaction history.  The sweep varies those parameters (e.g. the $50K floor
for Activity Deviation or the 5-std dev multiplier) and shows how SAR catch rate
and FP count change.

Public API (called from application.py):
    load_rule_sweep_data()            -> df  (call once at startup)
    list_rules_text(df)               -> pre-computed text for the model
    compute_rule_sar_sweep(df, ...)   -> pre-computed text for the model
"""

import json
import math
import os
import pandas as pd
import numpy as np

_HERE        = os.path.dirname(os.path.abspath(__file__))
_THRESHOLDS  = os.path.join(_HERE, "rule_thresholds.json")
_CATALOGUE   = os.path.join(_HERE, "config", "rule_catalogue.json")

from config import RULE_SWEEP_CSV as _CSV

MAX_SWEEP_ROWS = 12

# ── Rule catalogue ─────────────────────────────────────────────────────────────
# Loaded from config/rule_catalogue.json — add new rules there, no code changes needed.
# Maps lower-cased keyword -> canonical risk_factor name + sweepable params

def _load_catalogue() -> dict:
    with open(_CATALOGUE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Convert default_2d lists to tuples (preserves original type used in code)
    for entry in raw.values():
        if isinstance(entry.get("default_2d"), list):
            entry["default_2d"] = tuple(entry["default_2d"])
    return raw

RULE_CATALOGUE = _load_catalogue()

# ── Legacy inline catalogue (kept for reference, no longer used) ───────────────
_LEGACY_CATALOGUE = {
    "activity deviation (ach)": {
        "name":        "Activity Deviation (ACH)",
        "current":     "Monthly Outgoing ACH >= $50K AND >= 5 std dev above 12-month profile mean",
        "sweep_params": {
            "floor_amount": {
                "col":       "trigger_amt",
                "label":     "monthly ACH floor ($)",
                "current":   50_000,
                "direction": "gte",
                "sweep_min": 10_000,
                "desc":      "Minimum monthly Outgoing ACH sum to trigger (currently $50K)",
            },
            "z_threshold": {
                "col":          "z_score",
                "label":        "std dev multiplier",
                "current":      5.0,
                "direction":    "gte",
                "sweep_min":    0.0,
                "integer_axis": True,
                "desc":         "Std-dev multiplier above 12-month ACH profile mean (currently 5)",
            },
        },
        "default_sweep": "floor_amount",
        "default_2d":   ("floor_amount", "z_threshold"),
    },
    "activity deviation (check)": {
        "name":        "Activity Deviation (Check)",
        "current":     "Monthly Outgoing Check >= $50K AND >= 2 std dev above 12-month profile mean",
        "sweep_params": {
            "floor_amount": {
                "col":       "trigger_amt",
                "label":     "monthly Check floor ($)",
                "current":   50_000,
                "direction": "gte",
                "sweep_min": 10_000,
                "desc":      "Minimum monthly Outgoing Check sum to trigger (currently $50K)",
            },
            "z_threshold": {
                "col":          "z_score",
                "label":        "std dev multiplier",
                "current":      2.0,
                "direction":    "gte",
                "sweep_min":    0.0,
                "integer_axis": True,
                "desc":         "Std-dev multiplier above 12-month Check profile mean (currently 2)",
            },
        },
        "default_sweep": "floor_amount",
        "default_2d":   ("floor_amount", "z_threshold"),
    },
    "elder abuse": {
        "name":        "Elder Abuse",
        "current":     "Age >= 60 AND 14-day outgoing >= $5K AND >= 3 std dev above 90-day mean",
        "sweep_params": {
            "floor_amount": {
                "col":       "trigger_amt",
                "label":     "14-day outgoing floor ($)",
                "current":   5_000,
                "direction": "gte",
                "sweep_min": 500,
                "desc":      "Minimum 14-day aggregated outgoing to trigger (currently $5K)",
            },
            "z_threshold": {
                "col":          "z_score",
                "label":        "std dev multiplier",
                "current":      3.0,
                "direction":    "gte",
                "sweep_min":    0.0,
                "integer_axis": True,
                "desc":         "Std-dev multiplier above 90-day mean (currently 3)",
            },
            "age_threshold": {
                "col":          "age",
                "label":        "minimum age (years)",
                "current":      60,
                "direction":    "gte",
                "sweep_min":    50,
                "integer_axis": True,
                "desc":         "Minimum customer age to trigger (currently 60)",
            },
        },
        "default_sweep": "z_threshold",
        "default_2d":   ("floor_amount", "age_threshold"),
    },
    "velocity single": {
        "name":        "Velocity Single",
        "current":     ">=1 pair (in+out) within 14 days, out=90-110% of in, pair total >= $20K",
        "sweep_params": {
            "pair_total": {
                "col":       "pair_total",
                "label":     "pair total ($)",
                "current":   20_000,
                "direction": "gte",
                "sweep_min": 5_000,
                # sweep_max derived from data p95 — no catalogue cap
                "desc":      "Minimum combined in+out pair total to trigger (currently $20K)",
            },
            "ratio_tolerance": {
                "col":        "ratio",
                "label":      "ratio tolerance",
                "current":    0.10,
                "direction":  "abs_lte",
                "format_pct": True,
                "values":     [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20],
                "desc":       "Max deviation of out/in ratio from 1.0 to trigger (currently 10% = 90-110%)",
            },
        },
        "default_sweep":   "pair_total",
        "default_2d":      ("pair_total", "ratio_tolerance"),
    },
    "detect excessive": {
        "name":        "Detect Excessive Transaction Activity",
        "current":     "5-day incoming Cash+Check sum > $10K",
        "sweep_params": {
            "floor_amount": {
                "col":       "trigger_amt",
                "label":     "sum threshold ($)",
                "current":   10_000,
                "direction": "gte",
                "sweep_min": 2_000,
                # sweep_max derived from data p95 — no catalogue cap
                "desc":      "Minimum N-day incoming Cash+Check sum to trigger (currently $10K over 5 days)",
            },
            "time_window": {
                "col":       None,
                "label":     "time window (days)",
                "current":   5,
                "direction": "window",
                "values":    [3, 5, 7, 10, 14],
                "desc":      "Aggregation window in days (currently 5 days); options: 3, 5, 7, 10, 14",
            },
        },
        "default_sweep":   "floor_amount",
        "default_2d":      ("floor_amount", "time_window"),
    },
    "structuring (incoming cash)": {
        "name":        "Structuring (Incoming Cash)",
        "current":     "3 qualifying days within 14-day window, each day's Cash CashIn total $3K-$40K",
        "sweep_params": {
            "daily_floor": {
                "col":       "trigger_amt",
                "label":     "min daily Cash amount ($)",
                "current":   3_000,
                "direction": "gte",
                "sweep_min": 500,
                "desc":      "Minimum daily Cash CashIn total for a qualifying day (currently $3K)",
            },
            "days_required": {
                "col":          "days_observed",
                "label":        "qualifying days required",
                "current":      3,
                "direction":    "gte",
                "sweep_min":    1,
                "integer_axis": True,
                "desc":         "Minimum number of qualifying days in the window (currently 3)",
            },
        },
        "default_sweep": "daily_floor",
        "default_2d":    ("daily_floor", "days_required"),
    },
    "structuring (outgoing cash)": {
        "name":        "Structuring (Outgoing Cash)",
        "current":     "3 qualifying days within 14-day window, each day's Cash CashOut total $7K-$30K",
        "sweep_params": {
            "daily_floor": {
                "col":       "trigger_amt",
                "label":     "min daily Cash amount ($)",
                "current":   7_000,
                "direction": "gte",
                "sweep_min": 1_000,
                "desc":      "Minimum daily Cash CashOut total for a qualifying day (currently $7K)",
            },
            "days_required": {
                "col":          "days_observed",
                "label":        "qualifying days required",
                "current":      3,
                "direction":    "gte",
                "sweep_min":    1,
                "integer_axis": True,
                "desc":         "Minimum number of qualifying days in the window (currently 3)",
            },
        },
        "default_sweep": "daily_floor",
        "default_2d":    ("daily_floor", "days_required"),
    },
    "ctr client": {
        "name":        "CTR Client",
        "current":     "Cash + Currency Exchange in/out total > $10K",
        "sweep_params": {
            "floor_amount": {
                "col":       "trigger_amt",
                "label":     "total Cash threshold ($)",
                "current":   10_000,
                "direction": "gte",
                "sweep_min": 5_000,
                "desc":      "Minimum Cash/Currency Exchange total to trigger (currently $10K)",
            },
        },
        "default_sweep": "floor_amount",
        "default_2d":    ("floor_amount", "floor_amount"),
    },
    "burst in originator activity": {
        "name":        "Burst in Originator Activity",
        "current":     "5-day Incoming Wire/ACH sum >= $5K, >= 3 transactions, >= 2 distinct counterparties",
        "sweep_params": {
            "floor_amount": {
                "col":       "trigger_amt",
                "label":     "5-day incoming sum ($)",
                "current":   5_000,
                "direction": "gte",
                "sweep_min": 1_000,
                "desc":      "Minimum 5-day incoming Wire/ACH sum to trigger (currently $5K)",
            },
            "min_transactions": {
                "col":          "txn_count",
                "label":        "minimum transaction count",
                "current":      3,
                "direction":    "gte",
                "sweep_min":    1,
                "integer_axis": True,
                "desc":         "Minimum number of transactions in the 5-day window (currently 3)",
            },
        },
        "default_sweep": "floor_amount",
        "default_2d":    ("floor_amount", "min_transactions"),
    },
    "burst in beneficiary activity": {
        "name":        "Burst in Beneficiary Activity",
        "current":     "5-day Outgoing Wire/ACH sum >= $5K, >= 3 transactions, >= 2 distinct counterparties",
        "sweep_params": {
            "floor_amount": {
                "col":       "trigger_amt",
                "label":     "5-day outgoing sum ($)",
                "current":   5_000,
                "direction": "gte",
                "sweep_min": 1_000,
                "desc":      "Minimum 5-day outgoing Wire/ACH sum to trigger (currently $5K)",
            },
            "min_transactions": {
                "col":          "txn_count",
                "label":        "minimum transaction count",
                "current":      3,
                "direction":    "gte",
                "sweep_min":    1,
                "integer_axis": True,
                "desc":         "Minimum number of transactions in the 5-day window (currently 3)",
            },
        },
        "default_sweep": "floor_amount",
        "default_2d":    ("floor_amount", "min_transactions"),
    },
    "risky international transfer": {
        "name":        "Risky International Transfer",
        "current":     "Single Wire to/from medium-risk country >= $300K or >= $500K",
        "sweep_params": {
            "floor_amount": {
                "col":       "trigger_amt",
                "label":     "single Wire floor ($)",
                "current":   300_000,
                "direction": "gte",
                "sweep_min": 100_000,
                "desc":      "Minimum single Wire amount to a risky country to trigger (currently $300K)",
            },
        },
        "default_sweep": "floor_amount",
        "default_2d":    ("floor_amount", "floor_amount"),
    },
}  # end _LEGACY_CATALOGUE — not used; kept for reference only


def _load_thresholds():
    """
    Patch RULE_CATALOGUE with current thresholds from rule_thresholds.json.
    When a rule's operational threshold changes, only the JSON needs updating.
    """
    if not os.path.exists(_THRESHOLDS):
        return
    with open(_THRESHOLDS, "r") as f:
        config = json.load(f)
    for rule_key, rule_cfg in config.items():
        if rule_key.startswith("_"):
            continue
        if rule_key not in RULE_CATALOGUE:
            continue
        entry = RULE_CATALOGUE[rule_key]
        if "current_condition" in rule_cfg:
            entry["current"] = rule_cfg["current_condition"]
        for param_name, value in rule_cfg.get("thresholds", {}).items():
            if param_name in entry["sweep_params"]:
                entry["sweep_params"][param_name]["current"] = value

_load_thresholds()


def _match_rule(keyword):
    """Return (canonical_name, catalogue_entry) or (None, None)."""
    kw = keyword.strip().lower()
    for key, entry in RULE_CATALOGUE.items():
        if kw in key or key in kw or kw in entry["name"].lower():
            return entry["name"], entry
    return None, None


# ── Load ───────────────────────────────────────────────────────────────────────

def load_rule_sweep_data():
    """Load rule sweep data — production CSV or synthetic (when ARIA_SYNTH_DATA_DIR is set)."""
    if not os.path.exists(_CSV):
        return None
    df = pd.read_csv(_CSV)
    try:
        from column_mapper import normalize_columns
        df = normalize_columns(df, verbose=True)
    except Exception:
        pass
    # Normalise customer_id column name (synth uses aria_customer_id)
    if "aria_customer_id" in df.columns and "customer_id" not in df.columns:
        df = df.rename(columns={"aria_customer_id": "customer_id"})
    # Derive dynamic_segment from customer_type if missing (needed for cluster filter)
    if "dynamic_segment" not in df.columns and "customer_type" in df.columns:
        df["dynamic_segment"] = df["customer_type"].str.upper().map(
            {"BUSINESS": 0, "INDIVIDUAL": 1}
        )
    source = "synthetic" if os.getenv("ARIA_SYNTH_DATA_DIR") else "production"
    print(f"[rule_analysis] loaded {len(df):,} rows from {source} data ({_CSV})")
    return df


# ── list_rules ─────────────────────────────────────────────────────────────────

def list_rules_text(df):
    """
    Pre-computed text listing available rules with SAR/FP counts and sweep options.
    """
    lines = ["=== PRE-COMPUTED RULE LIST (copy this verbatim) ==="]
    lines.append("Available AML rules with SAR/FP performance (detailed table shown in chart below):")
    lines.append("NOTE: This is the COMPLETE list of rules in the system. Do NOT add or infer any rules not listed here.")

    for _, entry in RULE_CATALOGUE.items():
        rf   = entry["name"]
        grp  = df[df["risk_factor"] == rf] if df is not None else pd.DataFrame()
        n    = len(grp)
        sar  = int(grp["is_sar"].sum()) if n > 0 else 0
        fp   = int((grp["is_sar"] == 0).sum()) if n > 0 else 0
        prec = f"{round(100*sar/(sar+fp),1)}%" if (sar + fp) > 0 else "n/a"
        sweep_keys = ", ".join(entry["sweep_params"].keys())
        lines.append(f"  {rf}: alerts={n}, SAR={sar}, FP={fp}, precision={prec}, sweep_params=[{sweep_keys}]")

    lines.append("=== END RULE LIST ===")
    return "\n".join(lines)


# ── cluster_rule_summary ───────────────────────────────────────────────────────

def cluster_rule_summary_text(df, cluster_n):
    """
    Pre-computed SAR/FP/precision for ALL rules filtered to customers in one cluster.

    Parameters
    ----------
    df        : DataFrame already filtered to the target cluster by _filter_by_cluster()
    cluster_n : int — cluster number (1-based), used only for display
    """
    if df is None or len(df) == 0:
        return (
            f"=== PRE-COMPUTED CLUSTER RULE SUMMARY (copy this verbatim) ===\n"
            f"Cluster {cluster_n} — no alert data found for this cluster.\n"
            f"=== END CLUSTER RULE SUMMARY ==="
        )

    n_customers = df["customer_id"].nunique() if "customer_id" in df.columns else len(df)

    lines = ["=== PRE-COMPUTED CLUSTER RULE SUMMARY (copy this verbatim) ==="]
    lines.append(f"Cluster {cluster_n} — {n_customers:,} customers in rule alert data")
    lines.append("SAR/FP performance for all rules filtered to this cluster:")
    lines.append("NOTE: alerts=0 means no alerts from this rule for customers in this cluster.")

    active_lines, inactive = [], []
    for _, entry in RULE_CATALOGUE.items():
        rf  = entry["name"]
        grp = df[df["risk_factor"] == rf]
        n   = len(grp)
        if n == 0:
            inactive.append(rf)
            continue
        sar  = int(grp["is_sar"].sum())
        fp   = int((grp["is_sar"] == 0).sum())
        prec = f"{round(100 * sar / (sar + fp), 1)}%" if (sar + fp) > 0 else "n/a"
        active_lines.append(f"  {rf}: alerts={n}, SAR={sar}, FP={fp}, precision={prec}")

    lines.extend(active_lines)
    if inactive:
        lines.append(f"Rules with alerts=0 in Cluster {cluster_n}: {', '.join(inactive)}")
    lines.append("=== END CLUSTER RULE SUMMARY ===")
    return "\n".join(lines)


# ── rule_sar_backtest ──────────────────────────────────────────────────────────

def compute_rule_sar_sweep(df, risk_factor_keyword, sweep_param=None, max_rows=MAX_SWEEP_ROWS):
    """
    Sweep a rule condition parameter and show SAR caught / FP remaining / SAR missed
    at each threshold level.

    Parameters
    ----------
    df                  : DataFrame from load_rule_sweep_data()
    risk_factor_keyword : str — e.g. "Activity Deviation" or "elder"
    sweep_param         : str — key from rule's sweep_params dict (or None for default)
    """
    if df is None:
        return "Rule sweep data not loaded. Run python prepare_rule_sweep_data.py first."

    rf_name, entry = _match_rule(risk_factor_keyword)
    if entry is None:
        known = [e["name"] for e in RULE_CATALOGUE.values()]
        return (
            f"No rule matched '{risk_factor_keyword}'. "
            f"Call list_rules to see available rules. Known: {known}"
        )

    # Resolve sweep parameter — normalize spaces/hyphens to underscores
    if sweep_param is not None:
        sweep_param = str(sweep_param).replace(" ", "_").replace("-", "_")
    if sweep_param is None or sweep_param not in entry["sweep_params"]:
        sweep_param = entry["default_sweep"]
    sp = entry["sweep_params"][sweep_param]
    col = sp["col"]

    # Filter to this rule, only rows with SAR label and sweep column
    rule_df = df[df["risk_factor"] == rf_name].copy()
    known   = rule_df.dropna(subset=["is_sar", col]).copy()

    total_sars = int((known["is_sar"] == 1).sum())
    total_fps  = int((known["is_sar"] == 0).sum())
    total      = len(known)
    precision  = round(100 * total_sars / total, 1) if total > 0 else 0.0

    header = [
        "=== PRE-COMPUTED RULE SWEEP (copy this verbatim, do not alter numbers) ===",
        f"Rule: {rf_name}",
        f"Current condition: {entry['current']}",
        f"Sweep parameter: {sweep_param} - {sp['desc']}",
        f"Current value: {sp['current']:,}",
        f"Labeled population: {total} customers (TP+FN pool={total_sars} SAR, FP+TN pool={total_fps} non-SAR, precision={precision}%)",
    ]

    if total == 0:
        header.append("No labelled customers with this sweep column. Cannot sweep.")
        header.append("=== END RULE SAR SWEEP ===")
        return "\n".join(header)

    if total_sars == 0:
        header.append("No SAR customers in this rule's alerted population — all alerts are false positives.")
        header.append("=== END RULE SAR SWEEP ===")
        return "\n".join(header)

    # Use same nice-step logic as _sweep_points for consistency
    cur = float(sp["current"])
    raw_points = _sweep_points(known, sp, n_steps=16)

    sweep = []
    for t in raw_points:
        if sp["direction"] == "gte":
            mask_caught = (known[col] >= t) & (known["is_sar"] == 1)
            mask_fp     = (known[col] >= t) & (known["is_sar"] == 0)
        else:  # lte
            mask_caught = (known[col] <= t) & (known["is_sar"] == 1)
            mask_fp     = (known[col] <= t) & (known["is_sar"] == 0)
        caught = int(mask_caught.sum())
        fp_rem = int(mask_fp.sum())
        missed = total_sars - caught
        sweep.append((round(t, 2), caught, fp_rem, missed))

    # Key thresholds
    def last_point_above_rate(target_rate):
        target = int(total_sars * target_rate)
        candidates = [s for s in sweep if s[1] > target]
        return candidates[-1] if candidates else sweep[0]

    t90 = last_point_above_rate(0.90)
    t50 = last_point_above_rate(0.50)

    # Find current-condition row
    cur_row = next((s for s in sweep if s[0] == round(cur, 2)), None)

    def _prec(tp, fp):
        return round(100 * tp / (tp + fp), 1) if (tp + fp) > 0 else 0.0

    # TP=caught SAR, FP=caught non-SAR, FN=missed SAR, TN=uncaught non-SAR
    first_tp  = sweep[0][1];  first_fp  = sweep[0][2]
    first_fn  = total_sars - first_tp; first_tn = total_fps - first_fp
    first_pct = round(100 * first_tp / total_sars, 1)

    lines = header + [""]
    lines.append(
        f"At the lowest value ({sweep[0][0]:,.2f}): "
        f"TP={first_tp}, FP={first_fp}, FN={first_fn}, TN={first_tn} "
        f"(TP rate={first_pct}%, precision={_prec(first_tp, first_fp)}%)."
    )
    if cur_row:
        cur_tp  = cur_row[1]; cur_fp  = cur_row[2]
        cur_fn  = total_sars - cur_tp; cur_tn = total_fps - cur_fp
        cur_pct = round(100 * cur_tp / total_sars, 1)
        lines.append(
            f"At current condition ({cur_row[0]:,.2f}): "
            f"TP={cur_tp}, FP={cur_fp}, FN={cur_fn}, TN={cur_tn} "
            f"(TP rate={cur_pct}%, precision={_prec(cur_tp, cur_fp)}%)."
        )
    t90_tp = t90[1]; t90_fp = t90[2]; t90_fn = total_sars - t90_tp; t90_tn = total_fps - t90_fp
    t50_tp = t50[1]; t50_fp = t50[2]; t50_fn = total_sars - t50_tp; t50_tn = total_fps - t50_fp
    lines.append(f"To keep TP rate >=90%: {sweep_param} <= {t90[0]:,.2f} => TP={t90_tp}, FP={t90_fp}, FN={t90_fn}, TN={t90_tn}, precision={_prec(t90_tp, t90_fp)}%.")
    lines.append(f"To keep TP rate >=50%: {sweep_param} <= {t50[0]:,.2f} => TP={t50_tp}, FP={t50_fp}, FN={t50_fn}, TN={t50_tn}, precision={_prec(t50_tp, t50_fp)}%.")
    last = sweep[-1]
    last_fn = total_sars - last[1]; last_tn = total_fps - last[2]
    lines.append(f"At the highest value ({last[0]:,.2f}): TP={last[1]}, FP={last[2]}, FN={last_fn}, TN={last_tn}, precision={_prec(last[1], last[2])}%.")
    lines.append("=== END RULE SWEEP ===")
    lines.append("(Detailed sweep table shown in the chart below.)")

    return "\n".join(lines)


# ── rule_2d_sweep ──────────────────────────────────────────────────────────────

def _get_mask(df, sp, val):
    """Return boolean mask for a single sweep param at a given value."""
    direction = sp["direction"]
    if direction == "gte":
        return df[sp["col"]] >= val
    elif direction == "abs_lte":
        return (df[sp["col"]] - 1.0).abs() <= val
    elif direction == "window":
        col = f"max_rolling_{int(val)}d"
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col] >= df["trigger_amt"]   # fires if rolling sum >= current floor
    return pd.Series(False, index=df.index)


def _sweep_points(df, sp, n_steps=15):
    """Return a list of sweep values for a param.

    For dollar-amount columns: generates nice round steps centered on the current
    operational threshold (e.g. current=$20K → 5K,10K,15K,20K,25K,30K,35K,40K).
    For z-score columns: fixed 0–10 integer range.
    For integer-axis columns (age, count): integer steps.
    For fixed-value params (ratio_tolerance, time_window): use catalogue values list.
    """
    direction = sp["direction"]
    if direction == "window":
        return sp["values"]
    # Fixed value list (e.g. ratio_tolerance percentage steps)
    if "values" in sp:
        pts = [float(v) for v in sp["values"]]
        cur = float(sp["current"])
        if cur not in pts:
            pts.append(cur)
        return sorted(set(pts))

    col = sp["col"]
    raw_vals = df[col].dropna()
    if len(raw_vals) == 0:
        return []

    cur = float(sp["current"])

    # z-score: fixed 0–10 integer range
    if col in ("z_score",):
        i_max = 10
        total_ints = i_max + 1
        step = max(1, math.ceil(total_ints / (n_steps + 1)))
        pts = list(range(0, i_max + 1, step))
        cur_i = int(round(cur))
        if cur_i not in pts:
            pts.append(cur_i)
        return sorted(set(pts))

    # Integer-axis params (age, count): integer steps centered on current
    if sp.get("integer_axis"):
        hard_min = int(sp.get("sweep_min", 1))
        i_min = max(hard_min, int(round(cur)) - n_steps // 2)
        i_max = int(round(cur)) + n_steps // 2
        total_ints = i_max - i_min + 1
        step = max(1, math.ceil(total_ints / (n_steps + 1)))
        pts = list(range(i_min, i_max + 1, step))
        cur_i = int(round(cur))
        if cur_i not in pts:
            pts.append(cur_i)
        return sorted(set(pts))

    # Dollar-amount columns: nice round steps centered on current threshold.
    # Step size is chosen to give ~8 steps either side of current.
    def _nice_step(value, half_steps=4):
        """Round step giving ~half_steps increments on each side of value."""
        raw = value / half_steps
        if raw <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(raw))
        n = raw / mag
        if n < 1.5:
            nice = 1
        elif n < 3.5:
            nice = 2
        elif n < 7.5:
            nice = 5
        else:
            nice = 10
        return nice * mag

    # If explicit sweep_min AND sweep_max are both given, distribute evenly between them.
    # Round step UP (not down) so the grid stays compact (~6–9 steps).
    if "sweep_min" in sp and "sweep_max" in sp:
        t_min = float(sp["sweep_min"])
        t_max = float(sp["sweep_max"])
        target = max(5, min(n_steps, 9))
        raw_step = (t_max - t_min) / (target - 1)
        mag = 10 ** math.floor(math.log10(raw_step))
        n = raw_step / mag
        if n <= 1.5:
            nice = 2    # round up from 1×
        elif n <= 3.5:
            nice = 5    # round up from 2×
        elif n <= 7.5:
            nice = 10   # round up from 5×
        else:
            nice = 10
        step = nice * mag
        n_pts = int(round((t_max - t_min) / step)) + 1
        pts = [int(round(t_min + i * step)) for i in range(n_pts)]
        if int(round(cur)) not in pts and t_min <= cur <= t_max:
            pts.append(int(round(cur)))
        return sorted(set(pts))

    step = _nice_step(cur)
    hard_min = float(sp.get("sweep_min", step))
    t_min = max(hard_min, cur - 4 * step)
    t_max = cur + 4 * step

    n_pts = int(round((t_max - t_min) / step)) + 1
    pts = [int(round(t_min + i * step)) for i in range(n_pts)]
    if int(round(cur)) not in pts:
        pts.append(int(round(cur)))
    return sorted(set(pts))


def _fmt_cur(sp):
    """Format a param's current value cleanly: integers as int, pct as %, floats strip trailing zeros."""
    v = sp["current"]
    if sp.get("format_pct"):
        pct = v * 100
        return f"{pct:g}%"
    if sp.get("integer_axis") or (isinstance(v, (int, float)) and v == int(v)):
        return str(int(v))
    return str(v)


def _fmt_v(v, sp):
    """Format any sweep value for PRE-COMPUTED text output."""
    if sp.get("format_pct"):
        pct = v * 100
        return f"{pct:g}%"
    if sp.get("integer_axis"):
        return str(int(round(v)))
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.2f}"


def compute_rule_2d_sweep(df, risk_factor_keyword, param1=None, param2=None):
    """
    2D grid sweep: for each (param1_val, param2_val) combination, count
    SAR caught and FP remaining when both conditions are applied simultaneously.

    Returns (result_text, grid_dict) where grid_dict has keys:
      p1_vals, p2_vals, sar_grid (list of lists), fp_grid, total_sars, p1_label, p2_label
    """
    if df is None:
        return "Rule sweep data not loaded.", None

    rf_name, entry = _match_rule(risk_factor_keyword)
    if entry is None:
        return f"No rule matched '{risk_factor_keyword}'. Call list_rules to see available rules.", None

    params = entry["sweep_params"]
    keys   = list(params.keys())

    # Resolve param1 / param2
    default_2d = entry.get("default_2d", (keys[0], keys[1] if len(keys) > 1 else keys[0]))
    if param1 is None:
        param1 = default_2d[0]
    if param2 is None:
        param2 = default_2d[1]

    # Normalize param names: model may use spaces instead of underscores or pass integers
    if param1 is not None:
        param1 = str(param1).replace(" ", "_").replace("-", "_")
    if param2 is not None:
        param2 = str(param2).replace(" ", "_").replace("-", "_")

    if param1 not in params:
        return f"Unknown sweep_param_1 '{param1}'. Valid: {keys}", None
    if param2 not in params:
        return f"Unknown sweep_param_2 '{param2}'. Valid: {keys}", None
    if param1 == param2:
        return "sweep_param_1 and sweep_param_2 must be different.", None

    sp1 = params[param1]
    sp2 = params[param2]

    rule_df = df[df["risk_factor"] == rf_name].copy()
    # For window direction, need trigger_amt to be present
    needed = []
    if sp1["direction"] != "window" and sp1["col"]:
        needed.append(sp1["col"])
    if sp2["direction"] != "window" and sp2["col"]:
        needed.append(sp2["col"])
    needed.append("is_sar")
    known = rule_df.dropna(subset=needed).copy()

    total_sars = int((known["is_sar"] == 1).sum())
    total_fps  = int((known["is_sar"] == 0).sum())

    if total_sars == 0:
        return f"No SAR customers in {rf_name}. Cannot build 2D grid.", None

    p1_vals = _sweep_points(known, sp1)
    p2_vals = _sweep_points(known, sp2)

    # Build grids: rows = p1, cols = p2
    sar_grid, fp_grid = [], []
    for v1 in p1_vals:
        sar_row, fp_row = [], []
        m1 = _get_mask(known, sp1, v1)
        for v2 in p2_vals:
            m2 = _get_mask(known, sp2, v2)
            both = m1 & m2
            sar_row.append(int((both & (known["is_sar"] == 1)).sum()))
            fp_row.append(int((both & (known["is_sar"] == 0)).sum()))
        sar_grid.append(sar_row)
        fp_grid.append(fp_row)

    grid_dict = {
        "p1_vals":       p1_vals,
        "p2_vals":       p2_vals,
        "sar_grid":      sar_grid,
        "fp_grid":       fp_grid,
        "total_sars":    total_sars,
        "total_fps":     total_fps,
        "p1_label":      sp1["label"],
        "p2_label":      sp2["label"],
        "p1_current":    sp1["current"],
        "p2_current":    sp2["current"],
        "p1_format_pct": sp1.get("format_pct", False),
        "p2_format_pct": sp2.get("format_pct", False),
        "rf_name":       rf_name,
        "param1":        param1,
        "param2":        param2,
    }

    # Text summary — compute current condition directly using actual thresholds
    # (not via nearest sweep point, since current may be outside the sweep range)
    m1_cur = _get_mask(known, sp1, float(sp1["current"]))
    m2_cur = _get_mask(known, sp2, float(sp2["current"]))
    cur_both = m1_cur & m2_cur
    cur_sar = int((cur_both & (known["is_sar"] == 1)).sum())
    cur_fp  = int((cur_both & (known["is_sar"] == 0)).sum())
    cur_pct = round(100 * cur_sar / total_sars, 1)
    cur_fn  = total_sars - cur_sar
    cur_tn  = total_fps  - cur_fp

    lines = [
        "=== PRE-COMPUTED 2D SWEEP (copy this verbatim, do not alter numbers) ===",
        f"Rule: {rf_name}",
        f"Axis 1 ({param1}): {sp1['desc']}",
        f"Axis 2 ({param2}): {sp2['desc']}",
        f"Grid: {len(p1_vals)} x {len(p2_vals)} = {len(p1_vals)*len(p2_vals)} combinations",
        f"SAR pool: {total_sars}",
        f"Non-SAR pool: {total_fps}",
        "",
        f"At current condition ({param1}={_fmt_cur(sp1)}, {param2}={_fmt_cur(sp2)}): "
        f"TP={cur_sar}, FP={cur_fp}, FN={cur_fn}, TN={cur_tn} (TP rate={cur_pct}%).",
    ]

    # Best FP reduction cell: lowest FP count with TP rate >= 50%
    # (operationally: biggest workload reduction while keeping half of SARs)
    best_fp_val, best_v1, best_v2, best_tp, best_fp = float("inf"), None, None, 0, 0
    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            tp, fp = sar_grid[i][j], fp_grid[i][j]
            if tp < total_sars * 0.5:
                continue
            if fp < best_fp_val:
                best_fp_val, best_v1, best_v2, best_tp, best_fp = fp, v1, v2, tp, fp
    if best_v1 is not None and best_v1 != p1_vals[0]:
        best_fn = total_sars - best_tp
        best_tn = total_fps - best_fp
        best_pct = round(100 * best_tp / total_sars, 1)
        best_prec = round(100 * best_tp / (best_tp + best_fp), 1) if (best_tp + best_fp) > 0 else 0
        lines.append(
            f"Best FP reduction (TP rate >=50%): {param1}={_fmt_v(best_v1, sp1)}, {param2}={_fmt_v(best_v2, sp2)} "
            f"=> TP={best_tp}, FP={best_fp}, FN={best_fn}, TN={best_tn}, TP rate={best_pct}%, precision={best_prec}%."
        )

    lines.append("=== END 2D SWEEP ===")
    lines.append("(Heatmap shown in the chart below.)")

    return "\n".join(lines), grid_dict


def compute_2d_drilldown(df, risk_factor_keyword, param1, p1_val, param2, p2_val):
    """
    For a specific (p1_val, p2_val) cell in the 2D grid, return the customer-level breakdown:
      - tp_df : SAR customers who ARE alerted (true positives)
      - fp_df : non-SAR customers who ARE alerted (false positives)
      - fn_df : SAR customers who are NOT alerted (false negatives — missed)
      - tn_df : non-SAR customers who are NOT alerted (true negatives)
    Also returns the column names for param1 / param2 for display.
    Returns (tp_df, fp_df, fn_df, tn_df, col1, col2) or (None,)*6 on error.
    """
    rf_name, entry = _match_rule(risk_factor_keyword)
    if entry is None:
        return (None,) * 6

    params = entry["sweep_params"]
    if param1 not in params or param2 not in params:
        return (None,) * 6

    sp1 = params[param1]
    sp2 = params[param2]
    col1 = sp1.get("col", param1)
    col2 = sp2.get("col", param2)

    rule_df = df[df["risk_factor"] == rf_name].copy()
    needed  = [c for c in [sp1.get("col"), sp2.get("col"), "is_sar"] if c]
    known   = rule_df.dropna(subset=needed).copy()

    m1      = _get_mask(known, sp1, float(p1_val))
    m2      = _get_mask(known, sp2, float(p2_val))
    alerted = m1 & m2
    is_sar  = known["is_sar"] == 1

    tp_df = known[ alerted &  is_sar].copy()
    fp_df = known[ alerted & ~is_sar].copy()
    fn_df = known[~alerted &  is_sar].copy()
    tn_df = known[~alerted & ~is_sar].copy()

    return tp_df, fp_df, fn_df, tn_df, col1, col2
