"""
column_mapper.py — Maps user CSV column names to ARIA canonical names.

Resolution order:
  1. Exact match — column already named correctly, pass through
  2. column_map.yaml — explicit user mapping (highest priority override)
  3. Fuzzy match — difflib against known aliases, logs what was auto-resolved
  4. Warn and leave unmapped columns as-is

Usage:
    from column_mapper import normalize_columns
    df = normalize_columns(df)

To configure for your data, edit column_map.yaml in the repo root.
"""

import difflib
import os
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical column names and their known aliases
# ---------------------------------------------------------------------------

_CANONICAL_ALIASES: dict[str, list[str]] = {
    # Identifiers — must be listed first so fuzzy match never steals customer_id
    "customer_id":       ["customer_id", "cust_id", "client_id"],
    "account_id":        ["account_id", "acct_id", "acc_id"],

    # Rule analysis
    "risk_factor":       ["risk_factor", "rule_name", "rule", "alert_type", "risk factor",
                          "aml_rule", "monitoring_rule"],
    "is_sar":            ["is_sar", "sar", "sar_flag", "is_suspicious", "filed_sar",
                          "sar_filed", "suspicious_activity"],

    # Customer / segment
    "customer_type":     ["customer_type", "cust_type", "entity_type", "customer_segment",
                          "segment_type", "client_type"],
    "dynamic_segment":   ["dynamic_segment", "smart_segment_id", "segment_id", "cluster_id",
                          "segment", "cluster"],

    # Threshold tuning columns
    "TRXN_AMT_MONTHLY":  ["trxn_amt_monthly", "monthly_txn_amount", "monthly_transaction_amount",
                          "trxn_amt_month", "monthly_amount", "total_monthly_txn"],
    "AVG_TRXN_AMT":      ["avg_trxn_amt", "avg_transaction_amount", "average_transaction_amount",
                          "avg_txn_amt", "mean_trxn_amt", "avg_trxn_amount"],
    "AVG_TRXNS_WEEK":    ["avg_trxns_week", "avg_weekly_transactions", "weekly_txn_count",
                          "avg_num_trxns", "avg_weekly_trxn_count", "weekly_transactions"],

    # Alert flags
    "alerts":            ["alerts", "alert_flag", "is_alert", "has_alert", "alerted"],
    "false_positives":   ["false_positives", "fp", "is_fp", "false_positive",
                          "fp_flag", "false_pos"],

    # OFAC / customer profile
    "citizenship":       ["citizenship", "country_of_citizenship", "nationality",
                          "citizen_country", "country"],
    "ofac":              ["ofac", "ofac_flag", "ofac_hit", "sanctions_hit",
                          "is_ofac", "ofac_match"],
}

_FUZZY_THRESHOLD = 0.80  # minimum similarity score to auto-accept a fuzzy match

# ---------------------------------------------------------------------------
# Load column_map.yaml (explicit overrides)
# ---------------------------------------------------------------------------

def _load_yaml_map() -> dict[str, str]:
    """Load column_map.yaml from repo root. Returns {} if not found."""
    yaml_path = Path(__file__).parent / "column_map.yaml"
    if not yaml_path.exists():
        return {}
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # yaml format: canonical_name: your_column_name
        # invert to: your_column_name -> canonical_name
        return {str(v).strip(): str(k).strip() for k, v in data.items()
                if v and not str(k).startswith("#")}
    except Exception as e:
        print(f"[column_mapper] Warning: could not load column_map.yaml: {e}")
        return {}


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy_match(col: str, canonical_aliases: dict[str, list[str]]) -> str | None:
    """Return canonical name if col fuzzy-matches any alias above threshold."""
    col_lower = col.lower().strip()
    best_score = 0.0
    best_canonical = None

    for canonical, aliases in canonical_aliases.items():
        for alias in aliases:
            score = difflib.SequenceMatcher(None, col_lower, alias.lower()).ratio()
            if score > best_score:
                best_score = score
                best_canonical = canonical

    if best_score >= _FUZZY_THRESHOLD and best_canonical:
        return best_canonical, best_score
    return None, 0.0


# ---------------------------------------------------------------------------
# Main normalization function
# ---------------------------------------------------------------------------

def normalize_columns(df, verbose: bool = True):
    """
    Normalize DataFrame column names to ARIA canonical names.

    Parameters
    ----------
    df      : pandas DataFrame
    verbose : print resolution log (default True)

    Returns
    -------
    DataFrame with renamed columns (copy)
    """
    import pandas as pd

    df = df.copy()
    yaml_map = _load_yaml_map()

    # Build exact-match lookup: alias_lower -> canonical
    exact_lookup: dict[str, str] = {}
    for canonical, aliases in _CANONICAL_ALIASES.items():
        for alias in aliases:
            exact_lookup[alias.lower()] = canonical

    rename_map: dict[str, str] = {}

    for col in df.columns:
        col_lower = col.lower().strip()

        # 1. Already canonical — no rename needed
        if col in _CANONICAL_ALIASES:
            continue

        # 2. Explicit yaml override (highest priority)
        if col in yaml_map:
            target = yaml_map[col]
            if verbose:
                print(f"[column_mapper] yaml map:   '{col}' -> '{target}'")
            rename_map[col] = target
            continue

        # 3. Exact alias match
        if col_lower in exact_lookup:
            target = exact_lookup[col_lower]
            if target != col:
                if verbose:
                    print(f"[column_mapper] exact map:  '{col}' -> '{target}'")
                rename_map[col] = target
            continue

        # 4. Fuzzy match
        target, score = _fuzzy_match(col, _CANONICAL_ALIASES)
        if target and target != col:
            if verbose:
                print(f"[column_mapper] fuzzy map:  '{col}' -> '{target}' (score={score:.2f})")
            rename_map[col] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    return df
