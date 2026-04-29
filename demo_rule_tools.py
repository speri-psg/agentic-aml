"""
demo_rule_tools.py — Show rule SAR sweep tool outputs without needing the LLM.

Calls tool_executor() directly for each new rule tool and prints results.

Run:
    python demo_rule_tools.py
"""

import os, sys
os.environ.setdefault("OLLAMA_MODEL",    "demo")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:1234/v1")

# Suppress Dash/Flask startup noise
import logging
logging.disable(logging.CRITICAL)

# ── Bootstrap the same data loading as application.py ────────────────────────
import pandas as pd
from config import SAR_CSV, DS_CSV, ALERTS_CSV
import lambda_rule_analysis

print("Loading data...")
DF_SAR = pd.read_csv(SAR_CSV)
DF_RULE_SWEEP = lambda_rule_analysis.load_rule_sweep_data()

if DF_RULE_SWEEP is None:
    print("ERROR: docs/rule_sweep_data.csv not found.")
    print("Run:  python prepare_rule_sweep_data.py")
    sys.exit(1)

print(f"  Rule sweep data: {len(DF_RULE_SWEEP)} rows, {DF_RULE_SWEEP['risk_factor'].nunique()} rules")
print()

DIVIDER = "=" * 80

def show(title, result):
    print(DIVIDER)
    print(f"  TOOL CALL: {title}")
    print(DIVIDER)
    print(result)
    print()


# ── 1. list_rules ─────────────────────────────────────────────────────────────
show(
    "list_rules()",
    lambda_rule_analysis.list_rules_text(DF_RULE_SWEEP)
)

# ── 2. Activity Deviation — sweep the floor amount ───────────────────────────
show(
    "rule_sar_backtest(risk_factor='Activity Deviation', sweep_param='floor_amount')",
    lambda_rule_analysis.compute_rule_sar_sweep(DF_RULE_SWEEP, "Activity Deviation", "floor_amount")
)

# ── 3. Activity Deviation — sweep the z-score multiplier ─────────────────────
show(
    "rule_sar_backtest(risk_factor='Activity Deviation', sweep_param='z_threshold')",
    lambda_rule_analysis.compute_rule_sar_sweep(DF_RULE_SWEEP, "Activity Deviation", "z_threshold")
)

# ── 4. Elder Abuse — sweep the dollar floor ──────────────────────────────────
show(
    "rule_sar_backtest(risk_factor='Elder Abuse', sweep_param='floor_amount')",
    lambda_rule_analysis.compute_rule_sar_sweep(DF_RULE_SWEEP, "Elder Abuse", "floor_amount")
)

# ── 5. Elder Abuse — sweep the age threshold ─────────────────────────────────
show(
    "rule_sar_backtest(risk_factor='Elder Abuse', sweep_param='age_threshold')",
    lambda_rule_analysis.compute_rule_sar_sweep(DF_RULE_SWEEP, "Elder Abuse", "age_threshold")
)

# ── 6. Velocity Single — sweep the pair total ────────────────────────────────
show(
    "rule_sar_backtest(risk_factor='Velocity Single', sweep_param='pair_total')",
    lambda_rule_analysis.compute_rule_sar_sweep(DF_RULE_SWEEP, "Velocity Single", "pair_total")
)

# ── 7. Detect Excessive — sweep the 5-day sum threshold ──────────────────────
show(
    "rule_sar_backtest(risk_factor='Detect Excessive', sweep_param='floor_amount')",
    lambda_rule_analysis.compute_rule_sar_sweep(DF_RULE_SWEEP, "Detect Excessive", "floor_amount")
)

print(DIVIDER)
print("Done. These are the exact text blocks the model receives and copies into chat.")
print(DIVIDER)
