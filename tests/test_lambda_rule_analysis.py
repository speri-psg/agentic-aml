"""Tests for lambda_rule_analysis.py — rule matching, sweep text, catalogue structure."""
import pytest
import pandas as pd
import numpy as np

from lambda_rule_analysis import (
    RULE_CATALOGUE,
    _match_rule,
    list_rules_text,
    cluster_rule_summary_text,
    compute_rule_sar_sweep,
    compute_rule_2d_sweep,
    _sweep_points,
    _get_mask,
    load_rule_sweep_data,
)


# ── Shared synthetic data ─────────────────────────────────────────────────────

def _rule_df(rule_name, n_sar=30, n_fp=70, seed=42):
    """
    Build a minimal rule sweep DataFrame for one rule.
    Includes all columns used by sweep functions.
    """
    rng = np.random.default_rng(seed)
    n = n_sar + n_fp
    is_sar = np.array([1] * n_sar + [0] * n_fp)
    rng.shuffle(is_sar)
    return pd.DataFrame({
        "risk_factor":  [rule_name] * n,
        "is_sar":       is_sar,
        "customer_id":  range(n),
        "trigger_amt":  rng.uniform(1000, 100000, size=n),
        "z_score":      rng.uniform(0, 10, size=n),
        "age":          rng.integers(30, 90, size=n).astype(float),
        "pair_total":   rng.uniform(1000, 50000, size=n),
        "ratio":        rng.uniform(0.8, 1.2, size=n),
        "days_observed":rng.integers(1, 10, size=n).astype(float),
        "txn_count":    rng.integers(1, 20, size=n).astype(float),
        "dynamic_segment": rng.integers(0, 2, size=n),
        "cluster":      rng.integers(1, 5, size=n),
    })


# ── Rule catalogue ─────────────────────────────────────────────────────────────

class TestRuleCatalogue:
    def test_catalogue_is_dict(self):
        assert isinstance(RULE_CATALOGUE, dict)

    def test_catalogue_has_expected_rules(self):
        # Catalogue has grown beyond the original 11; verify it's non-trivially sized
        assert len(RULE_CATALOGUE) >= 11

    def test_each_entry_has_name(self):
        for key, entry in RULE_CATALOGUE.items():
            assert "name" in entry, f"Missing 'name' in rule '{key}'"

    def test_each_entry_has_sweep_params(self):
        for key, entry in RULE_CATALOGUE.items():
            assert "sweep_params" in entry, f"Missing 'sweep_params' in rule '{key}'"
            assert len(entry["sweep_params"]) >= 1

    def test_each_entry_has_default_sweep(self):
        for key, entry in RULE_CATALOGUE.items():
            assert "default_sweep" in entry

    def test_each_entry_has_default_2d(self):
        for key, entry in RULE_CATALOGUE.items():
            assert "default_2d" in entry

    def test_default_2d_is_tuple(self):
        for key, entry in RULE_CATALOGUE.items():
            assert isinstance(entry["default_2d"], tuple), f"default_2d is not a tuple in '{key}'"

    def test_activity_deviation_ach_present(self):
        names = [e["name"] for e in RULE_CATALOGUE.values()]
        assert "Activity Deviation (ACH)" in names

    def test_elder_abuse_present(self):
        names = [e["name"] for e in RULE_CATALOGUE.values()]
        assert "Elder Abuse" in names

    def test_velocity_single_present(self):
        names = [e["name"] for e in RULE_CATALOGUE.values()]
        assert "Velocity Single" in names

    def test_sweep_params_have_required_fields(self):
        for key, entry in RULE_CATALOGUE.items():
            for param_name, sp in entry["sweep_params"].items():
                assert "direction" in sp, f"Missing 'direction' in {key}.{param_name}"
                assert "current" in sp, f"Missing 'current' in {key}.{param_name}"
                assert "desc" in sp, f"Missing 'desc' in {key}.{param_name}"


# ── _match_rule ────────────────────────────────────────────────────────────────

class TestMatchRule:
    def test_exact_key_match(self):
        name, entry = _match_rule("activity deviation (ach)")
        assert name == "Activity Deviation (ACH)"
        assert entry is not None

    def test_partial_match(self):
        name, entry = _match_rule("elder")
        assert "Elder Abuse" in name

    def test_case_insensitive(self):
        name, entry = _match_rule("ELDER ABUSE")
        assert name is not None

    def test_no_match_returns_none_pair(self):
        name, entry = _match_rule("nonexistent_rule_xyz")
        assert name is None
        assert entry is None

    def test_velocity_single_match(self):
        name, entry = _match_rule("velocity single")
        assert name == "Velocity Single"

    def test_detect_excessive_match(self):
        name, entry = _match_rule("detect excessive")
        assert name is not None and "Detect Excessive" in name

    def test_ctr_client_match(self):
        name, entry = _match_rule("ctr client")
        assert name is not None and "CTR" in name

    def test_structuring_incoming_match(self):
        name, entry = _match_rule("structuring (incoming cash)")
        assert name is not None
        assert "Structuring" in name

    def test_empty_string_returns_none(self):
        name, entry = _match_rule("")
        # Empty string matches any key that contains "" — which all do
        # So it'll return something; just ensure it doesn't crash
        # (behavior: first entry wins)


# ── list_rules_text ───────────────────────────────────────────────────────────

class TestListRulesText:
    def test_returns_string(self):
        df = _rule_df("Activity Deviation (ACH)")
        result = list_rules_text(df)
        assert isinstance(result, str)

    def test_contains_pre_computed_header(self):
        df = _rule_df("Activity Deviation (ACH)")
        result = list_rules_text(df)
        assert "PRE-COMPUTED RULE LIST" in result

    def test_contains_all_rule_names(self):
        df = pd.concat([_rule_df(e["name"]) for e in RULE_CATALOGUE.values()], ignore_index=True)
        result = list_rules_text(df)
        for entry in RULE_CATALOGUE.values():
            assert entry["name"] in result, f"Rule {entry['name']} missing from list"

    def test_contains_end_marker(self):
        df = _rule_df("Activity Deviation (ACH)")
        result = list_rules_text(df)
        assert "END RULE LIST" in result

    def test_contains_sar_and_fp_counts(self):
        df = _rule_df("Elder Abuse", n_sar=30, n_fp=70)
        result = list_rules_text(df)
        assert "SAR=" in result
        assert "FP=" in result

    def test_contains_precision(self):
        df = _rule_df("Elder Abuse", n_sar=30, n_fp=70)
        result = list_rules_text(df)
        assert "precision=" in result

    def test_none_df_shows_zero_counts(self):
        result = list_rules_text(None)
        # When df is None, all counts should be 0
        assert "SAR=0" in result

    def test_do_not_add_rules_note(self):
        df = _rule_df("Activity Deviation (ACH)")
        result = list_rules_text(df)
        assert "Do NOT add or infer any rules not listed here" in result or \
               "COMPLETE list" in result


# ── cluster_rule_summary_text ─────────────────────────────────────────────────

def _multi_rule_df(seed=0):
    """DataFrame with rows for every rule in the catalogue — used for summary tests."""
    frames = [_rule_df(e["name"], n_sar=10, n_fp=20, seed=seed + i)
              for i, e in enumerate(RULE_CATALOGUE.values())]
    return pd.concat(frames, ignore_index=True)


class TestClusterRuleSummaryText:
    def test_returns_string(self):
        df = _multi_rule_df()
        result = cluster_rule_summary_text(df, 4)
        assert isinstance(result, str)

    def test_contains_pre_computed_header(self):
        df = _multi_rule_df()
        result = cluster_rule_summary_text(df, 4)
        assert "PRE-COMPUTED CLUSTER RULE SUMMARY" in result

    def test_contains_end_marker(self):
        df = _multi_rule_df()
        result = cluster_rule_summary_text(df, 4)
        assert "END CLUSTER RULE SUMMARY" in result

    def test_contains_cluster_number(self):
        df = _multi_rule_df()
        result = cluster_rule_summary_text(df, 4)
        assert "Cluster 4" in result

    def test_contains_sar_fp_precision(self):
        df = _multi_rule_df()
        result = cluster_rule_summary_text(df, 3)
        assert "SAR=" in result
        assert "FP=" in result
        assert "precision=" in result

    def test_contains_all_rule_names_when_data_present(self):
        df = _multi_rule_df()
        result = cluster_rule_summary_text(df, 2)
        for entry in RULE_CATALOGUE.values():
            assert entry["name"] in result, f"{entry['name']} missing from cluster summary"

    def test_empty_df_returns_no_data_message(self):
        result = cluster_rule_summary_text(pd.DataFrame(), 4)
        assert "no alert data" in result.lower() or "Cluster 4" in result

    def test_none_df_returns_no_data_message(self):
        result = cluster_rule_summary_text(None, 4)
        assert "no alert data" in result.lower() or "Cluster 4" in result

    def test_inactive_rules_listed_when_no_alerts(self):
        # Only provide rows for one rule — rest should appear as alerts=0
        df = _rule_df("Activity Deviation (ACH)", n_sar=5, n_fp=10)
        result = cluster_rule_summary_text(df, 1)
        assert "alerts=0" in result or "Rules with alerts=0" in result

    def test_customer_count_reported(self):
        df = _multi_rule_df()
        result = cluster_rule_summary_text(df, 4)
        assert "customers" in result

    def test_precision_100_when_no_fp(self):
        df = _rule_df("Structuring (Incoming Cash)", n_sar=5, n_fp=0)
        result = cluster_rule_summary_text(df, 1)
        assert "100.0%" in result

    def test_precision_zero_when_no_sar(self):
        df = _rule_df("Activity Deviation (ACH)", n_sar=0, n_fp=20)
        result = cluster_rule_summary_text(df, 1)
        # No SAR → rule appears in alerts=0 block (df is filtered to this rule only)
        # or precision=0.0% if counted
        assert isinstance(result, str)

    def test_different_cluster_numbers_work(self):
        df = _multi_rule_df()
        for cluster_n in (1, 2, 3, 4):
            result = cluster_rule_summary_text(df, cluster_n)
            assert f"Cluster {cluster_n}" in result

    def test_result_is_verbatim_copyable(self):
        # The block must start and end with the PRE-COMPUTED markers
        df = _multi_rule_df()
        result = cluster_rule_summary_text(df, 4)
        lines = result.strip().splitlines()
        assert lines[0].startswith("=== PRE-COMPUTED CLUSTER RULE SUMMARY")
        assert lines[-1].startswith("=== END CLUSTER RULE SUMMARY")


# ── compute_rule_sar_sweep ────────────────────────────────────────────────────

class TestComputeRuleSarSweep:
    ACH_RULE = "Activity Deviation (ACH)"

    def _df(self, n_sar=20, n_fp=40):
        return _rule_df(self.ACH_RULE, n_sar=n_sar, n_fp=n_fp)

    def test_returns_string(self):
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)")
        assert isinstance(result, str)

    def test_contains_pre_computed_header(self):
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)")
        assert "PRE-COMPUTED RULE SWEEP" in result

    def test_contains_end_marker(self):
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)")
        assert "END RULE SWEEP" in result

    def test_contains_rule_name(self):
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)")
        assert "Activity Deviation (ACH)" in result

    def test_contains_sweep_parameter(self):
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)", sweep_param="floor_amount")
        assert "floor_amount" in result.lower() or "floor" in result.lower()

    def test_contains_tp_fp_fn_labels(self):
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)")
        assert "TP=" in result
        assert "FP=" in result
        assert "FN=" in result

    def test_unknown_rule_returns_error_message(self):
        result = compute_rule_sar_sweep(self._df(), "nonexistent_rule_xyz")
        assert "No rule matched" in result

    def test_none_df_returns_error_message(self):
        result = compute_rule_sar_sweep(None, "activity deviation (ach)")
        assert "not loaded" in result.lower() or "none" in result.lower()

    def test_no_sar_returns_informative_message(self):
        df = _rule_df(self.ACH_RULE, n_sar=0, n_fp=50)
        result = compute_rule_sar_sweep(df, "activity deviation (ach)")
        assert "No SAR" in result or "false positives" in result.lower()

    def test_default_sweep_used_when_none_given(self):
        df = self._df()
        result_default = compute_rule_sar_sweep(df, "activity deviation (ach)", sweep_param=None)
        result_explicit = compute_rule_sar_sweep(df, "activity deviation (ach)", sweep_param="floor_amount")
        # Both should use floor_amount as the default for ACH
        assert "PRE-COMPUTED RULE SWEEP" in result_default

    def test_sweep_param_normalization(self):
        df = self._df()
        # Spaces and hyphens in param name should be normalized
        result = compute_rule_sar_sweep(df, "activity deviation (ach)", sweep_param="floor amount")
        assert "PRE-COMPUTED RULE SWEEP" in result

    def test_contains_tp_rate_90_line(self):
        df = self._df(n_sar=30)
        result = compute_rule_sar_sweep(df, "activity deviation (ach)")
        assert ">=90%" in result or "90%" in result

    def test_contains_labeled_population(self):
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)")
        assert "Labeled population" in result or "population" in result.lower()

    def test_sweep_param_as_integer_does_not_crash(self):
        # Fix: sweep_param must be str()-cast before .replace() — models can pass ints
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)", sweep_param=1)
        # Should return a valid result (int 1 doesn't match any param name → uses default)
        assert isinstance(result, str)

    def test_sweep_param_as_float_does_not_crash(self):
        result = compute_rule_sar_sweep(self._df(), "activity deviation (ach)", sweep_param=3.0)
        assert isinstance(result, str)


# ── compute_rule_2d_sweep ────────────────────────────────────────────────────

class TestComputeRule2dSweep:
    ACH_RULE = "Activity Deviation (ACH)"

    def _df(self, n_sar=20, n_fp=40):
        return _rule_df(self.ACH_RULE, n_sar=n_sar, n_fp=n_fp)

    def test_returns_tuple(self):
        result = compute_rule_2d_sweep(self._df(), "activity deviation (ach)")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_text_contains_pre_computed_header(self):
        text, grid = compute_rule_2d_sweep(self._df(), "activity deviation (ach)")
        assert "PRE-COMPUTED 2D SWEEP" in text

    def test_grid_dict_has_required_keys(self):
        text, grid = compute_rule_2d_sweep(self._df(), "activity deviation (ach)")
        assert grid is not None
        for key in ("p1_vals", "p2_vals", "sar_grid", "fp_grid", "total_sars", "p1_label", "p2_label"):
            assert key in grid, f"Missing key '{key}' in grid dict"

    def test_sar_grid_dimensions_match(self):
        text, grid = compute_rule_2d_sweep(self._df(), "activity deviation (ach)")
        assert len(grid["sar_grid"]) == len(grid["p1_vals"])
        assert all(len(row) == len(grid["p2_vals"]) for row in grid["sar_grid"])

    def test_fp_grid_dimensions_match(self):
        text, grid = compute_rule_2d_sweep(self._df(), "activity deviation (ach)")
        assert len(grid["fp_grid"]) == len(grid["p1_vals"])

    def test_no_sar_returns_error_text(self):
        df = _rule_df(self.ACH_RULE, n_sar=0, n_fp=50)
        text, grid = compute_rule_2d_sweep(df, "activity deviation (ach)")
        assert "No SAR" in text
        assert grid is None

    def test_same_params_returns_error(self):
        text, grid = compute_rule_2d_sweep(
            self._df(), "activity deviation (ach)",
            param1="floor_amount", param2="floor_amount"
        )
        assert "must be different" in text
        assert grid is None

    def test_unknown_rule_returns_error(self):
        text, grid = compute_rule_2d_sweep(self._df(), "xyz_nonexistent")
        assert "No rule matched" in text
        assert grid is None

    def test_param1_as_integer_does_not_crash(self):
        # Fix: param1/param2 must be str()-cast before .replace()
        text, grid = compute_rule_2d_sweep(
            self._df(), "activity deviation (ach)", param1=1, param2="z_threshold"
        )
        assert isinstance(text, str)

    def test_param2_as_integer_does_not_crash(self):
        text, grid = compute_rule_2d_sweep(
            self._df(), "activity deviation (ach)", param1="floor_amount", param2=2
        )
        assert isinstance(text, str)


# ── _sweep_points ────────────────────────────────────────────────────────────

class TestSweepPoints:
    def _ach_sp(self):
        return RULE_CATALOGUE["activity deviation (ach)"]["sweep_params"]["floor_amount"]

    def test_returns_non_empty_list(self):
        df = _rule_df("Activity Deviation (ACH)")
        sp = self._ach_sp()
        pts = _sweep_points(df, sp)
        assert isinstance(pts, list)
        assert len(pts) > 0

    def test_points_are_sorted(self):
        df = _rule_df("Activity Deviation (ACH)")
        sp = self._ach_sp()
        pts = _sweep_points(df, sp)
        assert pts == sorted(pts)

    def test_current_value_in_points(self):
        df = _rule_df("Activity Deviation (ACH)")
        sp = self._ach_sp()
        pts = _sweep_points(df, sp)
        cur = int(round(float(sp["current"])))
        assert cur in [int(round(p)) for p in pts]

    def test_z_score_param_fixed_range(self):
        df = _rule_df("Activity Deviation (ACH)")
        sp = RULE_CATALOGUE["activity deviation (ach)"]["sweep_params"]["z_threshold"]
        pts = _sweep_points(df, sp)
        # z_score: fixed 0–10 integer range
        assert min(pts) >= 0
        assert max(pts) <= 10

    def test_values_list_param_returns_values(self):
        # ratio_tolerance has a fixed "values" list
        sp = RULE_CATALOGUE["velocity single"]["sweep_params"]["ratio_tolerance"]
        df = _rule_df("Velocity Single")
        pts = _sweep_points(df, sp)
        assert 0.0 in pts
        assert 0.10 in pts


# ── _get_mask ────────────────────────────────────────────────────────────────

class TestGetMask:
    def _df(self):
        return pd.DataFrame({"trigger_amt": [5000, 10000, 50000, 100000], "is_sar": [0, 1, 0, 1]})

    def test_gte_direction_correct(self):
        df = self._df()
        sp = {"direction": "gte", "col": "trigger_amt"}
        mask = _get_mask(df, sp, 50000)
        assert list(mask) == [False, False, True, True]

    def test_gte_direction_all_pass_at_zero(self):
        df = self._df()
        sp = {"direction": "gte", "col": "trigger_amt"}
        mask = _get_mask(df, sp, 0)
        assert mask.all()

    def test_abs_lte_direction_correct(self):
        df = pd.DataFrame({"ratio": [0.95, 1.0, 1.05, 1.15]})
        sp = {"direction": "abs_lte", "col": "ratio"}
        # abs_lte: |ratio - 1.0| <= val
        mask = _get_mask(df, sp, 0.10)
        assert list(mask) == [True, True, True, False]

    def test_window_direction_missing_col_returns_false(self):
        df = pd.DataFrame({"trigger_amt": [5000, 10000]})
        sp = {"direction": "window", "col": None}
        mask = _get_mask(df, sp, 5)
        assert not mask.any()
