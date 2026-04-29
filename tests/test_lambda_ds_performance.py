"""Tests for lambda_ds_performance.py — analytics functions (no LLM required)."""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from lambda_ds_performance import (
    alerts_distribution,
    plot_thresholds_tuning,
    discover_dims,
    _cluster_title,
    perform_clustering,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _alerts_df():
    """Minimal alerts DataFrame with dynamic_segment, alerts, and false_positives."""
    return pd.DataFrame({
        "dynamic_segment": [0, 0, 0, 1, 1, 1],
        "alerts":          [1, 1, 0, 1, 0, 0],
        "false_positives": [1, 0, 0, 1, 0, 0],
        "AVG_TRXNS_WEEK":  [50, 80, 120, 30, 60, 90],
        "false_negatives": [0, 0, 0, 0, 1, 1],
    })


# ── alerts_distribution ────────────────────────────────────────────────────────

class TestAlertsDistribution:
    def test_returns_plotly_figure(self):
        fig = alerts_distribution(_alerts_df())
        assert isinstance(fig, go.Figure)

    def test_figure_has_two_traces(self):
        fig = alerts_distribution(_alerts_df())
        assert len(fig.data) == 2

    def test_figure_has_business_and_individual_x(self):
        fig = alerts_distribution(_alerts_df())
        x_values = set()
        for trace in fig.data:
            x_values.update(trace.x)
        assert "Business" in x_values
        assert "Individual" in x_values

    def test_total_alerts_count_correct(self):
        df = _alerts_df()
        fig = alerts_distribution(df)
        # First trace is "Total Alerts"; Business should have 2 alerts
        total_trace = fig.data[0]
        biz_idx = list(total_trace.x).index("Business")
        assert total_trace.y[biz_idx] == 2

    def test_false_positives_count_correct(self):
        df = _alerts_df()
        fig = alerts_distribution(df)
        fp_trace = fig.data[1]
        biz_idx = list(fp_trace.x).index("Business")
        assert fp_trace.y[biz_idx] == 1

    def test_empty_df_returns_figure(self):
        df = pd.DataFrame({
            "dynamic_segment": [], "alerts": [], "false_positives": []
        })
        fig = alerts_distribution(df)
        assert isinstance(fig, go.Figure)


# ── plot_thresholds_tuning ─────────────────────────────────────────────────────

class TestPlotThresholdsTuning:
    def test_returns_figure_and_dataframe(self, tmp_path, monkeypatch):
        # Redirect CSV output to tmp_path so test doesn't write to /tmp
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        df = _alerts_df()
        df_segment = df[df["dynamic_segment"] == 0].copy()
        fig, returned_df = plot_thresholds_tuning(df_segment, "AVG_TRXNS_WEEK", 0.1, "Business")
        assert isinstance(fig, go.Figure)
        assert isinstance(returned_df, pd.DataFrame)

    def test_figure_has_fp_and_fn_traces(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        df = _alerts_df()
        df_segment = df[df["dynamic_segment"] == 0].copy()
        fig, _ = plot_thresholds_tuning(df_segment, "AVG_TRXNS_WEEK", 0.1, "Business")
        trace_names = [t.name for t in fig.data]
        assert "False Positives" in trace_names
        assert "False Negatives" in trace_names

    def test_figure_has_threshold_annotations(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        df = _alerts_df()
        df_segment = df[df["dynamic_segment"] == 0].copy()
        fig, _ = plot_thresholds_tuning(df_segment, "AVG_TRXNS_WEEK", 0.1, "Business")
        annotation_texts = " ".join(a.text for a in fig.layout.annotations)
        assert "Threshold Min" in annotation_texts
        assert "Threshold Max" in annotation_texts


# ── discover_dims ──────────────────────────────────────────────────────────────

def _segmentation_df():
    """Minimal segmentation DataFrame with mix of good and bad columns."""
    n = 100
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "customer_id":    range(n),
        "account_id":     range(n),
        "customer_type":  rng.choice(["BUSINESS", "INDIVIDUAL"], size=n),
        "ACCOUNT_TYPE":   rng.choice(["Checking", "Savings", "Loan"], size=n),
        "GENDER":         rng.choice(["M", "F", "Unknown"], size=n),
        "AGE_CATEGORY":   rng.choice(["18-30", "31-45", "46-60", "60+"], size=n),
        "avg_num_trxns":  rng.uniform(0, 100, size=n),       # numeric — excluded
        "cluster":        rng.integers(0, 4, size=n),         # excluded
        "income":         rng.uniform(0, 200000, size=n),     # excluded (lowercase)
        "HIGH_CARDINALITY": [str(i) for i in range(n)],       # >20 unique → excluded
    })


class TestDiscoverDims:
    def test_returns_list(self):
        df = _segmentation_df()
        result = discover_dims(df)
        assert isinstance(result, list)

    def test_excludes_id_columns(self):
        df = _segmentation_df()
        dims = discover_dims(df)
        assert "customer_id" not in dims
        assert "account_id" not in dims

    def test_excludes_numeric_excluded_columns(self):
        df = _segmentation_df()
        dims = discover_dims(df)
        assert "avg_num_trxns" not in dims
        assert "income" not in dims

    def test_excludes_cluster_column(self):
        df = _segmentation_df()
        dims = discover_dims(df)
        assert "cluster" not in dims

    def test_excludes_customer_type_column(self):
        df = _segmentation_df()
        dims = discover_dims(df)
        assert "customer_type" not in dims

    def test_includes_account_type(self):
        df = _segmentation_df()
        dims = discover_dims(df)
        assert "ACCOUNT_TYPE" in dims

    def test_includes_gender(self):
        df = _segmentation_df()
        dims = discover_dims(df)
        assert "GENDER" in dims

    def test_excludes_high_cardinality_column(self):
        df = _segmentation_df()
        dims = discover_dims(df)
        assert "HIGH_CARDINALITY" not in dims

    def test_segment_filter_business(self):
        df = _segmentation_df()
        dims_all = discover_dims(df, segment=None)
        dims_biz = discover_dims(df, segment="BUSINESS")
        # Both should be non-empty lists (exact dims may differ by segment)
        assert isinstance(dims_biz, list)

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame(columns=["customer_type", "ACCOUNT_TYPE"])
        result = discover_dims(df, segment="BUSINESS")
        assert result == []

    def test_availability_threshold_respected(self):
        n = 100
        df = pd.DataFrame({
            "LOW_AVAIL": [None] * 80 + ["A"] * 20,  # only 20% available → excluded
            "HIGH_AVAIL": ["X"] * 100,               # 100% available, but only 1 unique → excluded
            "GOOD_COL":  ["A", "B"] * 50,             # 100% available, 2 unique → included
        })
        dims = discover_dims(df, availability=0.70)
        assert "LOW_AVAIL" not in dims
        assert "HIGH_AVAIL" not in dims
        assert "GOOD_COL" in dims


# ── _cluster_title ─────────────────────────────────────────────────────────────

class TestClusterTitle:
    def test_high_freq_high_value(self):
        title = _cluster_title(10, 1000, 5, 500)  # both 2x overall → High
        assert "High Freq" in title
        assert "High Value" in title

    def test_low_freq_low_value(self):
        title = _cluster_title(1, 100, 5, 500)  # both <0.85x overall → Low
        assert "Low Freq" in title
        assert "Low Value" in title

    def test_mid_freq_mid_value(self):
        title = _cluster_title(5, 500, 5, 500)  # exactly equal → Mid
        assert "Mid Freq" in title
        assert "Mid Value" in title

    def test_mixed_high_low(self):
        title = _cluster_title(10, 100, 5, 500)  # high freq, low value
        assert "High Freq" in title
        assert "Low Value" in title


# ── perform_clustering ─────────────────────────────────────────────────────────

def _clustering_df(n=200, seed=0):
    """Minimal DataFrame suitable for perform_clustering."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dynamic_segment":     rng.integers(0, 2, size=n),
        "avg_num_trxns":       rng.uniform(1, 50, size=n),
        "avg_weekly_trxn_amt": rng.uniform(100, 10000, size=n),
        "trxn_amt_monthly":    rng.uniform(500, 50000, size=n),
        "ACCT_AGE_YEARS":      rng.uniform(0.1, 15, size=n),
        "ACCOUNT_TYPE":        rng.choice(["Checking", "Savings", "Loan"], size=n),
        "AGE":                 rng.uniform(20, 80, size=n),
    })


class TestPerformClustering:
    def test_returns_tuple_of_three(self):
        df = _clustering_df()
        result = perform_clustering(df, n_clusters=3)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_figure_stats_df(self):
        df = _clustering_df()
        fig, stats, df_active = perform_clustering(df, n_clusters=3)
        assert isinstance(fig, go.Figure)
        assert isinstance(stats, str)
        assert isinstance(df_active, pd.DataFrame)

    def test_stats_contains_pre_computed_header(self):
        df = _clustering_df()
        _, stats, _ = perform_clustering(df, n_clusters=3)
        assert "PRE-COMPUTED CLUSTER STATS" in stats

    def test_stats_contains_cluster_count(self):
        df = _clustering_df()
        _, stats, _ = perform_clustering(df, n_clusters=3)
        assert "Clusters: 3" in stats

    def test_business_filter(self):
        df = _clustering_df()
        _, _, df_active = perform_clustering(df, customer_type="Business", n_clusters=2)
        # All rows should be Business segment (dynamic_segment == 0)
        assert (df_active["dynamic_segment"] == 0).all()

    def test_individual_filter(self):
        df = _clustering_df()
        _, _, df_active = perform_clustering(df, customer_type="Individual", n_clusters=2)
        assert (df_active["dynamic_segment"] == 1).all()

    def test_cluster_column_added(self):
        df = _clustering_df()
        _, _, df_active = perform_clustering(df, n_clusters=3)
        assert "cluster" in df_active.columns

    def test_cluster_values_in_range(self):
        df = _clustering_df()
        n_k = 3
        _, _, df_active = perform_clustering(df, n_clusters=n_k)
        assert df_active["cluster"].min() >= 0
        assert df_active["cluster"].max() < n_k

    def test_auto_cluster_selection(self):
        df = _clustering_df(n=200)
        fig, stats, df_active = perform_clustering(df, n_clusters=0)
        # Auto-selection should pick K between 2 and 8
        k_found = df_active["cluster"].nunique()
        assert 2 <= k_found <= 8
