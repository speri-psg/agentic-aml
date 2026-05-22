"""
lambda_cluster_threshold.py — Per-cluster adaptive threshold analysis.

Joins K-Means behavioral clusters with SAR labels to compute cluster-specific
threshold recommendations. Demonstrates how behavioral segmentation adapts alert
sensitivity per customer group rather than applying a single uniform threshold.

Option 2 hook: pass target_sar_rate as a parameter (default 0.90).
Option 3 hook: persist results externally and call again to compare across runs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lambda_ds_performance import perform_clustering

# Map tool parameter names → SAR CSV column names (mirrors SAR_COL_MAP in application.py)
_SAR_COL = {
    "AVG_TRXNS_WEEK":   "avg_num_trxns",
    "AVG_TRXN_AMT":     "avg_weekly_trxn_amt",
    "TRXN_AMT_MONTHLY": "trxn_amt_monthly",
}

_COL_LABEL = {
    "AVG_TRXNS_WEEK":   "Avg Weekly Transactions",
    "AVG_TRXN_AMT":     "Avg Weekly Txn Amount",
    "TRXN_AMT_MONTHLY": "Monthly Txn Volume",
}

_IS_DOLLAR = {"AVG_TRXN_AMT", "TRXN_AMT_MONTHLY"}

_RISK_LABELS = ["High Volume", "Mid-High", "Mid-Low", "Low Volume",
                "Group 5", "Group 6"]  # extra labels if n_clusters > 4


def _fmt(val, col):
    return f"${val:,.0f}" if col in _IS_DOLLAR else f"{val:,.2f}"


def _sweep(df_cl, sar_col, target):
    """
    Sweep threshold values for a cluster. Returns (recommended_row | None, rows).
    Recommended = highest threshold where tp_rate >= target (most FP reduction possible
    while still meeting the SAR catch target).
    """
    sar_vals = df_cl.loc[df_cl["is_sar"] == 1, sar_col].dropna()
    non_vals = df_cl.loc[df_cl["is_sar"] == 0, sar_col].dropna()
    total_sar = len(sar_vals)
    total_non = len(non_vals)

    if total_sar == 0:
        return None, []

    all_vals = df_cl[sar_col].dropna()
    pcts = [0, 10, 25, 40, 55, 70, 82, 91, 96]
    thresholds = sorted({round(float(all_vals.quantile(p / 100)), 2) for p in pcts})

    rows = []
    for t in thresholds:
        tp = int((sar_vals >= t).sum())
        fp = int((non_vals >= t).sum())
        tp_rate   = tp / total_sar
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rows.append({"threshold": t, "tp": tp, "fp": fp,
                     "fn": total_sar - tp, "tn": total_non - fp,
                     "tp_rate": tp_rate, "precision": precision})

    qualifying = [r for r in rows if r["tp_rate"] >= target]
    recommended = qualifying[-1] if qualifying else rows[0]
    return recommended, rows


def cluster_threshold_analysis(df_ss, df_sar, segment, threshold_column,
                                n_clusters=4, target_sar_rate=0.90):
    """
    Run K-Means on df_ss, join with SAR labels in df_sar, sweep threshold per cluster.

    Returns
    -------
    text : str   pre-computed block for the model to copy verbatim
    fig  : go.Figure  grouped bar chart comparing uniform vs adaptive FP counts
    """
    sar_col = _SAR_COL.get(threshold_column)
    if sar_col is None:
        return f"Unknown threshold_column '{threshold_column}'.", None

    if sar_col not in df_sar.columns:
        return f"Column '{sar_col}' not found in SAR data.", None

    # ── K-Means ───────────────────────────────────────────────────────────────
    n_clusters = max(2, min(6, int(n_clusters)))
    _, _, df_clustered = perform_clustering(df_ss, segment, n_clusters)
    # df_clustered: customer_id, cluster (0-based KMeans labels), feature cols

    # ── Filter SAR data by segment ────────────────────────────────────────────
    seg_val = 0 if segment.lower() == "business" else 1
    df_seg_sar = df_sar[df_sar["dynamic_segment"] == seg_val][
        ["customer_id", "is_sar", sar_col]
    ].dropna(subset=[sar_col]).copy()

    # ── Join ──────────────────────────────────────────────────────────────────
    df_joined = df_clustered[["customer_id", "cluster"]].merge(
        df_seg_sar, on="customer_id", how="inner"
    )

    if df_joined.empty:
        return f"No matching customers found for {segment} / {threshold_column}.", None

    total_sar = int(df_joined["is_sar"].sum())
    if total_sar == 0:
        return f"No SAR-labeled customers in {segment} data for this column.", None

    # ── Uniform threshold (whole segment at target_sar_rate) ─────────────────
    _, uniform_rows = _sweep(df_joined, sar_col, target_sar_rate)
    qualifying = [r for r in uniform_rows if r["tp_rate"] >= target_sar_rate]
    uni_row    = qualifying[-1] if qualifying else uniform_rows[0]
    uni_t      = uni_row["threshold"]
    uni_tp     = int((df_joined.loc[df_joined["is_sar"] == 1, sar_col] >= uni_t).sum())
    uni_fp     = int((df_joined.loc[df_joined["is_sar"] == 0, sar_col] >= uni_t).sum())

    # ── Per-cluster sweep ─────────────────────────────────────────────────────
    # Order clusters by median threshold column value descending → Cluster 1 = highest activity
    medians = df_joined.groupby("cluster")[sar_col].median().sort_values(ascending=False)
    cluster_order = list(medians.index)

    cluster_results = []
    for rank, km_label in enumerate(cluster_order):
        df_cl  = df_joined[df_joined["cluster"] == km_label]
        n_tot  = len(df_cl)
        n_sar  = int(df_cl["is_sar"].sum())
        rec, _ = _sweep(df_cl, sar_col, target_sar_rate)

        # Baseline counts at the uniform threshold
        base_tp = int((df_cl.loc[df_cl["is_sar"] == 1, sar_col] >= uni_t).sum())
        base_fp = int((df_cl.loc[df_cl["is_sar"] == 0, sar_col] >= uni_t).sum())

        cluster_results.append({
            "rank":    rank + 1,
            "label":   _RISK_LABELS[rank] if rank < len(_RISK_LABELS) else f"Group {rank+1}",
            "n":       n_tot,
            "n_sar":   n_sar,
            "rec":     rec,
            "base_tp": base_tp,
            "base_fp": base_fp,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    adapt_tp  = sum(r["rec"]["tp"] for r in cluster_results if r["rec"])
    adapt_fp  = sum(r["rec"]["fp"] for r in cluster_results if r["rec"])
    fp_delta  = adapt_fp - uni_fp   # negative = fewer FPs (improvement)
    sar_delta = adapt_tp - uni_tp   # negative = fewer SARs caught

    # ── Pre-computed text ─────────────────────────────────────────────────────
    target_pct = int(target_sar_rate * 100)
    lines = [
        "=== CLUSTER THRESHOLD ANALYSIS ===",
        f"Segment: **{segment}** | Column: {threshold_column} ({_COL_LABEL[threshold_column]}) "
        f"| Clusters: {n_clusters} | Target SAR catch: ≥{target_pct}%",
        "",
    ]

    for r in cluster_results:
        rec = r["rec"]
        if rec is None:
            lines.append(f"**Cluster {r['rank']} — {r['label']}** ({r['n']:,} customers): no SAR data")
            lines.append("")
            continue

        fp_chg = rec["fp"] - r["base_fp"]
        tp_chg = rec["tp"] - r["base_tp"]
        base_prec = (r["base_tp"] / (r["base_tp"] + r["base_fp"]) * 100
                     if r["base_tp"] + r["base_fp"] > 0 else 0)
        base_tpr  = r["base_tp"] / r["n_sar"] * 100 if r["n_sar"] > 0 else 0

        lines += [
            f"**Cluster {r['rank']} — {r['label']}** ({r['n']:,} customers, SAR pool: {r['n_sar']:,})",
            f"- Uniform {_fmt(uni_t, threshold_column)}: "
            f"TP={r['base_tp']:,}, FP={r['base_fp']:,}, "
            f"TP rate={base_tpr:.1f}%, precision={base_prec:.1f}%",
            f"- Recommended {_fmt(rec['threshold'], threshold_column)}: "
            f"TP={rec['tp']:,}, FP={rec['fp']:,}, "
            f"TP rate={rec['tp_rate']*100:.1f}%, precision={rec['precision']*100:.1f}% "
            f"({fp_chg:+,} FP, {tp_chg:+,} SAR)",
            "",
        ]

    lines += [
        "**ADAPTIVE SENSITIVITY SUMMARY**",
        f"- Uniform {_fmt(uni_t, threshold_column)} applied to all {segment}: "
        f"TP={uni_tp:,}, FP={uni_fp:,}, TP rate={uni_tp/total_sar*100:.1f}%",
        f"- Cluster-adaptive thresholds: "
        f"TP={adapt_tp:,}, FP={adapt_fp:,}, TP rate={adapt_tp/total_sar*100:.1f}%",
        f"- Net change: **{fp_delta:+,} FP**, {sar_delta:+,} SARs "
        f"({adapt_tp/total_sar*100:.1f}% SAR retention)",
        "=== END CLUSTER THRESHOLD ANALYSIS ===",
    ]

    text = "\n".join(lines)

    # ── Bar chart: uniform vs adaptive FP per cluster ─────────────────────────
    c_names   = [f"C{r['rank']}: {r['label']}" for r in cluster_results]
    uni_fps   = [r["base_fp"] for r in cluster_results]
    adapt_fps = [r["rec"]["fp"] if r["rec"] else r["base_fp"] for r in cluster_results]

    uni_hover = [
        (f"Cluster {r['rank']}: {r['label']}<br>"
         f"Customers: {r['n']:,} | SARs: {r['n_sar']:,}<br>"
         f"Uniform threshold: {_fmt(uni_t, threshold_column)}<br>"
         f"FP={r['base_fp']:,}, TP={r['base_tp']:,}, "
         f"TP rate={r['base_tp']/r['n_sar']*100:.1f}%")
        for r in cluster_results
    ]
    adapt_hover = [
        (f"Cluster {r['rank']}: {r['label']}<br>"
         f"Recommended threshold: {_fmt(r['rec']['threshold'], threshold_column)}<br>"
         f"FP={r['rec']['fp']:,}, TP={r['rec']['tp']:,}, "
         f"TP rate={r['rec']['tp_rate']*100:.1f}%, precision={r['rec']['precision']*100:.1f}%<br>"
         f"Change vs uniform: {r['rec']['fp']-r['base_fp']:+,} FP, "
         f"{r['rec']['tp']-r['base_tp']:+,} SAR")
        if r["rec"] else f"Cluster {r['rank']}: no SAR data"
        for r in cluster_results
    ]

    fig = go.Figure(data=[
        go.Bar(name=f"Uniform {_fmt(uni_t, threshold_column)}",
               x=c_names, y=uni_fps, marker_color="#EF553B",
               hovertext=uni_hover, hoverinfo="text"),
        go.Bar(name="Cluster-Adaptive", x=c_names, y=adapt_fps,
               marker_color="#636EFA",
               hovertext=adapt_hover, hoverinfo="text"),
    ])
    fig.update_layout(
        barmode="group",
        title=(f"False Positives: Uniform vs Adaptive — {segment} / "
               f"{_COL_LABEL[threshold_column]}"),
        xaxis_title="Behavioral Cluster",
        yaxis_title="False Positives",
        legend=dict(x=0.01, y=0.99),
    )

    return text, fig
