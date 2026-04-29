"""
make_figures.py — Plotly table figures for all pre-computed tool results.

Each function returns a go.Figure (Table trace) styled consistently.
Called from application.py tool_executor alongside the text result.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Shared style ──────────────────────────────────────────────────────────────
_HEADER_FILL   = "#1a3a5c"
_HEADER_FONT   = "white"
_ROW_FILL_ODD  = "#f0f4f8"
_ROW_FILL_EVEN = "#ffffff"
_FONT_FAMILY   = "monospace"
_FONT_SIZE     = 12


def _alternating_fill(n, odd=_ROW_FILL_ODD, even=_ROW_FILL_EVEN):
    return [odd if i % 2 == 0 else even for i in range(n)]


def _table_height(n_rows, row_h=28, header_h=32, max_rows=10, margin=70):
    """Return figure height sized to actual rows, capped at max_rows (Plotly scrolls beyond that)."""
    visible = min(n_rows, max_rows)
    return header_h + visible * row_h + margin


# ── 1. list_rules ─────────────────────────────────────────────────────────────

def rule_list_figure(df_rule_sweep):
    """Table showing all rules with SAR/FP counts and precision."""
    from lambda_rule_analysis import RULE_CATALOGUE

    rows = []
    for _, entry in RULE_CATALOGUE.items():
        rf  = entry["name"]
        grp = df_rule_sweep[df_rule_sweep["risk_factor"] == rf]
        n   = len(grp)
        sar = int(grp["is_sar"].sum()) if n > 0 else 0
        fp  = int((grp["is_sar"] == 0).sum()) if n > 0 else 0
        nul = int(grp["is_sar"].isna().sum()) if n > 0 else 0
        prec = f"{round(100*sar/(sar+fp), 1)}%" if (sar + fp) > 0 else "n/a"
        sweep_keys = ", ".join(entry["sweep_params"].keys())
        rows.append([rf, n, sar, fp, prec, sweep_keys])

    cols = ["Risk Factor", "Alerted", "SAR", "FP", "Precision", "Sweep Params"]
    df = pd.DataFrame(rows, columns=cols)
    n  = len(df)

    fig = go.Figure(go.Table(
        columnwidth=[3, 1, 1, 1, 1, 3],
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color=_HEADER_FILL,
            font=dict(color=_HEADER_FONT, family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["left", "right", "right", "right", "right", "left"],
            height=32,
        ),
        cells=dict(
            values=[df[c].tolist() for c in cols],
            fill_color=[_alternating_fill(n)] * len(cols),
            font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["left", "right", "right", "right", "right", "left"],
            height=28,
        ),
    ))
    fig.update_layout(
        title="AML Rule Performance Overview",
        margin=dict(l=10, r=10, t=40, b=10),
        height=_table_height(n, row_h=28),
    )
    return fig


# ── 2. rule_sar_backtest ──────────────────────────────────────────────────────

def rule_sweep_figure(df_rule_sweep, risk_factor_keyword, sweep_param=None):
    """Sweep table for a specific rule and condition parameter."""
    from lambda_rule_analysis import RULE_CATALOGUE, _match_rule

    rf_name, entry = _match_rule(risk_factor_keyword)
    if entry is None:
        return None

    if sweep_param is None or sweep_param not in entry["sweep_params"]:
        sweep_param = entry["default_sweep"]
    sp           = entry["sweep_params"][sweep_param]
    col          = sp["col"]
    integer_axis = sp.get("integer_axis", False)

    rule_df = df_rule_sweep[df_rule_sweep["risk_factor"] == rf_name].copy()
    known   = rule_df.dropna(subset=["is_sar", col]).copy()

    total_sars = int((known["is_sar"] == 1).sum())
    if len(known) == 0 or total_sars == 0:
        return None

    from lambda_rule_analysis import _sweep_points as _lra_sweep_points
    cur        = float(sp["current"])
    raw_points = _lra_sweep_points(known, sp, n_steps=16)

    def _fmt_thresh(t, is_integer):
        if is_integer:
            return f"{int(t):,}"
        return f"{t:,.2f}"

    rows = []
    for t in raw_points:
        if sp["direction"] == "gte":
            caught = int(((known[col] >= t) & (known["is_sar"] == 1)).sum())
            fp_rem = int(((known[col] >= t) & (known["is_sar"] == 0)).sum())
        else:
            caught = int(((known[col] <= t) & (known["is_sar"] == 1)).sum())
            fp_rem = int(((known[col] <= t) & (known["is_sar"] == 0)).sum())
        missed    = total_sars - caught
        sar_pct   = round(100 * caught / total_sars, 1)
        precision = round(100 * caught / (caught + fp_rem), 1) if (caught + fp_rem) > 0 else 0.0
        is_cur    = (int(round(t)) == int(round(cur))) if integer_axis else (t == round(cur, 2))
        rows.append([
            _fmt_thresh(t, integer_axis) + (" *" if is_cur else ""),
            caught,
            fp_rem,
            missed,
            f"{sar_pct}%",
            f"{precision}%",
        ])

    cols = [f"{sp['label']} (* = current)", "SAR Caught", "FP Remain", "SAR Missed", "SAR Catch %", "Precision"]
    df_t = pd.DataFrame(rows, columns=cols)
    n    = len(df_t)

    # Color SAR% column: green when high, red when low
    # Color Precision column: green when high, red when low
    sar_pcts   = [float(r[4].replace("%", "")) for r in rows]
    prec_vals  = [float(r[5].replace("%", "")) for r in rows]
    cell_colors = []
    for c_idx, col_name in enumerate(cols):
        if col_name == "SAR Catch %":
            colors = []
            for pct in sar_pcts:
                if pct >= 90:
                    colors.append("#c8e6c9")
                elif pct >= 50:
                    colors.append("#fff9c4")
                else:
                    colors.append("#ffcdd2")
            cell_colors.append(colors)
        elif col_name == "Precision":
            colors = []
            for pct in prec_vals:
                if pct >= 50:
                    colors.append("#c8e6c9")
                elif pct >= 25:
                    colors.append("#fff9c4")
                else:
                    colors.append("#ffcdd2")
            cell_colors.append(colors)
        else:
            cell_colors.append(_alternating_fill(n))

    # Highlight current row
    cur_indices = [i for i, r in enumerate(rows) if r[0].endswith(" *")]
    for c_idx in range(len(cols)):
        for ci in cur_indices:
            cell_colors[c_idx][ci] = "#bbdefb"   # light blue

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color=_HEADER_FILL,
            font=dict(color=_HEADER_FONT, family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["right"] * len(cols),
            height=32,
        ),
        cells=dict(
            values=[df_t[c].tolist() for c in cols],
            fill_color=cell_colors,
            font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["right"] * len(cols),
            height=26,
        ),
    ))
    fig.update_layout(
        title=f"{rf_name} — {sweep_param} sweep (* = current condition value)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=_table_height(n, row_h=26),
    )
    return fig


# ── 3. threshold_tuning ───────────────────────────────────────────────────────

_INTEGER_THRESHOLD_COLS = {"AVG_TRXNS_WEEK", "avg_trxns_week"}

def threshold_tuning_figure(df_seg, threshold_col, segment_name):
    """FP/FN sweep table for threshold tuning."""
    import math as _math
    is_int_col = threshold_col in _INTEGER_THRESHOLD_COLS or df_seg[threshold_col].dropna().apply(lambda x: x == int(x)).all()
    cur_val   = float(df_seg[threshold_col].median())
    raw_step  = cur_val / 4
    mag       = 10 ** _math.floor(_math.log10(max(raw_step, 1)))
    _n        = raw_step / mag
    if _n < 1.5:   nice = 1
    elif _n < 3.5: nice = 2
    elif _n < 7.5: nice = 5
    else:          nice = 10
    step  = int(nice * mag)
    t_min = max(step, int(cur_val - 4 * step))
    t_max = int(cur_val + 4 * step)

    rows = []
    t = t_min
    while t <= t_max + step:
        t_r = round(t, 2)
        fp  = int(df_seg[(df_seg[threshold_col] >= t_r) & (df_seg["false_positives"] == 1)].shape[0])
        fn  = int(df_seg[(df_seg[threshold_col] <  t_r) & (df_seg["false_negatives"] == 1)].shape[0])
        label = f"{int(t_r):,}" if is_int_col else f"{t_r:,.2f}"
        rows.append([label, fp, fn, fp + fn])
        t += step

    cols = ["Threshold", "False Positives", "False Negatives", "Total FP+FN"]
    df_t = pd.DataFrame(rows, columns=cols)
    n    = len(df_t)

    fp_vals = [r[1] for r in rows]
    fn_vals = [r[2] for r in rows]
    max_fp  = max(fp_vals) if fp_vals else 1
    max_fn  = max(fn_vals) if fn_vals else 1

    # Colour FP: green when low, red when high; FN: inverse
    fp_colors, fn_colors = [], []
    for fp, fn in zip(fp_vals, fn_vals):
        fp_pct = fp / max_fp if max_fp > 0 else 0
        fn_pct = fn / max_fn if max_fn > 0 else 0
        fp_colors.append("#ffcdd2" if fp_pct > 0.5 else "#c8e6c9" if fp_pct < 0.2 else "#fff9c4")
        fn_colors.append("#ffcdd2" if fn_pct > 0.5 else "#c8e6c9" if fn_pct < 0.2 else "#fff9c4")

    cell_colors = [
        _alternating_fill(n),
        fp_colors,
        fn_colors,
        _alternating_fill(n),
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color=_HEADER_FILL,
            font=dict(color=_HEADER_FONT, family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["right", "right", "right", "right"],
            height=32,
        ),
        cells=dict(
            values=[df_t[c].tolist() for c in cols],
            fill_color=cell_colors,
            font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["right", "right", "right", "right"],
            height=26,
        ),
    ))
    fig.update_layout(
        title=f"Threshold Tuning — {segment_name} / {threshold_col}",
        margin=dict(l=10, r=10, t=40, b=10),
        height=_table_height(n, row_h=26),
    )
    return fig


# ── 4. sar_backtest ───────────────────────────────────────────────────────────

def sar_backtest_figure(df_sar_seg, sar_col, segment_name):
    """SAR catch rate sweep table."""
    df = df_sar_seg.dropna(subset=[sar_col]).copy()
    total_sars = int(df["is_sar"].sum())
    if total_sars == 0:
        return None

    t_min = df[sar_col].min()
    t_max = df[sar_col].max()
    is_int_col = sar_col in _INTEGER_THRESHOLD_COLS or df[sar_col].dropna().apply(lambda x: x == int(x)).all()
    step  = max(1, int((t_max - t_min) / 100))

    rows = []
    t = t_min
    while t <= t_max + step:
        t_r    = round(t, 2)
        caught = int(((df[sar_col] >= t_r) & (df["is_sar"] == 1)).sum())
        missed = total_sars - caught
        pct    = round(100 * caught / total_sars, 1)
        label  = f"{int(t_r):,}" if is_int_col else f"{t_r:,.2f}"
        rows.append([label, caught, missed, f"{pct}%"])
        t += step

    cols = ["Threshold", "SAR Caught", "SAR Missed", "Catch Rate"]
    df_t = pd.DataFrame(rows, columns=cols)
    n    = len(df_t)

    pcts = [float(r[3].replace("%", "")) for r in rows]
    catch_colors = []
    for pct in pcts:
        catch_colors.append("#c8e6c9" if pct >= 90 else "#fff9c4" if pct >= 50 else "#ffcdd2")

    cell_colors = [
        _alternating_fill(n),
        _alternating_fill(n),
        _alternating_fill(n),
        catch_colors,
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color=_HEADER_FILL,
            font=dict(color=_HEADER_FONT, family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["right", "right", "right", "right"],
            height=32,
        ),
        cells=dict(
            values=[df_t[c].tolist() for c in cols],
            fill_color=cell_colors,
            font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["right", "right", "right", "right"],
            height=26,
        ),
    ))
    fig.update_layout(
        title=f"SAR Backtest — {segment_name} / {sar_col}",
        margin=dict(l=10, r=10, t=40, b=10),
        height=_table_height(n, row_h=26),
    )
    return fig


# ── 5. segment_stats / alerts_distribution ────────────────────────────────────

def segment_stats_figure(df):
    """Segment-level alert and FP/FN summary table."""
    total_alerts   = int(df["alerts"].sum())
    total_accounts = len(df)

    rows = []
    for seg_id, name in [(0, "Business"), (1, "Individual")]:
        seg      = df[df["dynamic_segment"] == seg_id]
        n        = len(seg)
        alerts   = int(seg["alerts"].sum())
        fp       = int(seg["false_positives"].sum())
        fn       = int(seg["false_negatives"].sum())
        fp_rate  = f"{round(100*fp/alerts, 1)}%" if alerts > 0 else "n/a"
        acct_pct = f"{round(100*n/total_accounts, 1)}%"
        rows.append([name, f"{n:,}", acct_pct, f"{alerts:,}", f"{fp:,}", fp_rate, f"{fn:,}"])

    cols = ["Segment", "Accounts", "% of Total", "Alerts", "FP", "FP Rate", "FN"]
    df_t = pd.DataFrame(rows, columns=cols)

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color=_HEADER_FILL,
            font=dict(color=_HEADER_FONT, family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["left", "right", "right", "right", "right", "right", "right"],
            height=32,
        ),
        cells=dict(
            values=[df_t[c].tolist() for c in cols],
            fill_color=[["#f0f4f8", "#ffffff"]] * len(cols),
            font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["left", "right", "right", "right", "right", "right", "right"],
            height=28,
        ),
    ))
    n_seg = len(rows)
    fig.update_layout(
        title="Segment Statistics — Alerts, FP, FN",
        margin=dict(l=10, r=10, t=40, b=10),
        height=_table_height(n_seg, row_h=28),
    )
    return fig


# ── 7. rule_2d_sweep heatmap ──────────────────────────────────────────────────

def rule_2d_heatmap(grid_dict):
    """
    Single interactive heatmap: SAR catch % as colour, FP count in hover tooltip.
    Current condition cell marked with a visible annotation.
    Rows = param1 values (Y axis), columns = param2 values (X axis).
    """
    if grid_dict is None:
        return None

    p1_vals    = grid_dict["p1_vals"]
    p2_vals    = grid_dict["p2_vals"]
    sar_grid   = grid_dict["sar_grid"]
    fp_grid    = grid_dict["fp_grid"]
    total_sars = grid_dict["total_sars"]
    total_fps  = grid_dict["total_fps"]
    p1_label   = grid_dict["p1_label"]
    p2_label   = grid_dict["p2_label"]
    p1_cur     = grid_dict["p1_current"]
    p2_cur     = grid_dict["p2_current"]
    p1_fmt_pct = grid_dict.get("p1_format_pct", False)
    p2_fmt_pct = grid_dict.get("p2_format_pct", False)
    rf_name    = grid_dict["rf_name"]
    param1     = grid_dict["param1"]
    param2     = grid_dict["param2"]

    sar_pct = [[round(100 * v / total_sars, 1) for v in row] for row in sar_grid]

    def fmt(v, as_pct=False):
        if as_pct:
            pct = v * 100
            return f"{pct:g}%"
        if isinstance(v, float) and v == int(v):
            v = int(v)
        if isinstance(v, int):
            return f"{v:,}" if abs(v) >= 1000 else str(v)
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3f}".rstrip("0").rstrip(".")

    x_labels = [fmt(v, p2_fmt_pct) for v in p2_vals]
    y_labels = [fmt(v, p1_fmt_pct) for v in p1_vals]
    x_idx = list(range(len(p2_vals)))
    y_idx = list(range(len(p1_vals)))

    # Hover: axis coordinates + 2x2 confusion matrix (TP/FP/FN/TN)
    # TP = alerted SAR  (green)   FP = alerted non-SAR  (red)
    # FN = missed SAR   (red)     TN = correctly silent  (green)
    _G = "color:#27ae60;font-weight:bold"   # green
    _R = "color:#e74c3c;font-weight:bold"   # red

    hover = []
    for i, v1 in enumerate(p1_vals):
        row_hover = []
        for j, v2 in enumerate(p2_vals):
            tp  = sar_grid[i][j]
            fp  = fp_grid[i][j]
            fn  = total_sars - tp
            tn  = total_fps  - fp
            pct = sar_pct[i][j]
            prec = round(100 * tp / (tp + fp), 1) if (tp + fp) > 0 else 0.0
            row_hover.append(
                f"<b>{param1}</b>: {fmt(v1, p1_fmt_pct)}  |  <b>{param2}</b>: {fmt(v2, p2_fmt_pct)}<br><br>"
                f"<span style='{_G}'>TP: {tp}</span>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;"
                f"<span style='{_R}'>FP: {fp}</span><br>"
                f"<span style='{_R}'>FN: {fn}</span>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;"
                f"<span style='{_G}'>TN: {tn}</span><br><br>"
                f"TP rate: {pct}%  |  Precision: {prec}%"
            )
        hover.append(row_hover)

    # Current condition marker
    cur_xi = min(range(len(p2_vals)), key=lambda i: abs(p2_vals[i] - float(p2_cur)))
    cur_yi = min(range(len(p1_vals)), key=lambda i: abs(p1_vals[i] - float(p1_cur)))

    annotations = [dict(
        x=cur_xi,
        y=cur_yi,
        text="<b>NOW</b>",
        showarrow=False,
        font=dict(size=11, color="white"),
        bgcolor="rgba(0,0,0,0.55)",
        bordercolor="white",
        borderwidth=1,
        borderpad=2,
    )]

    cell_h = max(28, min(50, 600 // max(len(p1_vals), 1)))
    height = cell_h * len(p1_vals) + 140

    fig = go.Figure(go.Heatmap(
        z=sar_pct,
        x=x_idx,
        y=y_idx,
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(text="TP Rate %", side="right"),
            ticksuffix="%",
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{rf_name}</b> — 2D Sweep<br>"
                f"<sup>Y: {p1_label} &nbsp;|&nbsp; X: {p2_label} &nbsp;|&nbsp; "
                f"Color: TP rate % (green=high) &nbsp;|&nbsp; Hover: TP/FP/FN/TN matrix &nbsp;|&nbsp; "
                f"<b>NOW</b> = current condition</sup>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title=p2_label, tickangle=-35, tickmode="array",
                   tickvals=x_idx, ticktext=x_labels),
        yaxis=dict(title=p1_label, tickmode="array",
                   tickvals=y_idx, ticktext=y_labels),
        height=height,
        margin=dict(l=10, r=20, t=90, b=60),
        annotations=annotations,
    )
    return fig


# ── 8a. 2D Sweep ranked table ────────────────────────────────────────────────

def rule_2d_ranked_table(grid_dict, top_n=15):
    """
    Ranked table of 2D sweep combinations sorted by best FP reduction
    while keeping TP rate >= 50%. Easier for AML analysts than a heatmap.
    Highlights the current condition row.
    """
    if grid_dict is None:
        return None

    p1_vals    = grid_dict["p1_vals"]
    p2_vals    = grid_dict["p2_vals"]
    sar_grid   = grid_dict["sar_grid"]
    fp_grid    = grid_dict["fp_grid"]
    total_sars = grid_dict["total_sars"]
    total_fps  = grid_dict["total_fps"]
    p1_label   = grid_dict["p1_label"]
    p2_label   = grid_dict["p2_label"]
    p1_cur     = float(grid_dict["p1_current"])
    p2_cur     = float(grid_dict["p2_current"])
    p1_fmt_pct = grid_dict.get("p1_format_pct", False)
    p2_fmt_pct = grid_dict.get("p2_format_pct", False)
    rf_name    = grid_dict["rf_name"]
    param1     = grid_dict["param1"]
    param2     = grid_dict["param2"]

    def fmt(v, as_pct=False):
        if as_pct:
            return f"{v * 100:g}%"
        if isinstance(v, float) and v == int(v):
            v = int(v)
        return f"{v:,}" if isinstance(v, int) and abs(v) >= 1000 else (f"{v:,.0f}" if abs(v) >= 1000 else str(v))

    rows = []
    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            tp = sar_grid[i][j]
            fp = fp_grid[i][j]
            fn = total_sars - tp
            tn = total_fps  - fp
            tp_rate   = round(100 * tp / total_sars, 1) if total_sars > 0 else 0
            precision = round(100 * tp / (tp + fp), 1)  if (tp + fp) > 0  else 0
            fp_reduc  = round(100 * (total_fps - fp) / total_fps, 1) if total_fps > 0 else 0
            is_current = (abs(v1 - p1_cur) < 1e-6 or i == min(range(len(p1_vals)), key=lambda x: abs(p1_vals[x] - p1_cur))) and \
                         (abs(v2 - p2_cur) < 1e-6 or j == min(range(len(p2_vals)), key=lambda x: abs(p2_vals[x] - p2_cur)))
            rows.append({
                "v1": v1, "v2": v2,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "tp_rate": tp_rate, "precision": precision, "fp_reduc": fp_reduc,
                "is_current": is_current,
            })

    # Sort: highest TP rate first, then lowest FP
    rows.sort(key=lambda r: (-r["tp_rate"], r["fp"]))
    rows = rows[:top_n]

    p1_col = [fmt(r["v1"], p1_fmt_pct) for r in rows]
    p2_col = [fmt(r["v2"], p2_fmt_pct) for r in rows]
    tp_col    = [f"{r['tp']}" for r in rows]
    fp_col    = [f"{r['fp']}" for r in rows]
    fn_col    = [f"{r['fn']}" for r in rows]
    rate_col  = [f"{r['tp_rate']}%" for r in rows]
    prec_col  = [f"{r['precision']}%" for r in rows]
    reduc_col = [f"{r['fp_reduc']}%" for r in rows]

    # Row colours: current = blue, high TP rate = light green, low TP = light red
    fill_colors = []
    font_colors = []
    for r in rows:
        if r["is_current"]:
            fill_colors.append("#2980b9"); font_colors.append("white")
        elif r["tp_rate"] >= 80:
            fill_colors.append("#d5f5e3"); font_colors.append("#1a1a1a")
        elif r["tp_rate"] >= 50:
            fill_colors.append("#fef9e7"); font_colors.append("#1a1a1a")
        else:
            fill_colors.append("#fadbd8"); font_colors.append("#1a1a1a")

    header_vals = [p1_label, p2_label, "SARs Caught", "False Positives",
                   "Missed SARs", "SAR Catch %", "Precision", "FP Reduction"]

    fig = go.Figure(go.Table(
        columnwidth=[120, 120, 90, 100, 90, 90, 90, 100],
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="center",
            height=32,
        ),
        cells=dict(
            values=[p1_col, p2_col, tp_col, fp_col, fn_col, rate_col, prec_col, reduc_col],
            fill_color=[fill_colors] * 8,
            font=dict(color=[font_colors] * 8, size=11),
            align="center",
            height=28,
        ),
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{rf_name}</b> — Top {top_n} Parameter Combinations<br>"
                f"<sup>Sorted by SAR catch rate (highest first) &nbsp;|&nbsp; "
                f"<span style='color:#27ae60'>Green</span>=≥80% catch &nbsp;|&nbsp; "
                f"<span style='color:#f39c12'>Yellow</span>=50–80% &nbsp;|&nbsp; "
                f"<span style='color:#e74c3c'>Red</span>=&lt;50% &nbsp;|&nbsp; "
                f"<span style='color:#2980b9;font-weight:bold'>Blue</span>=current condition</sup>"
            ),
            font=dict(size=13),
        ),
        height=min(120 + 30 * len(rows), 580),
        margin=dict(l=10, r=10, t=90, b=10),
    )
    return fig


# ── 8. Rule alert distribution by cluster ────────────────────────────────────

def rule_alerts_by_cluster(df_rule_sweep, df_cluster_labels, rf_name, target_cluster):
    """
    Grouped bar chart: Alerts / SARs / FPs for rf_name broken down by cluster.
    target_cluster (1-indexed) is highlighted. df_cluster_labels has 1-indexed cluster column.
    """
    import plotly.graph_objects as go

    rule_df = df_rule_sweep[df_rule_sweep["risk_factor"] == rf_name].copy()
    if rule_df.empty or df_cluster_labels is None:
        return None

    merged = rule_df.merge(df_cluster_labels[["customer_id", "cluster"]], on="customer_id", how="left")
    merged = merged.dropna(subset=["cluster"])
    if merged.empty:
        return None

    merged["cluster"] = merged["cluster"].astype(int)
    groups = sorted(merged["cluster"].unique())

    cluster_labels = [f"Cluster {c}" for c in groups]
    alerts = [len(merged[merged["cluster"] == c]) for c in groups]
    sars   = [int(merged[merged["cluster"] == c]["is_sar"].sum()) for c in groups]
    fps    = [a - s for a, s in zip(alerts, sars)]

    # Highlight target cluster bar with a marker
    target_int = int(target_cluster)
    marker_colors_sar = ["#c0392b" if c == target_int else "#e74c3c" for c in groups]
    marker_colors_fp  = ["#e67e22" if c == target_int else "#f39c12" for c in groups]

    fig = go.Figure([
        go.Bar(name="SAR (TP)", x=cluster_labels, y=sars,
               marker_color=marker_colors_sar, text=sars, textposition="auto"),
        go.Bar(name="FP",       x=cluster_labels, y=fps,
               marker_color=marker_colors_fp,  text=fps,  textposition="auto"),
    ])
    fig.update_layout(
        barmode="stack",
        title=f"{rf_name} — Alert Distribution by Cluster (Cluster {target_int} selected)",
        xaxis_title="Cluster",
        yaxis_title="Alerted Customers",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=70, b=40),
        height=340,
        font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
    )
    return fig


# ── 9. Cluster profile table (shown alongside cluster-filtered sweeps) ─────────

def cluster_profile_table(df_clustered, target_cluster):
    """
    Table of per-cluster stats with target_cluster row highlighted in gold.
    df_clustered has 0-indexed cluster labels; display as 1-indexed.
    target_cluster is 1-indexed (as the user specifies it).
    """
    numeric_cols = [c for c in [
        "avg_weekly_trxn_amt", "trxn_amt_monthly",
        "ACCT_AGE_YEARS",
    ] if c in df_clustered.columns]

    rows = []
    for cid, grp in sorted(df_clustered.groupby("cluster")):
        display_id = int(cid) + 1  # convert 0-indexed → 1-indexed
        row = {"Cluster": f"Cluster {display_id}", "Num Customers": len(grp)}
        for col in numeric_cols:
            row[col] = round(grp[col].median(), 1)
        rows.append(row)

    if not rows:
        return None

    df_t = pd.DataFrame(rows)
    cols = list(df_t.columns)

    # Highlight target cluster row
    highlight_gold = "#f5c518"
    normal_odd     = _ROW_FILL_ODD
    normal_even    = _ROW_FILL_EVEN
    fill_colors = []
    for col in cols:
        col_colors = []
        for i, row in df_t.iterrows():
            label = row["Cluster"]
            if label == f"Cluster {int(target_cluster)}":
                col_colors.append(highlight_gold)
            elif i % 2 == 0:
                col_colors.append(normal_odd)
            else:
                col_colors.append(normal_even)
        fill_colors.append(col_colors)

    # Format numeric values
    def _fmt_col(series, col_name):
        if col_name == "Num Customers":
            return [f"{int(v):,}" for v in series]
        if col_name in ("ACCT_AGE_YEARS", "AGE"):
            return [f"{v:.1f}" for v in series]
        return [f"{v:,.0f}" for v in series]

    cell_values = [_fmt_col(df_t[c], c) for c in cols]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color=_HEADER_FILL,
            font=dict(color=_HEADER_FONT, family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["left"] + ["right"] * (len(cols) - 1),
            height=32,
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["left"] + ["right"] * (len(cols) - 1),
            height=26,
        ),
    ))
    n_rows = len(rows)
    fig.update_layout(
        title=f"Cluster Profiles — Cluster {int(target_cluster)} highlighted",
        margin=dict(l=10, r=10, t=40, b=10),
        height=_table_height(n_rows, row_h=26),
    )
    return fig


# ── 9. Cluster stats summary table (shown with every clustering result) ────────

def cluster_stats_table(df_clustered, customer_type="All"):
    """
    Summary table of all cluster stats — replaces the hard-to-read text block.
    df_clustered has 0-indexed cluster labels; display as 1-indexed.
    """
    age_cols = [] if customer_type.upper() == "BUSINESS" else ["AGE"]
    numeric_cols = [c for c in [
        "avg_weekly_trxn_amt", "trxn_amt_monthly",
        "ACCT_AGE_YEARS",
    ] + age_cols if c in df_clustered.columns]

    total_n = len(df_clustered)

    rows = []
    for cid, grp in sorted(df_clustered.groupby("cluster")):
        display_id = int(cid) + 1  # convert 0-indexed → 1-indexed
        pct  = round(100 * len(grp) / total_n, 1) if total_n > 0 else 0.0
        row  = {"Cluster": f"Cluster {display_id}", "Num Customers": len(grp), "% of Active": f"{pct}%"}
        for col in numeric_cols:
            row[col] = round(grp[col].median(), 1)
        rows.append(row)

    if not rows:
        return None

    df_t = pd.DataFrame(rows)
    cols = list(df_t.columns)

    n_rows = len(rows)
    fill_colors = []
    for col in cols:
        fill_colors.append([_ROW_FILL_ODD if i % 2 == 0 else _ROW_FILL_EVEN for i in range(n_rows)])

    def _fmt_col(series, col_name):
        if col_name in ("Cluster", "% of Active"):
            return list(series)
        if col_name == "Num Customers":
            return [f"{int(v):,}" for v in series]
        if col_name in ("ACCT_AGE_YEARS", "AGE"):
            return [f"{v:.1f}" for v in series]
        return [f"{v:,.0f}" for v in series]

    cell_values = [_fmt_col(df_t[c], c) for c in cols]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in cols],
            fill_color=_HEADER_FILL,
            font=dict(color=_HEADER_FONT, family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["left", "right", "right"] + ["right"] * len(numeric_cols),
            height=32,
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            font=dict(family=_FONT_FAMILY, size=_FONT_SIZE),
            align=["left", "right", "right"] + ["right"] * len(numeric_cols),
            height=26,
        ),
    ))
    fig.update_layout(
        title=f"Cluster Summary — {customer_type}",
        margin=dict(l=10, r=10, t=40, b=10),
        height=_table_height(n_rows, row_h=26),
    )
    return fig
