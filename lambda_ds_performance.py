import math
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import json
#import kaleido
def alerts_distribution(df):
    segment_total_alerts = [
        df[(df['dynamic_segment'] == 0) & (df['alerts'] == 1)].shape[0],
        df[(df['dynamic_segment'] == 1) & (df['alerts'] == 1)].shape[0],
    ]
    segment_fps = [
        df[(df['dynamic_segment'] == 0) & (df['false_positives'] == 1)].shape[0],
        df[(df['dynamic_segment'] == 1) & (df['false_positives'] == 1)].shape[0],
    ]

    data = [
        go.Bar(name='Total Alerts', x=['Business', 'Individual'], y=segment_total_alerts),
        go.Bar(name='False Positives', x=['Business', 'Individual'], y=segment_fps),
    ]

    fig = go.Figure(data)
    fig.update_layout(barmode='group', title="Alerts distribution across Segments")
    return fig
def plot_thresholds_tuning(df_segment, threshold, bump_pct, segment):
    false_positives = []
    false_negatives = []
    thresholds = []
    threshold_min = df_segment[threshold].min()
    threshold_max = df_segment[threshold].max()
    step = max(1, int((threshold_max - threshold_min) / 100))
    threshold_bump = threshold_min
    while threshold_bump <= threshold_max + step:
        fp = df_segment[(df_segment[threshold] >= threshold_bump) & (df_segment['false_positives'] == 1)].shape[0]
        fn = df_segment[(df_segment[threshold] < threshold_bump) & (df_segment['false_negatives'] == 1)].shape[0]
        false_positives.append(fp)
        false_negatives.append(fn)
        thresholds.append(round(threshold_bump, 2))
        threshold_bump = threshold_bump + step
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=false_positives, mode='lines', name='False Positives',
                             line=dict(color='#EF553B', width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=false_negatives, mode='lines', name='False Negatives',
                             line=dict(color='#636EFA', width=2)))
    fig.update_layout(
        title=f'False Positives & False Negatives vs Threshold ({threshold}) — Segment: {segment}',
        xaxis_title=threshold,
        yaxis_title='Count',
        legend=dict(x=0.01, y=0.99),
    )
    fig.add_annotation(
        text=f"<b>Threshold Min: {round(threshold_min, 2)}<br><b>Threshold Max: {round(threshold_max, 2)}",
        xref="paper", yref="paper",
        x=1, y=0.5,
        showarrow=False, align="right", valign="middle"
    )
    df_thresholds = pd.DataFrame({f'{threshold}': thresholds, 'False Positives': false_positives, 'False Negatives': false_negatives})
    df_thresholds.to_csv(os.path.join("/tmp", f"Segment_{segment}_{threshold}.csv"), index=False)
    return fig, df_segment
def smartseg_tree():
    dtree = pd.read_csv('smartsegments.csv')
    dtree['SmartSegment'] = dtree['SmartSegment'].astype(int)

    agg = {
        'amount_MEAN':        'mean',
        'avg_num_trxns_MEAN': 'mean',
        'avg_trxn_amt_MEAN':  'mean',
        'NUM_COUNT':          'sum',
    }

    rows = []

    # Root node
    r = dtree.agg(agg)
    rows.append({'id': 'All', 'parent': '', 'label': 'AML Dynamic Segments',
                 'amount_MEAN': r['amount_MEAN'], 'avg_num_trxns_MEAN': r['avg_num_trxns_MEAN'],
                 'avg_trxn_amt_MEAN': r['avg_trxn_amt_MEAN'], 'NUM_COUNT': r['NUM_COUNT']})

    # SmartSegment level
    for _, g in dtree.groupby('SmartSegment').agg(agg).reset_index().iterrows():
        sid = f"SS_{int(g['SmartSegment'])}"
        rows.append({'id': sid, 'parent': 'All', 'label': f"Segment {int(g['SmartSegment'])}",
                     'amount_MEAN': g['amount_MEAN'], 'avg_num_trxns_MEAN': g['avg_num_trxns_MEAN'],
                     'avg_trxn_amt_MEAN': g['avg_trxn_amt_MEAN'], 'NUM_COUNT': g['NUM_COUNT']})

    # SmartSegment x customer_type level
    for _, g in dtree.groupby(['SmartSegment', 'customer_type']).agg(agg).reset_index().iterrows():
        sid = f"SS_{int(g['SmartSegment'])}"
        cid = f"{sid}_{g['customer_type']}"
        rows.append({'id': cid, 'parent': sid, 'label': g['customer_type'],
                     'amount_MEAN': g['amount_MEAN'], 'avg_num_trxns_MEAN': g['avg_num_trxns_MEAN'],
                     'avg_trxn_amt_MEAN': g['avg_trxn_amt_MEAN'], 'NUM_COUNT': g['NUM_COUNT']})

    # Leaf: SmartSegment x customer_type x acct_type
    for _, g in dtree.groupby(['SmartSegment', 'customer_type', 'acct_type']).agg(agg).reset_index().iterrows():
        sid = f"SS_{int(g['SmartSegment'])}"
        cid = f"{sid}_{g['customer_type']}"
        lid = f"{cid}_{g['acct_type']}"
        rows.append({'id': lid, 'parent': cid, 'label': g['acct_type'],
                     'amount_MEAN': g['amount_MEAN'], 'avg_num_trxns_MEAN': g['avg_num_trxns_MEAN'],
                     'avg_trxn_amt_MEAN': g['avg_trxn_amt_MEAN'], 'NUM_COUNT': g['NUM_COUNT']})

    tree_df = pd.DataFrame(rows)

    fig = go.Figure(go.Treemap(
        ids=tree_df['id'],
        labels=tree_df['label'],
        parents=tree_df['parent'],
        values=tree_df['NUM_COUNT'],
        customdata=np.column_stack([
            tree_df['avg_num_trxns_MEAN'].fillna(0),
            tree_df['avg_trxn_amt_MEAN'].fillna(0),
            tree_df['NUM_COUNT'].fillna(0),
            tree_df['amount_MEAN'].fillna(0),
        ]),
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Count: %{customdata[2]:.0f}<br>'
            'Avg Trxns/Week: %{customdata[0]:.0f}<br>'
            'Avg Trxn Amt: $%{customdata[1]:.0f}<br>'
            'Avg Monthly Amt: $%{customdata[3]:.0f}<br>'
            '<extra></extra>'
        ),
        texttemplate=(
            '<b>%{label}</b><br>'
            'n=%{customdata[2]:.0f}<br>'
            'trxns/wk=%{customdata[0]:.0f}<br>'
            'amt=$%{customdata[1]:.0f}'
        ),
        marker=dict(
            colors=tree_df['avg_num_trxns_MEAN'].fillna(0),
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title='Avg Trxns/Wk'),
        ),
    ))
    fig.update_layout(
        title='AML Dynamic Segments',
        font_size=14,
        margin=dict(t=50, l=25, r=25, b=25),
    )
    return fig, tree_df
# Remove rows with outliers in any of the specified columns using IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.10)
        Q3 = df[col].quantile(0.90)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= Q1) & (df[col] <= Q3)]
    return df

def plot_pct_metric(df, metric):
    scores=[]
    pcts = []
    Precision = []
    Recall = []
    for i in range(0, 101):
        df_pct = df.head(int(len(df)*(i/100)))
        TP = df_pct[df_pct['true_positives'] ==1].shape[0]
        FP = df_pct[df_pct['false_positives'] ==1].shape[0]
        TN = df_pct[df_pct['true_negatives'] ==1].shape[0]
        FN = df_pct[df_pct['false_negatives'] ==1].shape[0]
        if (metric == 'Jstat'):
            if ((TP+FN == 0) or (FP+TN == 0)):
                metric_J = 0
            else:
                 metric_J = (TP/(TP+FN))+(TN/(FP+TN)) - 1
            scores.append (metric_J)
        elif (metric == 'F1'):
            if ((TP+FP) == 0):
                P = 0
            else:
                P = TP / (TP+FP)
            if ((TP+FN) == 0):
                R = 0
            else:
                R = TP / (TP+FN)
            Precision.append(P)
            Recall.append(R)
            if (P+R != 0):
                metric_F1 = 2 * (P* R) / (P+R)
            else:
                metric_F1 = 0
            scores.append (metric_F1)
        pcts.append(i/100)
    maxJ = max(scores)
    max_index = scores.index(maxJ)
    if (metric == 'Jstat'):
        fig = px.line( x=pcts, y=scores)
        # Highlight the maximum point
        fig.add_scatter(x= [pcts[max_index]],y=[scores[max_index]],
                    mode='markers', marker=dict(color='red', size=10),
                    marker_symbol = ['star'],
                    name=f'Max J: ({scores[max_index]})')
        #fig.show()
        return fig
    else:
        fig1 = px.line( x=pcts, y=scores)
        # Highlight the maximum point
        fig1.add_scatter(x= [pcts[max_index]],y=[scores[max_index]],
            mode='markers', marker=dict(color='red', size=10),
            marker_symbol = ['star'],
            name=f'Max J: ({scores[max_index]})')
        fig2 = px.line( x= Recall, y = Precision)
        # Highlight the maximum point
        fig2.add_scatter(x= [Recall[max_index]],y=[Precision[max_index]],
                    mode='markers', marker=dict(color='red', size=10),
                    marker_symbol = ['star'],
                    name=f'Max J: ({scores[max_index]})')
        return fig1, fig2

def plot_thresholds_metric(df_segment, threshold, bump_pct, segment, metric):
    scores = []
    thresholds = []
    df_segment = remove_outliers_iqr(df_segment, [threshold])
    threshold_min = df_segment[threshold].min()
    threshold_max = df_segment[threshold].max()
    threshold_bump = threshold_min
    while threshold_bump < threshold_max:
        df_trxn_set = df_segment[df_segment[threshold] >= threshold_bump]
        TP = df_trxn_set[df_trxn_set['true_positives'] ==1].shape[0]
        FP = df_trxn_set[df_trxn_set['false_positives'] ==1].shape[0]
        TN = df_trxn_set[df_trxn_set['true_negatives'] ==1].shape[0]
        FN = df_trxn_set[df_trxn_set['false_negatives'] ==1].shape[0]
        if (metric == 'Jstat'):

            if ((TP+FN == 0) or (FP+TN == 0)):
                metric_J = 0
            else:
                metric_J = (TP/(TP+FN))+(TN/(FP+TN)) - 1
            scores.append (metric_J)
        elif (metric == 'F1'):
            if ((TP+FP) == 0):
                P = 0
            else:
                P = TP / (TP+FP)
            if ((TP+FN) == 0):
                R = 0
            else:
                R = TP / (TP+FN)
            if (P+R != 0):
                metric_F1 = 2 * (P* R) / (P+R)
            else:
                metric_F1 = 0
            scores.append (metric_F1)
        thresholds.append(round(threshold_bump, 2))
        threshold_bump = threshold_bump + (threshold_bump * bump_pct)
    fig = px.line( x=thresholds, y=scores)
    maxJ = max(scores)
    max_index = scores.index(maxJ)
    fig.add_scatter(x= [thresholds[max_index]],y=[scores[max_index]],
                mode='markers', marker=dict(color='red', size=10),
                marker_symbol = ['star'],
                name=f'Max J: ({scores[max_index]})')
    #fig.show()
    #write this out to a file for this segment for plotting later
    df_Jstats = pd.DataFrame({f'YJ_{threshold}':thresholds,'YJstats':scores})
    df_Jstats.to_csv(f"Jstats_segment_{segment}_{threshold}.csv", index=False)
    return fig

def tpr_fpr_plot(df):
    tpr = []
    fpr = []
    tp_cnts = 0
    fp_cnts = 0
    df_alerts = df[df['alert']==1].reset_index()
    tp_total = df_alerts[df_alerts['true_positives'] == 1].shape[0]
    fp_total = df_alerts[df_alerts['false_positives'] == 1].shape[0]
    total_alerts = df_alerts.shape[0]
    Jstat = 0
    max_index = 0
    for index, row in df_alerts.iterrows():
        if row['true_positives'] == 1:
            tp_cnts = tp_cnts+1
        elif row['false_positives'] == 1:
            fp_cnts = fp_cnts+1
        tpr.append(tp_cnts/tp_total)
        fpr.append(fp_cnts/fp_total)
        #J stat
        if ( ((tp_cnts/tp_total) - (index / total_alerts)) > Jstat):
            Jstat = ((tp_cnts/tp_total) - (index / total_alerts)) #second part is random guess value
            max_index = index

    fig = px.line( x=fpr, y=tpr)
    fig.add_scatter(x= [fpr[max_index]],y=[tpr[max_index]],
            mode='markers', marker=dict(color='red', size=10),
            marker_symbol = ['star'],
            name=f'Max J: ({Jstat})')
    #fig.show()
    return fig

def add_sub_plots(fig, subplot, row_id, col_id, x_title, y_title):
    for trace in subplot.data:
        fig.add_trace(trace, row=row_id, col=col_id)
        fig.update_xaxes(title_text=x_title, row=row_id, col=col_id)
        fig.update_yaxes(title_text=y_title, row=row_id, col=col_id)
    return fig

def perform_clustering(df, customer_type=None, n_clusters=4):
    """
    Cluster active customers (avg_num_trxns > 0) using numeric + categorical features.
    Inactive accounts are assigned to a 'No Activity' cluster (index = n_clusters).
    Returns (scatter_fig, stats_text, df_combined).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Filter by segment
    if customer_type == "Business":
        df_work = df[df['dynamic_segment'] == 0].copy()
    elif customer_type == "Individual":
        df_work = df[df['dynamic_segment'] == 1].copy()
    else:
        df_work = df.copy()

    seg_label = customer_type or "All"

    # ── Keep only accounts with transaction history ─────────────────────
    if 'avg_num_trxns' in df_work.columns:
        df_active = df_work[df_work['avg_num_trxns'].fillna(0) > 0].copy()
    else:
        df_active = df_work.copy()
    df_inactive = pd.DataFrame()   # not used — excluded entirely

    # ── Feature set (avg_weekly_trxn_amt replaces avg_trxn_amt) ────────
    numeric_cols = [c for c in [
        'avg_num_trxns', 'avg_weekly_trxn_amt', 'trxn_amt_monthly',
        'INCOME', 'CURRENT_BALANCE', 'ACCT_AGE_YEARS', 'AGE'
    ] if c in df_active.columns]

    cat_cols = [c for c in [
        'ACCOUNT_TYPE', 'GENDER', 'AGE_CATEGORY', 'ACCT_OPEN_CHANNEL',
        'NNM', 'OFAC', '314b', 'CITIZENSHIP', 'RESIDENCY_COUNTRY'
    ] if c in df_active.columns]

    df_encoded = pd.get_dummies(df_active[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=df_active.index)
    X_num   = df_active[numeric_cols].fillna(df_active[numeric_cols].median())
    X       = pd.concat([X_num.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1).fillna(0)
    feature_cols = list(X.columns)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Auto-select K via elbow ─────────────────────────────────────────
    if n_clusters == 0:
        inertias = []
        k_range  = range(2, 9)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        diffs  = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        diffs2 = [diffs[i] - diffs[i+1] for i in range(len(diffs)-1)]
        n_clusters = list(k_range)[diffs2.index(max(diffs2)) + 1]
        print(f"Auto-selected K={n_clusters} clusters")

    # ── K-Means on active accounts only ────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_active['cluster'] = labels

    # ── PCA scatter ─────────────────────────────────────────────────────
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var1  = pca.explained_variance_ratio_[0] * 100
    var2  = pca.explained_variance_ratio_[1] * 100

    scatter_df = pd.DataFrame({
        'PC1':     X_pca[:, 0],
        'PC2':     X_pca[:, 1],
        'Cluster': [f'Cluster {l+1}' for l in labels],
    })
    # Sort so legend appears in numeric order (1, 2, 3, 4) regardless of KMeans label assignment
    cluster_order = [f'Cluster {i+1}' for i in range(n_clusters)]
    scatter_df['Cluster'] = pd.Categorical(scatter_df['Cluster'], categories=cluster_order, ordered=True)
    scatter_df = scatter_df.sort_values('Cluster')

    fig = px.scatter(
        scatter_df, x='PC1', y='PC2', color='Cluster',
        category_orders={'Cluster': cluster_order},
        title=f"Dynamic Segmentation Clustering — {seg_label} ({n_clusters} clusters, active accounts only)",
        labels={
            'PC1': f'PC1 ({var1:.1f}% variance)',
            'PC2': f'PC2 ({var2:.1f}% variance)',
        },
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend=dict(itemsizing='constant'))

    # ── Stats ────────────────────────────────────────────────────────────
    _COL_DISPLAY = {
        'avg_num_trxns':        'Avg Weekly Transactions',
        'avg_weekly_trxn_amt':  'Avg Weekly Txn Amount ($)',
        'trxn_amt_monthly':     'Monthly Txn Volume ($)',
        'INCOME':               'Income ($)',
        'CURRENT_BALANCE':      'Current Balance ($)',
        'ACCT_AGE_YEARS':       'Account Age (years)',
        'AGE':                  'Age',
    }

    n_num         = len(numeric_cols)
    n_cat_encoded = len(df_encoded.columns)
    stats_lines = [
        f"=== PRE-COMPUTED CLUSTER STATS (copy verbatim, do not compute new numbers) ===",
        f"Segment: {seg_label} | Active accounts: {len(df_active):,} (excluded {len(df_work) - len(df_active):,} with no transactions)",
        f"Clusters: {n_clusters} | Features: {n_num} numeric + {n_cat_encoded} encoded categorical ({len(cat_cols)} original)",
        f"PCA variance explained: PC1={var1:.1f}%, PC2={var2:.1f}%",
        "",
    ]
    # Columns to skip in stats display per segment
    _skip_cols = set()
    if seg_label.upper() == "BUSINESS":
        _skip_cols.add("INCOME")   # income is individual-only
        _skip_cols.add("AGE")      # age not collected for businesses

    total_active = len(df_active)
    for i in range(n_clusters):
        c   = df_active[df_active['cluster'] == i]
        pct = 100 * len(c) / total_active if total_active > 0 else 0
        stats_lines.append(f"**Cluster {i+1}**")
        stats_lines.append(f"- Customers: **{len(c):,}** ({pct:.1f}% of active accounts)")
        for col in numeric_cols:
            if col in _skip_cols:
                continue
            val = c[col].median()
            if not (val != val):  # skip NaN
                label = _COL_DISPLAY.get(col, col)
                stats_lines.append(f"- {label}: **{val:,.1f}**")
        stats_lines.append("")  # blank line after each cluster block

    stats_lines.append("=== END PRE-COMPUTED CLUSTER STATS ===")
    return fig, "\n".join(stats_lines), df_active


def _cluster_title(trxns, amt, overall_trxns, overall_amt):
    """Generate a descriptive cluster title based on relative profile values."""
    freq  = "High Freq"  if trxns > overall_trxns * 1.15 else ("Low Freq"  if trxns < overall_trxns * 0.85 else "Mid Freq")
    value = "High Value" if amt   > overall_amt   * 1.15 else ("Low Value" if amt   < overall_amt   * 0.85 else "Mid Value")
    return f"{freq} / {value}"


# Columns excluded from treemap dimension discovery — IDs, numerics, internal flags
_DIM_EXCLUDE = {
    'customer_id', 'account_id', 'cluster', 'cluster_label', 'dynamic_segment',
    'is_sar', 'is_fp', 'is_alerted', 'is_fn', 'pct_active',
    'avg_num_trxns', 'avg_weekly_trxn_amt', 'trxn_amt_monthly', 'avg_trxn_amt',
    'income', 'current_balance', 'acct_age_years', 'age',
    'total_trxn_amt', 'cashout_count', 'sar_score', 'alert_count',
    'customer_type',  # used as the segment split level, not a sub-dimension
}


def discover_dims(df, segment=None, availability=0.70, max_cardinality=20):
    """
    Discover categorical columns suitable as treemap dimensions from df.

    Parameters
    ----------
    df             : segmentation DataFrame (output of DS_CSV load)
    segment        : 'BUSINESS' or 'INDIVIDUAL' — filter df before scanning, or None for all
    availability   : minimum fraction of non-null values required (default 0.70)
    max_cardinality: maximum number of unique values for a column to be considered categorical

    Returns
    -------
    List of column names suitable as treemap hierarchy dimensions, ordered by availability desc.
    """
    if segment and 'customer_type' in df.columns:
        sub = df[df['customer_type'].str.upper() == segment.upper()]
    else:
        sub = df

    if len(sub) == 0:
        return []

    n = len(sub)
    scored = []
    for col in sub.columns:
        if col.lower() in _DIM_EXCLUDE:
            continue
        col_data = sub[col].dropna()
        avail = len(col_data) / n
        if avail < availability:
            continue
        n_unique = sub[col].nunique(dropna=True)
        if 1 < n_unique <= max_cardinality:
            scored.append((col, avail))

    # Sort by availability descending so highest-coverage dims come first
    scored.sort(key=lambda x: -x[1])
    return [col for col, _ in scored]


def smartseg_tree_dynamic(df_clustered, seg_label="All", dims=None, df_rule_sweep=None):
    """
    Build a treemap from a cluster-labelled DataFrame (output of perform_clustering).

    dims can be:
      - None / list: same hierarchy path applied to all rows.
            e.g. ['customer_type', 'ACCOUNT_TYPE']
      - dict: customer_type is always the first level after Cluster;
            the dict maps each customer_type value to its own sub-dim path.
            e.g. {
                'BUSINESS':   ['ACCOUNT_TYPE', 'ACCOUNT_AGE_CATEGORY'],
                'INDIVIDUAL': ['ACCOUNT_TYPE', 'GENDER', 'AGE_CATEGORY', 'INCOME_BAND'],
            }

    Only columns actually present in df_clustered are used.
    Each cluster gets its own distinct color; no heatmap colorscale.
    """
    PALETTE = px.colors.qualitative.Set1

    if dims is None:
        dims = ['customer_type', 'ACCOUNT_TYPE']

    df = df_clustered.copy()

    # Enrich with SAR/alert info from rule sweep if provided
    if df_rule_sweep is not None and 'customer_id' in df.columns:
        sar_map   = df_rule_sweep.groupby('customer_id')['is_sar'].max()
        alerted   = set(df_rule_sweep['customer_id'].unique())
        df['is_sar']     = df['customer_id'].map(sar_map).fillna(0).astype(int)
        df['is_alerted'] = df['customer_id'].isin(alerted).astype(int)
        df['is_fp']      = ((df['is_alerted'] == 1) & (df['is_sar'] == 0)).astype(int)
    else:
        df['is_sar'] = 0; df['is_alerted'] = 0; df['is_fp'] = 0

    # Overall means over active accounts only for cluster title relative comparisons
    _active_all = df[df['avg_num_trxns'].fillna(0) > 0] if 'avg_num_trxns' in df.columns else df
    overall_trxns = _active_all['avg_num_trxns'].mean()       if len(_active_all) > 0 and 'avg_num_trxns'       in _active_all.columns else 1
    overall_amt   = _active_all['avg_weekly_trxn_amt'].mean() if len(_active_all) > 0 and 'avg_weekly_trxn_amt' in _active_all.columns else 1

    # Build indicative title per cluster (all clusters are active — inactive excluded before clustering)
    cluster_titles = {}
    for counter, (i, grp) in enumerate(df.groupby('cluster'), start=1):
        title = _cluster_title(
            grp['avg_num_trxns'].mean() if 'avg_num_trxns' in grp.columns else 0,
            grp['avg_weekly_trxn_amt'].mean() if 'avg_weekly_trxn_amt' in grp.columns else 0,
            overall_trxns, overall_amt,
        )
        cluster_titles[i] = f"C{counter}: {title}"

    df['cluster_label'] = df['cluster'].map(cluster_titles)

    rows = []

    def add_row(rid, parent, label, sub, cidx=None):
        # Filter to active accounts (with transactions) for transaction metrics
        active = sub[sub['avg_num_trxns'] > 0] if 'avg_num_trxns' in sub.columns else sub
        n_active = len(active)
        pct_active = round(100 * n_active / len(sub), 1) if len(sub) > 0 else 0
        rows.append({
            'id': rid, 'parent': parent, 'label': label,
            # Transaction frequency: median over active accounts (robust to outliers)
            'avg_num_trxns':       active['avg_num_trxns'].median()       if n_active > 0 and 'avg_num_trxns'       in active.columns else 0,
            # Transaction amounts: median to avoid single large-transaction accounts skewing results
            'avg_weekly_trxn_amt': active['avg_weekly_trxn_amt'].median() if n_active > 0 and 'avg_weekly_trxn_amt' in active.columns else 0,
            'trxn_amt_monthly':    active['trxn_amt_monthly'].median()    if n_active > 0 and 'trxn_amt_monthly'    in active.columns else 0,
            # Demographics: mean over all accounts in this node
            'INCOME':           sub['INCOME'].mean()              if 'INCOME' in sub.columns else 0,
            'AGE':              sub['AGE'].mean()                 if 'AGE'    in sub.columns else 0,
            'pct_active': pct_active,
            'NUM_COUNT': len(sub),
            'cidx': cidx,
            # AML risk counts
            'n_sar':     int(sub['is_sar'].sum()),
            'n_alerted': int(sub['is_alerted'].sum()),
            'n_fp':      int(sub['is_fp'].sum()),
        })

    def build_nodes(sub_df, parent_id, remaining_dims, cidx):
        """Recursively build treemap nodes for each dimension level."""
        if not remaining_dims:
            return
        dim = remaining_dims[0]
        if dim not in sub_df.columns:
            return
        for val, grp in sub_df.groupby(dim, dropna=True):
            val_str = str(val)
            node_id = f"{parent_id}__{dim}_{val_str}"
            add_row(node_id, parent_id, val_str, grp, cidx=cidx)
            build_nodes(grp, node_id, remaining_dims[1:], cidx)

    SMALL_CLUSTER_THRESHOLD = 0.01  # clusters < 1% of total go into a "Small Clusters" group

    total_rows = len(df)
    small_clusters = {cl for cl, grp in df.groupby('cluster_label')
                      if len(grp) / total_rows < SMALL_CLUSTER_THRESHOLD} if total_rows > 0 else set()

    # Root
    add_row('All', '', f'Dynamic Segments - {seg_label}', df, cidx=None)

    # Add a "Small Clusters" bucket if any clusters are below threshold
    if small_clusters:
        df_small = df[df['cluster_label'].isin(small_clusters)]
        add_row('SMALL', 'All', f'Small Clusters (<1%) — {len(df_small):,} accounts', df_small, cidx=None)

    # Cluster level
    for cl, grp in df.groupby('cluster_label'):
        cid  = f"CL__{cl}"
        cidx = next((k for k, v in cluster_titles.items() if v == cl), None)
        parent = 'SMALL' if cl in small_clusters else 'All'
        add_row(cid, parent, cl, grp, cidx=cidx)

        if isinstance(dims, dict):
            # customer_type is always the first level; each type gets its own sub-dims
            if 'customer_type' not in grp.columns:
                continue
            for ct, cgrp in grp.groupby('customer_type'):
                ctid = f"{cid}__ct_{ct}"
                add_row(ctid, cid, ct, cgrp, cidx=cidx)
                ct_sub_dims = [d for d in dims.get(ct, []) if d in cgrp.columns]
                build_nodes(cgrp, ctid, ct_sub_dims, cidx)
        else:
            # List mode: recurse through all dims uniformly
            active_dims = [d for d in dims if d in grp.columns]
            build_nodes(grp, cid, active_dims, cidx)

    tree_df = pd.DataFrame(rows)

    # Boost small cluster display values so they're visible in the treemap.
    # Use 5% of total as the minimum display size; actual counts are shown in hover labels.
    if small_clusters:
        min_display = int(max(total_rows * 0.05, 1))
        small_ids = {f"CL__{cl}" for cl in small_clusters} | {'SMALL'}
        tree_df.loc[tree_df['id'].isin(small_ids), 'NUM_COUNT'] = \
            tree_df.loc[tree_df['id'].isin(small_ids), 'NUM_COUNT'].clip(lower=min_display).astype(int)

    # Per-node colors: neutral grey for root, cluster color for all other nodes
    node_colors = []
    for _, r in tree_df.iterrows():
        if r['cidx'] is None or pd.isna(r['cidx']):
            node_colors.append('#CCCCCC')
        else:
            node_colors.append(PALETTE[int(r['cidx']) % len(PALETTE)])

    fig = go.Figure(go.Treemap(
        ids=tree_df['id'],
        labels=tree_df['label'],
        parents=tree_df['parent'],
        values=tree_df['NUM_COUNT'],
        customdata=np.column_stack([
            tree_df['avg_num_trxns'].fillna(0),       # 0
            tree_df['avg_weekly_trxn_amt'].fillna(0), # 1
            tree_df['NUM_COUNT'].fillna(0),            # 2
            tree_df['trxn_amt_monthly'].fillna(0),     # 3
            tree_df['INCOME'].fillna(0),               # 4
            tree_df['AGE'].fillna(0),                  # 5
            tree_df['pct_active'].fillna(0),           # 6
            tree_df['n_sar'].fillna(0),                # 7
            tree_df['n_alerted'].fillna(0),            # 8
            tree_df['n_fp'].fillna(0),                 # 9
        ]),
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Count: %{customdata[2]:.0f}<br>'
            'Active (w/ txns): %{customdata[6]:.1f}%<br>'
            'Avg Trxns/Week: %{customdata[0]:.1f}<br>'
            'Avg Weekly Trxn Amt: $%{customdata[1]:.0f}<br>'
            'Avg Monthly Trxn Amt: $%{customdata[3]:.0f}<br>'
            + ('' if seg_label.upper() == 'BUSINESS' else
               'Avg Income: $%{customdata[4]:.0f}<br>'
               'Avg Age: %{customdata[5]:.0f}<br>')
            + '─────────────────<br>'
            'Alerts: %{customdata[8]:.0f} | SARs: %{customdata[7]:.0f} | FPs: %{customdata[9]:.0f}<br>'
            '<extra></extra>'
        ),
        texttemplate=(
            '<b>%{label}</b><br>'
            'n=%{customdata[2]:.0f}<br>'
            'SAR=%{customdata[7]:.0f} FP=%{customdata[9]:.0f}<br>'
            'wk=$%{customdata[1]:.0f}'
        ),
        marker=dict(colors=node_colors),
    ))
    fig.update_layout(
        title=f'AML Dynamic Segments - {seg_label}',
        font_size=14,
        margin=dict(t=50, l=25, r=25, b=25),
    )
    return fig


if __name__ == "__main__":
    pass
