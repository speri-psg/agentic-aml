"""
lambda_ofac.py — OFAC SDN screening for ARIA.

Provides:
  ofac_screening(customers_df, filter_type=None) → (text, fig)

Screening logic:
  1. Country-based: customers whose citizenship is in a comprehensively
     sanctioned country (Iran, North Korea, Cuba, Syria) — blanket prohibition.
  2. Targeted sanctions: customers from Myanmar, Russia, Belarus, Venezuela
     who are flagged ofac=True in the customer record.
  3. SDN list stats: summary from the local OFAC SQLite DB.
"""

import sqlite3
import pandas as pd
import plotly.graph_objects as go
from config import OFAC_DB, CUSTOMERS_CSV
from ofac_fuzzy import screen_name as _fuzzy_screen, format_results as _fuzzy_format

# ── Sanctioned country classification ─────────────────────────────────────────
COMPREHENSIVE = {"Iran", "North Korea", "Cuba", "Syria"}
TARGETED      = {"Myanmar", "Russia", "Belarus", "Venezuela"}

# Corresponding OFAC sanctions program labels
PROGRAM_LABEL = {
    "Iran":        "IRAN / IFSR (Comprehensive)",
    "North Korea": "DPRK (Comprehensive / FinCEN Primary Concern)",
    "Cuba":        "CUBA (Comprehensive)",
    "Syria":       "SYRIA (Comprehensive)",
    "Myanmar":     "BURMA-EO14014 (Targeted)",
    "Russia":      "RUSSIA-EO14024 (Targeted)",
    "Belarus":     "BELARUS-EO14038 (Targeted)",
    "Venezuela":   "VENEZUELA-EO13850 (Targeted)",
}


def _load_customers() -> pd.DataFrame:
    df = pd.read_csv(CUSTOMERS_CSV, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _sdn_summary() -> dict:
    """Pull aggregate stats from the local OFAC SDN SQLite DB."""
    try:
        con = sqlite3.connect(OFAC_DB)
        cur = con.cursor()
        total       = cur.execute("SELECT COUNT(*) FROM sdn").fetchone()[0]
        individuals = cur.execute("SELECT COUNT(*) FROM sdn WHERE sdn_type='individual'").fetchone()[0]
        entities    = cur.execute("SELECT COUNT(*) FROM sdn WHERE sdn_type='entity'").fetchone()[0]
        vessels     = cur.execute("SELECT COUNT(*) FROM sdn WHERE sdn_type='vessel'").fetchone()[0]
        by_country  = cur.execute(
            "SELECT country, COUNT(*) as n FROM sdn GROUP BY country ORDER BY n DESC LIMIT 10"
        ).fetchall()
        by_type     = cur.execute(
            "SELECT sanction_type, COUNT(*) as n FROM sdn GROUP BY sanction_type ORDER BY n DESC"
        ).fetchall()
        con.close()
        return {
            "total": total, "individuals": individuals,
            "entities": entities, "vessels": vessels,
            "by_country": by_country, "by_type": by_type,
        }
    except Exception as e:
        return {"error": str(e)}


def ofac_screening(customers_df: pd.DataFrame = None,
                   filter_type: str = None) -> tuple:
    """
    Run OFAC screening against the customer population.

    Parameters
    ----------
    customers_df : optional override DataFrame (uses CUSTOMERS_CSV if None)
    filter_type  : "comprehensive" | "targeted" | "all" | None (= "all")

    Returns
    -------
    (markdown_text, plotly_figure)
    """
    # ── load customers ─────────────────────────────────────────────────────────
    try:
        df = _load_customers() if customers_df is None else customers_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
    except Exception as e:
        return f"Error loading customer data: {e}", None

    total_customers = len(df)

    # ── SDN DB summary ─────────────────────────────────────────────────────────
    sdn = _sdn_summary()

    # ── Country-based screening ────────────────────────────────────────────────
    if "citizenship" not in df.columns:
        df["citizenship"] = "Unknown"
    if "ofac" not in df.columns:
        df["ofac"] = "False"

    df["_comprehensive_hit"] = df["citizenship"].isin(COMPREHENSIVE)
    df["_targeted_hit"]      = df["citizenship"].isin(TARGETED) & \
                               (df["ofac"].astype(str).str.lower() == "true")
    df["_any_ofac"]          = df["_comprehensive_hit"] | df["_targeted_hit"] | \
                               (df["ofac"].astype(str).str.lower() == "true")

    comp_hits    = int(df["_comprehensive_hit"].sum())
    target_hits  = int(df["_targeted_hit"].sum())
    any_hits     = int(df["_any_ofac"].sum())

    # Breakdown by citizenship for flagged customers
    flagged = df[df["_any_ofac"]]
    by_ctz = (flagged.groupby("citizenship").size()
                      .reset_index(name="customers")
                      .sort_values("customers", ascending=False))

    # ── Build markdown text ────────────────────────────────────────────────────
    lines = []

    if "error" not in sdn:
        lines += [
            "## OFAC SDN List — Reference Database",
            f"- **Total SDN entries:** {sdn['total']:,}",
            f"  - Individuals: **{sdn['individuals']:,}** | "
            f"Entities: **{sdn['entities']:,}** | "
            f"Vessels: **{sdn['vessels']:,}**",
            "",
            "**SDN entries by sanctions program:**",
        ]
        for country, n in sdn["by_country"]:
            lines.append(f"  - {country}: **{n:,}**")
        lines.append("")

    lines += [
        "## Customer OFAC Screening Results",
        f"- Total customers screened: **{total_customers:,}**",
        f"- **Comprehensive sanctions hits** (Iran, North Korea, Cuba, Syria): "
        f"**{comp_hits:,}** ({comp_hits/total_customers*100:.2f}%)",
        f"- **Targeted sanctions hits** (Myanmar, Russia, Belarus, Venezuela): "
        f"**{target_hits:,}** ({target_hits/total_customers*100:.2f}%)",
        f"- **Total OFAC-flagged customers:** **{any_hits:,}** "
        f"({any_hits/total_customers*100:.2f}%)",
        "",
        "**Breakdown by citizenship (flagged customers):**",
    ]
    for _, row in by_ctz.iterrows():
        ctz = row["citizenship"]
        n   = row["customers"]
        prog = PROGRAM_LABEL.get(ctz, "Other")
        sanction_class = "Comprehensive" if ctz in COMPREHENSIVE else \
                         "Targeted" if ctz in TARGETED else "Other"
        lines.append(f"  - **{ctz}** [{sanction_class}]: **{n:,}** customers — {prog}")

    lines += [
        "",
        "> **Note:** Comprehensive sanctions (Iran, DPRK, Cuba, Syria) prohibit "
        "virtually all transactions. Targeted sanctions require transaction-level "
        "screening against the SDN list before processing.",
    ]

    text = "\n".join(lines)

    # ── Build figure ───────────────────────────────────────────────────────────
    fig = _ofac_bar_chart(by_ctz, total_customers)

    return text, fig


def _ofac_bar_chart(by_ctz: pd.DataFrame, total: int) -> go.Figure:
    """Horizontal bar chart of OFAC-flagged customers by citizenship."""
    colors = []
    for ctz in by_ctz["citizenship"]:
        if ctz in COMPREHENSIVE:
            colors.append("#d32f2f")   # red — comprehensive
        elif ctz in TARGETED:
            colors.append("#f57c00")   # orange — targeted
        else:
            colors.append("#78909c")   # grey — other

    fig = go.Figure(go.Bar(
        x=by_ctz["customers"],
        y=by_ctz["citizenship"],
        orientation="h",
        marker_color=colors,
        text=[f"{n:,} ({n/total*100:.2f}%)" for n in by_ctz["customers"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Customers: %{x:,}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="OFAC-Flagged Customers by Citizenship",
            font=dict(size=14),
        ),
        xaxis_title="Number of Customers",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(300, 60 + len(by_ctz) * 35),
        margin=dict(l=120, r=160, t=60, b=40),
        annotations=[
            dict(
                x=0.01, y=1.06, xref="paper", yref="paper",
                text="<b style='color:#d32f2f'>Red</b> = Comprehensive sanctions &nbsp;"
                     "<b style='color:#f57c00'>Orange</b> = Targeted sanctions",
                showarrow=False, font=dict(size=11),
            )
        ],
    )
    return fig


def screen_name(name: str, threshold: float = 85) -> tuple:
    """
    Screen a single name against the OFAC SDN list using fuzzy matching.

    Returns (text, None) — no chart for name lookups.
    """
    try:
        hits = _fuzzy_screen(name, OFAC_DB, threshold=threshold)
        text = _fuzzy_format(name, hits, threshold)
        return text, None
    except Exception as e:
        return f"OFAC name screening error: {e}", None
