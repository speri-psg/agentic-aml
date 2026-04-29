"""
ds_data_prep.py — Dynamic Segmentation Data Preparation

Schema:
  aml_s_customers.csv          id → customer_id
  aml_s_accounts.csv           id → account_id
  aml_s_account_relationship.csv  account_id, customer_id (many-to-many)
  aml_s_transactions.csv       subject_id = account_id, timestamp, amount

Output: docs/ds_segmentation_data.csv — one row per customer-account pair

Usage:
    python ds_data_prep.py
"""

import os
import pandas as pd
import numpy as np

SS_DIR      = "ss_files"
OUTPUT_FILE = "docs/ds_segmentation_data.csv"
REF_DATE    = pd.Timestamp.now()

# Synth mode: when ARIA_SYNTH_DATA_DIR is set, read from aria_synth/ and write to separate file
_SYNTH_DIR   = os.getenv("ARIA_SYNTH_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "aria_synth"))
_USE_SYNTH   = bool(os.getenv("ARIA_SYNTH_DATA_DIR")) and os.path.isdir(_SYNTH_DIR)
OUTPUT_SYNTH = "docs/ds_segmentation_synth.csv"


def load(filename):
    path = os.path.join(SS_DIR, filename)
    print(f"  Loading {filename} ...", end=" ")
    df = pd.read_csv(path)
    print(f"{len(df):,} rows, {len(df.columns)} cols")
    return df


def _strip_aria(df):
    """Strip 'aria_' prefix from all column names."""
    return df.rename(columns={c: c[5:] if c.startswith("aria_") else c for c in df.columns})


def _load_synth():
    """Load the four aria_synth/ files and normalise them to match the production schema."""
    def _read(fname):
        path = os.path.join(_SYNTH_DIR, fname)
        print(f"  Loading {fname} ...", end=" ")
        df = pd.read_csv(path, low_memory=False)
        print(f"{len(df):,} rows, {len(df.columns)} cols")
        return _strip_aria(df)

    cust = _read("aria_customers.csv")
    # aria_customer_id → customer_id (after strip already done), rename to match id→customer_id path
    if "customer_id" in cust.columns and "id" not in cust.columns:
        cust = cust.rename(columns={"customer_id": "id"})
    # residency_country → residency_country_id
    if "residency_country" in cust.columns and "residency_country_id" not in cust.columns:
        cust = cust.rename(columns={"residency_country": "residency_country_id"})

    acct = _read("aria_accounts.csv")
    # aria_account_id → account_id (after strip), rename to id for the id→account_id rename
    if "account_id" in acct.columns and "id" not in acct.columns:
        acct = acct.rename(columns={"account_id": "id"})

    rel  = _read("aria_account_relationships.csv")
    # relationship_type → type (production code reads rel['type'])
    if "relationship_type" in rel.columns and "type" not in rel.columns:
        rel = rel.rename(columns={"relationship_type": "type"})

    txn  = _read("aria_transactions.csv")
    # subject_id, timestamp, amount, cash_direction — already correct after strip

    return cust, acct, rel, txn


def normalize_account_type(x):
    """Normalize raw account type strings to 5 standard categories."""
    if pd.isna(x):
        return 'Other'
    t = str(x).lower()
    if 'checking' in t:
        return 'Checking'
    if 'certificate' in t or 'cert' in t:
        return 'Certificate of Deposit'
    if 'saving' in t or 'money market' in t:
        return 'Savings'
    if any(k in t for k in [
        'loan', 'mortgage', 'line of credit', 'home equity',
        'auto', 'credit card', 'visa', 'heloc', 'indirect',
        'preapproved', 'personal', 'boat', 'motorcycle',
        'mobile home', 'recreational', 'motorhome', 'prescreen',
        'workout', 'disaster'
    ]):
        return 'Loan'
    return 'Other'


def compute_txn_aggregates(txn):
    """Per-account transaction aggregates from transactions file."""
    df = txn.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Separate cash-in and cash-out
    df['is_cashin']  = (df['cash_direction'] == 'CashIn').astype(int)
    df['is_cashout'] = (df['cash_direction'] == 'CashOut').astype(int)

    grp = df.groupby('subject_id')

    # Date range for avg_trxns_week
    date_min = grp['timestamp'].min()
    date_max = grp['timestamp'].max()
    weeks_active = ((date_max - date_min).dt.days / 7).clip(lower=1)

    # Monthly amount: group by (account, year-month), then average across months
    df['_ym'] = df['timestamp'].dt.to_period('M')
    monthly_totals = df.groupby(['subject_id', '_ym'])['amount'].sum()
    avg_monthly = monthly_totals.groupby(level=0).mean().rename('avg_monthly_trxn_amt')

    agg = pd.DataFrame({
        'trxn_count':          grp['amount'].count(),
        'total_trxn_amt':      grp['amount'].sum(),
        'avg_trxn_amt':        grp['amount'].mean(),
        'max_trxn_amt':        grp['amount'].max(),
        'min_trxn_amt':        grp['amount'].min(),
        'std_trxn_amt':        grp['amount'].std().fillna(0),
        'cashin_count':        grp['is_cashin'].sum(),
        'cashout_count':       grp['is_cashout'].sum(),
        'avg_trxns_week':      (grp['amount'].count() / weeks_active).round(2),
        'avg_weekly_trxn_amt': (grp['amount'].sum()   / weeks_active).round(2),
        'weeks_active':        weeks_active.round(1),
    }).reset_index().rename(columns={'subject_id': 'account_id'})

    agg = agg.merge(avg_monthly.reset_index().rename(columns={'subject_id': 'account_id'}),
                    on='account_id', how='left')

    # Cash-in ratio
    agg['cashin_ratio'] = (agg['cashin_count'] / agg['trxn_count'].clip(lower=1)).round(3)

    return agg


def prepare_data():
    print(f"\n{'='*60}")
    mode = "SYNTHETIC" if _USE_SYNTH else "PRODUCTION"
    print(f"Dynamic Segmentation Data Prep  [{mode}]")
    print(f"{'='*60}\n")

    out_file = OUTPUT_SYNTH if _USE_SYNTH else OUTPUT_FILE

    # ── 1. Load files ────────────────────────────────────────────────
    print("--- Loading files ---")
    if _USE_SYNTH:
        cust, acct, rel, txn = _load_synth()
    else:
        cust = load("aml_s_customers.csv")
        acct = load("aml_s_accounts.csv")
        rel  = load("aml_s_account_relationship.csv")
        txn  = load("aml_s_transactions.csv")

    # ── 2. Rename IDs ────────────────────────────────────────────────
    cust = cust.rename(columns={'id': 'customer_id'})
    acct = acct.rename(columns={'id': 'account_id'})

    # ── 3. Derive customer features ──────────────────────────────────
    print("\n--- Deriving customer features ---")
    cust['birthdate'] = pd.to_datetime(cust['birthdate'], errors='coerce')
    cust['age'] = ((REF_DATE - cust['birthdate']).dt.days / 365.25).round(1)

    # Boolean flags → int
    bool_cols = ['internal_employee', 'entity', 'is_customer', 'exempt_cdd',
                 'exempt_sanctions', 'pep', 'subpeona', 'is_314b', 'negative_news', 'ofac']
    for c in bool_cols:
        if c in cust.columns:
            cust[c] = cust[c].fillna(False).astype(int)

    cust_keep = ['customer_id', 'age', 'gender', 'marital_status', 'occupation',
                 'citizenship', 'gross_annual_income', 'nationality_country_id',
                 'residency_country_id', 'pep', 'is_314b', 'negative_news', 'ofac',
                 'exempt_cdd', 'exempt_sanctions', 'naics', 'sic_code', 'entity']
    cust_keep = [c for c in cust_keep if c in cust.columns]
    cust = cust[cust_keep]
    print(f"  Customer features kept: {cust_keep}")

    # ── 4. Derive account features ───────────────────────────────────
    print("\n--- Deriving account features ---")
    acct['open_date'] = pd.to_datetime(acct['open_date'], errors='coerce')
    acct['acct_age_years'] = ((REF_DATE - acct['open_date']).dt.days / 365.25).round(2)

    acct_keep = ['account_id', 'account_type', 'product_name', 'status',
                 'initial_balance', 'current_balance', 'acct_age_years',
                 'loan_amount']
    acct_keep = [c for c in acct_keep if c in acct.columns]
    acct = acct[acct_keep]
    print(f"  Account features kept: {acct_keep}")

    # Normalize account_type to standard categories using keyword matching
    acct['account_type'] = acct['account_type'].apply(normalize_account_type)
    print(f"  Account type distribution: {acct['account_type'].value_counts().to_dict()}")

    # ── 5. Transaction aggregates ─────────────────────────────────────
    print("\n--- Computing transaction aggregates ---")
    txn_agg = compute_txn_aggregates(txn)
    print(f"  Aggregates computed for {len(txn_agg):,} accounts")
    print(f"  Cols: {[c for c in txn_agg.columns if c != 'account_id']}")

    # ── 6. Join everything ───────────────────────────────────────────
    print("\n--- Joining ---")
    # relationship: keep HOLDER or all types
    print(f"  Relationship types: {rel['type'].unique()}")
    # Use all relationship types (or filter to HOLDER if needed)
    mapping = rel[['account_id', 'customer_id', 'type']].drop_duplicates()

    df = mapping.merge(cust, on='customer_id', how='left')
    df = df.merge(acct, on='account_id',  how='left')
    df = df.merge(txn_agg, on='account_id', how='left')
    print(f"  Final: {len(df):,} customer-account rows, {len(df.columns)} columns")

    # ── 7. Fill missing transaction data (accounts with no transactions) ─
    txn_fill_cols = ['trxn_count', 'total_trxn_amt', 'avg_trxn_amt', 'avg_trxns_week',
                     'avg_monthly_trxn_amt', 'cashin_count', 'cashout_count', 'cashin_ratio']
    for c in txn_fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # ── 8. Normalize column names to match perform_clustering schema ──
    print("\n--- Normalizing column names ---")
    df = df.rename(columns={
        'avg_trxns_week':       'avg_num_trxns',
        'avg_monthly_trxn_amt': 'trxn_amt_monthly',
        'gross_annual_income':  'INCOME',
        'current_balance':      'CURRENT_BALANCE',
        'acct_age_years':       'ACCT_AGE_YEARS',
        'age':                  'AGE',
        'account_type':         'ACCOUNT_TYPE',
        'gender':               'GENDER',
        'citizenship':          'CITIZENSHIP',
        'residency_country_id': 'RESIDENCY_COUNTRY',
        'is_314b':              '314b',
        'ofac':                 'OFAC',
    })

    # Derive customer_type and dynamic_segment from entity flag
    # entity=1 (True) → Business, entity=0 (False) → Individual
    df['customer_type']    = df['entity'].map({1: 'BUSINESS', 0: 'INDIVIDUAL'}).fillna('INDIVIDUAL')
    df['dynamic_segment'] = df['entity'].map({1: 0, 0: 1}).fillna(1).astype(int)
    print(f"  customer_type distribution: {df['customer_type'].value_counts().to_dict()}")

    # Age category bins
    df['AGE_CATEGORY'] = pd.cut(
        df['AGE'],
        bins=[0, 30, 45, 60, 120],
        labels=['18-30', '31-45', '46-60', '60+'],
        right=True,
    ).astype(str).replace('nan', 'Unknown')

    # Income bands (mostly null in this dataset — filled as Unknown)
    df['INCOME_BAND'] = pd.cut(
        df['INCOME'],
        bins=[0, 50000, 100000, 200000, float('inf')],
        labels=['<50K', '50K-100K', '100K-200K', '200K+'],
        right=True,
    ).astype(str).replace('nan', 'Unknown')

    # Fill null GENDER for tree groupby
    df['GENDER'] = df['GENDER'].fillna('Unknown')

    # Account age category for business segmentation
    df['ACCOUNT_AGE_CATEGORY'] = pd.cut(
        df['ACCT_AGE_YEARS'],
        bins=[-0.01, 1.0, float('inf')],
        labels=['New (0-1yr)', 'Established (>1yr)'],
        right=True,
    ).astype(str).replace('nan', 'Unknown')

    # ── 9. Summary ───────────────────────────────────────────────────
    print("\n--- Output Summary ---")
    print(f"  Rows:        {len(df):,}")
    print(f"  Columns:     {len(df.columns)}")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"  Numeric:     {num_cols}")
    print(f"  Categorical: {cat_cols}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f"  Nulls:\n{nulls.to_string()}")

    # ── 10. Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"\n  Saved -> {out_file}")
    print(f"{'='*60}\n")
    return df


if __name__ == "__main__":
    prepare_data()
