"""Tests for ds_data_prep.py — data preparation utilities (pure functions, no I/O)."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ds_data_prep import (
    normalize_account_type,
    _strip_aria,
    compute_txn_aggregates,
)


# ── normalize_account_type ─────────────────────────────────────────────────────

class TestNormalizeAccountType:
    """5 canonical categories: Checking, Savings, Certificate of Deposit, Loan, Other."""

    # Checking
    @pytest.mark.parametrize("value", [
        "Checking", "CHECKING", "checking account", "DDA Checking", "personal checking"
    ])
    def test_checking_variants(self, value):
        assert normalize_account_type(value) == "Checking"

    # Savings
    @pytest.mark.parametrize("value", [
        "Savings", "SAVINGS", "savings account", "Money Market", "money market savings"
    ])
    def test_savings_variants(self, value):
        assert normalize_account_type(value) == "Savings"

    # Certificate of Deposit
    @pytest.mark.parametrize("value", [
        "Certificate of Deposit", "cert", "CERTIFICATE", "Cert Account"
    ])
    def test_cd_variants(self, value):
        assert normalize_account_type(value) == "Certificate of Deposit"

    # Loan
    @pytest.mark.parametrize("value", [
        "Loan", "LOAN", "Mortgage", "Auto Loan", "Line of Credit",
        "Home Equity", "Credit Card", "VISA", "HELOC", "Indirect Loan",
        "Boat Loan", "Motorcycle Loan", "Mobile Home", "Recreational Vehicle",
        "Motorhome", "Personal Loan", "Preapproved", "Workout Loan",
        "Disaster Loan", "Prescreen",
    ])
    def test_loan_variants(self, value):
        assert normalize_account_type(value) == "Loan"

    # Other
    @pytest.mark.parametrize("value", [
        "Unknown", "Trust", "IRA", "Brokerage", "escrow", "OTHER",
    ])
    def test_other_variants(self, value):
        assert normalize_account_type(value) == "Other"

    # Null/NaN
    def test_none_returns_other(self):
        assert normalize_account_type(None) == "Other"

    def test_nan_returns_other(self):
        assert normalize_account_type(float("nan")) == "Other"

    def test_empty_string_returns_other(self):
        assert normalize_account_type("") == "Other"

    def test_pd_na_returns_other(self):
        assert normalize_account_type(pd.NA) == "Other"


# ── _strip_aria ────────────────────────────────────────────────────────────────

class TestStripAria:
    def test_strips_aria_prefix(self):
        df = pd.DataFrame(columns=["aria_customer_id", "aria_name", "age"])
        result = _strip_aria(df)
        assert "customer_id" in result.columns
        assert "name" in result.columns

    def test_non_aria_columns_unchanged(self):
        df = pd.DataFrame(columns=["customer_id", "balance", "aria_account_id"])
        result = _strip_aria(df)
        assert "customer_id" in result.columns
        assert "balance" in result.columns
        assert "account_id" in result.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["aria_id"])
        result = _strip_aria(df)
        assert "id" in result.columns

    def test_all_columns_have_aria_prefix(self):
        df = pd.DataFrame({"aria_x": [1], "aria_y": [2]})
        result = _strip_aria(df)
        assert list(result.columns) == ["x", "y"]

    def test_no_columns_have_aria_prefix(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        result = _strip_aria(df)
        assert list(result.columns) == ["x", "y"]

    def test_data_values_preserved(self):
        df = pd.DataFrame({"aria_id": [10, 20], "value": [1, 2]})
        result = _strip_aria(df)
        assert list(result["id"]) == [10, 20]
        assert list(result["value"]) == [1, 2]


# ── compute_txn_aggregates ─────────────────────────────────────────────────────

def _make_txn_df(n_accounts=3, n_txns_per_account=10, seed=42):
    """Build a minimal transactions DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    ref = datetime(2024, 1, 1)
    for acct_id in range(1, n_accounts + 1):
        for i in range(n_txns_per_account):
            rows.append({
                "subject_id": acct_id,
                "timestamp": ref + timedelta(days=int(rng.integers(0, 365))),
                "amount": float(rng.integers(100, 10000)),
                "cash_direction": rng.choice(["CashIn", "CashOut"]),
            })
    return pd.DataFrame(rows)


class TestComputeTxnAggregates:
    @pytest.fixture
    def agg_df(self):
        txn = _make_txn_df()
        return compute_txn_aggregates(txn)

    def test_returns_one_row_per_account(self, agg_df):
        assert len(agg_df) == 3  # 3 accounts

    def test_account_id_column_present(self, agg_df):
        assert "account_id" in agg_df.columns

    def test_trxn_count_equals_n_txns(self, agg_df):
        for _, row in agg_df.iterrows():
            assert row["trxn_count"] == 10

    def test_avg_trxn_amt_between_min_and_max(self, agg_df):
        for _, row in agg_df.iterrows():
            assert row["min_trxn_amt"] <= row["avg_trxn_amt"] <= row["max_trxn_amt"]

    def test_cashin_plus_cashout_lte_total(self, agg_df):
        for _, row in agg_df.iterrows():
            assert row["cashin_count"] + row["cashout_count"] == row["trxn_count"]

    def test_cashin_ratio_between_0_and_1(self, agg_df):
        for _, row in agg_df.iterrows():
            assert 0.0 <= row["cashin_ratio"] <= 1.0

    def test_weeks_active_at_least_1(self, agg_df):
        for _, row in agg_df.iterrows():
            assert row["weeks_active"] >= 1.0

    def test_avg_trxns_week_positive(self, agg_df):
        for _, row in agg_df.iterrows():
            assert row["avg_trxns_week"] > 0

    def test_avg_monthly_trxn_amt_positive(self, agg_df):
        for _, row in agg_df.iterrows():
            assert row["avg_monthly_trxn_amt"] > 0

    def test_std_trxn_amt_non_negative(self, agg_df):
        for _, row in agg_df.iterrows():
            assert row["std_trxn_amt"] >= 0

    def test_single_transaction_std_is_zero(self):
        txn = pd.DataFrame([{
            "subject_id": 1,
            "timestamp": "2024-01-15",
            "amount": 500.0,
            "cash_direction": "CashIn",
        }])
        agg = compute_txn_aggregates(txn)
        assert agg.loc[0, "std_trxn_amt"] == 0.0

    def test_handles_non_numeric_amounts(self):
        txn = pd.DataFrame([
            {"subject_id": 1, "timestamp": "2024-01-01", "amount": "badvalue", "cash_direction": "CashIn"},
            {"subject_id": 1, "timestamp": "2024-01-02", "amount": 500.0, "cash_direction": "CashOut"},
        ])
        # Should not raise — bad values are coerced to NaN
        agg = compute_txn_aggregates(txn)
        assert len(agg) == 1

    def test_handles_empty_dataframe(self):
        txn = pd.DataFrame(columns=["subject_id", "timestamp", "amount", "cash_direction"])
        agg = compute_txn_aggregates(txn)
        assert len(agg) == 0

    def test_total_trxn_amt_equals_sum(self):
        txn = pd.DataFrame([
            {"subject_id": 1, "timestamp": "2024-01-01", "amount": 100.0, "cash_direction": "CashIn"},
            {"subject_id": 1, "timestamp": "2024-02-01", "amount": 200.0, "cash_direction": "CashOut"},
        ])
        agg = compute_txn_aggregates(txn)
        assert agg.loc[0, "total_trxn_amt"] == pytest.approx(300.0)
