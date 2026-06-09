"""
synth_recurring.py — Recurring transaction patterns + mobile/crypto rails.

Each pattern emits a list of (date_offset_days, direction, txn_type, amount,
counterparty, payment_method, channel) tuples spanning a 12-month window.

emit(key, cust, account, rng, start_date, end_date) → list[dict] of txn rows
matching the generate_synthetic_data._baseline_transactions schema.

Counterparty registries:
  CRYPTO_EXCHANGES — fixed names that AML rules can flag specifically
  P2P_APPS         — Zelle / Venmo / Cash App (used as txn_type values)
  PAYMENT_METHODS  — Apple/Google/Samsung Pay tokenization for card txns
"""

from __future__ import annotations
from typing import Callable, Dict, List
from datetime import datetime, timedelta


# ── Counterparty / channel registries ─────────────────────────────────────────

# Regulated US exchanges (KYC, FinCEN-registered)
CRYPTO_REGULATED = [
    "Coinbase", "Kraken", "Gemini", "Binance.US",
    "Crypto.com", "Robinhood Crypto", "PayPal Crypto",
]
# DeFi gateways / non-custodial on-ramps (less KYC, no exchange surveillance)
CRYPTO_DEFI_GATEWAYS = [
    "MoonPay", "Wyre", "Ramp Network", "MetaMask Bridge",
    "Simplex Onramp", "Transak",
]
# High-risk / privacy / mixer-adjacent (anything reaching these is presumptively suspicious)
CRYPTO_HIGH_RISK = [
    "Unknown Crypto Exchange", "PrivacySwap",
    "OffshoreCoin Exchange", "NotKYC Exchange",
    "Tornado Cash Mixer", "Wasabi Wallet Bridge",
]
# Crypto ATMs / kiosks
BITCOIN_KIOSKS = [
    "CoinFlip Kiosk", "Bitcoin Depot ATM", "RocketCoin ATM",
    "Coinhub ATM", "LibertyX Kiosk",
]
# Union — preserved as CRYPTO_EXCHANGES for backward compat with existing patterns
CRYPTO_EXCHANGES = CRYPTO_REGULATED + CRYPTO_DEFI_GATEWAYS + CRYPTO_HIGH_RISK

P2P_APPS = ["Zelle", "Venmo", "Cash App"]
P2P_APPS_WEIGHTS = [0.55, 0.30, 0.15]

PAYMENT_METHODS_CARD = [
    "Card Swipe", "Card Chip", "Card Online",
    "Apple Pay", "Google Pay", "Samsung Pay",
]
PAYMENT_METHODS_CARD_WEIGHTS = [0.18, 0.22, 0.30, 0.18, 0.10, 0.02]

# Common merchant / payee pools (for repeat counterparties)
UTILITY_PAYEES = ["ConEd Power", "PG&E", "Duke Energy", "Verizon Wireless",
                  "AT&T Mobility", "Comcast Xfinity", "Spectrum Cable",
                  "City Water", "Waste Mgmt"]
LANDLORD_PAYEES = ["Greystar Residential", "Equity Apartments", "AvalonBay",
                   "Maxwell Property Mgmt", "Bluestone REIT"]
RX_PAYEES = ["CVS Pharmacy", "Walgreens", "Express Scripts", "OptumRx", "AARP Pharmacy"]
VENDOR_POOL_BIZ = [f"Vendor {chr(65+i)} Supply Co" for i in range(15)]
TAX_AUTH = ["IRS - Federal", "State Tax Dept", "City Tax Authority"]
GOV_BENEFIT = ["SSA - Social Security", "VA Benefits", "State Unemployment"]
PAYROLL_EMPLOYER = ["Acme Corp Payroll", "Globex Industries Pay",
                    "Initech Payroll", "Stark Industries Pay",
                    "Wayne Enterprises Pay", "Hooli HR"]
MERCHANT_PROCESSORS = ["Stripe Settlement", "Square Payouts",
                       "Worldpay ACH", "Adyen Funding", "FIS Merchant Svcs"]
BROKERAGE_NAMES = ["Fidelity Brokerage", "Schwab Brokerage", "Vanguard Investments",
                   "Morgan Stanley Wealth", "JP Morgan Private Bank"]
CHARITABLE = ["Red Cross", "United Way", "Doctors Without Borders",
              "Salvation Army", "Local Community Fund"]

REF_DATE   = datetime(2026, 3, 31)
START_DATE = datetime(2025, 3, 31)


# ── Helper utilities ──────────────────────────────────────────────────────────

def _biweekly_days(rng, start: datetime, end: datetime, jitter=1) -> List[datetime]:
    """Return list of biweekly dates from start to end, jittered by ±jitter days."""
    out = []
    d   = start + timedelta(days=int(rng.integers(0, 14)))
    while d < end:
        out.append(d + timedelta(days=int(rng.integers(-jitter, jitter+1))))
        d += timedelta(days=14)
    return out


def _monthly_days(rng, start: datetime, end: datetime, dom_lo=1, dom_hi=5, jitter=2) -> List[datetime]:
    """Return list of monthly dates with day-of-month in [dom_lo, dom_hi]."""
    out = []
    yr, mo = start.year, start.month
    while True:
        try:
            base = datetime(yr, mo, int(rng.integers(dom_lo, dom_hi+1)))
        except ValueError:
            base = datetime(yr, mo, 1)
        d = base + timedelta(days=int(rng.integers(-jitter, jitter+1)))
        if d > end:
            break
        if d >= start:
            out.append(d)
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1
    return out


def _weekly_days(rng, start: datetime, end: datetime, n_per_week=1) -> List[datetime]:
    out = []
    d = start
    while d < end:
        for _ in range(n_per_week):
            out.append(d + timedelta(days=int(rng.integers(0, 7))))
        d += timedelta(days=7)
    return [x for x in out if start <= x < end]


def _daily_days(rng, start: datetime, end: datetime, probability=1.0) -> List[datetime]:
    out = []
    d = start
    while d < end:
        if rng.random() < probability:
            out.append(d)
        d += timedelta(days=1)
    return out


def _row(dt, direction, txn_type, amount, cust_id, acct_id, counterparty,
         payment_method=None, channel=2, country_id=267, rule_trigger=None,
         is_anomalous=False, pattern=None):
    """Produce a txn dict matching the aria_transactions.csv schema."""
    return {
        "aria_timestamp":          dt.strftime("%Y-%m-%d %H:%M:%S"),
        "aria_cash_direction":     direction,
        "aria_transaction_type":   txn_type,
        "aria_amount":             round(float(amount), 2),
        "aria_subject_id":         acct_id,
        "aria_customer_id":        cust_id,
        "aria_channel_id":         channel,
        "aria_currency_id":        1,
        "aria_country_id":         country_id,
        "aria_other_party_id":     counterparty,
        "aria_is_anomalous":       bool(is_anomalous),
        "aria_rule_trigger":       rule_trigger,
        "aria_payment_method":     payment_method,
        "aria_pattern":            pattern,
    }


def _amount(base, jitter_pct, rng):
    return max(1.0, base * (1 + (rng.random() - 0.5) * 2 * jitter_pct))


# ── Pattern implementations ───────────────────────────────────────────────────
# All return list[dict] of txn rows.

def _income_monthly(cust):
    return max(1500.0, (cust.get("aria_gross_annual_income") or 30_000) / 12)


def _income_biweekly(cust):
    return _income_monthly(cust) * 12 / 26


# Individual income / benefits inflows
def payroll_inflow(cust, acct, rng, s, e):
    base = _income_biweekly(cust)
    emp  = PAYROLL_EMPLOYER[int(rng.integers(0, len(PAYROLL_EMPLOYER)))]
    return [_row(d, "CashIn", "ACH", _amount(base, 0.02, rng),
                 cust["_id"], acct, emp, channel=2, pattern="payroll_inflow")
            for d in _biweekly_days(rng, s, e)]


def payroll_inflow_large(cust, acct, rng, s, e):
    base = max(_income_biweekly(cust), 5000.0)
    emp  = PAYROLL_EMPLOYER[int(rng.integers(0, len(PAYROLL_EMPLOYER)))]
    return [_row(d, "CashIn", "ACH", _amount(base, 0.03, rng),
                 cust["_id"], acct, emp, channel=2, pattern="payroll_inflow_large")
            for d in _biweekly_days(rng, s, e)]


def part_time_payroll_inflow(cust, acct, rng, s, e):
    base = max(_income_biweekly(cust) * 0.5, 300.0)
    emp  = "Part-Time Employer Inc"
    return [_row(d, "CashIn", "ACH", _amount(base, 0.05, rng),
                 cust["_id"], acct, emp, channel=2, pattern="part_time_payroll")
            for d in _biweekly_days(rng, s, e)]


def gov_benefit_inflow(cust, acct, rng, s, e):
    src  = GOV_BENEFIT[int(rng.integers(0, len(GOV_BENEFIT)))]
    amt  = max(900.0, min(_income_monthly(cust), 2500.0))
    return [_row(d, "CashIn", "ACH", _amount(amt, 0.01, rng),
                 cust["_id"], acct, src, channel=2, pattern="gov_benefit")
            for d in _monthly_days(rng, s, e, 1, 3)]


def ssa_pension_inflow(cust, acct, rng, s, e):
    src = "SSA - Social Security" if rng.random() < 0.85 else "Defined-Benefit Pension"
    amt = max(1200.0, min(_income_monthly(cust), 3500.0))
    return [_row(d, "CashIn", "ACH", _amount(amt, 0.005, rng),
                 cust["_id"], acct, src, channel=2, pattern="ssa_pension")
            for d in _monthly_days(rng, s, e, 1, 3)]


def allowance_inflow(cust, acct, rng, s, e):
    return [_row(d, "CashIn", "Misc Deposit", _amount(rng.uniform(20, 80), 0.2, rng),
                 cust["_id"], acct, "Parent Transfer", channel=2, pattern="allowance")
            for d in _weekly_days(rng, s, e, 1)]


# Outflows — rent / mortgage / utilities / loans
def rent_outflow(cust, acct, rng, s, e):
    landlord = LANDLORD_PAYEES[int(rng.integers(0, len(LANDLORD_PAYEES)))]
    base = max(800.0, _income_monthly(cust) * rng.uniform(0.25, 0.35))
    return [_row(d, "CashOut", "ACH", _amount(base, 0.01, rng),
                 cust["_id"], acct, landlord, channel=2, pattern="rent")
            for d in _monthly_days(rng, s, e, 1, 5)]


def mortgage_outflow(cust, acct, rng, s, e):
    lender = "Wells Fargo Mortgage" if rng.random() < 0.4 else "Chase Home Lending"
    base = max(1500.0, _income_monthly(cust) * rng.uniform(0.20, 0.30))
    return [_row(d, "CashOut", "ACH", _amount(base, 0.005, rng),
                 cust["_id"], acct, lender, channel=2, pattern="mortgage")
            for d in _monthly_days(rng, s, e, 1, 5)]


def rent_or_mortgage_outflow(cust, acct, rng, s, e):
    return mortgage_outflow(cust, acct, rng, s, e) if rng.random() < 0.5 \
           else rent_outflow(cust, acct, rng, s, e)


def utility_outflow(cust, acct, rng, s, e):
    """3-4 recurring utility payees per customer, monthly."""
    n_payees = int(rng.integers(3, 5))
    payees   = list(rng.choice(UTILITY_PAYEES, size=n_payees, replace=False))
    out = []
    for payee in payees:
        base = rng.uniform(60, 280)
        for d in _monthly_days(rng, s, e, 10, 25, jitter=3):
            out.append(_row(d, "CashOut", "ACH", _amount(base, 0.10, rng),
                            cust["_id"], acct, payee, channel=2, pattern="utility"))
    return out


def medical_rx_outflow(cust, acct, rng, s, e):
    n_payees = int(rng.integers(1, 3))
    payees   = list(rng.choice(RX_PAYEES, size=n_payees, replace=False))
    out = []
    for payee in payees:
        for d in _monthly_days(rng, s, e, 5, 25, jitter=4):
            out.append(_row(d, "CashOut", "Debit Card",
                            _amount(rng.uniform(20, 220), 0.3, rng),
                            cust["_id"], acct, payee,
                            payment_method=_card_pm(rng), channel=4, pattern="medical_rx"))
    return out


def fixed_check_outflow(cust, acct, rng, s, e):
    out = []
    for d in _monthly_days(rng, s, e, 1, 28, jitter=4):
        if rng.random() < 0.4:
            out.append(_row(d, "CashOut", "Check",
                            _amount(rng.uniform(50, 600), 0.2, rng),
                            cust["_id"], acct, "Personal Payee", channel=1,
                            pattern="senior_check"))
    return out


def tuition_outflow(cust, acct, rng, s, e):
    school = "State University Bursar"
    # Tuition is quarterly-ish
    out = []
    for k in range(4):
        d = s + timedelta(days=int((k + 0.5) * (e - s).days / 4) + int(rng.integers(-7, 7)))
        if d < e:
            out.append(_row(d, "CashOut", "ACH", _amount(rng.uniform(2500, 8000), 0.05, rng),
                            cust["_id"], acct, school, channel=2, pattern="tuition"))
    return out


def loan_monthly_outflow(cust, acct, rng, s, e):
    lender = "Wells Fargo Auto" if rng.random() < 0.5 else "Capital One Auto"
    base = rng.uniform(380, 720)
    return [_row(d, "CashOut", "ACH", _amount(base, 0.002, rng),
                 cust["_id"], acct, lender, channel=2, pattern="auto_loan")
            for d in _monthly_days(rng, s, e, 3, 8)]


# Credit card / P2P / brokerage
def _card_pm(rng):
    idx = rng.choice(range(len(PAYMENT_METHODS_CARD)), p=PAYMENT_METHODS_CARD_WEIGHTS)
    return PAYMENT_METHODS_CARD[int(idx)]


def cc_purchases_outflow(cust, acct, rng, s, e):
    """Daily-ish small card purchases."""
    out = []
    income_factor = max(0.5, min(3.0, (cust.get("aria_gross_annual_income") or 50_000) / 60_000))
    p_per_day = min(0.7, 0.25 + 0.1 * income_factor)
    for d in _daily_days(rng, s, e, p_per_day):
        merch = f"Merchant {int(rng.integers(1, 200))}"
        out.append(_row(d, "CashOut", "Debit Card",
                        _amount(rng.uniform(8, 220), 0.5, rng),
                        cust["_id"], acct, merch,
                        payment_method=_card_pm(rng), channel=4, pattern="cc_purchase"))
    return out


def cc_autopay_outflow(cust, acct, rng, s, e):
    base = max(200.0, _income_monthly(cust) * rng.uniform(0.06, 0.18))
    return [_row(d, "CashOut", "ACH", _amount(base, 0.4, rng),
                 cust["_id"], acct, "Chase Credit Card Payment", channel=2,
                 pattern="cc_autopay")
            for d in _monthly_days(rng, s, e, 15, 22)]


def cc_autopay_inflow(cust, acct, rng, s, e):
    base = max(200.0, _income_monthly(cust) * rng.uniform(0.06, 0.18))
    return [_row(d, "CashIn", "ACH", _amount(base, 0.4, rng),
                 cust["_id"], acct, "Customer Payment", channel=2,
                 pattern="cc_payment_inflow")
            for d in _monthly_days(rng, s, e, 15, 22)]


def p2p_outflow(cust, acct, rng, s, e):
    """Recurring P2P outflows (Zelle/Venmo/Cash App) to friends/family."""
    app_idx = rng.choice(range(3), p=P2P_APPS_WEIGHTS)
    app = P2P_APPS[int(app_idx)]
    # 3-8 recurring P2P payees per customer
    payees = [f"P2P Payee {chr(65+i)}" for i in range(int(rng.integers(3, 9)))]
    out = []
    for d in _weekly_days(rng, s, e, n_per_week=1):
        if rng.random() < 0.55:
            payee = payees[int(rng.integers(0, len(payees)))]
            amt   = rng.uniform(15, 250) if app == "Zelle" else rng.uniform(8, 120)
            out.append(_row(d, "CashOut", app, _amount(amt, 0.2, rng),
                            cust["_id"], acct, payee, channel=4, pattern="p2p_routine"))
    return out


def p2p_outflow_small(cust, acct, rng, s, e):
    """Teen-style P2P — Venmo/Cash App preferred, smaller amounts."""
    app_idx = rng.choice(range(3), p=[0.15, 0.5, 0.35])
    app = P2P_APPS[int(app_idx)]
    out = []
    for d in _weekly_days(rng, s, e, n_per_week=2):
        if rng.random() < 0.6:
            out.append(_row(d, "CashOut", app, _amount(rng.uniform(5, 60), 0.4, rng),
                            cust["_id"], acct, f"Friend {int(rng.integers(1, 30))}",
                            channel=4, pattern="p2p_teen"))
    return out


def brokerage_sweep_outflow(cust, acct, rng, s, e):
    bk = BROKERAGE_NAMES[int(rng.integers(0, len(BROKERAGE_NAMES)))]
    base = max(500.0, _income_monthly(cust) * rng.uniform(0.05, 0.18))
    return [_row(d, "CashOut", "ACH", _amount(base, 0.05, rng),
                 cust["_id"], acct, bk, channel=2, pattern="brokerage_sweep")
            for d in _monthly_days(rng, s, e, 1, 5)]


def brokerage_transfer_outflow(cust, acct, rng, s, e):
    bk = BROKERAGE_NAMES[int(rng.integers(0, len(BROKERAGE_NAMES)))]
    base = max(5000.0, _income_monthly(cust) * rng.uniform(0.10, 0.30))
    return [_row(d, "CashOut", "Wire", _amount(base, 0.20, rng),
                 cust["_id"], acct, bk, channel=2, pattern="brokerage_transfer")
            for d in _monthly_days(rng, s, e, 1, 10)]


def intl_wire_outflow(cust, acct, rng, s, e):
    countries = ["United Kingdom", "Switzerland", "Singapore", "Hong Kong",
                 "Cayman Islands", "Luxembourg", "Germany"]
    out = []
    for d in _monthly_days(rng, s, e, 1, 28, jitter=8):
        if rng.random() < 0.4:
            ctry = countries[int(rng.integers(0, len(countries)))]
            amt  = rng.uniform(8_000, 80_000)
            out.append(_row(d, "CashOut", "Wire", _amount(amt, 0.15, rng),
                            cust["_id"], acct, f"Intl Beneficiary - {ctry}",
                            channel=2, country_id=999, pattern="intl_wire"))
    return out


def charitable_outflow(cust, acct, rng, s, e):
    out = []
    for d in _monthly_days(rng, s, e, 15, 28):
        if rng.random() < 0.5:
            org = CHARITABLE[int(rng.integers(0, len(CHARITABLE)))]
            out.append(_row(d, "CashOut", "ACH", _amount(rng.uniform(50, 800), 0.3, rng),
                            cust["_id"], acct, org, channel=2, pattern="charitable"))
    return out


# Savings/Retirement
def savings_sweep_inflow(cust, acct, rng, s, e):
    base = max(100.0, _income_monthly(cust) * rng.uniform(0.05, 0.20))
    return [_row(d, "CashIn", "ACH", _amount(base, 0.10, rng),
                 cust["_id"], acct, "Internal Transfer - Checking", channel=2,
                 pattern="savings_sweep")
            for d in _monthly_days(rng, s, e, 1, 5)]


def cd_open_inflow(cust, acct, rng, s, e):
    amt = rng.uniform(5000, 60000)
    return [_row(s + timedelta(days=10), "CashIn", "ACH", round(amt, 2),
                 cust["_id"], acct, "CD Initial Deposit", channel=2, pattern="cd_open")]


def retirement_contribution_inflow(cust, acct, rng, s, e):
    base = max(100.0, _income_biweekly(cust) * rng.uniform(0.05, 0.15))
    return [_row(d, "CashIn", "ACH", _amount(base, 0.02, rng),
                 cust["_id"], acct, "401(k) Employer Contribution", channel=2,
                 pattern="retirement_contribution")
            for d in _biweekly_days(rng, s, e)]


# ── Business patterns ─────────────────────────────────────────────────────────

def _biz_headcount(cust):
    size = (cust.get("aria_business_size") or "Small")
    return {"Small": 12, "Medium": 75, "Large": 350}.get(size, 12)


def _biz_payroll_batch(cust, acct, rng, s, e, mult=1.0):
    headcount = _biz_headcount(cust)
    avg_salary_biweekly = 1900  # ~$50K annual / 26
    base = headcount * avg_salary_biweekly * mult
    return [_row(d, "CashOut", "ACH", _amount(base, 0.04, rng),
                 cust["_id"], acct, "ADP Payroll Services", channel=2,
                 pattern="biz_payroll_batch")
            for d in _biweekly_days(rng, s, e)]


def biz_payroll_batch_small(c, a, r, s, e): return _biz_payroll_batch(c, a, r, s, e, 1.0)
def biz_payroll_batch_med(c, a, r, s, e):   return _biz_payroll_batch(c, a, r, s, e, 1.0)
def biz_payroll_batch_large(c, a, r, s, e): return _biz_payroll_batch(c, a, r, s, e, 1.0)


def vendor_payments_monthly(cust, acct, rng, s, e):
    n_vendors = int(rng.integers(3, 9))
    vendors   = list(rng.choice(VENDOR_POOL_BIZ, size=n_vendors, replace=False))
    out = []
    for v in vendors:
        base = rng.uniform(800, 12000)
        for d in _monthly_days(rng, s, e, 25, 30, jitter=3):
            out.append(_row(d, "CashOut", "ACH", _amount(base, 0.10, rng),
                            cust["_id"], acct, v, channel=2, pattern="vendor_payment"))
    return out


def vendor_payments_weekly(cust, acct, rng, s, e):
    n_vendors = int(rng.integers(8, 15))
    vendors   = list(rng.choice(VENDOR_POOL_BIZ, size=min(n_vendors, len(VENDOR_POOL_BIZ)), replace=False))
    out = []
    for v in vendors:
        base = rng.uniform(2000, 25000)
        for d in _weekly_days(rng, s, e, 1):
            if rng.random() < 0.6:
                out.append(_row(d, "CashOut", "ACH", _amount(base, 0.15, rng),
                                cust["_id"], acct, v, channel=2, pattern="vendor_payment"))
    return out


def daily_vendor_wire(cust, acct, rng, s, e):
    out = []
    for d in _daily_days(rng, s, e, 0.4):
        v = VENDOR_POOL_BIZ[int(rng.integers(0, len(VENDOR_POOL_BIZ)))]
        amt = rng.uniform(15_000, 250_000)
        out.append(_row(d, "CashOut", "Wire", _amount(amt, 0.20, rng),
                        cust["_id"], acct, v, channel=2, pattern="daily_vendor_wire"))
    return out


def fx_payment(cust, acct, rng, s, e):
    countries = ["China", "Germany", "United Kingdom", "Japan", "Mexico", "India"]
    out = []
    for d in _weekly_days(rng, s, e, 1):
        if rng.random() < 0.5:
            ctry = countries[int(rng.integers(0, len(countries)))]
            out.append(_row(d, "CashOut", "Wire", _amount(rng.uniform(20_000, 400_000), 0.30, rng),
                            cust["_id"], acct, f"Intl Supplier - {ctry}", channel=2,
                            country_id=999, pattern="fx_payment"))
    return out


def biz_wire_outflow(cust, acct, rng, s, e):
    out = []
    for d in _monthly_days(rng, s, e, 1, 28, jitter=8):
        if rng.random() < 0.4:
            v = VENDOR_POOL_BIZ[int(rng.integers(0, len(VENDOR_POOL_BIZ)))]
            out.append(_row(d, "CashOut", "Wire", _amount(rng.uniform(8_000, 75_000), 0.20, rng),
                            cust["_id"], acct, v, channel=2, pattern="biz_wire"))
    return out


def quarterly_tax_outflow(cust, acct, rng, s, e):
    out = []
    for k in range(4):
        d = s + timedelta(days=int((k + 0.5) * (e - s).days / 4))
        if d < e:
            auth = TAX_AUTH[int(rng.integers(0, len(TAX_AUTH)))]
            amt = rng.uniform(3000, 90_000) * (_biz_headcount(cust) / 12)
            out.append(_row(d, "CashOut", "ACH", _amount(amt, 0.10, rng),
                            cust["_id"], acct, auth, channel=2, pattern="tax"))
    return out


def monthly_tax_outflow(cust, acct, rng, s, e):
    return [_row(d, "CashOut", "ACH",
                 _amount(rng.uniform(5000, 80_000), 0.15, rng) * (_biz_headcount(cust)/12),
                 cust["_id"], acct, TAX_AUTH[int(rng.integers(0, len(TAX_AUTH)))],
                 channel=2, pattern="tax")
            for d in _monthly_days(rng, s, e, 10, 18)]


def merchant_settlement_inflow(cust, acct, rng, s, e):
    out = []
    proc = MERCHANT_PROCESSORS[int(rng.integers(0, len(MERCHANT_PROCESSORS)))]
    for d in _daily_days(rng, s, e, 0.7):
        out.append(_row(d, "CashIn", "ACH", _amount(rng.uniform(800, 35_000), 0.6, rng),
                        cust["_id"], acct, proc, channel=2, pattern="merchant_settlement"))
    return out


def loc_interest_outflow(cust, acct, rng, s, e):
    base = rng.uniform(2000, 25_000)
    return [_row(d, "CashOut", "ACH", _amount(base, 0.10, rng),
                 cust["_id"], acct, "Commercial LOC Interest", channel=2,
                 pattern="loc_interest")
            for d in _monthly_days(rng, s, e, 1, 5)]


def biz_savings_sweep_inflow(cust, acct, rng, s, e):
    base = rng.uniform(20_000, 500_000)
    return [_row(d, "CashIn", "ACH", _amount(base, 0.20, rng),
                 cust["_id"], acct, "Internal Sweep - Operating", channel=2,
                 pattern="biz_sweep")
            for d in _monthly_days(rng, s, e, 1, 5)]


def escrow_irregular_wire(cust, acct, rng, s, e):
    out = []
    for d in _monthly_days(rng, s, e, 1, 28, jitter=10):
        if rng.random() < 0.4:
            direction = "CashIn" if rng.random() < 0.5 else "CashOut"
            amt = rng.uniform(80_000, 2_500_000)
            out.append(_row(d, direction, "Wire", _amount(amt, 0.30, rng),
                            cust["_id"], acct, "Real Estate Escrow Party",
                            channel=2, pattern="escrow"))
    return out


# Crypto patterns
def crypto_ramp_recurring(cust, acct, rng, s, e):
    """Recurring on-ramps to crypto exchange + occasional cash-out."""
    exch = CRYPTO_EXCHANGES[int(rng.integers(0, len(CRYPTO_EXCHANGES) - 1))]  # not unknown
    out = []
    # Recurring buys (biweekly small)
    for d in _biweekly_days(rng, s, e):
        if rng.random() < 0.7:
            out.append(_row(d, "CashOut", "ACH", _amount(rng.uniform(50, 600), 0.3, rng),
                            cust["_id"], acct, exch, channel=2, pattern="crypto_on_ramp"))
    # Occasional sell-side inflow
    for d in _monthly_days(rng, s, e, 1, 28, jitter=10):
        if rng.random() < 0.10:
            out.append(_row(d, "CashIn", "ACH", _amount(rng.uniform(300, 4000), 0.5, rng),
                            cust["_id"], acct, exch, channel=2, pattern="crypto_off_ramp"))
    return out


def msb_high_velocity(cust, acct, rng, s, e):
    """MSB-style high-velocity: daily P2P inflows + ACH out to crypto exchanges."""
    out = []
    # Many daily P2P inflows
    for d in _daily_days(rng, s, e, 0.8):
        app = P2P_APPS[int(rng.choice(range(3), p=P2P_APPS_WEIGHTS))]
        out.append(_row(d, "CashIn", app, _amount(rng.uniform(100, 2500), 0.5, rng),
                        cust["_id"], acct, f"P2P Sender {int(rng.integers(1, 200))}",
                        channel=4, pattern="msb_p2p_in"))
    # ACH out to exchanges
    for d in _weekly_days(rng, s, e, 1):
        if rng.random() < 0.6:
            exch = CRYPTO_EXCHANGES[int(rng.integers(0, len(CRYPTO_EXCHANGES)))]
            out.append(_row(d, "CashOut", "ACH", _amount(rng.uniform(2000, 35_000), 0.4, rng),
                            cust["_id"], acct, exch, channel=2, pattern="msb_to_exchange"))
    return out


# ── Pattern registry — key → callable ─────────────────────────────────────────

RECURRING_PATTERNS: Dict[str, Callable] = {
    # Individual income
    "payroll_inflow":               payroll_inflow,
    "payroll_inflow_large":         payroll_inflow_large,
    "part_time_payroll_inflow":     part_time_payroll_inflow,
    "gov_benefit_inflow":           gov_benefit_inflow,
    "ssa_pension_inflow":           ssa_pension_inflow,
    "allowance_inflow":             allowance_inflow,
    # Individual outflows
    "rent_outflow":                 rent_outflow,
    "mortgage_outflow":             mortgage_outflow,
    "rent_or_mortgage_outflow":     rent_or_mortgage_outflow,
    "utility_outflow":              utility_outflow,
    "medical_rx_outflow":           medical_rx_outflow,
    "fixed_check_outflow":          fixed_check_outflow,
    "tuition_outflow":              tuition_outflow,
    "loan_monthly_outflow":         loan_monthly_outflow,
    # Card / P2P / brokerage
    "cc_purchases_outflow":         cc_purchases_outflow,
    "cc_autopay_outflow":           cc_autopay_outflow,
    "cc_autopay_inflow":            cc_autopay_inflow,
    "p2p_outflow":                  p2p_outflow,
    "p2p_outflow_small":            p2p_outflow_small,
    "brokerage_sweep_outflow":      brokerage_sweep_outflow,
    "brokerage_transfer_outflow":   brokerage_transfer_outflow,
    "intl_wire_outflow":            intl_wire_outflow,
    "charitable_outflow":           charitable_outflow,
    # Savings / retirement
    "savings_sweep_inflow":         savings_sweep_inflow,
    "cd_open_inflow":               cd_open_inflow,
    "retirement_contribution_inflow": retirement_contribution_inflow,
    # Business
    "biz_payroll_batch_small":      biz_payroll_batch_small,
    "biz_payroll_batch_med":        biz_payroll_batch_med,
    "biz_payroll_batch_large":      biz_payroll_batch_large,
    "vendor_payments_monthly":      vendor_payments_monthly,
    "vendor_payments_weekly":       vendor_payments_weekly,
    "daily_vendor_wire":            daily_vendor_wire,
    "fx_payment":                   fx_payment,
    "biz_wire_outflow":             biz_wire_outflow,
    "quarterly_tax_outflow":        quarterly_tax_outflow,
    "monthly_tax_outflow":          monthly_tax_outflow,
    "merchant_settlement_inflow":   merchant_settlement_inflow,
    "loc_interest_outflow":         loc_interest_outflow,
    "biz_savings_sweep_inflow":     biz_savings_sweep_inflow,
    "escrow_irregular_wire":        escrow_irregular_wire,
    # Crypto
    "crypto_ramp_recurring":        crypto_ramp_recurring,
    "msb_high_velocity":            msb_high_velocity,
}


def emit(key: str, cust: dict, acct_id: str, rng, start: datetime = None,
         end: datetime = None) -> List[dict]:
    """Generate recurring transactions for a pattern key, or return []."""
    fn = RECURRING_PATTERNS.get(key)
    if fn is None:
        return []
    s = start or START_DATE
    e = end   or REF_DATE
    return fn(cust, acct_id, rng, s, e)
