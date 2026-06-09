"""
synth_products.py — 22-product taxonomy for synthetic bank data.

Each Product entry binds:
  - segment / account_type    : where the product slots into the existing schema
  - eligibility(cust_dict)    : returns True if a customer qualifies
  - target_share              : fraction of eligible customers who hold it
  - recurring_set             : pattern keys consumed by synth_recurring.emit()
  - is_primary                : at most one primary product per customer

assign_products(cust, rng) returns the list a customer actually holds.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

# Account-type values must match ds_data_prep.normalize_account_type buckets
TYPE_CHECKING = "Checking"
TYPE_SAVINGS  = "Savings"
TYPE_LOAN     = "Loan"
TYPE_CC       = "Credit Card"
TYPE_CD       = "Certificate Deposit"


@dataclass(frozen=True)
class Product:
    code: str
    name: str
    segment: str          # 'Individual' | 'Business' | 'Crypto'
    account_type: str
    eligibility: Callable[[dict], bool]
    target_share: float
    recurring_set: tuple
    is_primary: bool = False


def _is_indiv(c):  return not bool(c.get("aria_entity", False))
def _is_biz(c):    return bool(c.get("aria_entity", False))
def _age(c):       return c.get("aria_age") or 0
def _income(c):    return c.get("aria_gross_annual_income") or 0
def _naics(c):     return str(c.get("aria_naics") or "")
def _bsize(c):     return c.get("aria_business_size") or ""
def _crypto(c):    return bool(c.get("aria_crypto_user", False))


PRODUCTS: List[Product] = [
    # ── Individual checking (primary — exactly one) ───────────────────────
    Product("HSC", "High-School Checking",   "Individual", TYPE_CHECKING,
            lambda c: _is_indiv(c) and 14 <= _age(c) <= 19,
            1.00, ("allowance_inflow", "p2p_outflow_small"), is_primary=True),
    Product("STC", "Student Checking",       "Individual", TYPE_CHECKING,
            lambda c: _is_indiv(c) and 18 <= _age(c) <= 25 and _income(c) < 35_000,
            1.00, ("part_time_payroll_inflow", "tuition_outflow", "p2p_outflow", "cc_autopay_outflow"),
            is_primary=True),
    Product("SBK", "Secure Banking",         "Individual", TYPE_CHECKING,
            lambda c: _is_indiv(c) and _income(c) < 40_000 and _age(c) >= 20,
            1.00, ("gov_benefit_inflow", "rent_outflow", "utility_outflow"),
            is_primary=True),
    Product("TCK", "Total Checking",         "Individual", TYPE_CHECKING,
            lambda c: _is_indiv(c) and 40_000 <= _income(c) < 150_000 and _age(c) >= 22,
            1.00, ("payroll_inflow", "rent_or_mortgage_outflow", "utility_outflow",
                   "cc_autopay_outflow", "p2p_outflow"),
            is_primary=True),
    Product("PPC", "Premier Plus Checking",  "Individual", TYPE_CHECKING,
            lambda c: _is_indiv(c) and 150_000 <= _income(c) < 400_000,
            1.00, ("payroll_inflow_large", "mortgage_outflow", "utility_outflow",
                   "brokerage_sweep_outflow", "p2p_outflow"),
            is_primary=True),
    Product("PCC", "Private Client Checking", "Individual", TYPE_CHECKING,
            lambda c: _is_indiv(c) and _income(c) >= 400_000,
            1.00, ("payroll_inflow_large", "brokerage_transfer_outflow",
                   "intl_wire_outflow", "charitable_outflow"),
            is_primary=True),
    Product("SNK", "Senior Checking",        "Individual", TYPE_CHECKING,
            lambda c: _is_indiv(c) and _age(c) >= 60,
            1.00, ("ssa_pension_inflow", "medical_rx_outflow", "fixed_check_outflow"),
            is_primary=True),

    # ── Individual savings / investment / credit (non-primary) ────────────
    Product("HYS", "High-Yield Savings",     "Individual", TYPE_SAVINGS,
            lambda c: _is_indiv(c) and _income(c) >= 40_000,
            0.55, ("savings_sweep_inflow",)),
    Product("CDA", "Certificate of Deposit", "Individual", TYPE_CD,
            lambda c: _is_indiv(c) and (_income(c) >= 60_000 or _age(c) >= 55),
            0.18, ("cd_open_inflow",)),
    Product("RAA", "401(K) / IRA Contribution", "Individual", TYPE_SAVINGS,
            lambda c: _is_indiv(c) and 25 <= _age(c) <= 70 and _income(c) >= 40_000,
            0.40, ("retirement_contribution_inflow",)),
    Product("CCR", "Credit Card (Rewards)",  "Individual", TYPE_CC,
            lambda c: _is_indiv(c) and _income(c) >= 30_000,
            0.55, ("cc_purchases_outflow", "cc_autopay_inflow")),
    Product("MOA", "Mortgage / Auto Loan",   "Individual", TYPE_LOAN,
            lambda c: _is_indiv(c) and _age(c) >= 25 and _income(c) >= 50_000,
            0.35, ("loan_monthly_outflow",)),

    # ── Business checking (primary — exactly one by size) ─────────────────
    Product("BCC", "Business Complete Checking",  "Business", TYPE_CHECKING,
            lambda c: _is_biz(c) and _bsize(c) == "Small",
            1.00, ("biz_payroll_batch_small", "vendor_payments_monthly", "quarterly_tax_outflow"),
            is_primary=True),
    Product("BPC", "Business Performance Checking", "Business", TYPE_CHECKING,
            lambda c: _is_biz(c) and _bsize(c) == "Medium",
            1.00, ("biz_payroll_batch_med", "vendor_payments_weekly", "monthly_tax_outflow",
                   "biz_wire_outflow"),
            is_primary=True),
    Product("BPL", "Business Platinum Checking",  "Business", TYPE_CHECKING,
            lambda c: _is_biz(c) and _bsize(c) == "Large",
            1.00, ("biz_payroll_batch_large", "daily_vendor_wire", "fx_payment",
                   "intl_wire_outflow"),
            is_primary=True),
    Product("BMM", "Business Money Market",   "Business", TYPE_SAVINGS,
            lambda c: _is_biz(c),
            0.55, ("biz_savings_sweep_inflow",)),
    Product("MSV", "Merchant Services Settlement", "Business", TYPE_CHECKING,
            lambda c: _is_biz(c) and _naics(c) in ("722511", "441110", "812990", "713210"),
            0.80, ("merchant_settlement_inflow",)),
    Product("BCL", "Commercial Line of Credit", "Business", TYPE_LOAN,
            lambda c: _is_biz(c) and _bsize(c) in ("Medium", "Large"),
            0.55, ("loc_interest_outflow",)),
    Product("RES", "Real Estate / Escrow Account", "Business", TYPE_CHECKING,
            lambda c: _is_biz(c) and _naics(c) in ("531110", "236220"),
            0.60, ("escrow_irregular_wire",)),

    # ── Crypto-related (cross-segment) ────────────────────────────────────
    Product("CTA", "Crypto Trading Account", "Crypto", TYPE_CHECKING,
            lambda c: _is_indiv(c) and _crypto(c),
            1.00, ("crypto_ramp_recurring",)),
    Product("MSB", "MSB / Crypto-Adjacent Business Account", "Crypto", TYPE_CHECKING,
            lambda c: _is_biz(c) and _naics(c) == "522110",
            0.50, ("msb_high_velocity",)),
]

PRODUCT_BY_CODE: Dict[str, Product] = {p.code: p for p in PRODUCTS}
PRODUCT_BY_NAME: Dict[str, Product] = {p.name: p for p in PRODUCTS}


# Primary-product priority when multiple match (more specific wins)
_PRIMARY_PRIORITY = ["PCC", "SNK", "PPC", "TCK", "SBK", "STC", "HSC",
                     "BPL", "BPC", "BCC"]


def assign_crypto_user(cust: dict, rng) -> bool:
    """Probabilistic crypto-user flag based on age/income/segment."""
    if _is_biz(cust):
        return _naics(cust) == "522110" and rng.random() < 0.10
    a, inc = _age(cust), _income(cust)
    if a < 30:                       return rng.random() < 0.40
    if a < 50 and inc >= 100_000:    return rng.random() < 0.25
    if a < 50:                       return rng.random() < 0.10
    return rng.random() < 0.05


def assign_products(cust: dict, rng) -> List[Product]:
    """Return the product list a customer actually holds."""
    held: List[Product] = []
    primary_matches = [p for p in PRODUCTS if p.is_primary and p.eligibility(cust)]
    if primary_matches:
        primary_matches.sort(key=lambda p: _PRIMARY_PRIORITY.index(p.code)
                                            if p.code in _PRIMARY_PRIORITY else 99)
        held.append(primary_matches[0])

    for p in PRODUCTS:
        if p.is_primary:
            continue
        if not p.eligibility(cust):
            continue
        if rng.random() < p.target_share:
            held.append(p)
    return held


def occupation_group(occ: str) -> str:
    """Bucket the 23 individual occupations into ~10 industries for segmentation."""
    if not occ or occ == "" or str(occ).lower() == "nan":
        return "Unknown"
    o = occ.lower()
    if any(k in o for k in ("nurse", "medical", "physician", "physical therapist")):
        return "Healthcare"
    if any(k in o for k in ("teacher", "social worker")):
        return "Education & Social"
    if any(k in o for k in ("engineer", "software", "analyst", "accountant")):
        return "Tech & Finance"
    if any(k in o for k in ("attorney", "marketing", "office manager", "graphic")):
        return "Professional Services"
    if any(k in o for k in ("plumber", "electrician", "construction", "truck", "chef")):
        return "Skilled Trades"
    if any(k in o for k in ("retail", "sales", "food service")):
        return "Retail & Hospitality"
    if any(k in o for k in ("security", "administrative")):
        return "Administrative"
    if any(k in o for k in ("ceo", "cfo", "director", "controller", "vice president",
                            "operations manager", "general manager", "business owner")):
        return "Business Executive"
    return "Other"


def revenue_band(rev) -> str:
    """Bucket business revenues into 4 bands."""
    if rev is None or rev == "" or (isinstance(rev, float) and rev != rev):
        return "Unknown"
    try:
        r = float(rev)
    except (TypeError, ValueError):
        return "Unknown"
    if r <= 0:                  return "Unknown"
    if r < 5_000_000:           return "<$5M"
    if r < 50_000_000:          return "$5M-$50M"
    if r < 250_000_000:         return "$50M-$250M"
    return "$250M+"
