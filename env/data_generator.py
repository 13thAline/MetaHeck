import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from env.models import CustomerProfile, Transaction, TransactionType


NATIONALITIES = ["US", "UK", "DE", "FR", "IN", "CN", "NG", "AE", "RU", "BR"]
OCCUPATIONS = ["Software Engineer", "Doctor", "Teacher", "Accountant", "Student",
               "Business Owner", "Retired", "Lawyer", "Sales Manager", "Consultant"]
HIGH_RISK_COUNTRIES = ["IR", "KP", "SY", "CU", "VE", "MM"]
SHELL_COMPANY_NAMES = ["Apex Holdings LLC", "Blue Ridge Ventures", "Pacific Star Ltd",
                       "Horizon Capital Group", "Nexus Global Corp"]
DOCUMENT_TYPES = ["passport", "national_id", "drivers_license", "utility_bill",
                  "bank_statement", "employment_letter", "tax_return"]


def random_date(start_year: int, end_year: int) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")


def random_txn(customer_id: str, date_str: str, amount: float,
               txn_type: TransactionType, counterparty: str = None,
               country: str = None, description: str = None) -> Transaction:
    return Transaction(
        txn_id=f"TXN-{uuid.uuid4().hex[:8].upper()}",
        date=date_str,
        amount_usd=round(amount, 2),
        type=txn_type,
        counterparty=counterparty,
        country=country,
        description=description,
    )


def generate_clean_customer(customer_id: str = None) -> CustomerProfile:
    """A fully legitimate customer — no red flags."""
    cid = customer_id or f"CUST-{uuid.uuid4().hex[:6].upper()}"
    income = random.randint(40000, 120000)
    base_date = datetime(2024, 1, 1)

    txns = []
    for i in range(random.randint(5, 12)):
        d = (base_date + timedelta(days=random.randint(0, 364))).strftime("%Y-%m-%d")
        txns.append(random_txn(
            cid, d,
            random.uniform(100, income / 12 * 0.3),
            random.choice([TransactionType.WIRE_TRANSFER, TransactionType.INTERNAL]),
            counterparty=f"Payroll Inc #{random.randint(100,999)}",
            country="US",
        ))

    return CustomerProfile(
        customer_id=cid,
        name=f"{random.choice(['James','Maria','Chen','Aisha','Carlos'])} {random.choice(['Smith','Patel','Zhang','Johnson','Mueller'])}",
        dob=random_date(1960, 1995),
        nationality=random.choice(["US", "UK", "DE", "IN", "FR"]),
        occupation=random.choice(["Software Engineer", "Doctor", "Accountant", "Lawyer"]),
        annual_income_usd=income,
        account_open_date=random_date(2018, 2022),
        documents_present=["passport", "utility_bill", "employment_letter"],
        documents_expired=[],
        documents_missing=[],
        transactions=txns,
        linked_entity_ids=[],
        address=f"{random.randint(1,999)} Main St, Springfield, US",
        phone=f"+1-555-{random.randint(1000,9999)}",
        pep_flag=False,
        sanctions_flag=False,
        adverse_media=[],
    )


# ─── Task 1 generators ──────────────────────────────────────────────────────

def generate_task1_profiles() -> Tuple[List[CustomerProfile], Dict]:
    """
    3 profiles for document-level KYC check.
    1 clean, 1 with expired/missing docs, 1 income mismatch.
    Returns profiles + ground truth decisions.
    """
    profiles = []
    ground_truth = {}

    # Profile A — clean
    a = generate_clean_customer("CUST-T1-A")
    a.name = "Emma Richardson"
    profiles.append(a)
    ground_truth["CUST-T1-A"] = {
        "decision": "clear_customer", "risk_tier": "low",
        "red_flags": [], "missing_docs": []
    }

    # Profile B — expired passport + missing utility bill
    b = CustomerProfile(
        customer_id="CUST-T1-B",
        name="Marcus Webb",
        dob="1985-03-22",
        nationality="US",
        occupation="Sales Manager",
        annual_income_usd=65000,
        account_open_date="2021-06-01",
        documents_present=["national_id"],
        documents_expired=["passport"],
        documents_missing=["utility_bill", "bank_statement"],
        transactions=[
            random_txn("CUST-T1-B", "2024-03-10", 2000, TransactionType.WIRE_TRANSFER,
                       "Employer Corp", "US", "Salary"),
        ],
        address="45 Oak Avenue, Denver, US",
        phone="+1-555-7821",
        pep_flag=False, sanctions_flag=False, adverse_media=[],
    )
    profiles.append(b)
    ground_truth["CUST-T1-B"] = {
        "decision": "request_documents", "risk_tier": "medium",
        "red_flags": ["expired_passport"],
        "missing_docs": ["utility_bill", "bank_statement"]
    }

    # Profile C — student occupation but $180k income (mismatch)
    c = CustomerProfile(
        customer_id="CUST-T1-C",
        name="Li Wei",
        dob="2000-07-14",
        nationality="CN",
        occupation="Student",
        annual_income_usd=180000,
        account_open_date="2023-01-15",
        documents_present=["passport", "national_id"],
        documents_expired=[],
        documents_missing=["utility_bill", "employment_letter", "tax_return"],
        transactions=[
            random_txn("CUST-T1-C", "2024-01-05", 15000, TransactionType.WIRE_TRANSFER,
                       None, "CN", "Transfer"),
            random_txn("CUST-T1-C", "2024-01-20", 12000, TransactionType.CASH_DEPOSIT,
                       None, None, "Cash"),
        ],
        address="12B University Rd, Boston, US",
        phone="+1-555-3344",
        pep_flag=False, sanctions_flag=False, adverse_media=[],
    )
    profiles.append(c)
    ground_truth["CUST-T1-C"] = {
        "decision": "flag_for_review", "risk_tier": "high",
        "red_flags": ["income_occupation_mismatch", "missing_source_of_funds"],
        "missing_docs": ["utility_bill", "employment_letter", "tax_return"]
    }

    return profiles, ground_truth


# ─── Task 2 generators ──────────────────────────────────────────────────────

def generate_structuring_transactions(cid: str) -> List[Transaction]:
    """Cash deposits just under $10,000 — classic structuring."""
    base = datetime(2024, 1, 1)
    txns = []
    amounts = [9800, 9750, 9900, 9650, 9500, 9850]
    for i, amt in enumerate(amounts):
        d = (base + timedelta(days=i * 5)).strftime("%Y-%m-%d")
        txns.append(random_txn(cid, d, amt, TransactionType.CASH_DEPOSIT,
                               description="Cash deposit"))
    return txns


def generate_roundtrip_transactions(cid: str, other_cid: str) -> List[Transaction]:
    """Money sent out and returned within 48 hours."""
    txns = [
        random_txn(cid, "2024-02-10", 50000, TransactionType.WIRE_TRANSFER,
                   other_cid, "AE", "Business payment"),
        random_txn(cid, "2024-02-12", 49500, TransactionType.WIRE_TRANSFER,
                   other_cid, "AE", "Refund"),
    ]
    return txns


def generate_task2_profiles() -> Tuple[List[CustomerProfile], Dict]:
    """
    5 customers with 30-day transaction histories.
    Mix of clean + structuring + round-trip + high velocity.
    """
    profiles = []
    ground_truth = {}

    # Customer 1 — clean
    c1 = generate_clean_customer("CUST-T2-1")
    c1.name = "Sarah O'Brien"
    profiles.append(c1)
    ground_truth["CUST-T2-1"] = {
        "decision": "clear_customer", "risk_tier": "low", "red_flags": []
    }

    # Customer 2 — structuring
    c2 = CustomerProfile(
        customer_id="CUST-T2-2",
        name="Robert Finch",
        dob="1978-11-05",
        nationality="US",
        occupation="Consultant",
        annual_income_usd=90000,
        account_open_date="2020-03-10",
        documents_present=["passport", "utility_bill", "employment_letter"],
        documents_expired=[], documents_missing=[],
        transactions=generate_structuring_transactions("CUST-T2-2"),
        address="88 Pine St, Miami, US",
        phone="+1-555-2200",
        pep_flag=False, sanctions_flag=False, adverse_media=[],
    )
    profiles.append(c2)
    ground_truth["CUST-T2-2"] = {
        "decision": "file_sar", "risk_tier": "high",
        "red_flags": ["structuring", "smurfing"]
    }

    # Customer 3 — round-trip with Customer 4
    c3 = CustomerProfile(
        customer_id="CUST-T2-3",
        name="Amara Osei",
        dob="1990-04-20",
        nationality="NG",
        occupation="Business Owner",
        annual_income_usd=120000,
        account_open_date="2019-07-22",
        documents_present=["passport", "utility_bill"],
        documents_expired=[], documents_missing=[],
        transactions=generate_roundtrip_transactions("CUST-T2-3", "CUST-T2-4"),
        address="200 Harbor Blvd, Houston, US",
        phone="+1-555-4411",
        pep_flag=False, sanctions_flag=False, adverse_media=[],
    )
    profiles.append(c3)
    ground_truth["CUST-T2-3"] = {
        "decision": "flag_for_review", "risk_tier": "high",
        "red_flags": ["round_trip_transfer", "high_value_wire"]
    }

    # Customer 4 — counterparty to round-trip, also suspicious
    c4 = CustomerProfile(
        customer_id="CUST-T2-4",
        name="Dubai Trade LLC",
        dob="2015-01-01",
        nationality="AE",
        occupation="Import/Export",
        annual_income_usd=500000,
        account_open_date="2023-11-01",
        documents_present=["registration_cert"],
        documents_expired=[], documents_missing=["beneficial_owner_declaration"],
        transactions=[
            random_txn("CUST-T2-4", "2024-02-10", 50000, TransactionType.WIRE_TRANSFER,
                       "CUST-T2-3", "US", "Payment received"),
            random_txn("CUST-T2-4", "2024-02-12", 49500, TransactionType.WIRE_TRANSFER,
                       "CUST-T2-3", "US", "Refund sent"),
        ],
        linked_entity_ids=["CUST-T2-3"],
        address="Dubai Free Zone, AE",
        phone="+971-4-5556677",
        pep_flag=False, sanctions_flag=False, adverse_media=[],
    )
    profiles.append(c4)
    ground_truth["CUST-T2-4"] = {
        "decision": "flag_for_review", "risk_tier": "high",
        "red_flags": ["new_account_high_value", "missing_beneficial_owner", "round_trip_transfer"]
    }

    # Customer 5 — high velocity (50+ txns, dormant before)
    base = datetime(2024, 3, 1)
    rapid_txns = []
    for i in range(52):
        d = (base + timedelta(hours=i * 3)).strftime("%Y-%m-%d")
        rapid_txns.append(random_txn(
            "CUST-T2-5", d, random.uniform(500, 3000),
            TransactionType.WIRE_TRANSFER,
            f"Vendor-{random.randint(1,30)}", random.choice(["US", "MX", "CO"]),
        ))

    c5 = CustomerProfile(
        customer_id="CUST-T2-5",
        name="Kevin Marsh",
        dob="1982-09-17",
        nationality="US",
        occupation="Retired",
        annual_income_usd=35000,
        account_open_date="2015-05-05",
        documents_present=["passport", "utility_bill"],
        documents_expired=[], documents_missing=[],
        transactions=rapid_txns,
        address="34 Maple Lane, Phoenix, US",
        phone="+1-555-8890",
        pep_flag=False, sanctions_flag=False, adverse_media=[],
    )
    profiles.append(c5)
    ground_truth["CUST-T2-5"] = {
        "decision": "flag_for_review", "risk_tier": "critical",
        "red_flags": ["high_velocity", "dormant_account_reactivation", "income_mismatch"]
    }

    return profiles, ground_truth


# ─── Task 3 generators ──────────────────────────────────────────────────────

def generate_task3_profiles() -> Tuple[List[CustomerProfile], Dict]:
    """
    10 customers with hidden network relationships.
    Shell companies, chain layering, PEP links.
    """
    profiles = []
    ground_truth = {}
    shared_address = "Suite 400, 1 Corporate Plaza, Wilmington, DE"

    # Shell company hub
    shell = CustomerProfile(
        customer_id="CUST-T3-SHELL",
        name="Apex Holdings LLC",
        dob="2020-06-01",
        nationality="US",
        occupation="Holding Company",
        annual_income_usd=0,
        account_open_date="2020-07-01",
        documents_present=["registration_cert"],
        documents_expired=[], documents_missing=["beneficial_owner_declaration", "audited_accounts"],
        transactions=[
            random_txn("CUST-T3-SHELL", "2024-01-15", 250000, TransactionType.WIRE_TRANSFER,
                       "CUST-T3-A", "US", "Management fee"),
            random_txn("CUST-T3-SHELL", "2024-02-01", 180000, TransactionType.WIRE_TRANSFER,
                       "CUST-T3-B", "US", "Consulting"),
            random_txn("CUST-T3-SHELL", "2024-03-05", 95000, TransactionType.WIRE_TRANSFER,
                       "CUST-T3-C", "VE", "Services"),
        ],
        linked_entity_ids=["CUST-T3-A", "CUST-T3-B", "CUST-T3-C"],
        address=shared_address,
        phone="+1-302-5550100",
        pep_flag=False, sanctions_flag=False, adverse_media=[],
    )
    profiles.append(shell)
    ground_truth["CUST-T3-SHELL"] = {
        "decision": "freeze_account", "risk_tier": "critical",
        "red_flags": ["shell_company", "missing_beneficial_owner", "high_value_outflows"],
        "network_links": ["CUST-T3-A", "CUST-T3-B", "CUST-T3-C"]
    }

    # Three individuals linked to shell, share address
    for cid, name, flag in [
        ("CUST-T3-A", "Victor Morano", False),
        ("CUST-T3-B", "Natalia Voss", False),
        ("CUST-T3-C", "Hassan Al-Radi", True),  # PEP
    ]:
        p = CustomerProfile(
            customer_id=cid,
            name=name,
            dob=random_date(1970, 1985),
            nationality=random.choice(["US", "DE", "AE"]),
            occupation="Director",
            annual_income_usd=random.randint(80000, 150000),
            account_open_date=random_date(2020, 2021),
            documents_present=["passport", "utility_bill"],
            documents_expired=[], documents_missing=[],
            transactions=[
                random_txn(cid, "2024-01-16", random.randint(50000, 200000),
                           TransactionType.WIRE_TRANSFER, "CUST-T3-SHELL", "US", "Received"),
                random_txn(cid, "2024-01-20", random.randint(40000, 180000),
                           TransactionType.WIRE_TRANSFER, None, "CH", "Outbound"),
            ],
            linked_entity_ids=["CUST-T3-SHELL"],
            address=shared_address,
            phone=f"+1-302-555{random.randint(1000,9999)}",
            pep_flag=flag, sanctions_flag=False,
            adverse_media=["Linked to offshore scheme (2022)" if flag else ""],
        )
        profiles.append(p)
        decision = "file_sar" if flag else "flag_for_review"
        risk = "critical" if flag else "high"
        red_flags = ["shared_address_shell", "receives_shell_funds"]
        if flag:
            red_flags += ["pep_flag", "adverse_media"]
        ground_truth[cid] = {
            "decision": decision, "risk_tier": risk,
            "red_flags": red_flags,
            "network_links": ["CUST-T3-SHELL"]
        }

    # Layering chain: D → E → F → cash out
    chain = [
        ("CUST-T3-D", "Elena Petrov", "RU", 2024, 1, 10, 120000),
        ("CUST-T3-E", "Omar Khalid",  "AE", 2024, 1, 11, 118000),
        ("CUST-T3-F", "Jin-Ho Park",  "KR", 2024, 1, 12, 115000),
    ]
    for i, (cid, name, nat, yr, mo, day, amt) in enumerate(chain):
        next_cid = chain[i+1][0] if i < len(chain)-1 else None
        prev_cid = chain[i-1][0] if i > 0 else "CUST-T3-SHELL"
        txns = []
        if i > 0:
            txns.append(random_txn(cid, f"{yr}-0{mo}-{day}", amt + 5000,
                                   TransactionType.WIRE_TRANSFER, prev_cid, nat, "Received"))
        if next_cid:
            txns.append(random_txn(cid, f"{yr}-0{mo}-{day+1}", amt,
                                   TransactionType.WIRE_TRANSFER, next_cid, "KR", "Forwarded"))
        else:
            txns.append(random_txn(cid, f"{yr}-0{mo}-{day+1}", amt,
                                   TransactionType.CASH_WITHDRAWAL, None, None, "Cash out"))

        p = CustomerProfile(
            customer_id=cid,
            name=name,
            dob=random_date(1975, 1990),
            nationality=nat,
            occupation="Trader",
            annual_income_usd=60000,
            account_open_date=random_date(2023, 2024),
            documents_present=["passport"],
            documents_expired=[], documents_missing=["utility_bill"],
            transactions=txns,
            linked_entity_ids=[c for c in [prev_cid, next_cid] if c],
            address=f"{random.randint(1,500)} Trade Blvd, {nat}",
            phone=f"+{random.randint(1,99)}-555-{random.randint(1000,9999)}",
            pep_flag=False, sanctions_flag=False, adverse_media=[],
        )
        profiles.append(p)
        ground_truth[cid] = {
            "decision": "file_sar", "risk_tier": "critical",
            "red_flags": ["layering", "rapid_forwarding", "chain_transfer"],
            "network_links": [c for c in [prev_cid, next_cid] if c]
        }

    # 3 clean customers as noise
    for j in range(3):
        cid = f"CUST-T3-CLEAN{j+1}"
        c = generate_clean_customer(cid)
        profiles.append(c)
        ground_truth[cid] = {
            "decision": "clear_customer", "risk_tier": "low", "red_flags": []
        }

    return profiles, ground_truth