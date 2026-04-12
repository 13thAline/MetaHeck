"""
Procedural Data Engine — Synthetic KYC/AML Episode Generator.

Every call to `generate_episode(task_id, seed)` produces a fresh, randomized
set of customers, transactions, documents, and device signals.  Fraud patterns
are injected per-task and the affected IDs are secretly logged into a ground-
truth manifest so the deterministic grader can evaluate the agent without
relying on static data.

Design constraints:
  - stdlib only (random, uuid, hashlib) — no numpy/pandas to keep Docker slim
  - Memory-efficient: ~50-200 transactions per episode, well under 8 GB limit
  - Seeded RNG for reproducibility: same seed → identical episode

v3.0 Changes:
  - All timestamps are now exact `datetime` (ISO 8601 with time component)
  - evidence_keywords replaced with expected_typologies (discrete regulatory codes)
  - New fraud pattern: Burst Velocity (dormant → micro-transaction storm)
  - New profile type: Ambiguous "Grey" customers (exploit-resistant)
"""

from __future__ import annotations

import hashlib
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from env.models import CustomerProfile, EpisodeManifest

# ---------------------------------------------------------------------------
# Name / address / occupation pools (curated for realism)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bob", "Carlos", "Diana", "Evelyn", "Frank", "Grace", "Hector",
    "Irene", "James", "Kira", "Liam", "Maya", "Nathan", "Olivia", "Pavel",
    "Quinn", "Rosa", "Stefan", "Tanya", "Ulrich", "Vera", "Winston", "Xena",
    "Yuki", "Zara", "Amara", "Dmitri", "Fatima", "Giovanni",
]

LAST_NAMES = [
    "Smith", "Chen", "Patel", "Garcia", "Novak", "Williams", "Kim", "O'Brien",
    "Hassan", "Johansson", "Petrov", "Nakamura", "Torres", "Müller", "Singh",
    "Cooper", "Ivanov", "Reyes", "Okonkwo", "Bergström", "Fischer", "Rao",
    "Brooks", "Lombardi", "Andersen", "Volkov", "DaSilva", "Tanaka", "Moreau",
    "Ndlovu",
]

STREETS = [
    "Maple Ave", "Oak Lane", "Industrial Pkwy", "Broadway", "Pine St",
    "Cedar Blvd", "Elm Court", "River Road", "Market St", "Highland Drive",
    "Prestige Plaza", "Commerce Way", "Lakefront Dr", "Harbor View",
    "University Ave", "Corporate Center Blvd",
]

CITIES = [
    ("Seattle", "WA"), ("Austin", "TX"), ("Miami", "FL"), ("New York", "NY"),
    ("Chicago", "IL"), ("Denver", "CO"), ("Portland", "OR"), ("Atlanta", "GA"),
    ("Boston", "MA"), ("San Diego", "CA"), ("Phoenix", "AZ"), ("Detroit", "MI"),
]

OCCUPATIONS_CLEAN = [
    "Graphic Designer", "Software Engineer", "Teacher", "Nurse",
    "Accountant", "Retail Manager", "Marketing Analyst", "Chef",
    "Mechanic", "Dentist", "Social Worker", "Pharmacist",
]

OCCUPATIONS_RISKY = [
    "Import/Export Director", "Freelance Consultant", "Private Investor",
    "International Art Dealer", "Crypto Trader", "Real Estate Developer",
    "Director of Offshore Operations", "Venture Capital Associate",
]

HIGH_RISK_JURISDICTIONS = [
    "Cyprus", "Cayman Islands", "Panama", "British Virgin Islands",
    "Liechtenstein", "Seychelles", "Malta", "Dubai", "Bermuda",
]

DOC_TYPES = ["passport", "drivers_license", "utility_bill", "bank_statement",
             "tax_return", "employment_letter"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_from_episode_id(episode_id: str) -> int:
    """Deterministic seed from an episode_id string."""
    return int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16)


def _make_txn_id(rng: random.Random) -> str:
    """Generate a unique transaction ID."""
    return f"TXN-{uuid.UUID(int=rng.getrandbits(128)).hex[:8].upper()}"


def _random_datetime(rng: random.Random, start: datetime, end: datetime) -> datetime:
    """Random datetime between start and end (inclusive), with second precision."""
    delta_seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, max(delta_seconds, 1)))


def _random_address(rng: random.Random) -> str:
    num = rng.randint(100, 9999)
    street = rng.choice(STREETS)
    city, state = rng.choice(CITIES)
    return f"{num} {street}, {city}, {state}"


def _random_ip(rng: random.Random, local: bool = True) -> str:
    if local:
        return f"192.168.{rng.randint(0,255)}.{rng.randint(1,254)}"
    return f"{rng.randint(1,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"


# ---------------------------------------------------------------------------
# Customer generation
# ---------------------------------------------------------------------------

def _generate_clean_customer(
    rng: random.Random, cid: str
) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
    """Generate a legitimate customer with no fraud signals.

    Returns (profile, db_record, ground_truth_entry, expected_typologies).
    """
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    addr = _random_address(rng)
    occ = rng.choice(OCCUPATIONS_CLEAN)

    # Documents — all valid
    docs_text = f"ID: Valid {rng.choice(['Passport', 'Drivers License'])}. Utility Bill: Address matches ({addr})."

    # Clean transactions — salary, rent, groceries
    txns = []
    base_dt = datetime(2025, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    salary = rng.randint(2500, 6000)
    for month_offset in range(rng.randint(2, 4)):
        d = base_dt + timedelta(days=30 * month_offset + rng.randint(0, 5),
                                hours=rng.randint(8, 17),
                                minutes=rng.randint(0, 59))
        tid = _make_txn_id(rng)
        txns.append({"timestamp": d.isoformat(), "amount": salary, "type": "deposit",
                      "id": tid, "description": f"Salary - {'TechCorp' if rng.random() > 0.5 else 'Acme Inc'}"})
        # Expenses
        for _ in range(rng.randint(1, 3)):
            tid2 = _make_txn_id(rng)
            exp_dt = d + timedelta(days=rng.randint(1, 15),
                                   hours=rng.randint(0, 12),
                                   minutes=rng.randint(0, 59))
            txns.append({"timestamp": exp_dt.isoformat(),
                          "amount": rng.randint(50, 1200), "type": "withdrawal",
                          "id": tid2, "description": rng.choice(["Rent Payment", "Grocery Store", "Utilities", "Gas Station"])})

    ip = _random_ip(rng, local=True)
    device = f"IP: {ip} (Local)"

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": (base_dt - timedelta(days=rng.randint(30, 365))).isoformat(),
                       "dob": f"{rng.randint(1960, 2000)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                       "address": addr, "occupation": occ, "pep_status": "None"},
    )

    db_record = {
        "docs": docs_text,
        "txns": txns,
        "watchlists": "CLEAR",
        "device": device,
    }

    gt = {
        "expected_decision": "approve",
        "expected_txns": [],
        "expected_docs": [],
        "decision": "clear_customer",
        "risk_tier": "low",
        "red_flags": [],
    }

    typologies = ["CLEAN_PROFILE"]

    return profile, db_record, gt, typologies


def _generate_address_mismatch_customer(
    rng: random.Random, cid: str
) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
    """Task1-style fraud: document address mismatch."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    real_addr = _random_address(rng)
    # Mismatch: utility bill shows a P.O. Box
    po_box = f"P.O. Box {rng.randint(100, 9999)}, {rng.choice(CITIES)[0]}"

    docs_text = f"ID: Valid Driver's License. Utility Bill: Mismatch (Address is {po_box})."

    txns = []
    base_dt = datetime(2025, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
    for i in range(rng.randint(1, 3)):
        d = base_dt + timedelta(days=10 * i + rng.randint(0, 5),
                                hours=rng.randint(0, 8),
                                minutes=rng.randint(0, 59))
        tid = _make_txn_id(rng)
        txns.append({"timestamp": d.isoformat(), "amount": rng.randint(1000, 9500),
                      "type": "deposit", "id": tid, "description": "Wire Transfer"})

    ip = _random_ip(rng, local=True)
    device = f"IP: {ip} (Local)"

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": "2025-01-01T00:00:00+00:00",
                       "dob": f"{rng.randint(1965, 1998)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                       "address": real_addr, "occupation": rng.choice(OCCUPATIONS_CLEAN),
                       "pep_status": "None"},
    )

    db_record = {
        "docs": docs_text,
        "txns": txns,
        "watchlists": "CLEAR",
        "device": device,
    }

    gt = {
        "expected_decision": "escalate",
        "expected_txns": [],
        "expected_docs": ["utility_bill"],
        "decision": "escalate",
        "risk_tier": "medium",
        "red_flags": ["address_mismatch"],
    }

    typologies = ["ADDRESS_MISMATCH"]

    return profile, db_record, gt, typologies


def _generate_structuring_customer(
    rng: random.Random, cid: str
) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
    """Task2-style fraud: structuring deposits just under $10,000 threshold."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    addr = _random_address(rng)
    occ = rng.choice(OCCUPATIONS_RISKY)

    docs_text = f"ID: Passport (Expired). Employment verification: {occ}."

    # Structuring: multiple deposits between $9,000 and $9,999
    fraud_txn_ids = []
    txns = []
    base_dt = datetime(2025, 2, 1, 14, 0, 0, tzinfo=timezone.utc)
    num_structured = rng.randint(2, 5)
    for i in range(num_structured):
        d = base_dt + timedelta(days=i + rng.randint(0, 2),
                                hours=rng.randint(0, 6),
                                minutes=rng.randint(0, 59))
        tid = _make_txn_id(rng)
        amount = rng.randint(9000, 9999)
        txns.append({"timestamp": d.isoformat(), "amount": amount, "type": "deposit",
                      "id": tid, "description": "Wire Transfer - Unknown Origin"})
        fraud_txn_ids.append(tid)

    # Sprinkle a couple clean transactions
    for i in range(rng.randint(1, 2)):
        tid = _make_txn_id(rng)
        txns.append({"timestamp": (base_dt + timedelta(days=10 + i * 5, hours=rng.randint(8, 17))).isoformat(),
                      "amount": rng.randint(50, 500), "type": "withdrawal",
                      "id": tid, "description": rng.choice(["ATM Withdrawal", "Grocery Store"])})

    foreign_ip = _random_ip(rng, local=False)
    device = f"IP: {foreign_ip} (VPN Detected)"

    pep_hit = rng.choice(["", "PEP HIT (Uncle is Minister of Finance)",
                           "PEP HIT (Cousin is Governor)", ""])

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": "2025-02-01T00:00:00+00:00",
                       "dob": f"{rng.randint(1970, 1995)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                       "address": addr, "occupation": occ, "pep_status": pep_hit or "None"},
    )

    db_record = {
        "docs": docs_text,
        "txns": txns,
        "watchlists": pep_hit if pep_hit else "CLEAR",
        "device": device,
    }

    gt = {
        "expected_decision": "freeze_account",
        "expected_txns": fraud_txn_ids,
        "expected_docs": ["passport"],
        "decision": "freeze_account",
        "risk_tier": "high",
        "red_flags": ["structuring", "vpn_detected"],
    }

    typologies = ["STRUCTURING_314A", "SMURFING"]
    if pep_hit:
        typologies.append("PEP_UNDISCLOSED")

    return profile, db_record, gt, typologies


def _generate_layering_customer(
    rng: random.Random, cid: str,
    is_shell: bool = False,
    linked_cids: Optional[List[str]] = None,
) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
    """Task3-style fraud: chain-layering / shell company / circular transactions."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    jurisdiction = rng.choice(HIGH_RISK_JURISDICTIONS)

    if is_shell:
        name = f"{rng.choice(['Apex', 'Garantia', 'Meridian', 'Zenith', 'Pinnacle'])} {rng.choice(['Holdings', 'LLC', 'Enterprises', 'Capital', 'Group'])}"
        occ = "Shell Company / Holding Entity"
        addr = f"{rng.randint(1, 500)} Corporate Plaza, Suite {rng.randint(100, 999)}, Wilmington, DE"
    else:
        occ = rng.choice(OCCUPATIONS_RISKY)
        addr = _random_address(rng)

    # Suspicious docs
    doc_issues = rng.choice([
        "Suspicious formatting, possible Deepfake ID.",
        "PDF metadata authored yesterday by unknown software.",
        "Document serial number not in issuing authority database.",
    ])
    docs_text = f"ID: Passport (Uploaded). ALERT: {doc_issues}"

    # Chain-layering transactions
    fraud_txn_ids = []
    txns = []
    base_dt = datetime(2025, 3, 1, 2, 30, 0, tzinfo=timezone.utc)
    big_amount = rng.randint(30000, 120000)

    # Incoming wire
    tid_in = _make_txn_id(rng)
    txns.append({"timestamp": base_dt.isoformat(), "amount": big_amount, "type": "deposit",
                  "id": tid_in, "description": f"{'Garantia LLC' if is_shell else 'Wire Transfer'} - Consulting"})
    fraud_txn_ids.append(tid_in)

    # Splits to associates
    linked = linked_cids or []
    remaining = big_amount
    for i, linked_cid in enumerate(linked[:3]):
        split_amt = rng.randint(8000, min(remaining - 5000, 30000)) if remaining > 15000 else remaining
        remaining -= split_amt
        d = base_dt + timedelta(days=i + 1, hours=rng.randint(0, 4),
                                minutes=rng.randint(0, 59))
        tid = _make_txn_id(rng)
        txns.append({"timestamp": d.isoformat(), "amount": split_amt, "type": "withdrawal",
                      "id": tid, "description": f"Transfer to {linked_cid}"})
        fraud_txn_ids.append(tid)

    # Offshore outflow
    if remaining > 0:
        tid_off = _make_txn_id(rng)
        off_dt = base_dt + timedelta(days=len(linked) + 1, hours=rng.randint(1, 6))
        txns.append({"timestamp": off_dt.isoformat(),
                      "amount": remaining, "type": "withdrawal",
                      "id": tid_off, "description": f"Offshore Account ({jurisdiction})"})
        fraud_txn_ids.append(tid_off)

    # Circular loop: money comes back
    if rng.random() > 0.4:
        tid_loop = _make_txn_id(rng)
        loop_amt = rng.randint(int(big_amount * 0.6), big_amount)
        loop_dt = base_dt + timedelta(days=rng.randint(4, 7),
                                      hours=rng.randint(0, 5),
                                      minutes=rng.randint(0, 59))
        txns.append({"timestamp": loop_dt.isoformat(),
                      "amount": loop_amt, "type": "deposit",
                      "id": tid_loop, "description": f"{'Associate Repayment' if not is_shell else f'{name} - Refund'} (Circular Loop)"})
        fraud_txn_ids.append(tid_loop)

    spoofed_ip = _random_ip(rng, local=False)
    device = f"IP: {spoofed_ip} ({jurisdiction})"

    pep = rng.choice(["Potential Match", "Adverse Media Hit", "Sanctions List - Partial Match", ""])

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": "2025-03-01T00:00:00+00:00",
                       "dob": f"{rng.randint(1955, 1990)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                       "address": addr, "occupation": occ,
                       "pep_status": pep if pep else "None"},
    )

    db_record = {
        "docs": docs_text,
        "txns": txns,
        "watchlists": f"PEP: {pep}" if pep else "CLEAR",
        "device": device,
    }

    gt = {
        "expected_decision": "file_sar",
        "expected_txns": fraud_txn_ids,
        "expected_docs": ["passport"],
        "decision": "file_sar",
        "risk_tier": "critical",
        "red_flags": ["layering", "circular_transactions", "shell_company" if is_shell else "chain_transfer"],
    }

    typologies = ["LAYERING_FATF_02", "CIRCULAR_TRANSACTION", "RAPID_MOVEMENT"]
    if is_shell:
        typologies.extend(["SHELL_COMPANY_FATF_04", "BENEFICIAL_OWNER_CONCEALMENT"])
    else:
        typologies.append("STRUCTURING_314A")
    if "Deepfake" in doc_issues:
        typologies.append("DEEPFAKE_DOCUMENT")

    return profile, db_record, gt, typologies


# ---------------------------------------------------------------------------
# Burst Velocity fraud pattern (Phase 2)
# ---------------------------------------------------------------------------

def _generate_burst_velocity_customer(
    rng: random.Random, cid: str
) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
    """Dormant account that erupts with 30-50 micro-transactions at 3 AM."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    addr = _random_address(rng)
    occ = rng.choice(OCCUPATIONS_CLEAN)

    docs_text = f"ID: Valid Passport. Utility Bill: Address matches ({addr})."

    txns = []
    fraud_txn_ids = []

    # Phase A: dormant period — 1-2 small transactions over 3-6 months
    dormant_start = datetime(2024, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(rng.randint(1, 2)):
        d = dormant_start + timedelta(days=rng.randint(30, 150),
                                      hours=rng.randint(9, 17),
                                      minutes=rng.randint(0, 59))
        tid = _make_txn_id(rng)
        txns.append({"timestamp": d.isoformat(), "amount": rng.randint(20, 200),
                      "type": "withdrawal", "id": tid,
                      "description": rng.choice(["Coffee Shop", "Gas Station", "Grocery Store"])})

    # Phase B: burst — 30-50 micro-transactions within a 15-minute window at 3:00 AM
    burst_date = datetime(2025, 2, 15, 3, 0, 0, tzinfo=timezone.utc)
    num_burst = rng.randint(30, 50)
    burst_window_seconds = 15 * 60  # 15 minutes

    for i in range(num_burst):
        offset_seconds = rng.randint(0, burst_window_seconds)
        d = burst_date + timedelta(seconds=offset_seconds)
        tid = _make_txn_id(rng)
        amount = rng.randint(5, 50)
        txns.append({"timestamp": d.isoformat(), "amount": amount, "type": "withdrawal",
                      "id": tid, "description": f"POS Terminal #{rng.randint(1000, 9999)}"})
        fraud_txn_ids.append(tid)

    jurisdiction = rng.choice(HIGH_RISK_JURISDICTIONS)
    foreign_ip = _random_ip(rng, local=False)
    device = f"IP: {foreign_ip} ({jurisdiction}, 3:00 AM local)"

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": dormant_start.isoformat(),
                       "dob": f"{rng.randint(1970, 2000)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                       "address": addr, "occupation": occ, "pep_status": "None"},
    )

    db_record = {
        "docs": docs_text,
        "txns": txns,
        "watchlists": "CLEAR",
        "device": device,
    }

    gt = {
        "expected_decision": "freeze_account",
        "expected_txns": fraud_txn_ids,
        "expected_docs": [],
        "decision": "freeze_account",
        "risk_tier": "high",
        "red_flags": ["burst_velocity", "dormant_activation", "foreign_ip_3am"],
    }

    typologies = ["BURST_VELOCITY", "DORMANT_ACTIVATION"]

    return profile, db_record, gt, typologies


# ---------------------------------------------------------------------------
# Ambiguous "Grey" customer (Phase 3 — exploit resistance)
# ---------------------------------------------------------------------------

def _generate_ambiguous_customer(
    rng: random.Random, cid: str
) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
    """High-net-worth individual with massive red flags that are ACTUALLY LEGITIMATE.

    Clearance evidence (probate/inheritance docs) is hidden in the document
    dossier and interview responses — the agent must explicitly pull them.
    """
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    addr = _random_address(rng)

    # High-profile clean occupation
    occ = rng.choice(["Retired Surgeon", "Estate Attorney", "University Professor",
                       "Retired Military Officer", "Non-Profit Director"])

    # --- Red-flag surface (looks terrible) ---
    inheritance_amount = rng.randint(250_000, 1_000_000)
    jurisdiction = rng.choice(HIGH_RISK_JURISDICTIONS)
    foreign_ip = _random_ip(rng, local=False)

    # Documents: surface-level alarm + hidden clearance
    clearance_doc = rng.choice([
        f"Probate Court Order #{rng.randint(10000, 99999)}: Estate of {rng.choice(LAST_NAMES)}, "
        f"approved {rng.randint(2023, 2024)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}. "
        f"Beneficiary: {name}. Total estate value: ${inheritance_amount:,}.",

        f"Inheritance Certificate: {name} is sole heir to estate of late "
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)} (deceased "
        f"{rng.randint(2023, 2024)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}). "
        f"Notarized by {rng.choice(CITIES)[0]} County Probate Court.",

        f"Real Estate Closing Statement: Property at {_random_address(rng)} sold for "
        f"${inheritance_amount:,} on {rng.randint(2023, 2024)}-{rng.randint(1,12):02d}-"
        f"{rng.randint(1,28):02d}. Proceeds transferred to {name}'s account.",
    ])

    docs_text = (
        f"ID: Valid Passport. Employment: {occ}. "
        f"ALERT: Large offshore wire (${inheritance_amount:,}) from {jurisdiction}. "
        f"VPN detected on multiple logins. "
        f"--- SUPPLEMENTAL DOCUMENTS --- {clearance_doc}"
    )

    # Interview response reveals legitimacy
    interview_response = (
        f"I recently inherited funds from a family member's estate. "
        f"The probate court approved the distribution. I use a VPN because "
        f"I travel frequently for {rng.choice(['medical conferences', 'academic seminars', 'charity work'])}. "
        f"I can provide the full probate court order and notarized inheritance certificate."
    )

    # Transactions: large but explainable
    txns = []
    base_dt = datetime(2025, 1, 15, 11, 0, 0, tzinfo=timezone.utc)

    # Inheritance deposit
    tid_inherit = _make_txn_id(rng)
    txns.append({"timestamp": base_dt.isoformat(), "amount": inheritance_amount,
                  "type": "deposit", "id": tid_inherit,
                  "description": f"Wire Transfer from {jurisdiction} - Estate Distribution"})

    # Normal spending after inheritance
    for i in range(rng.randint(3, 6)):
        d = base_dt + timedelta(days=rng.randint(1, 60),
                                hours=rng.randint(8, 20),
                                minutes=rng.randint(0, 59))
        tid = _make_txn_id(rng)
        amount = rng.randint(500, 15000)
        txns.append({"timestamp": d.isoformat(), "amount": amount,
                      "type": "withdrawal", "id": tid,
                      "description": rng.choice([
                          "Property Tax Payment", "Home Renovation", "Charity Donation",
                          "Investment Brokerage", "Medical Expenses", "University Tuition"
                      ])})

    device = f"IP: {foreign_ip} (VPN Detected - {jurisdiction})"

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": (base_dt - timedelta(days=rng.randint(365, 3650))).isoformat(),
                       "dob": f"{rng.randint(1950, 1980)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                       "address": addr, "occupation": occ, "pep_status": "None",
                       "net_worth_declared": f"${rng.randint(500_000, 5_000_000):,}"},
    )

    db_record = {
        "docs": docs_text,
        "txns": txns,
        "watchlists": "CLEAR",
        "device": device,
        "interview_response": interview_response,
    }

    gt = {
        "expected_decision": "approve",
        "expected_txns": [],
        "expected_docs": [],
        "decision": "approve",
        "risk_tier": "low",
        "red_flags": [],
        "is_ambiguous": True,
    }

    typologies = ["CLEAN_PROFILE"]

    return profile, db_record, gt, typologies


# ---------------------------------------------------------------------------
# High-level episode generators (one per task difficulty)
# ---------------------------------------------------------------------------

def _generate_task1_easy(rng: random.Random) -> EpisodeManifest:
    """2-3 customers: 1 clean + 1-2 with address mismatch."""
    customers = []
    database = {}
    ground_truth = {}
    expected_typologies = {}

    num_customers = rng.randint(2, 3)
    num_flagged = rng.randint(1, min(2, num_customers - 1))

    for i in range(num_customers):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        if i < num_flagged:
            prof, db, gt, typo = _generate_address_mismatch_customer(rng, cid)
        else:
            prof, db, gt, typo = _generate_clean_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        expected_typologies[cid] = typo

    rng.shuffle(customers)

    return EpisodeManifest(
        task_id="task1_easy", seed=0,
        customers=customers, database=database,
        ground_truth=ground_truth, expected_typologies=expected_typologies,
    )


def _generate_task2_medium(rng: random.Random) -> EpisodeManifest:
    """3-5 customers: 1-2 clean + 2-3 structuring + 1 burst velocity + 1 ambiguous."""
    customers = []
    database = {}
    ground_truth = {}
    expected_typologies = {}

    # Structuring fraud
    num_fraud = rng.randint(2, 3)
    for i in range(num_fraud):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        prof, db, gt, typo = _generate_structuring_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        expected_typologies[cid] = typo

    # Burst velocity fraud
    cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
    prof, db, gt, typo = _generate_burst_velocity_customer(rng, cid)
    customers.append(prof)
    database[cid] = db
    ground_truth[cid] = gt
    expected_typologies[cid] = typo

    # Ambiguous grey case
    cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
    prof, db, gt, typo = _generate_ambiguous_customer(rng, cid)
    customers.append(prof)
    database[cid] = db
    ground_truth[cid] = gt
    expected_typologies[cid] = typo

    # Clean
    num_clean = rng.randint(1, 2)
    for i in range(num_clean):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        prof, db, gt, typo = _generate_clean_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        expected_typologies[cid] = typo

    rng.shuffle(customers)

    return EpisodeManifest(
        task_id="task2_medium", seed=0,
        customers=customers, database=database,
        ground_truth=ground_truth, expected_typologies=expected_typologies,
    )


def _generate_task3_hard(rng: random.Random) -> EpisodeManifest:
    """5-8 customers: shell company network + layering chain + burst + ambiguous + clean."""
    customers = []
    database = {}
    ground_truth = {}
    expected_typologies = {}
    network_truth: Dict[str, List[str]] = {}

    # --- Shell company cluster (1 shell + 2-3 associates) ---
    shell_cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
    associate_cids = [f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
                      for _ in range(rng.randint(2, 3))]

    # Shell company
    prof, db, gt, typo = _generate_layering_customer(
        rng, shell_cid, is_shell=True, linked_cids=associate_cids)
    customers.append(prof)
    database[shell_cid] = db
    ground_truth[shell_cid] = gt
    expected_typologies[shell_cid] = typo
    network_truth[shell_cid] = associate_cids

    # Associates
    for acid in associate_cids:
        prof, db, gt, typo = _generate_layering_customer(
            rng, acid, is_shell=False, linked_cids=[shell_cid])
        customers.append(prof)
        database[acid] = db
        ground_truth[acid] = gt
        expected_typologies[acid] = typo
        network_truth[acid] = [shell_cid]

    # --- Layering chain (2-3 nodes) ---
    chain_len = rng.randint(2, 3)
    chain_cids = [f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
                  for _ in range(chain_len)]

    for idx, ccid in enumerate(chain_cids):
        linked = []
        if idx > 0:
            linked.append(chain_cids[idx - 1])
        if idx < chain_len - 1:
            linked.append(chain_cids[idx + 1])

        prof, db, gt, typo = _generate_layering_customer(
            rng, ccid, is_shell=False, linked_cids=linked)
        customers.append(prof)
        database[ccid] = db
        ground_truth[ccid] = gt
        expected_typologies[ccid] = typo
        network_truth[ccid] = linked

    # --- Burst velocity ---
    cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
    prof, db, gt, typo = _generate_burst_velocity_customer(rng, cid)
    customers.append(prof)
    database[cid] = db
    ground_truth[cid] = gt
    expected_typologies[cid] = typo

    # --- Ambiguous grey case(s) ---
    num_ambiguous = rng.randint(1, 2)
    for _ in range(num_ambiguous):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        prof, db, gt, typo = _generate_ambiguous_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        expected_typologies[cid] = typo

    # --- Clean customers ---
    num_clean = rng.randint(1, 2)
    for _ in range(num_clean):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        prof, db, gt, typo = _generate_clean_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        expected_typologies[cid] = typo

    rng.shuffle(customers)

    return EpisodeManifest(
        task_id="task3_hard", seed=0,
        customers=customers, database=database,
        ground_truth=ground_truth, network_truth=network_truth,
        expected_typologies=expected_typologies,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_episode(task_id: str, seed: Optional[int] = None,
                     episode_id: Optional[str] = None) -> EpisodeManifest:
    """Generate a complete randomized episode for the given task.

    Args:
        task_id: one of 'task1_easy', 'task2_medium', 'task3_hard'
        seed: explicit RNG seed for reproducibility.  If None, derived from
              episode_id (or random if both are None).
        episode_id: used to derive seed when `seed` is not provided.

    Returns:
        EpisodeManifest with customers, database, and secret ground truth.
    """
    if seed is None:
        if episode_id:
            seed = _seed_from_episode_id(episode_id)
        else:
            seed = random.randint(0, 2**31)

    rng = random.Random(seed)

    generators = {
        "task1_easy": _generate_task1_easy,
        "task2_medium": _generate_task2_medium,
        "task3_hard": _generate_task3_hard,
    }

    if task_id not in generators:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(generators.keys())}")

    manifest = generators[task_id](rng)
    # Stamp the actual seed used (for trajectory logging / reproducibility)
    manifest.seed = seed
    return manifest
