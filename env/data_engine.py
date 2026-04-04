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
"""

from __future__ import annotations

import hashlib
import random
import uuid
from datetime import date, timedelta
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


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta, 1)))


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

def _generate_clean_customer(rng: random.Random, cid: str) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
    """Generate a legitimate customer with no fraud signals.
    
    Returns (profile, db_record, ground_truth_entry, evidence_keywords).
    """
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    addr = _random_address(rng)
    city_state = addr.split(", ", 1)[1] if ", " in addr else addr
    occ = rng.choice(OCCUPATIONS_CLEAN)

    # Documents — all valid
    docs_text = f"ID: Valid {rng.choice(['Passport', 'Drivers License'])}. Utility Bill: Address matches ({addr})."
    docs_list = [
        {"type": "passport", "status": "valid", "address": addr},
        {"type": "utility_bill", "status": "valid", "address": addr},
    ]

    # Clean transactions — salary, rent, groceries
    txns = []
    base_date = date(2025, 1, 1)
    salary = rng.randint(2500, 6000)
    for month_offset in range(rng.randint(2, 4)):
        d = base_date + timedelta(days=30 * month_offset + rng.randint(0, 5))
        tid = _make_txn_id(rng)
        txns.append({"date": d.isoformat(), "amount": salary, "type": "deposit",
                      "id": tid, "description": f"Salary - {'TechCorp' if rng.random() > 0.5 else 'Acme Inc'}"})
        # Expenses
        for _ in range(rng.randint(1, 3)):
            tid2 = _make_txn_id(rng)
            txns.append({"date": (d + timedelta(days=rng.randint(1, 15))).isoformat(),
                          "amount": rng.randint(50, 1200), "type": "withdrawal",
                          "id": tid2, "description": rng.choice(["Rent Payment", "Grocery Store", "Utilities", "Gas Station"])})

    ip = _random_ip(rng, local=True)
    device = f"IP: {ip} (Local)"

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": (base_date - timedelta(days=rng.randint(30, 365))).isoformat(),
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

    keywords = ["clean", "normal", "legitimate", "regular", "consistent",
                 name.lower().split()[0], occ.lower()]

    return profile, db_record, gt, keywords


def _generate_address_mismatch_customer(rng: random.Random, cid: str) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
    """Task1-style fraud: document address mismatch."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    real_addr = _random_address(rng)
    # Mismatch: utility bill shows a P.O. Box
    po_box = f"P.O. Box {rng.randint(100, 9999)}, {rng.choice(CITIES)[0]}"

    docs_text = f"ID: Valid Driver's License. Utility Bill: Mismatch (Address is {po_box})."
    docs_list = [
        {"type": "drivers_license", "status": "valid", "address": real_addr},
        {"type": "utility_bill", "status": "mismatch", "address": po_box,
         "notes": f"Listed address differs from ID. Shows {po_box} vs {real_addr}."},
    ]

    txns = []
    base_date = date(2025, 1, 1)
    for i in range(rng.randint(1, 3)):
        d = base_date + timedelta(days=10 * i + rng.randint(0, 5))
        tid = _make_txn_id(rng)
        txns.append({"date": d.isoformat(), "amount": rng.randint(1000, 9500),
                      "type": "deposit", "id": tid, "description": "Wire Transfer"})

    ip = _random_ip(rng, local=True)
    device = f"IP: {ip} (Local)"

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": "2025-01-01",
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

    keywords = ["mismatch", "p.o. box", "address", "utility", "differs",
                 po_box.lower(), name.lower().split()[0]]

    return profile, db_record, gt, keywords


def _generate_structuring_customer(rng: random.Random, cid: str) -> Tuple[CustomerProfile, Dict[str, Any], Dict[str, Any], List[str]]:
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
    base_date = date(2025, 2, 1)
    num_structured = rng.randint(2, 5)
    for i in range(num_structured):
        d = base_date + timedelta(days=i + rng.randint(0, 2))
        tid = _make_txn_id(rng)
        amount = rng.randint(9000, 9999)
        txns.append({"date": d.isoformat(), "amount": amount, "type": "deposit",
                      "id": tid, "description": "Wire Transfer - Unknown Origin"})
        fraud_txn_ids.append(tid)

    # Sprinkle a couple clean transactions
    for i in range(rng.randint(1, 2)):
        tid = _make_txn_id(rng)
        txns.append({"date": (base_date + timedelta(days=10 + i * 5)).isoformat(),
                      "amount": rng.randint(50, 500), "type": "withdrawal",
                      "id": tid, "description": rng.choice(["ATM Withdrawal", "Grocery Store"])})

    foreign_ip = _random_ip(rng, local=False)
    device = f"IP: {foreign_ip} (VPN Detected)"

    pep_hit = rng.choice(["", "PEP HIT (Uncle is Minister of Finance)",
                           "PEP HIT (Cousin is Governor)", ""])

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": "2025-02-01",
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

    keywords = ["structuring", "smurfing", "threshold", "under", "10000",
                "$10,000", "9900", "9800", "vpn", "expired", "unknown origin",
                name.lower().split()[0]]

    return profile, db_record, gt, keywords


def _generate_layering_customer(rng: random.Random, cid: str,
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
    base_date = date(2025, 3, 1)
    big_amount = rng.randint(30000, 120000)

    # Incoming wire
    tid_in = _make_txn_id(rng)
    txns.append({"date": base_date.isoformat(), "amount": big_amount, "type": "deposit",
                  "id": tid_in, "description": f"{'Garantia LLC' if is_shell else 'Wire Transfer'} - Consulting"})
    fraud_txn_ids.append(tid_in)

    # Splits to associates
    linked = linked_cids or []
    remaining = big_amount
    for i, linked_cid in enumerate(linked[:3]):
        split_amt = rng.randint(8000, min(remaining - 5000, 30000)) if remaining > 15000 else remaining
        remaining -= split_amt
        d = base_date + timedelta(days=i + 1)
        tid = _make_txn_id(rng)
        txns.append({"date": d.isoformat(), "amount": split_amt, "type": "withdrawal",
                      "id": tid, "description": f"Transfer to {linked_cid}"})
        fraud_txn_ids.append(tid)

    # Offshore outflow
    if remaining > 0:
        tid_off = _make_txn_id(rng)
        txns.append({"date": (base_date + timedelta(days=len(linked) + 1)).isoformat(),
                      "amount": remaining, "type": "withdrawal",
                      "id": tid_off, "description": f"Offshore Account ({jurisdiction})"})
        fraud_txn_ids.append(tid_off)

    # Circular loop: money comes back
    if rng.random() > 0.4:
        tid_loop = _make_txn_id(rng)
        loop_amt = rng.randint(int(big_amount * 0.6), big_amount)
        txns.append({"date": (base_date + timedelta(days=rng.randint(4, 7))).isoformat(),
                      "amount": loop_amt, "type": "deposit",
                      "id": tid_loop, "description": f"{'Associate Repayment' if not is_shell else f'{name} - Refund'} (Circular Loop)"})
        fraud_txn_ids.append(tid_loop)

    spoofed_ip = _random_ip(rng, local=False)
    device = f"IP: {spoofed_ip} ({jurisdiction})"

    pep = rng.choice(["Potential Match", "Adverse Media Hit", "Sanctions List - Partial Match", ""])

    profile = CustomerProfile(
        customer_id=cid, status="pending_review",
        personal_info={"name": name, "opened": "2025-03-01",
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

    kw_base = ["layering", "chain", "circular", "shell", "offshore",
                jurisdiction.lower(), name.lower().split()[0]]
    if is_shell:
        kw_base.extend(["beneficial owner", "holding", "missing", "outflow",
                         "corporate plaza", "wilmington"])
    else:
        kw_base.extend(["forward", "rapid", "transfer", "associate"])

    return profile, db_record, gt, kw_base


# ---------------------------------------------------------------------------
# High-level episode generators (one per task difficulty)
# ---------------------------------------------------------------------------

def _generate_task1_easy(rng: random.Random) -> EpisodeManifest:
    """2-3 customers: 1 clean + 1-2 with address mismatch."""
    customers = []
    database = {}
    ground_truth = {}
    evidence_keywords = {}

    num_customers = rng.randint(2, 3)
    num_flagged = rng.randint(1, min(2, num_customers - 1))

    for i in range(num_customers):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        if i < num_flagged:
            prof, db, gt, kw = _generate_address_mismatch_customer(rng, cid)
        else:
            prof, db, gt, kw = _generate_clean_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        evidence_keywords[cid] = kw

    rng.shuffle(customers)

    return EpisodeManifest(
        task_id="task1_easy", seed=0,
        customers=customers, database=database,
        ground_truth=ground_truth, evidence_keywords=evidence_keywords,
    )


def _generate_task2_medium(rng: random.Random) -> EpisodeManifest:
    """3-5 customers: 1-2 clean + 2-3 structuring."""
    customers = []
    database = {}
    ground_truth = {}
    evidence_keywords = {}

    num_customers = rng.randint(3, 5)
    num_clean = rng.randint(1, 2)
    num_fraud = num_customers - num_clean

    for i in range(num_fraud):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        prof, db, gt, kw = _generate_structuring_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        evidence_keywords[cid] = kw

    for i in range(num_clean):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        prof, db, gt, kw = _generate_clean_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        evidence_keywords[cid] = kw

    rng.shuffle(customers)

    return EpisodeManifest(
        task_id="task2_medium", seed=0,
        customers=customers, database=database,
        ground_truth=ground_truth, evidence_keywords=evidence_keywords,
    )


def _generate_task3_hard(rng: random.Random) -> EpisodeManifest:
    """5-8 customers: shell company network + layering chain + 1-2 clean."""
    customers = []
    database = {}
    ground_truth = {}
    evidence_keywords = {}
    network_truth: Dict[str, List[str]] = {}

    # --- Shell company cluster (1 shell + 2-3 associates) ---
    shell_cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
    associate_cids = [f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
                      for _ in range(rng.randint(2, 3))]

    # Shell company
    prof, db, gt, kw = _generate_layering_customer(
        rng, shell_cid, is_shell=True, linked_cids=associate_cids)
    customers.append(prof)
    database[shell_cid] = db
    ground_truth[shell_cid] = gt
    evidence_keywords[shell_cid] = kw
    network_truth[shell_cid] = associate_cids

    # Associates
    for acid in associate_cids:
        prof, db, gt, kw = _generate_layering_customer(
            rng, acid, is_shell=False, linked_cids=[shell_cid])
        customers.append(prof)
        database[acid] = db
        ground_truth[acid] = gt
        evidence_keywords[acid] = kw
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

        prof, db, gt, kw = _generate_layering_customer(
            rng, ccid, is_shell=False, linked_cids=linked)
        customers.append(prof)
        database[ccid] = db
        ground_truth[ccid] = gt
        evidence_keywords[ccid] = kw
        network_truth[ccid] = linked

    # --- Clean customers ---
    num_clean = rng.randint(1, 2)
    for _ in range(num_clean):
        cid = f"CUST-{uuid.UUID(int=rng.getrandbits(128)).hex[:6].upper()}"
        prof, db, gt, kw = _generate_clean_customer(rng, cid)
        customers.append(prof)
        database[cid] = db
        ground_truth[cid] = gt
        evidence_keywords[cid] = kw

    rng.shuffle(customers)

    return EpisodeManifest(
        task_id="task3_hard", seed=0,
        customers=customers, database=database,
        ground_truth=ground_truth, network_truth=network_truth,
        evidence_keywords=evidence_keywords,
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
