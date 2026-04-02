from typing import Dict, Any
from env.models import CustomerProfile

def generate_easy_task() -> CustomerProfile:
    return CustomerProfile(
        customer_id="cust_easy_001",
        personal_info={
            "name": "Sarah Jenkins",
            "dob": "1992-05-14",
            "address": "452 Maple Ave, Apt 4B, Seattle, WA",
            "email": "sarah.j@example.com",
            "occupation": "Graphic Designer",
            "pep_status": "None"
        },
        documents=[
            {"type": "passport", "status": "uploaded", "address": "452 Maple Ave, Seattle, WA", "notes": "Minor apt # mismatch"},
            {"type": "utility_bill", "status": "uploaded", "address": "452 Maple Ave, Apartment 4-B, Seattle, WA"}
        ],
        transaction_history=[
            {"date": "2025-01-10", "type": "deposit", "amount": 3200, "description": "Salary - TechCorp"},
            {"date": "2025-01-15", "type": "withdrawal", "amount": 800, "description": "Rent Payment"},
            {"date": "2025-01-20", "type": "withdrawal", "amount": 120, "description": "Grocery Store"}
        ],
        device_signals={
            "ip_location": "Seattle, WA",
            "device_hash": "a1b2c3d4-genuine",
            "vpn_detected": False,
            "emulator_detected": False
        },
        behavioral_signals={
            "login_velocity": "normal",
            "session_duration_avg": "12m",
            "typing_cadence": "human"
        }
    )

def generate_medium_task() -> CustomerProfile:
    return CustomerProfile(
        customer_id="cust_medium_002",
        personal_info={
            "name": "Marcus Vance",
            "dob": "1988-11-22",
            "address": "880 Industrial Pkwy, Austin, TX",
            "occupation": "Freelance Consultant",
            "pep_status": "None"
        },
        documents=[
            {"type": "drivers_license", "status": "verified", "address": "880 Industrial Pkwy, Austin, TX"}
        ],
        transaction_history=[
            {"date": "2025-02-01", "type": "deposit", "amount": 9500, "description": "Wire Transfer - Unknown Origin"},
            {"date": "2025-02-02", "type": "withdrawal", "amount": 9400, "description": "Crypto Exchange XYZ"},
            {"date": "2025-02-15", "type": "deposit", "amount": 8800, "description": "Wire Transfer - Unknown Origin"},
            {"date": "2025-02-16", "type": "withdrawal", "amount": 8700, "description": "Crypto Exchange XYZ"}
        ],
        device_signals={
            "ip_location": "Miami, FL", # Mismatch with home address
            "device_hash": "new_device_999",
            "vpn_detected": True,
            "emulator_detected": False
        },
        behavioral_signals={
            "login_velocity": "high_frequency_transfers",
            "session_duration_avg": "2m", # In and out quickly
            "typing_cadence": "human"
        }
    )

def generate_hard_task() -> CustomerProfile:
    return CustomerProfile(
        customer_id="cust_hard_003",
        personal_info={
            "name": "Alexei Volkov",
            "dob": "1975-03-01",
            "address": "100 Prestige Plaza, Suite 500, New York, NY",
            "occupation": "Director of Imports/Exports",
            "pep_status": "Potential Match"
        },
        documents=[
            {"type": "passport", "status": "uploaded", "details": "Suspicious formatting, possible Deepfake ID."},
            {"type": "bank_statement", "status": "uploaded", "details": "PDF metadata authored yesterday by unknown software."}
        ],
        transaction_history=[
            {"date": "2025-03-01", "type": "deposit", "amount": 49000, "description": "Garantia LLC - Consulting"},
            {"date": "2025-03-01", "type": "withdrawal", "amount": 10000, "description": "Transfer to Associate 1"},
            {"date": "2025-03-02", "type": "withdrawal", "amount": 10000, "description": "Transfer to Associate 2"},
            {"date": "2025-03-03", "type": "withdrawal", "amount": 29000, "description": "Offshore Account XYZ (High Risk Jurisdiction)"},
            {"date": "2025-03-04", "type": "deposit", "amount": 49000, "description": "Associate 3 - Repayment"},
            {"date": "2025-03-05", "type": "withdrawal", "amount": 49000, "description": "Garantia LLC - Refund (Circular Loop)"}
        ],
        device_signals={
            "ip_location": "Cyprus", 
            "device_hash": "spoofed_hash_0000000",
            "vpn_detected": True,
            "emulator_detected": True
        },
        behavioral_signals={
            "login_velocity": "bot_like",
            "session_duration_avg": "0.5m",
            "typing_cadence": "automated_script"
        },
        watchlist_report="Unchecked"
    )

def get_customer_for_task(task_id: str) -> CustomerProfile:
    if task_id == "task1_easy":
        return generate_easy_task()
    elif task_id == "task2_medium":
        return generate_medium_task()
    elif task_id == "task3_hard":
        return generate_hard_task()
    raise ValueError(f"Unknown task: {task_id}")