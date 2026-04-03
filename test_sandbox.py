import requests
import json

BASE_URL = "http://localhost:8080"

print("--- 1. STARTING NEW EPISODE ---")
reset_res = requests.get(f"{BASE_URL}/reset?task_id=task1_easy").json()
episode_id = reset_res.get("episode_id")
obs = reset_res.get("observation", {})

print("Did the environment leak data? (investigation_context should be empty):")
print(f"Context: '{obs.get('investigation_context')}'")
print(f"Queue size: {len(obs.get('customer_queue', []))} customers waiting.\n")


print("--- 2. TESTING COMPLIANCE PENALTY ---")
print("Agent attempts to blindly FREEZE_ACCOUNT without checking docs...")
bad_action = {
    "action_type": "freeze_account",
    "target_customer_id": "CUST-001",
    "decision_reasoning": "Vibes felt off.",
    "flagged_transaction_ids": [],
    "flagged_document_ids": []
}
step1_res = requests.post(f"{BASE_URL}/step", json={"episode_id": episode_id, "action": bad_action}).json()

print(f"Message from Env: {step1_res['observation']['message']}")
print(f"Reward (Should be negative): {step1_res['reward']}\n")


print("--- 3. TESTING DISCOVERY ACTION ---")
print("Agent pulls the document dossier for CUST-002...")
good_action = {
    "action_type": "pull_document_dossier",
    "target_customer_id": "CUST-002",
    "decision_reasoning": "Pulling documents to check for expiration or mismatch." # Added this
}
step2_res = requests.post(f"{BASE_URL}/step", json={"episode_id": episode_id, "action": good_action}).json()

print(f"Message from Env: {step2_res['observation']['message']}")
print(f"Injected Context: {step2_res['observation']['investigation_context']}")
print(f"Reward (Should be positive): {step2_res['reward']}")