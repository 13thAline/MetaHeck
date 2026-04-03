from env.graders.grader1 import grade

# Simulation 1: Agent freezes the account, but fails to extract the transaction IDs
sloppy_actions = [{
    "action_type": "freeze_account",
    "target_customer_id": "CUST-002",
    "decision_reasoning": "This guy looks fraudulent.",
    "flagged_transaction_ids": [], # FAIL
    "flagged_document_ids": ["passport"]
}]

# Simulation 2: Agent gets the decision right AND extracts the exact evidence
perfect_actions = [{
    "action_type": "freeze_account",
    "target_customer_id": "CUST-002",
    "decision_reasoning": "VPN detected, expired passport, structured deposits.",
    "flagged_transaction_ids": ["TXN-M1", "TXN-M2"], # CORRECT
    "flagged_document_ids": ["passport"]
}]

print("--- Sloppy Agent Score ---")
res1 = grade(sloppy_actions)
print(f"Score: {res1['score']}\nFeedback: {res1['feedback']}\n")

print("--- Perfect Agent Score ---")
res2 = grade(perfect_actions)
print(f"Score: {res2['score']}\nFeedback: {res2['feedback']}")