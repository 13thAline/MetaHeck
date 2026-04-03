"""
Deterministic Grader - Evaluates exact matches of flagged evidence IDs.
Scores 0.0 - 1.0 based on precision/recall (F1 Score) of flagged evidence and final decision.
"""
from typing import Dict, Any, List

GROUND_TRUTH = {
    "CUST-001": {
        "expected_decision": "escalate", # Mismatched utility bill warrants escalation
        "expected_txns": [],
        "expected_docs": ["utility_bill"] # Agent must explicitly flag this string/ID
    },
    "CUST-002": {
        "expected_decision": "freeze_account", # VPN + PEP hit + Structured deposits
        "expected_txns": ["TXN-M1", "TXN-M2"], # Agent MUST extract these exact IDs
        "expected_docs": ["passport"]
    }
}

def calculate_f1(predicted: List[str], actual: List[str]) -> float:
    """Calculates F1 score between two lists of strings."""
    if not predicted and not actual:
        return 1.0 # Correctly flagged nothing
    if not predicted or not actual:
        return 0.0 # Failed to flag, or hallucinated flags
        
    pred_set = set(p.lower().strip() for p in predicted)
    actual_set = set(a.lower().strip() for a in actual)
    
    true_positives = len(pred_set & actual_set)
    false_positives = len(pred_set - actual_set)
    false_negatives = len(actual_set - pred_set)
    
    if true_positives == 0:
        return 0.0
        
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    return 2 * (precision * recall) / (precision + recall)

def grade(actions: List[Dict[str, Any]], task_id: str = "task1_easy") -> Dict[str, Any]:
    if not actions:
        return {"score": 0.0, "feedback": "No actions taken."}

    total_score = 0.0
    feedback_parts = []
    
    # Filter out discovery actions. We only grade terminal decisions.
    terminal_actions = [a for a in actions if a.get("action_type") in ["approve", "reject", "escalate", "freeze_account", "file_sar"]]
    
    if not terminal_actions:
        return {"score": 0.0, "feedback": "No terminal decisions made. Auto-fail."}

    for action in terminal_actions:
        cid = action.get("target_customer_id")
        if cid not in GROUND_TRUTH:
            continue
            
        gt = GROUND_TRUTH[cid]
        decision = action.get("action_type")
        flagged_txns = action.get("flagged_transaction_ids", [])
        flagged_docs = action.get("flagged_document_ids", [])
        
        cust_score = 0.0
        
        # 1. Base Score for Correct Decision (40%)
        if decision == gt["expected_decision"]:
            cust_score += 0.40
            feedback_parts.append(f"{cid}: Correct decision.")
        else:
            feedback_parts.append(f"{cid}: Wrong decision (Got {decision}, Expected {gt['expected_decision']}).")
            
        # 2. Evidence Score (F1 of Flagged IDs) (60%)
        txn_f1 = calculate_f1(flagged_txns, gt["expected_txns"])
        doc_f1 = calculate_f1(flagged_docs, gt["expected_docs"])
        
        # Average the F1 scores
        evidence_score = ((txn_f1 + doc_f1) / 2) * 0.60
        cust_score += evidence_score
        
        feedback_parts.append(f"[{cid} Evidence F1 -> Txn:{txn_f1:.2f}, Doc:{doc_f1:.2f}]")
        total_score += cust_score

    # Normalize by number of customers in the Ground Truth
    final_score = total_score / len(GROUND_TRUTH)
    
    return {
        "score": round(final_score, 4),
        "feedback": " | ".join(feedback_parts)
    }