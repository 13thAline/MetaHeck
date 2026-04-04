"""
Deterministic Grader (Task 1) — Evaluates exact matches of flagged evidence IDs.
Scores 0.0 - 1.0 based on precision/recall (F1 Score) of flagged evidence and final decision.

Ground truth is now passed in dynamically from the procedural data engine,
not hardcoded at module level.
"""
from typing import Dict, Any, List


def calculate_f1(predicted: List[str], actual: List[str]) -> float:
    """Calculates F1 score between two lists of strings."""
    if not predicted and not actual:
        return 1.0  # Correctly flagged nothing
    if not predicted or not actual:
        return 0.0  # Failed to flag, or hallucinated flags
        
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


def grade(actions: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Grade agent actions against dynamically generated ground truth.

    Args:
        actions: list of action dicts from the agent's session.
        ground_truth: dict keyed by customer_id with expected_decision,
                      expected_txns, and expected_docs.

    Returns:
        dict with 'score' (float 0-1) and 'feedback' (str).
    """
    if not actions:
        return {"score": 0.0, "feedback": "No actions taken."}

    total_score = 0.0
    feedback_parts = []
    
    # Filter out discovery actions. We only grade terminal decisions.
    terminal_actions = [a for a in actions if a.get("action_type") in
                        ["approve", "reject", "escalate", "freeze_account", "file_sar"]]
    
    if not terminal_actions:
        return {"score": 0.0, "feedback": "No terminal decisions made. Auto-fail."}

    for action in terminal_actions:
        cid = action.get("target_customer_id")
        if cid not in ground_truth:
            continue
            
        gt = ground_truth[cid]
        decision = action.get("action_type")
        flagged_txns = action.get("flagged_transaction_ids", [])
        flagged_docs = action.get("flagged_document_ids", [])
        
        cust_score = 0.0
        
        # 1. Base Score for Correct Decision (40%)
        expected = gt.get("expected_decision", gt.get("decision", ""))
        if decision == expected:
            cust_score += 0.40
            feedback_parts.append(f"{cid}: Correct decision.")
        else:
            feedback_parts.append(f"{cid}: Wrong decision (Got {decision}, Expected {expected}).")
            
        # 2. Evidence Score (F1 of Flagged IDs) (60%)
        txn_f1 = calculate_f1(flagged_txns, gt.get("expected_txns", []))
        doc_f1 = calculate_f1(flagged_docs, gt.get("expected_docs", []))
        
        # Average the F1 scores
        evidence_score = ((txn_f1 + doc_f1) / 2) * 0.60
        cust_score += evidence_score
        
        feedback_parts.append(f"[{cid} Evidence F1 -> Txn:{txn_f1:.2f}, Doc:{doc_f1:.2f}]")
        total_score += cust_score

    # Normalize by number of customers in the Ground Truth
    final_score = total_score / len(ground_truth) if ground_truth else 0.0
    
    return {
        "score": round(final_score, 4),
        "feedback": " | ".join(feedback_parts)
    }