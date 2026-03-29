"""
Task 1 Grader — Document completeness and identity verification.
Scores 0.0 – 1.0 based on correct decisions, risk tier, and evidence quality.
"""
from typing import Dict, Any
from env.models import Action, ActionType, RiskTier


DECISION_SCORES = {
    "clear_customer": 0.45,
    "request_documents": 0.45,
    "flag_for_review": 0.45,
    "file_sar": 0.45,
    "freeze_account": 0.45,
}

# Keywords that indicate good reasoning for each customer
EVIDENCE_KEYWORDS = {
    "CUST-T1-A": ["valid", "complete", "documents", "verified", "clear"],
    "CUST-T1-B": ["expired", "passport", "missing", "utility", "bank statement"],
    "CUST-T1-C": ["student", "income", "mismatch", "180", "source of funds", "occupation"],
}


def grade(actions: list[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grade a set of actions against ground truth for Task 1.
    Returns score (0.0-1.0) and detailed breakdown.
    """
    if not actions:
        return {"score": 0.0, "breakdown": {}, "feedback": "No actions taken."}

    total_score = 0.0
    breakdown = {}
    feedback_parts = []
    customers_acted_on = set()

    for action in actions:
        cid = action.get("customer_id")
        if cid not in ground_truth or cid in customers_acted_on:
            continue
        customers_acted_on.add(cid)

        gt = ground_truth[cid]
        action_type = action.get("action_type")
        risk_tier = action.get("risk_tier")
        reason = (action.get("reason") or "").lower()
        docs_requested = action.get("documents_requested") or []

        cust_score = 0.0

        # 1. Correct decision (45%)
        if action_type == gt["decision"]:
            cust_score += 0.45
            feedback_parts.append(f"{cid}: Correct decision ({action_type}).")
        else:
            feedback_parts.append(
                f"{cid}: Wrong decision — expected '{gt['decision']}', got '{action_type}'."
            )

        # 2. Risk tier (20%)
        if risk_tier and risk_tier == gt["risk_tier"]:
            cust_score += 0.20
        elif risk_tier:
            feedback_parts.append(f"{cid}: Risk tier mismatch — expected '{gt['risk_tier']}'.")

        # 3. Evidence quality in reason (25%) — keyword matching
        keywords = EVIDENCE_KEYWORDS.get(cid, [])
        matched = sum(1 for kw in keywords if kw in reason)
        evidence_score = min(matched / max(len(keywords), 1), 1.0) * 0.25
        cust_score += evidence_score

        # 4. Document handling (10%) — for request_documents action
        if gt["decision"] == "request_documents" and action_type == "request_documents":
            expected_docs = set(gt.get("missing_docs", []))
            requested = set(docs_requested)
            if expected_docs:
                overlap = len(expected_docs & requested) / len(expected_docs)
                cust_score += overlap * 0.10
            else:
                cust_score += 0.10

        breakdown[cid] = round(cust_score, 3)
        total_score += cust_score

    # Normalize by number of customers in ground truth
    num_customers = len(ground_truth)
    final_score = round(total_score / num_customers, 4) if num_customers else 0.0

    # Penalize if not all customers were reviewed
    unreviewed = set(ground_truth.keys()) - customers_acted_on
    if unreviewed:
        penalty = 0.1 * len(unreviewed)
        final_score = max(0.0, final_score - penalty)
        feedback_parts.append(f"Penalty: {len(unreviewed)} customer(s) not reviewed: {unreviewed}")

    return {
        "score": min(final_score, 1.0),
        "breakdown": breakdown,
        "feedback": " | ".join(feedback_parts) if feedback_parts else "No valid actions graded.",
    }