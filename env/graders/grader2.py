"""
Task 2 Grader — Transaction pattern analysis.
Scores agents on detecting structuring, round-trips, and high velocity.
"""
from typing import Dict, Any, List


EVIDENCE_KEYWORDS = {
    "CUST-T2-1": ["clean", "normal", "legitimate", "regular", "consistent"],
    "CUST-T2-2": ["structuring", "smurfing", "9800", "9750", "9900", "threshold",
                  "10000", "$10,000", "reporting", "under"],
    "CUST-T2-3": ["round", "trip", "return", "refund", "same amount", "48", "wire",
                  "50000", "dubai", "ae"],
    "CUST-T2-4": ["new account", "high value", "beneficial owner", "missing",
                  "round trip", "counterparty"],
    "CUST-T2-5": ["velocity", "rapid", "dormant", "retired", "income", "52",
                  "frequent", "mismatch", "reactivat"],
}

# Partial credit mapping: if agent decision is "close enough"
PARTIAL_CREDIT = {
    ("file_sar", "flag_for_review"): 0.5,
    ("flag_for_review", "file_sar"): 0.7,
    ("flag_for_review", "freeze_account"): 0.6,
    ("clear_customer", "flag_for_review"): 0.0,
}


def grade(actions: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
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

        cust_score = 0.0

        # 1. Decision accuracy (40%)
        if action_type == gt["decision"]:
            cust_score += 0.40
            feedback_parts.append(f"{cid}: Correct decision.")
        else:
            partial = PARTIAL_CREDIT.get((gt["decision"], action_type), 0.0)
            cust_score += 0.40 * partial
            if partial > 0:
                feedback_parts.append(
                    f"{cid}: Partial credit — expected '{gt['decision']}', got '{action_type}'."
                )
            else:
                feedback_parts.append(
                    f"{cid}: Wrong — expected '{gt['decision']}', got '{action_type}'."
                )

        # 2. Risk tier (20%)
        tier_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        gt_tier = gt.get("risk_tier", "low")
        if risk_tier == gt_tier:
            cust_score += 0.20
        elif risk_tier:
            diff = abs(tier_map.get(risk_tier, 0) - tier_map.get(gt_tier, 0))
            tier_score = max(0.0, 0.20 - diff * 0.07)
            cust_score += tier_score
            feedback_parts.append(f"{cid}: Risk tier off by {diff} level(s).")

        # 3. Evidence / reasoning quality (30%)
        keywords = EVIDENCE_KEYWORDS.get(cid, [])
        matched = sum(1 for kw in keywords if kw in reason)
        evidence_score = min(matched / max(len(keywords), 1), 1.0) * 0.30
        cust_score += evidence_score

        # 4. Red flag identification bonus (10%)
        gt_flags = set(gt.get("red_flags", []))
        flag_hits = sum(1 for flag in gt_flags if flag.replace("_", " ") in reason
                        or flag in reason)
        if gt_flags:
            cust_score += (flag_hits / len(gt_flags)) * 0.10

        breakdown[cid] = round(cust_score, 3)
        total_score += cust_score

    num_customers = len(ground_truth)
    final_score = round(total_score / num_customers, 4) if num_customers else 0.0

    unreviewed = set(ground_truth.keys()) - customers_acted_on
    if unreviewed:
        penalty = 0.08 * len(unreviewed)
        final_score = max(0.0, final_score - penalty)
        feedback_parts.append(f"Unreviewed: {unreviewed}")

    return {
        "score": min(final_score, 1.0),
        "breakdown": breakdown,
        "feedback": " | ".join(feedback_parts),
    }