"""
Task 2 Grader — Transaction pattern analysis.
Scores agents on detecting structuring, round-trips, and high velocity.

Ground truth and evidence keywords are now passed in dynamically from
the procedural data engine.
"""
from typing import Dict, Any, List, Optional


# Partial credit mapping: if agent decision is "close enough"
PARTIAL_CREDIT = {
    ("file_sar", "flag_for_review"): 0.5,
    ("flag_for_review", "file_sar"): 0.7,
    ("flag_for_review", "freeze_account"): 0.6,
    ("freeze_account", "file_sar"): 0.8,
    ("file_sar", "freeze_account"): 0.8,
    ("freeze_account", "escalate"): 0.5,
    ("clear_customer", "flag_for_review"): 0.0,
    ("clear_customer", "approve"): 1.0,
    ("approve", "clear_customer"): 1.0,
}


def grade(actions: List[Dict[str, Any]], ground_truth: Dict[str, Any],
          evidence_keywords: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Grade agent actions against dynamically generated ground truth.

    Args:
        actions: list of action dicts from the agent's session.
        ground_truth: dict keyed by customer_id with decision, risk_tier,
                      expected_txns, expected_docs, and red_flags.
        evidence_keywords: optional dict keyed by customer_id mapping to
                           keyword lists for reasoning quality scoring.

    Returns:
        dict with 'score', 'breakdown', and 'feedback'.
    """
    if not actions:
        return {"score": 0.0, "breakdown": {}, "feedback": "No actions taken."}

    evidence_keywords = evidence_keywords or {}
    total_score = 0.0
    breakdown = {}
    feedback_parts = []
    customers_acted_on = set()

    for action in actions:
        cid = action.get("target_customer_id") or action.get("customer_id")
        if cid not in ground_truth or cid in customers_acted_on:
            continue
        # Skip discovery actions — only grade terminal decisions
        if action.get("action_type") in ["pull_document_dossier", "query_transactions",
                                          "check_watchlists", "pull_device_signals",
                                          "interview_customer"]:
            continue
        customers_acted_on.add(cid)

        gt = ground_truth[cid]
        action_type = action.get("action_type")
        risk_tier = action.get("risk_tier")
        reason = (action.get("decision_reasoning") or action.get("reason") or "").lower()

        cust_score = 0.0

        # 1. Decision accuracy (40%)
        expected = gt.get("expected_decision", gt.get("decision", ""))
        if action_type == expected:
            cust_score += 0.40
            feedback_parts.append(f"{cid}: Correct decision.")
        else:
            partial = PARTIAL_CREDIT.get((expected, action_type), 0.0)
            cust_score += 0.40 * partial
            if partial > 0:
                feedback_parts.append(
                    f"{cid}: Partial credit — expected '{expected}', got '{action_type}'."
                )
            else:
                feedback_parts.append(
                    f"{cid}: Wrong — expected '{expected}', got '{action_type}'."
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
        keywords = evidence_keywords.get(cid, [])
        if keywords:
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