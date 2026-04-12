"""
Task 2 Grader — Transaction pattern analysis.
Scores agents on detecting structuring, round-trips, burst velocity, and
ambiguous grey cases.

All keyword-based reasoning scoring has been removed.  Agents are now graded
ONLY on:
  - Correct terminal decision   (35%)
  - Risk tier accuracy           (15%)
  - Typology F1 (discrete codes) (30%)
  - Evidence F1 (txn + doc IDs)  (10%)
  - Confidence calibration       (10%)

Ground truth and expected typologies are passed in dynamically from the
procedural data engine.
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


def _typology_f1(submitted: List[str], expected: List[str]) -> float:
    """F1 between agent-submitted typology codes and ground-truth typologies."""
    if not submitted and not expected:
        return 1.0
    if not submitted or not expected:
        return 0.0

    sub = set(t.upper().strip() for t in submitted)
    exp = set(t.upper().strip() for t in expected)

    tp = len(sub & exp)
    if tp == 0:
        return 0.0

    precision = tp / len(sub)
    recall = tp / len(exp)
    return 2 * (precision * recall) / (precision + recall)


def _evidence_f1(predicted: List[str], actual: List[str]) -> float:
    """F1 between predicted and actual flagged IDs."""
    if not predicted and not actual:
        return 1.0
    if not predicted or not actual:
        return 0.0

    pred_set = set(p.lower().strip() for p in predicted)
    actual_set = set(a.lower().strip() for a in actual)

    tp = len(pred_set & actual_set)
    if tp == 0:
        return 0.0

    precision = tp / (tp + len(pred_set - actual_set))
    recall = tp / (tp + len(actual_set - pred_set))
    return 2 * (precision * recall) / (precision + recall)


def _confidence_modifier(confidence: float, decision_correct: bool,
                          is_ambiguous: bool) -> float:
    """Return a score modifier based on the agent's stated confidence."""
    if decision_correct:
        return 0.10 * confidence

    if is_ambiguous and confidence >= 0.9:
        return -1.0
    if confidence <= 0.3:
        return -0.10 * confidence
    return -0.5 * confidence


def grade(actions: List[Dict[str, Any]], ground_truth: Dict[str, Any],
          expected_typologies: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Grade agent actions against dynamically generated ground truth.

    Args:
        actions: list of action dicts from the agent's session.
        ground_truth: dict keyed by customer_id with decision, risk_tier,
                      expected_txns, expected_docs, and red_flags.
        expected_typologies: dict keyed by customer_id mapping to lists of
                             RegulatoryTypology string values.

    Returns:
        dict with 'score', 'breakdown', and 'feedback'.
    """
    if not actions:
        return {"score": 0.0, "breakdown": {}, "feedback": "No actions taken."}

    expected_typologies = expected_typologies or {}
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
                                          "interview_customer", "link_entities",
                                          "intercept_transaction"]:
            continue
        customers_acted_on.add(cid)

        gt = ground_truth[cid]
        action_type = action.get("action_type")
        risk_tier = action.get("risk_tier")
        confidence = float(action.get("confidence_score", 0.5))
        submitted_typos = action.get("regulatory_typology", [])

        cust_score = 0.0

        # 1. Decision accuracy (35%)
        expected = gt.get("expected_decision", gt.get("decision", ""))
        decision_correct = action_type == expected
        if decision_correct:
            cust_score += 0.35
            feedback_parts.append(f"{cid}: Correct decision.")
        else:
            partial = PARTIAL_CREDIT.get((expected, action_type), 0.0)
            cust_score += 0.35 * partial
            if partial > 0:
                feedback_parts.append(
                    f"{cid}: Partial credit — expected '{expected}', got '{action_type}'."
                )
            else:
                feedback_parts.append(
                    f"{cid}: Wrong — expected '{expected}', got '{action_type}'."
                )

        # 2. Risk tier (15%)
        tier_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        gt_tier = gt.get("risk_tier", "low")
        if risk_tier == gt_tier:
            cust_score += 0.15
        elif risk_tier:
            diff = abs(tier_map.get(risk_tier, 0) - tier_map.get(gt_tier, 0))
            tier_score = max(0.0, 0.15 - diff * 0.05)
            cust_score += tier_score
            feedback_parts.append(f"{cid}: Risk tier off by {diff} level(s).")

        # 3. Typology F1 (30%) — replaces keyword matching
        exp_typos = expected_typologies.get(cid, [])
        typo_f1 = _typology_f1(submitted_typos, exp_typos)
        cust_score += typo_f1 * 0.30
        feedback_parts.append(f"[{cid} Typology F1: {typo_f1:.2f}]")

        # 4. Evidence F1 (10%)
        flagged_txns = action.get("flagged_transaction_ids", [])
        flagged_docs = action.get("flagged_document_ids", [])
        txn_f1 = _evidence_f1(flagged_txns, gt.get("expected_txns", []))
        doc_f1 = _evidence_f1(flagged_docs, gt.get("expected_docs", []))
        cust_score += ((txn_f1 + doc_f1) / 2) * 0.10

        # 5. Confidence calibration (10%)
        is_ambiguous = gt.get("is_ambiguous", False)
        conf_mod = _confidence_modifier(confidence, decision_correct, is_ambiguous)
        cust_score += conf_mod * 0.10
        feedback_parts.append(f"[{cid} Confidence: {confidence:.2f}, mod: {conf_mod:+.2f}]")

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
        "score": min(max(final_score, 0.0), 1.0),
        "breakdown": breakdown,
        "feedback": " | ".join(feedback_parts),
    }