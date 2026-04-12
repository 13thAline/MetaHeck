"""
Task 3 Grader — Network fraud investigation and SAR filing.
Scores entity linking accuracy, typology classification, and network discovery.

All keyword-based reasoning scoring has been removed.  Agents are now graded
ONLY on:
  - Correct terminal decision   (30%)
  - Risk tier accuracy           (10%)
  - Typology F1 (discrete codes) (25%)
  - Network / entity linking F1  (20%)
  - SAR filing when required      (5%)
  - Confidence calibration       (10%)

Network truth and expected typologies are passed in dynamically from the
procedural data engine.
"""
from typing import Dict, Any, List, Optional, Set


# Partial credit for close-enough decisions
PARTIAL_CREDIT_T3 = {
    ("file_sar", "freeze_account"): 0.8,
    ("file_sar", "flag_for_review"): 0.4,
    ("freeze_account", "file_sar"): 0.9,
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
          network_truth: Optional[Dict[str, List[str]]] = None,
          expected_typologies: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Grade agent actions against dynamically generated ground truth.

    Args:
        actions: list of action dicts from the agent's session.
        ground_truth: dict keyed by customer_id with decision, risk_tier,
                      expected_txns, expected_docs, and red_flags.
        network_truth: adjacency dict keyed by customer_id mapping to
                       list of linked customer_ids.
        expected_typologies: dict keyed by customer_id mapping to lists of
                             RegulatoryTypology string values.

    Returns:
        dict with 'score', 'breakdown', and 'feedback'.
    """
    if not actions:
        return {"score": 0.0, "breakdown": {}, "feedback": "No actions taken."}

    network_truth = network_truth or {}
    expected_typologies = expected_typologies or {}

    feedback_parts = []
    breakdown = {}
    customers_acted_on: Set[str] = set()
    agent_links: Dict[str, Set[str]] = {k: set() for k in ground_truth}
    sar_filed: Set[str] = set()
    total_score = 0.0

    # Collect link_entities actions
    for action in actions:
        cid = action.get("target_customer_id") or action.get("customer_id")
        if action.get("action_type") == "link_entities":
            target = action.get("target_customer_id") or action.get("linked_customer_id")
            source = action.get("customer_id") or action.get("source_customer_id")
            if source and target:
                agent_links.setdefault(source, set()).add(target)
                agent_links.setdefault(target, set()).add(source)

        if action.get("action_type") == "file_sar":
            sar_filed.add(cid)

    for action in actions:
        cid = action.get("target_customer_id") or action.get("customer_id")
        if cid not in ground_truth or cid in customers_acted_on:
            continue
        # Skip discovery and link actions — only grade terminal decisions
        if action.get("action_type") in ["link_entities", "pull_document_dossier",
                                          "query_transactions", "check_watchlists",
                                          "pull_device_signals", "interview_customer",
                                          "intercept_transaction"]:
            continue
        customers_acted_on.add(cid)

        gt = ground_truth[cid]
        action_type = action.get("action_type")
        risk_tier = action.get("risk_tier")
        confidence = float(action.get("confidence_score", 0.5))
        submitted_typos = action.get("regulatory_typology", [])

        cust_score = 0.0

        # 1. Decision (30%)
        expected = gt.get("expected_decision", gt.get("decision", ""))
        decision_correct = action_type == expected
        if decision_correct:
            cust_score += 0.30
            feedback_parts.append(f"{cid}: Correct decision ({action_type}).")
        else:
            partial = PARTIAL_CREDIT_T3.get((expected, action_type), 0.0)
            if partial > 0:
                cust_score += 0.30 * partial
                feedback_parts.append(
                    f"{cid}: Partial ({action_type} vs {expected}, credit={partial:.0%})."
                )
            else:
                feedback_parts.append(
                    f"{cid}: Expected '{expected}', got '{action_type}'."
                )

        # 2. Risk tier (10%)
        tier_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        gt_tier = gt.get("risk_tier", "low")
        if risk_tier == gt_tier:
            cust_score += 0.10
        elif risk_tier:
            diff = abs(tier_map.get(risk_tier, 0) - tier_map.get(gt_tier, 0))
            cust_score += max(0.0, 0.10 - diff * 0.03)

        # 3. Typology F1 (25%) — replaces keyword matching
        exp_typos = expected_typologies.get(cid, [])
        typo_f1 = _typology_f1(submitted_typos, exp_typos)
        cust_score += typo_f1 * 0.25
        feedback_parts.append(f"[{cid} Typology F1: {typo_f1:.2f}]")

        # 4. Network / entity linking (20%)
        gt_links = set(network_truth.get(cid, []))
        if gt_links:
            found_links = agent_links.get(cid, set())
            if found_links:
                precision = len(found_links & gt_links) / max(len(found_links), 1)
                recall = len(found_links & gt_links) / len(gt_links)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                cust_score += f1 * 0.20
                if found_links & gt_links:
                    feedback_parts.append(
                        f"{cid}: Found {len(found_links & gt_links)}/{len(gt_links)} network links."
                    )

        # 5. SAR filed when required (5%)
        if expected == "file_sar" and cid in sar_filed:
            cust_score += 0.05

        # 6. Confidence calibration (10%)
        is_ambiguous = gt.get("is_ambiguous", False)
        conf_mod = _confidence_modifier(confidence, decision_correct, is_ambiguous)
        cust_score += conf_mod * 0.10
        feedback_parts.append(f"[{cid} Confidence: {confidence:.2f}, mod: {conf_mod:+.2f}]")

        breakdown[cid] = round(cust_score, 3)
        total_score += cust_score

    num_customers = len(ground_truth)
    final_score = round(total_score / num_customers, 4) if num_customers else 0.0

    # False positive penalty — clean/ambiguous customers wrongly flagged
    for action in actions:
        cid = action.get("target_customer_id") or action.get("customer_id")
        gt = ground_truth.get(cid, {})
        expected = gt.get("expected_decision", gt.get("decision", ""))
        confidence = float(action.get("confidence_score", 0.5))
        is_ambiguous = gt.get("is_ambiguous", False)
        if expected in ["clear_customer", "approve"] and \
           action.get("action_type") in ["file_sar", "freeze_account", "flag_for_review"]:
            # Scale false positive penalty by confidence
            fp_penalty = 0.05 * confidence if not is_ambiguous else confidence
            final_score = max(0.0, final_score - fp_penalty)
            feedback_parts.append(f"False positive penalty for {cid} (conf={confidence:.2f}).")

    unreviewed = set(ground_truth.keys()) - customers_acted_on
    if unreviewed:
        penalty = 0.06 * len(unreviewed)
        final_score = max(0.0, final_score - penalty)

    return {
        "score": min(max(final_score, 0.0), 1.0),
        "breakdown": breakdown,
        "feedback": " | ".join(feedback_parts) if feedback_parts else "Graded.",
    }