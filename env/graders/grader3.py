"""
Task 3 Grader — Network fraud investigation and SAR filing.
Scores entity linking accuracy, SAR quality, and network discovery.

Network truth and evidence keywords are now passed in dynamically from
the procedural data engine, not hardcoded at module level.
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


def grade(actions: List[Dict[str, Any]], ground_truth: Dict[str, Any],
          network_truth: Optional[Dict[str, List[str]]] = None,
          evidence_keywords: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Grade agent actions against dynamically generated ground truth.

    Args:
        actions: list of action dicts from the agent's session.
        ground_truth: dict keyed by customer_id with decision, risk_tier,
                      expected_txns, expected_docs, and red_flags.
        network_truth: adjacency dict keyed by customer_id mapping to
                       list of linked customer_ids.
        evidence_keywords: optional dict keyed by customer_id mapping to
                           keyword lists for reasoning quality scoring.

    Returns:
        dict with 'score', 'breakdown', and 'feedback'.
    """
    if not actions:
        return {"score": 0.0, "breakdown": {}, "feedback": "No actions taken."}

    network_truth = network_truth or {}
    evidence_keywords = evidence_keywords or {}

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
                                          "pull_device_signals", "interview_customer"]:
            continue
        customers_acted_on.add(cid)

        gt = ground_truth[cid]
        action_type = action.get("action_type")
        risk_tier = action.get("risk_tier")
        reason = (action.get("decision_reasoning") or action.get("reason") or "").lower()

        cust_score = 0.0

        # 1. Decision (35%)
        expected = gt.get("expected_decision", gt.get("decision", ""))
        if action_type == expected:
            cust_score += 0.35
            feedback_parts.append(f"{cid}: Correct decision ({action_type}).")
        else:
            partial = PARTIAL_CREDIT_T3.get((expected, action_type), 0.0)
            if partial > 0:
                cust_score += 0.35 * partial
                feedback_parts.append(
                    f"{cid}: Partial ({action_type} vs {expected}, credit={partial:.0%})."
                )
            else:
                feedback_parts.append(
                    f"{cid}: Expected '{expected}', got '{action_type}'."
                )

        # 2. Risk tier (15%)
        gt_tier = gt.get("risk_tier", "low")
        if risk_tier == gt_tier:
            cust_score += 0.15

        # 3. Evidence reasoning (25%)
        keywords = evidence_keywords.get(cid, [])
        if keywords:
            matched = sum(1 for kw in keywords if kw in reason)
            cust_score += min(matched / len(keywords), 1.0) * 0.25

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

        breakdown[cid] = round(cust_score, 3)
        total_score += cust_score

    num_customers = len(ground_truth)
    final_score = round(total_score / num_customers, 4) if num_customers else 0.0

    # False positive penalty — clean customers wrongly flagged
    for action in actions:
        cid = action.get("target_customer_id") or action.get("customer_id")
        gt = ground_truth.get(cid, {})
        expected = gt.get("expected_decision", gt.get("decision", ""))
        if expected in ["clear_customer", "approve"] and \
           action.get("action_type") in ["file_sar", "freeze_account", "flag_for_review"]:
            final_score = max(0.0, final_score - 0.05)
            feedback_parts.append(f"False positive penalty for {cid}.")

    unreviewed = set(ground_truth.keys()) - customers_acted_on
    if unreviewed:
        penalty = 0.06 * len(unreviewed)
        final_score = max(0.0, final_score - penalty)

    return {
        "score": min(final_score, 1.0),
        "breakdown": breakdown,
        "feedback": " | ".join(feedback_parts) if feedback_parts else "Graded.",
    }