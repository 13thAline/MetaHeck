"""
Task 3 Grader — Network fraud investigation and SAR filing.
Scores entity linking accuracy, SAR quality, and network discovery.
"""
from typing import Dict, Any, List, Set


GROUND_TRUTH_NETWORK = {
    "CUST-T3-SHELL": {"CUST-T3-A", "CUST-T3-B", "CUST-T3-C"},
    "CUST-T3-A": {"CUST-T3-SHELL"},
    "CUST-T3-B": {"CUST-T3-SHELL"},
    "CUST-T3-C": {"CUST-T3-SHELL"},
    "CUST-T3-D": {"CUST-T3-E"},
    "CUST-T3-E": {"CUST-T3-D", "CUST-T3-F"},
    "CUST-T3-F": {"CUST-T3-E"},
}

EVIDENCE_KEYWORDS = {
    "CUST-T3-SHELL": ["shell", "holding", "beneficial owner", "missing", "outflow",
                      "apex", "corporate plaza", "wilmington"],
    "CUST-T3-A": ["shared address", "shell", "apex", "receives", "funds", "corporate plaza"],
    "CUST-T3-B": ["shared address", "shell", "apex", "receives", "funds"],
    "CUST-T3-C": ["pep", "politically exposed", "adverse media", "shared address",
                  "shell", "offshore"],
    "CUST-T3-D": ["layering", "chain", "forward", "rapid", "120000"],
    "CUST-T3-E": ["layering", "chain", "forward", "rapid", "middle"],
    "CUST-T3-F": ["layering", "chain", "cash out", "withdrawal", "end"],
}


def grade(actions: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    if not actions:
        return {"score": 0.0, "breakdown": {}, "feedback": "No actions taken."}

    feedback_parts = []
    breakdown = {}
    customers_acted_on: Set[str] = set()
    agent_links: Dict[str, Set[str]] = {k: set() for k in ground_truth}
    sar_filed: Set[str] = set()
    total_score = 0.0

    # Collect link_entities actions
    for action in actions:
        if action.get("action_type") == "link_entities":
            cid = action.get("customer_id")
            target = action.get("target_customer_id")
            if cid and target:
                agent_links.setdefault(cid, set()).add(target)
                agent_links.setdefault(target, set()).add(cid)

        if action.get("action_type") == "file_sar":
            sar_filed.add(action.get("customer_id"))

    for action in actions:
        cid = action.get("customer_id")
        if cid not in ground_truth or cid in customers_acted_on:
            continue
        if action.get("action_type") == "link_entities":
            continue  # handled separately
        customers_acted_on.add(cid)

        gt = ground_truth[cid]
        action_type = action.get("action_type")
        risk_tier = action.get("risk_tier")
        reason = (action.get("reason") or "").lower()

        cust_score = 0.0

        # 1. Decision (35%)
        if action_type == gt["decision"]:
            cust_score += 0.35
            feedback_parts.append(f"{cid}: Correct decision ({action_type}).")
        else:
            if gt["decision"] in ["file_sar", "freeze_account"] and \
               action_type in ["flag_for_review", "file_sar"]:
                cust_score += 0.15
            feedback_parts.append(
                f"{cid}: Expected '{gt['decision']}', got '{action_type}'."
            )

        # 2. Risk tier (15%)
        if risk_tier == gt.get("risk_tier"):
            cust_score += 0.15

        # 3. Evidence reasoning (25%)
        keywords = EVIDENCE_KEYWORDS.get(cid, [])
        if keywords:
            matched = sum(1 for kw in keywords if kw in reason)
            cust_score += min(matched / len(keywords), 1.0) * 0.25

        # 4. Network / entity linking (20%)
        gt_links = GROUND_TRUTH_NETWORK.get(cid, set())
        if gt_links:
            found_links = agent_links.get(cid, set())
            precision = len(found_links & gt_links) / max(len(found_links), 1)
            recall = len(found_links & gt_links) / len(gt_links)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            cust_score += f1 * 0.20
            if found_links & gt_links:
                feedback_parts.append(
                    f"{cid}: Found {len(found_links & gt_links)}/{len(gt_links)} network links."
                )

        # 5. SAR filed when required (5%)
        if gt["decision"] == "file_sar" and cid in sar_filed:
            cust_score += 0.05

        breakdown[cid] = round(cust_score, 3)
        total_score += cust_score

    num_customers = len(ground_truth)
    final_score = round(total_score / num_customers, 4) if num_customers else 0.0

    # False positive penalty — clean customers wrongly flagged
    for action in actions:
        cid = action.get("customer_id")
        gt = ground_truth.get(cid, {})
        if gt.get("decision") == "clear_customer" and \
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