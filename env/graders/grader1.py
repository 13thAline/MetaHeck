"""
Deterministic Grader (Task 1) — Evaluates exact matches of flagged evidence IDs
and discrete regulatory typology codes.

Scoring breakdown (per customer, normalised to 1.0):
  - Correct terminal decision:  40%
  - Evidence F1 (txn + doc IDs): 30%
  - Typology F1:                 20%
  - Confidence calibration:      10% (bonus / penalty)

Ground truth and expected typologies are passed in dynamically from the
procedural data engine, not hardcoded at module level.
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
    """Return a score modifier based on the agent's stated confidence.

    Correct + high confidence   → small bonus
    Wrong   + high confidence   → catastrophic penalty
    Wrong   + low confidence    → minor penalty
    """
    if decision_correct:
        # Bonus for being right and confident (up to +0.10)
        return 0.10 * confidence

    # Wrong decision — penalise proportional to confidence
    if is_ambiguous and confidence >= 0.9:
        return -1.0  # Catastrophic: froze a legitimately-clean grey case
    if confidence <= 0.3:
        return -0.10 * confidence  # Minor: at least the agent was unsure
    return -0.5 * confidence  # Moderate


def grade(actions: List[Dict[str, Any]], ground_truth: Dict[str, Any],
          expected_typologies: Dict[str, List[str]] | None = None) -> Dict[str, Any]:
    """Grade agent actions against dynamically generated ground truth.

    Args:
        actions: list of action dicts from the agent's session.
        ground_truth: dict keyed by customer_id with expected_decision,
                      expected_txns, and expected_docs.
        expected_typologies: dict keyed by customer_id mapping to lists of
                             RegulatoryTypology string values.

    Returns:
        dict with 'score' (float 0-1) and 'feedback' (str).
    """
    if not actions:
        return {"score": 0.0, "feedback": "No actions taken."}

    expected_typologies = expected_typologies or {}
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
        submitted_typos = action.get("regulatory_typology", [])
        confidence = float(action.get("confidence_score", 0.5))

        cust_score = 0.0

        # 1. Base Score for Correct Decision (40%)
        expected = gt.get("expected_decision", gt.get("decision", ""))
        decision_correct = decision == expected
        if decision_correct:
            cust_score += 0.40
            feedback_parts.append(f"{cid}: Correct decision.")
        else:
            feedback_parts.append(f"{cid}: Wrong decision (Got {decision}, Expected {expected}).")

        # 2. Evidence Score (F1 of Flagged IDs) (30%)
        txn_f1 = calculate_f1(flagged_txns, gt.get("expected_txns", []))
        doc_f1 = calculate_f1(flagged_docs, gt.get("expected_docs", []))
        evidence_score = ((txn_f1 + doc_f1) / 2) * 0.30
        cust_score += evidence_score
        feedback_parts.append(f"[{cid} Evidence F1 -> Txn:{txn_f1:.2f}, Doc:{doc_f1:.2f}]")

        # 3. Typology F1 (20%)
        expected_typos = expected_typologies.get(cid, [])
        typo_f1 = _typology_f1(submitted_typos, expected_typos)
        cust_score += typo_f1 * 0.20
        feedback_parts.append(f"[{cid} Typology F1: {typo_f1:.2f}]")

        # 4. Confidence calibration (10% — bonus or penalty)
        is_ambiguous = gt.get("is_ambiguous", False)
        conf_mod = _confidence_modifier(confidence, decision_correct, is_ambiguous)
        cust_score += conf_mod * 0.10  # Scale the modifier into the 10% band
        feedback_parts.append(f"[{cid} Confidence: {confidence:.2f}, mod: {conf_mod:+.2f}]")

        total_score += cust_score

    # Normalize by number of customers in the Ground Truth
    final_score = total_score / len(ground_truth) if ground_truth else 0.0

    return {
        "score": round(max(min(final_score, 1.0), 0.0), 4),
        "feedback": " | ".join(feedback_parts)
    }