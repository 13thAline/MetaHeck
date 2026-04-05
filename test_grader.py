"""
Test suite for graders with dynamic ground truth from the procedural data engine.
"""
from env.data_engine import generate_episode
from env.graders.grader1 import grade as grade1
from env.graders.grader2 import grade as grade2
from env.graders.grader3 import grade as grade3


def test_grader1_dynamic():
    """Generate a task1 episode, simulate perfect + sloppy agents, check scores."""
    manifest = generate_episode("task1_easy", seed=42)
    gt = manifest.ground_truth

    print("=== Task 1 (Easy) — Dynamic Ground Truth ===")
    print(f"Customers: {list(gt.keys())}")
    for cid, entry in gt.items():
        print(f"  {cid}: decision={entry['expected_decision']}, txns={entry['expected_txns']}, docs={entry['expected_docs']}")

    # Perfect agent: matches every expected decision + evidence
    perfect_actions = []
    for cid, entry in gt.items():
        perfect_actions.append({
            "action_type": entry["expected_decision"],
            "target_customer_id": cid,
            "decision_reasoning": "Thorough investigation completed.",
            "flagged_transaction_ids": entry["expected_txns"],
            "flagged_document_ids": entry["expected_docs"],
        })

    result = grade1(perfect_actions, gt)
    print(f"\nPerfect Agent Score: {result['score']}")
    print(f"Feedback: {result['feedback']}")
    assert result["score"] > 0.8, f"Perfect agent should score > 0.8, got {result['score']}"

    # Sloppy agent: right decision, no evidence
    sloppy_actions = []
    for cid, entry in gt.items():
        sloppy_actions.append({
            "action_type": entry["expected_decision"],
            "target_customer_id": cid,
            "decision_reasoning": "Gut feeling.",
            "flagged_transaction_ids": [],
            "flagged_document_ids": [],
        })

    result2 = grade1(sloppy_actions, gt)
    print(f"\nSloppy Agent Score: {result2['score']}")
    print(f"Feedback: {result2['feedback']}")
    assert result2["score"] < result["score"], "Sloppy agent should score lower than perfect"
    print("\n✅ Grader 1 passed.\n")


def test_grader2_dynamic():
    """Generate a task2 episode, simulate perfect agent."""
    manifest = generate_episode("task2_medium", seed=99)
    gt = manifest.ground_truth

    print("=== Task 2 (Medium) — Dynamic Ground Truth ===")
    print(f"Customers: {list(gt.keys())}")

    perfect_actions = []
    for cid, entry in gt.items():
        expected = entry.get("expected_decision", entry.get("decision"))
        perfect_actions.append({
            "action_type": expected,
            "target_customer_id": cid,
            "decision_reasoning": " ".join(manifest.evidence_keywords.get(cid, [])),
            "risk_tier": entry.get("risk_tier", "low"),
            "flagged_transaction_ids": entry.get("expected_txns", []),
            "flagged_document_ids": entry.get("expected_docs", []),
        })

    result = grade2(perfect_actions, gt, evidence_keywords=manifest.evidence_keywords)
    print(f"Perfect Agent Score: {result['score']}")
    print(f"Breakdown: {result['breakdown']}")
    print(f"Feedback: {result['feedback']}")
    assert result["score"] > 0.5, f"Perfect agent should score > 0.5, got {result['score']}"
    print("\n✅ Grader 2 passed.\n")


def test_grader3_dynamic():
    """Generate a task3 episode, simulate perfect agent."""
    manifest = generate_episode("task3_hard", seed=777)
    gt = manifest.ground_truth

    print("=== Task 3 (Hard) — Dynamic Ground Truth ===")
    print(f"Customers: {list(gt.keys())}")
    print(f"Network: {manifest.network_truth}")

    perfect_actions = []
    for cid, entry in gt.items():
        expected = entry.get("expected_decision", entry.get("decision"))
        perfect_actions.append({
            "action_type": expected,
            "target_customer_id": cid,
            "decision_reasoning": " ".join(manifest.evidence_keywords.get(cid, [])),
            "risk_tier": entry.get("risk_tier", "low"),
            "flagged_transaction_ids": entry.get("expected_txns", []),
            "flagged_document_ids": entry.get("expected_docs", []),
        })

    result = grade3(perfect_actions, gt,
                    network_truth=manifest.network_truth,
                    evidence_keywords=manifest.evidence_keywords)
    print(f"Perfect Agent Score: {result['score']}")
    print(f"Breakdown: {result['breakdown']}")
    print(f"Feedback: {result['feedback']}")
    assert result["score"] > 0.3, f"Perfect agent should score > 0.3, got {result['score']}"
    print("\n✅ Grader 3 passed.\n")


def test_reproducibility():
    """Same seed → same episode."""
    m1 = generate_episode("task2_medium", seed=12345)
    m2 = generate_episode("task2_medium", seed=12345)

    assert [c.customer_id for c in m1.customers] == [c.customer_id for c in m2.customers], \
        "Same seed should produce same customer IDs"
    assert list(m1.ground_truth.keys()) == list(m2.ground_truth.keys()), \
        "Same seed should produce same ground truth keys"

    # Different seed → different episode
    m3 = generate_episode("task2_medium", seed=99999)
    assert [c.customer_id for c in m1.customers] != [c.customer_id for c in m3.customers], \
        "Different seeds should produce different customer IDs"

    print("✅ Reproducibility test passed.\n")


def test_no_empty_ground_truth():
    """Every generated episode must have non-empty ground truth."""
    for task in ["task1_easy", "task2_medium", "task3_hard"]:
        for seed in range(10):
            m = generate_episode(task, seed=seed)
            assert len(m.ground_truth) > 0, f"Empty ground truth for {task} seed={seed}"
            assert len(m.customers) > 0, f"Empty customers for {task} seed={seed}"
            assert len(m.database) > 0, f"Empty database for {task} seed={seed}"

            # Verify all fraud txn IDs in ground truth actually exist in the database
            for cid, gt in m.ground_truth.items():
                db_txn_ids = {t["id"] for t in m.database[cid]["txns"]}
                for expected_tid in gt.get("expected_txns", []):
                    assert expected_tid in db_txn_ids, \
                        f"Ground truth TXN {expected_tid} missing from DB for {cid} in {task} seed={seed}"

    print("✅ No-empty-ground-truth test passed (30 episodes verified).\n")


if __name__ == "__main__":
    test_reproducibility()
    test_no_empty_ground_truth()
    test_grader1_dynamic()
    test_grader2_dynamic()
    test_grader3_dynamic()
    print("🎉 All tests passed!")