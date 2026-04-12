"""
Test suite for graders with dynamic ground truth from the procedural data engine.

Updated for v3.0:
  - Uses regulatory_typology instead of decision_reasoning
  - Uses expected_typologies instead of evidence_keywords
  - Tests burst velocity and ambiguous customer patterns
  - Tests confidence-weighted scoring
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

    # Perfect agent: matches every expected decision + evidence + typology
    perfect_actions = []
    for cid, entry in gt.items():
        perfect_actions.append({
            "action_type": entry["expected_decision"],
            "target_customer_id": cid,
            "regulatory_typology": manifest.expected_typologies.get(cid, []),
            "confidence_score": 0.9,
            "flagged_transaction_ids": entry["expected_txns"],
            "flagged_document_ids": entry["expected_docs"],
        })

    result = grade1(perfect_actions, gt, expected_typologies=manifest.expected_typologies)
    print(f"\nPerfect Agent Score: {result['score']}")
    print(f"Feedback: {result['feedback']}")
    assert result["score"] > 0.7, f"Perfect agent should score > 0.7, got {result['score']}"

    # Sloppy agent: right decision, no evidence, no typology, low confidence
    sloppy_actions = []
    for cid, entry in gt.items():
        sloppy_actions.append({
            "action_type": entry["expected_decision"],
            "target_customer_id": cid,
            "regulatory_typology": [],
            "confidence_score": 0.3,
            "flagged_transaction_ids": [],
            "flagged_document_ids": [],
        })

    result2 = grade1(sloppy_actions, gt, expected_typologies=manifest.expected_typologies)
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
    print(f"Expected typologies: {manifest.expected_typologies}")

    perfect_actions = []
    for cid, entry in gt.items():
        expected = entry.get("expected_decision", entry.get("decision"))
        perfect_actions.append({
            "action_type": expected,
            "target_customer_id": cid,
            "regulatory_typology": manifest.expected_typologies.get(cid, []),
            "risk_tier": entry.get("risk_tier", "low"),
            "confidence_score": 0.85,
            "flagged_transaction_ids": entry.get("expected_txns", []),
            "flagged_document_ids": entry.get("expected_docs", []),
        })

    result = grade2(perfect_actions, gt, expected_typologies=manifest.expected_typologies)
    print(f"Perfect Agent Score: {result['score']}")
    print(f"Breakdown: {result['breakdown']}")
    print(f"Feedback: {result['feedback']}")
    assert result["score"] > 0.4, f"Perfect agent should score > 0.4, got {result['score']}"
    print("\n✅ Grader 2 passed.\n")


def test_grader3_dynamic():
    """Generate a task3 episode, simulate perfect agent."""
    manifest = generate_episode("task3_hard", seed=777)
    gt = manifest.ground_truth

    print("=== Task 3 (Hard) — Dynamic Ground Truth ===")
    print(f"Customers: {list(gt.keys())}")
    print(f"Network: {manifest.network_truth}")
    print(f"Expected typologies: {manifest.expected_typologies}")

    perfect_actions = []
    for cid, entry in gt.items():
        expected = entry.get("expected_decision", entry.get("decision"))
        perfect_actions.append({
            "action_type": expected,
            "target_customer_id": cid,
            "regulatory_typology": manifest.expected_typologies.get(cid, []),
            "risk_tier": entry.get("risk_tier", "low"),
            "confidence_score": 0.9,
            "flagged_transaction_ids": entry.get("expected_txns", []),
            "flagged_document_ids": entry.get("expected_docs", []),
        })

    result = grade3(perfect_actions, gt,
                    network_truth=manifest.network_truth,
                    expected_typologies=manifest.expected_typologies)
    print(f"Perfect Agent Score: {result['score']}")
    print(f"Breakdown: {result['breakdown']}")
    print(f"Feedback: {result['feedback']}")
    assert result["score"] > 0.2, f"Perfect agent should score > 0.2, got {result['score']}"
    print("\n✅ Grader 3 passed.\n")


def test_confidence_penalty():
    """Test that high-confidence wrong decisions on ambiguous cases are catastrophically penalised."""
    manifest = generate_episode("task2_medium", seed=55)
    gt = manifest.ground_truth

    print("=== Confidence Penalty Test ===")

    # Find an ambiguous customer (expected_decision == "approve" with is_ambiguous)
    ambiguous_cid = None
    for cid, entry in gt.items():
        if entry.get("is_ambiguous", False):
            ambiguous_cid = cid
            break

    if not ambiguous_cid:
        print("No ambiguous customer in this seed — skipping confidence penalty test.")
        print("✅ Confidence penalty test skipped (no ambiguous case generated).\n")
        return

    print(f"Ambiguous customer: {ambiguous_cid}")

    # Agent wrongly freezes with HIGH confidence → catastrophic penalty
    high_conf_wrong = [{
        "action_type": "freeze_account",
        "target_customer_id": ambiguous_cid,
        "regulatory_typology": ["STRUCTURING_314A"],
        "confidence_score": 0.95,
        "flagged_transaction_ids": [],
        "flagged_document_ids": [],
    }]

    # Agent wrongly flags with LOW confidence → minor penalty
    low_conf_wrong = [{
        "action_type": "freeze_account",
        "target_customer_id": ambiguous_cid,
        "regulatory_typology": ["STRUCTURING_314A"],
        "confidence_score": 0.25,
        "flagged_transaction_ids": [],
        "flagged_document_ids": [],
    }]

    result_high = grade2(high_conf_wrong, {ambiguous_cid: gt[ambiguous_cid]},
                         expected_typologies={ambiguous_cid: manifest.expected_typologies.get(ambiguous_cid, [])})
    result_low = grade2(low_conf_wrong, {ambiguous_cid: gt[ambiguous_cid]},
                        expected_typologies={ambiguous_cid: manifest.expected_typologies.get(ambiguous_cid, [])})

    print(f"High confidence wrong score: {result_high['score']}")
    print(f"Low confidence wrong score:  {result_low['score']}")
    assert result_high["score"] <= result_low["score"], \
        "High-confidence wrong decision should score worse than low-confidence"
    print("✅ Confidence penalty test passed.\n")


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
            assert len(m.expected_typologies) > 0, f"Empty expected_typologies for {task} seed={seed}"

            # Verify all fraud txn IDs in ground truth actually exist in the database
            for cid, gt in m.ground_truth.items():
                db_txn_ids = {t["id"] for t in m.database[cid]["txns"]}
                for expected_tid in gt.get("expected_txns", []):
                    assert expected_tid in db_txn_ids, \
                        f"Ground truth TXN {expected_tid} missing from DB for {cid} in {task} seed={seed}"

            # Verify expected_typologies keys match ground_truth keys
            for cid in m.ground_truth:
                assert cid in m.expected_typologies, \
                    f"Customer {cid} missing from expected_typologies in {task} seed={seed}"

    print("✅ No-empty-ground-truth test passed (30 episodes verified).\n")


def test_burst_velocity_pattern():
    """Verify burst velocity customers generate correct transaction patterns."""
    manifest = generate_episode("task2_medium", seed=42)
    gt = manifest.ground_truth

    print("=== Burst Velocity Pattern Test ===")
    found_burst = False
    for cid, entry in gt.items():
        if "burst_velocity" in entry.get("red_flags", []):
            found_burst = True
            txns = manifest.database[cid]["txns"]
            burst_txns = [t for t in txns if t["id"] in entry["expected_txns"]]
            print(f"  {cid}: {len(burst_txns)} burst transactions found")
            assert len(burst_txns) >= 30, f"Burst should have >= 30 txns, got {len(burst_txns)}"
            assert "BURST_VELOCITY" in manifest.expected_typologies[cid]
            break

    if not found_burst:
        print("  No burst velocity customer in this seed — checking alternate seed...")
        manifest2 = generate_episode("task2_medium", seed=100)
        for cid, entry in manifest2.ground_truth.items():
            if "burst_velocity" in entry.get("red_flags", []):
                found_burst = True
                burst_txns = [t for t in manifest2.database[cid]["txns"]
                              if t["id"] in entry["expected_txns"]]
                print(f"  {cid}: {len(burst_txns)} burst transactions found (seed=100)")
                assert len(burst_txns) >= 30
                break

    assert found_burst, "Should find at least one burst velocity customer across seeds"
    print("✅ Burst velocity pattern test passed.\n")


if __name__ == "__main__":
    test_reproducibility()
    test_no_empty_ground_truth()
    test_grader1_dynamic()
    test_grader2_dynamic()
    test_grader3_dynamic()
    test_confidence_penalty()
    test_burst_velocity_pattern()
    print("🎉 All tests passed!")