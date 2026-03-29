"""
KYC Audit Environment — OpenEnv compliant implementation.
Simulates a bank KYC/AML compliance desk.
"""
import uuid
from typing import Dict, Any, Tuple, Optional
from env.models import (
    Action, Observation, Reward, RewardBreakdown, EnvironmentState,
    ActionType
)
from env.data_generator import (
    generate_task1_profiles,
    generate_task2_profiles,
    generate_task3_profiles,
)
from env.graders import grader1, grader2, grader3


TASK_CONFIG = {
    "task1_doc_check": {
        "description": (
            "You are a KYC analyst. Review each customer's identity documents, "
            "occupation, and account details. Decide: clear, flag, or request more docs. "
            "Always provide a reason explaining your decision."
        ),
        "max_steps": 15,
        "generator": generate_task1_profiles,
        "grader": grader1,
    },
    "task2_txn_analysis": {
        "description": (
            "You are an AML analyst. Analyze 30-day transaction histories for each customer. "
            "Identify structuring, round-trips, high-velocity, and income mismatches. "
            "Flag suspicious customers and file SARs where required."
        ),
        "max_steps": 25,
        "generator": generate_task2_profiles,
        "grader": grader2,
    },
    "task3_network_fraud": {
        "description": (
            "You are a senior fraud investigator. Ten customers may be connected through "
            "shell companies, shared addresses, and chain transfers. "
            "Use link_entities to map the network, file SARs for critical cases, "
            "and freeze accounts where funds are at risk."
        ),
        "max_steps": 40,
        "generator": generate_task3_profiles,
        "grader": grader3,
    },
}

AVAILABLE_ACTIONS = [
    "clear_customer", "flag_for_review", "request_documents",
    "file_sar", "freeze_account", "link_entities", "add_note",
]


class KYCEnvironment:
    """OpenEnv-compliant KYC Audit environment."""

    def __init__(self, task_id: str = "task1_doc_check"):
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASK_CONFIG)}")
        self.task_id = task_id
        self._state: Optional[EnvironmentState] = None

    # ── Core OpenEnv API ───────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode. Returns initial observation."""
        config = TASK_CONFIG[self.task_id]
        customers, ground_truth = config["generator"]()

        self._state = EnvironmentState(
            task_id=self.task_id,
            episode_id=str(uuid.uuid4()),
            step=0,
            max_steps=config["max_steps"],
            customers=customers,
            actions_taken=[],
            ground_truth=ground_truth,
            cumulative_score=0.0,
            done=False,
        )

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action. Returns (observation, reward, done, info).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._state.step += 1

        # Validate action customer exists
        cust_ids = {c.customer_id for c in self._state.customers}
        if action.customer_id not in cust_ids:
            reward = Reward(
                step_score=0.0,
                cumulative_score=self._state.cumulative_score,
                breakdown=RewardBreakdown(penalty=-0.05),
                feedback=f"Invalid customer_id '{action.customer_id}'.",
                done=False,
            )
            return self._build_observation(), reward, False, {}

        # Record the action
        self._state.actions_taken.append(action.model_dump())

        # Compute step reward
        step_score, breakdown, feedback = self._compute_step_reward(action)

        # Update cumulative score (running average)
        n = self._state.step
        self._state.cumulative_score = (
            (self._state.cumulative_score * (n - 1) + step_score) / n
        )

        # Check done conditions
        done = self._check_done()
        self._state.done = done

        reward = Reward(
            step_score=round(step_score, 4),
            cumulative_score=round(self._state.cumulative_score, 4),
            breakdown=breakdown,
            feedback=feedback,
            done=done,
        )

        return self._build_observation(), reward, done, {"episode_id": self._state.episode_id}

    def state(self) -> EnvironmentState:
        """Return full internal state (for debugging/logging)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ── Final episode grading ──────────────────────────────────────────────

    def grade_episode(self) -> Dict[str, Any]:
        """
        Run the task grader over all actions taken this episode.
        Returns final score + detailed breakdown.
        """
        if self._state is None:
            return {"score": 0.0, "feedback": "No episode started."}

        grader = TASK_CONFIG[self.task_id]["grader"]
        result = grader.grade(self._state.actions_taken, self._state.ground_truth)
        return result

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        s = self._state
        return Observation(
            task_id=s.task_id,
            episode_id=s.episode_id,
            step=s.step,
            max_steps=s.max_steps,
            customers=s.customers,
            available_actions=AVAILABLE_ACTIONS,
            completed_actions=s.actions_taken,
            task_description=TASK_CONFIG[s.task_id]["description"],
            message=self._step_message(),
        )

    def _step_message(self) -> str:
        s = self._state
        remaining = s.max_steps - s.step
        acted = {a["customer_id"] for a in s.actions_taken
                 if a.get("action_type") != "link_entities"}
        total = len(s.customers)
        reviewed = len(acted)
        return (
            f"Step {s.step}/{s.max_steps}. "
            f"Customers reviewed: {reviewed}/{total}. "
            f"Steps remaining: {remaining}."
        )

    def _compute_step_reward(self, action: Action) -> Tuple[float, RewardBreakdown, str]:
        """
        Lightweight step-level reward for immediate feedback.
        Full episode scoring is done by grade_episode().
        """
        bd = RewardBreakdown()
        feedback_parts = []

        cid = action.customer_id
        gt = self._state.ground_truth.get(cid, {})
        action_type = action.action_type.value

        # Correct decision signal
        if gt and action_type == gt.get("decision"):
            bd.correct_decision = 0.5
            feedback_parts.append("Correct decision.")
        elif gt and action_type in ["flag_for_review", "file_sar"] \
                and gt.get("decision") in ["flag_for_review", "file_sar", "freeze_account"]:
            bd.correct_decision = 0.25
            feedback_parts.append("Partially correct — suspicious customer detected.")
        elif gt and gt.get("decision") == "clear_customer" \
                and action_type in ["flag_for_review", "file_sar", "freeze_account"]:
            bd.penalty = -0.3
            feedback_parts.append("False positive — this customer is clean.")
        elif gt:
            feedback_parts.append(f"Incorrect decision. Expected: {gt.get('decision')}.")

        # Risk tier signal
        if action.risk_tier and gt:
            if action.risk_tier.value == gt.get("risk_tier"):
                bd.risk_tier_accuracy = 0.2
            else:
                feedback_parts.append(f"Risk tier mismatch. Expected: {gt.get('risk_tier')}.")

        # Reward for providing reasoning
        if action.reason and len(action.reason) > 20:
            bd.evidence_quality = 0.1
        else:
            bd.penalty += -0.05
            feedback_parts.append("Provide more detailed reasoning.")

        # Entity linking signal
        if action.action_type == ActionType.LINK_ENTITIES:
            gt_links = self._state.ground_truth.get(cid, {}).get("network_links", [])
            if action.target_customer_id in gt_links:
                bd.entity_linking = 0.3
                feedback_parts.append("Correct entity link identified.")
            elif action.target_customer_id:
                bd.penalty += -0.1
                feedback_parts.append("Incorrect entity link.")

        # Procedural: freeze without SAR first
        if action.action_type == ActionType.FREEZE_ACCOUNT:
            prior_sars = [a for a in self._state.actions_taken
                          if a.get("action_type") == "file_sar"
                          and a.get("customer_id") == cid]
            if not prior_sars:
                bd.procedural_compliance = -0.2
                feedback_parts.append("Procedural: file SAR before freezing account.")

        # Step penalty for going over 80% of max steps without finishing
        progress = self._state.step / self._state.max_steps
        if progress > 0.8:
            reviewed = {a["customer_id"] for a in self._state.actions_taken
                        if a.get("action_type") != "link_entities"}
            total = len(self._state.customers)
            if len(reviewed) < total * 0.8:
                bd.penalty += -0.05
                feedback_parts.append("Warning: running out of steps.")

        step_score = max(0.0, min(1.0,
            bd.correct_decision + bd.risk_tier_accuracy + bd.evidence_quality +
            bd.entity_linking + bd.document_handling + bd.procedural_compliance + bd.penalty
        ))

        return step_score, bd, " | ".join(feedback_parts) or "Action recorded."

    def _check_done(self) -> bool:
        s = self._state
        # Done if max steps reached
        if s.step >= s.max_steps:
            return True
        # Done if all customers have a final decision
        final_decisions = {
            "clear_customer", "file_sar", "freeze_account", "flag_for_review"
        }
        acted_final = {
            a["customer_id"] for a in s.actions_taken
            if a.get("action_type") in final_decisions
        }
        all_customers = {c.customer_id for c in s.customers}
        return acted_final >= all_customers