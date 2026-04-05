import os
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from openenv.core import Environment
from env.models import (
    Action, Observation, RewardBreakdown, EnvironmentState,
    ActionType, CustomerProfile, EpisodeManifest
)
from env.data_engine import generate_episode

TASK_CONFIG = {
    "task1_easy": {"max_steps": 15, "description": "Easy: Process the compliance queue. Verify docs, check mismatches."},
    "task2_medium": {"max_steps": 20, "description": "Medium: Suspicious txns. Query the ledger, check source of funds."},
    "task3_hard": {"max_steps": 30, "description": "Hard: Multi-customer mule network. Find the overlapping IP addresses and circular chains."},
}

# Task → grader routing
GRADER_MAP = {
    "task1_easy": "env.graders.grader1",
    "task2_medium": "env.graders.grader2",
    "task3_hard": "env.graders.grader3",
}

# Optional trajectory logging directory (for future fine-tuning data)
TRAJECTORY_LOG_DIR = os.getenv("TRAJECTORY_LOG_DIR", "")


class BankKYCAuditEnv(Environment[Action, Observation, EnvironmentState]):
    """OpenEnv-compliant 2025 Bank KYC Audit Sandbox.
    
    Now powered by a procedural data engine — every reset() generates
    fresh, randomized customers and transactions with injected fraud
    patterns and secretly tracked ground truth.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: str = "task1_easy"):
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id '{task_id}'.")
        self.task_id = task_id
        self._state: Optional[EnvironmentState] = None
        self._manifest: Optional[EpisodeManifest] = None
        self._investigation_tracker: Dict[str, set] = {}
        self._trajectory: List[Dict[str, Any]] = []
        super().__init__()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        config = TASK_CONFIG[self.task_id]
        episode_id = episode_id or str(uuid.uuid4())

        # Parse optional seed from env var (for debugging reproducibility)
        if seed is None:
            env_seed = os.getenv("EPISODE_SEED", "")
            if env_seed:
                try:
                    seed = int(env_seed)
                except ValueError:
                    pass

        # --- Generate fresh episode via the procedural data engine ---
        self._manifest = generate_episode(
            task_id=self.task_id, seed=seed, episode_id=episode_id
        )

        # Build the customer queue from the manifest (agent sees these)
        queue = self._manifest.customers

        self._state = EnvironmentState(
            task_id=self.task_id,
            episode_id=episode_id,
            step=0,
            max_steps=config["max_steps"],
            customers=queue,
            actions_taken=[],
            cumulative_score=0.0,
            done=False,
            reward_breakdown=RewardBreakdown()
        )

        # Track which data sources each customer has been investigated on
        self._investigation_tracker = {c.customer_id: set() for c in queue}
        self._trajectory = []
        
        return self._build_observation(
            message="System initialized. Compliance Queue loaded.",
            context=""
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step().")

        self._state.step += 1
        self._state.actions_taken.append(action.model_dump())
        
        breakdown = RewardBreakdown()
        message = ""
        context = ""
        cid = action.target_customer_id

        # Query the procedurally generated database
        db = self._manifest.database

        if cid not in db:
            breakdown.penalty -= 0.1
            obs = self._build_observation(f"Error: Customer {cid} not found in system.", "")
            obs.reward = round(breakdown.penalty, 4)
            self._log_trajectory(action, obs, breakdown.penalty)
            return obs

        db_record = db[cid]
        a_type = action.action_type

        if a_type == ActionType.PULL_DOCUMENT_DOSSIER:
            context = db_record["docs"]
            message = f"Dossier retrieved for {cid}."
            if "docs" not in self._investigation_tracker.get(cid, set()):
                breakdown.data_gathering += 0.05
                self._investigation_tracker.setdefault(cid, set()).add("docs")

        elif a_type == ActionType.QUERY_TRANSACTIONS:
            if not action.start_date or not action.end_date:
                message = "API Error: QUERY_TRANSACTIONS requires start_date and end_date."
                breakdown.penalty -= 0.05
            else:
                context = f"Ledger for {cid} ({action.start_date} to {action.end_date}): {db_record['txns']}"
                message = "Transactions retrieved."
                if "txns" not in self._investigation_tracker.get(cid, set()):
                    breakdown.data_gathering += 0.05
                    self._investigation_tracker.setdefault(cid, set()).add("txns")

        elif a_type == ActionType.CHECK_WATCHLISTS:
            context = f"Watchlist Scan {cid}: {db_record['watchlists']}"
            message = "Scan complete."
            if "watchlists" not in self._investigation_tracker.get(cid, set()):
                breakdown.data_gathering += 0.05
                self._investigation_tracker.setdefault(cid, set()).add("watchlists")

        elif a_type == ActionType.PULL_DEVICE_SIGNALS:
            context = f"Device Signals for {cid}: {db_record['device']}"
            message = "Device signals retrieved."
            if "device" not in self._investigation_tracker.get(cid, set()):
                breakdown.data_gathering += 0.05
                self._investigation_tracker.setdefault(cid, set()).add("device")

        elif a_type == ActionType.INTERVIEW_CUSTOMER:
            question = action.interview_question or "General inquiry"
            context = f"Interview response from {cid}: 'I have nothing unusual to report regarding: {question}'"
            message = "Interview recorded."
            if "interview" not in self._investigation_tracker.get(cid, set()):
                breakdown.data_gathering += 0.03
                self._investigation_tracker.setdefault(cid, set()).add("interview")

        elif a_type in [ActionType.APPROVE, ActionType.REJECT, ActionType.FREEZE_ACCOUNT,
                        ActionType.FILE_SAR, ActionType.ESCALATE]:
            investigated_items = self._investigation_tracker.get(cid, set())
            if not investigated_items:
                message = f"COMPLIANCE VIOLATION: {a_type.value} issued blindly without reviewing docs or ledgers."
                breakdown.penalty -= 0.30
            else:
                message = f"Decision '{a_type.value}' recorded for {cid}. Awaiting batch grading."
                breakdown.final_decision += 0.10
                
            for c in self._state.customers:
                if c.customer_id == cid:
                    c.status = f"processed_{a_type.value}"

        all_processed = all(c.status.startswith("processed") for c in self._state.customers)
        if all_processed:
            self._state.done = True
            message += " | All queue items processed. Run /grade for final score."
            
        if self._state.step >= self._state.max_steps and not self._state.done:
            self._state.done = True
            message = "Max steps reached. Shift ended with items in queue."
            breakdown.penalty -= 0.20

        step_total = sum([
            breakdown.data_gathering, breakdown.accurate_flagging,
            breakdown.final_decision, breakdown.penalty
        ])
        
        self._state.cumulative_score += step_total
        
        obs = self._build_observation(message, context)
        obs.reward = round(step_total, 4)

        self._log_trajectory(action, obs, step_total)

        return obs

    def _build_observation(self, message: str, context: str) -> Observation:
        config = TASK_CONFIG[self.task_id]
        avail = [a.value for a in ActionType]

        return Observation(
            task_id=self.task_id,
            episode_id=self._state.episode_id,
            step=self._state.step,
            max_steps=self._state.max_steps,
            customer_queue=self._state.customers,
            investigation_context=context, 
            available_actions=avail,
            completed_actions=self._state.actions_taken,
            task_description=config["description"],
            message=message
        )

    def _log_trajectory(self, action: Action, obs: Observation, reward: float) -> None:
        """Record state-action-reward tuple for trajectory logging (fine-tuning prep)."""
        self._trajectory.append({
            "step": self._state.step,
            "action": action.model_dump(),
            "reward": reward,
            "done": self._state.done,
            "message": obs.message,
        })

    def _save_trajectory(self) -> None:
        """Write trajectory to disk if TRAJECTORY_LOG_DIR is configured."""
        if not TRAJECTORY_LOG_DIR or not self._trajectory:
            return
        try:
            os.makedirs(TRAJECTORY_LOG_DIR, exist_ok=True)
            filename = f"{self._state.episode_id}_{self.task_id}.jsonl"
            filepath = os.path.join(TRAJECTORY_LOG_DIR, filename)
            with open(filepath, "w") as f:
                for entry in self._trajectory:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Don't crash the environment over logging failures

    @property
    def state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Environment not initialized.")
        return self._state
        
    def grade(self) -> float:
        """Run the task-appropriate grader with the dynamically generated ground truth."""
        import importlib

        grader_module = GRADER_MAP.get(self.task_id, "env.graders.grader1")
        mod = importlib.import_module(grader_module)

        # All graders now accept (actions, ground_truth)
        ground_truth = self._manifest.ground_truth

        # For task3, also pass network_truth and evidence_keywords via ground_truth entries
        if self.task_id == "task3_hard":
            result = mod.grade(
                self._state.actions_taken,
                ground_truth,
                network_truth=self._manifest.network_truth,
                evidence_keywords=self._manifest.evidence_keywords,
            )
        elif self.task_id == "task2_medium":
            result = mod.grade(
                self._state.actions_taken,
                ground_truth,
                evidence_keywords=self._manifest.evidence_keywords,
            )
        else:
            result = mod.grade(self._state.actions_taken, ground_truth)

        if self._state.reward_breakdown:
            self._state.reward_breakdown.reasoning = result.get("feedback", "")

        # Save trajectory after grading (episode complete)
        self._save_trajectory()

        return result["score"]