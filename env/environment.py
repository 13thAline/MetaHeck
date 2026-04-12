import os
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

from openenv.core import Environment
from env.models import (
    Action, Observation, RewardBreakdown, EnvironmentState,
    ActionType, CustomerProfile, EpisodeManifest
)
from env.data_engine import generate_episode

TASK_CONFIG = {
    "task1_easy": {"max_steps": 15, "description": "Easy: Process the compliance queue. Verify docs, check mismatches."},
    "task2_medium": {"max_steps": 20, "description": "Medium: Suspicious txns. Query the ledger, check source of funds. Watch for burst velocity."},
    "task3_hard": {"max_steps": 30, "description": "Hard: Multi-customer mule network. Find overlapping IPs, circular chains, and dormant-burst patterns. Beware ambiguous grey cases."},
}

# Task → grader routing
GRADER_MAP = {
    "task1_easy": "env.graders.grader1",
    "task2_medium": "env.graders.grader2",
    "task3_hard": "env.graders.grader3",
}

# Optional trajectory logging directory (for future fine-tuning data)
TRAJECTORY_LOG_DIR = os.getenv("TRAJECTORY_LOG_DIR", "")

# Penalty for each fraudulent transaction that escapes (funds not intercepted)
FUNDS_ESCAPE_PENALTY = -0.15

# How much time each discovery action advances the global clock
CLOCK_ADVANCE_HOURS = 1


class BankKYCAuditEnv(Environment[Action, Observation, EnvironmentState]):
    """OpenEnv-compliant 2026 Bank KYC Audit Sandbox.

    Now powered by a procedural data engine with live time-series dynamics.
    Every reset() generates fresh, randomized customers and transactions
    with injected fraud patterns, burst velocity, ambiguous grey cases,
    and secretly tracked ground truth.

    Key features:
      - Global simulation clock that advances with discovery actions
      - Funds escape penalty when fraud transactions are not intercepted
      - Confidence-weighted grading to resist exploit attempts
      - Discrete typology codes instead of free-text reasoning
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
        self._current_time: Optional[datetime] = None
        self._intercepted_txn_ids: set = set()
        self._escaped_checked: set = set()  # txn IDs already penalised
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

        # --- Initialise the global simulation clock ---
        # Set to the earliest transaction timestamp in the episode
        self._current_time = self._find_earliest_timestamp()
        self._intercepted_txn_ids = set()
        self._escaped_checked = set()

        self._state = EnvironmentState(
            task_id=self.task_id,
            episode_id=episode_id,
            step=0,
            max_steps=config["max_steps"],
            customers=queue,
            actions_taken=[],
            cumulative_score=0.0,
            done=False,
            reward_breakdown=RewardBreakdown(),
            current_time=self._current_time.isoformat() if self._current_time else "",
            escaped_funds=0.0,
            intercepted_txn_ids=[],
        )

        # Track which data sources each customer has been investigated on
        self._investigation_tracker = {c.customer_id: set() for c in queue}
        self._trajectory = []

        return self._build_observation(
            message="System initialized. Compliance Queue loaded. Simulation clock active.",
            context=""
        )

    def _find_earliest_timestamp(self) -> datetime:
        """Find the earliest transaction timestamp across all customers."""
        earliest = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for cid, db_record in self._manifest.database.items():
            for txn in db_record.get("txns", []):
                ts_str = txn.get("timestamp", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts < earliest:
                            earliest = ts
                    except (ValueError, TypeError):
                        continue
        return earliest

    def _advance_clock(self, hours: int = CLOCK_ADVANCE_HOURS) -> None:
        """Advance the global simulation clock."""
        if self._current_time:
            self._current_time += timedelta(hours=hours)
            self._state.current_time = self._current_time.isoformat()

    def _check_funds_escaped(self) -> float:
        """Check for fraudulent transactions whose timestamp has passed.

        Returns the total escape penalty for this step.
        """
        if not self._current_time:
            return 0.0

        penalty = 0.0
        gt = self._manifest.ground_truth

        for cid, entry in gt.items():
            fraud_txn_ids = set(entry.get("expected_txns", []))
            if not fraud_txn_ids:
                continue

            db_record = self._manifest.database.get(cid, {})
            for txn in db_record.get("txns", []):
                tid = txn.get("id", "")
                if tid not in fraud_txn_ids:
                    continue
                if tid in self._intercepted_txn_ids:
                    continue
                if tid in self._escaped_checked:
                    continue

                ts_str = txn.get("timestamp", "")
                if not ts_str:
                    continue

                try:
                    txn_time = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    continue

                if txn_time <= self._current_time:
                    penalty += FUNDS_ESCAPE_PENALTY
                    self._escaped_checked.add(tid)

        self._state.escaped_funds += abs(penalty)
        return penalty

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
            # Advance clock for discovery action
            self._advance_clock()

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
            # Advance clock for discovery action
            self._advance_clock()

        elif a_type == ActionType.CHECK_WATCHLISTS:
            context = f"Watchlist Scan {cid}: {db_record['watchlists']}"
            message = "Scan complete."
            if "watchlists" not in self._investigation_tracker.get(cid, set()):
                breakdown.data_gathering += 0.05
                self._investigation_tracker.setdefault(cid, set()).add("watchlists")
            self._advance_clock()

        elif a_type == ActionType.PULL_DEVICE_SIGNALS:
            context = f"Device Signals for {cid}: {db_record['device']}"
            message = "Device signals retrieved."
            if "device" not in self._investigation_tracker.get(cid, set()):
                breakdown.data_gathering += 0.05
                self._investigation_tracker.setdefault(cid, set()).add("device")
            self._advance_clock()

        elif a_type == ActionType.INTERVIEW_CUSTOMER:
            question = action.interview_question or "General inquiry"
            # Return hidden interview response if available, else generic
            interview_data = db_record.get("interview_response",
                f"I have nothing unusual to report regarding: {question}")
            context = f"Interview response from {cid}: '{interview_data}'"
            message = "Interview recorded."
            if "interview" not in self._investigation_tracker.get(cid, set()):
                breakdown.data_gathering += 0.03
                self._investigation_tracker.setdefault(cid, set()).add("interview")
            self._advance_clock()

        elif a_type == ActionType.LINK_ENTITIES:
            source = action.source_customer_id or cid
            linked = action.linked_customer_id
            if not linked:
                message = "API Error: LINK_ENTITIES requires linked_customer_id."
                breakdown.penalty -= 0.05
            elif linked not in db:
                message = f"Error: Linked customer {linked} not found in system."
                breakdown.penalty -= 0.05
            else:
                message = f"Entity link recorded: {source} <-> {linked}."
                breakdown.data_gathering += 0.03

        elif a_type == ActionType.INTERCEPT_TRANSACTION:
            txn_ids = action.transaction_ids_to_intercept
            if not txn_ids:
                message = "API Error: INTERCEPT_TRANSACTION requires transaction_ids_to_intercept."
                breakdown.penalty -= 0.05
            else:
                newly_intercepted = []
                for tid in txn_ids:
                    if tid not in self._intercepted_txn_ids:
                        self._intercepted_txn_ids.add(tid)
                        newly_intercepted.append(tid)
                self._state.intercepted_txn_ids = list(self._intercepted_txn_ids)
                message = f"Intercepted {len(newly_intercepted)} transaction(s): {newly_intercepted}"
                if newly_intercepted:
                    breakdown.accurate_flagging += 0.05

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

        # --- Check for funds escaping after clock advancement ---
        escape_penalty = self._check_funds_escaped()
        if escape_penalty < 0:
            breakdown.funds_escaped = escape_penalty
            message += f" | ⚠️ FUNDS ESCAPED: {abs(escape_penalty):.2f} penalty applied (unintercepted fraud)."

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
            breakdown.final_decision, breakdown.penalty, breakdown.funds_escaped
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
            "current_time": self._state.current_time,
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

        # All graders now accept (actions, ground_truth, expected_typologies=…)
        ground_truth = self._manifest.ground_truth
        expected_typologies = self._manifest.expected_typologies

        if self.task_id == "task3_hard":
            result = mod.grade(
                self._state.actions_taken,
                ground_truth,
                network_truth=self._manifest.network_truth,
                expected_typologies=expected_typologies,
            )
        elif self.task_id == "task2_medium":
            result = mod.grade(
                self._state.actions_taken,
                ground_truth,
                expected_typologies=expected_typologies,
            )
        else:
            result = mod.grade(
                self._state.actions_taken,
                ground_truth,
                expected_typologies=expected_typologies,
            )

        if self._state.reward_breakdown:
            self._state.reward_breakdown.reasoning = result.get("feedback", "")

        # Apply cumulative funds_escaped penalty to the final score
        final_score = result["score"]
        if self._state.escaped_funds > 0:
            final_score = max(0.0, final_score - self._state.escaped_funds * 0.1)

        # Save trajectory after grading (episode complete)
        self._save_trajectory()

        return final_score