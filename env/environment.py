import uuid
from typing import Dict, Any, Optional, List

from openenv.core import Environment
from env.models import (
    Action, Observation, RewardBreakdown, EnvironmentState,
    ActionType, CustomerProfile
)

TASK_CONFIG = {
    "task1_easy": {"max_steps": 15, "description": "Easy: Process the compliance queue. Verify docs, check mismatches."},
    "task2_medium": {"max_steps": 20, "description": "Medium: Suspicious txns. Query the ledger, check source of funds."},
    "task3_hard": {"max_steps": 30, "description": "Hard: Multi-customer mule network. Find the overlapping IP addresses and circular chains."},
}

MOCK_DATABASE = {
    "CUST-001": {
        "docs": "ID: Valid Driver's License. Utility Bill: Mismatch (Address is P.O. Box).",
        "txns": [{"date": "2025-01-10", "amount": 9500, "type": "deposit", "id": "TXN-A1"}],
        "watchlists": "CLEAR",
        "device": "IP: 192.168.1.5 (Local)"
    },
    "CUST-002": {
        "docs": "ID: Passport (Expired).",
        "txns": [
            {"date": "2025-02-01", "amount": 9900, "type": "deposit", "id": "TXN-M1"},
            {"date": "2025-02-02", "amount": 9900, "type": "deposit", "id": "TXN-M2"}
        ],
        "watchlists": "PEP HIT (Uncle is Minister of Finance)",
        "device": "IP: 45.33.22.1 (VPN Detected)"
    }
}

class BankKYCAuditEnv(Environment[Action, Observation, EnvironmentState]):
    """OpenEnv-compliant 2025 Bank KYC Audit Sandbox."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: str = "task1_easy"):
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id '{task_id}'.")
        self.task_id = task_id
        self._state: Optional[EnvironmentState] = None
        super().__init__()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        config = TASK_CONFIG[self.task_id]
        
        queue = [
            CustomerProfile(customer_id="CUST-001", status="pending_review", personal_info={"name": "Alice Smith", "opened": "2025-01-01"}),
            CustomerProfile(customer_id="CUST-002", status="pending_review", personal_info={"name": "Bob Jones", "opened": "2025-02-01"})
        ]
        
        self._state = EnvironmentState(
            task_id=self.task_id,
            episode_id=episode_id or str(uuid.uuid4()),
            step=0,
            max_steps=config["max_steps"],
            customers=queue,
            actions_taken=[],
            cumulative_score=0.0,
            done=False,
            reward_breakdown=RewardBreakdown()
        )
        self._investigation_tracker = {"CUST-001": set(), "CUST-002": set()}
        
        return self._build_observation(message="System initialized. Compliance Queue loaded.", context="")

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step().")

        self._state.step += 1
        self._state.actions_taken.append(action.model_dump())
        
        breakdown = RewardBreakdown()
        message = ""
        context = ""
        cid = action.target_customer_id

        if cid not in MOCK_DATABASE:
            breakdown.penalty -= 0.1
            return self._build_observation(f"Error: Customer {cid} not found in system.", "")

        db_record = MOCK_DATABASE[cid]
        a_type = action.action_type

        if a_type == ActionType.PULL_DOCUMENT_DOSSIER:
            context = db_record["docs"]
            message = f"Dossier retrieved for {cid}."
            if "docs" not in self._investigation_tracker[cid]:
                breakdown.data_gathering += 0.05
                self._investigation_tracker[cid].add("docs")

        elif a_type == ActionType.QUERY_TRANSACTIONS:
            if not action.start_date or not action.end_date:
                message = "API Error: QUERY_TRANSACTIONS requires start_date and end_date."
                breakdown.penalty -= 0.05
            else:
                context = f"Ledger for {cid} ({action.start_date} to {action.end_date}): {db_record['txns']}"
                message = "Transactions retrieved."
                if "txns" not in self._investigation_tracker[cid]:
                    breakdown.data_gathering += 0.05
                    self._investigation_tracker[cid].add("txns")

        elif a_type == ActionType.CHECK_WATCHLISTS:
            context = f"Watchlist Scan {cid}: {db_record['watchlists']}"
            message = "Scan complete."
            if "watchlists" not in self._investigation_tracker[cid]:
                breakdown.data_gathering += 0.05
                self._investigation_tracker[cid].add("watchlists")

        elif a_type in [ActionType.APPROVE, ActionType.REJECT, ActionType.FREEZE_ACCOUNT, ActionType.FILE_SAR]:
            
            investigated_items = self._investigation_tracker[cid]
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

    @property
    def state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Environment not initialized.")
        return self._state
        
    def grade(self) -> float:
        from env.graders.grader1 import grade as run_grade
        result = run_grade(self._state.actions_taken, self.task_id)
        if self._state.reward_breakdown:
            self._state.reward_breakdown.reasoning = result.get("feedback", "")
        return result["score"]