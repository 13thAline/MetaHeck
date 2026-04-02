import uuid
from typing import Dict, Any, Optional, Tuple

from openenv.core import Environment
from env.models import (
    Action, Observation, RewardBreakdown, EnvironmentState,
    ActionType, CustomerProfile
)
from env.data_generator import get_customer_for_task
from env.graders import grade_episode

# Task Config
TASK_CONFIG = {
    "task1_easy": {"max_steps": 10, "description": "Easy: Clean customer, address mismatch."},
    "task2_medium": {"max_steps": 15, "description": "Medium: Suspicious txns, source of funds needed."},
    "task3_hard": {"max_steps": 25, "description": "Hard: Synthetic ID, mule network, deepfakes."},
}

class BankKYCAuditEnv(Environment[Action, Observation, EnvironmentState]):
    """OpenEnv-compliant 2025 Bank KYC Audit environment passing dense multi-step reward logic."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: str = "task1_easy"):
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id '{task_id}'.")
        self.task_id = task_id
        self._state: Optional[EnvironmentState] = None
        super().__init__()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        config = TASK_CONFIG[self.task_id]
        customer = get_customer_for_task(self.task_id)
        
        self._state = EnvironmentState(
            task_id=self.task_id,
            episode_id=episode_id or str(uuid.uuid4()),
            step=0,
            max_steps=config["max_steps"],
            customer=customer,
            actions_taken=[],
            cumulative_score=0.0,
            done=False,
            reward_breakdown=RewardBreakdown(),
            investigation_flags_found=[]
        )
        return self._build_observation("New case loaded. Waiting for investigator action.")

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step().")

        self._state.step += 1
        self._state.actions_taken.append(action.model_dump())
        
        breakdown = RewardBreakdown()
        message = ""
        done = False

        # --- DENSE REWARD LOGIC ---
        a_type = action.action_type
        
        if a_type in [ActionType.REQUEST_ADDITIONAL_DOCUMENTS, ActionType.VERIFY_DOCUMENT_AUTHENTICITY]:
            message = "Documents verified via secure enclave. ID confirmed."
            breakdown.correct_sequencing += 0.05
            
        elif a_type == ActionType.ANALYZE_TRANSACTION_PATTERNS:
            message = "Transaction graph analyzed. Detected 2 circular loops in history."
            breakdown.accurate_flag_detection += 0.10
            self._state.investigation_flags_found.append("circular_txns")
            
        elif a_type == ActionType.CHECK_WATCHLISTS:
            message = "Checked OFAC/PEP lists. No direct hits."
            breakdown.correct_sequencing += 0.05
            
        elif a_type == ActionType.INTERVIEW_CUSTOMER:
            q = getattr(action, "question", None) or getattr(action, "interview_question", None) or ""
            message = f"Customer replied: 'I received those funds from my uncle regarding the {q}'"
            self._state.customer.interview_log.append(f"Q: {q} | A: {message}")
            breakdown.professional_interviewing += 0.10
            
        elif a_type == ActionType.PERFORM_RISK_SCORING:
            if not self._state.investigation_flags_found:
                breakdown.penalty -= 0.10
                message = "Risk score performed without gathering evidence. Penalty applied."
            else:
                breakdown.correct_sequencing += 0.10
                message = "Risk safely calculated based on current indicators."
                
        elif a_type in [ActionType.APPROVE, ActionType.REJECT, ActionType.ESCALATE, ActionType.FREEZE_ACCOUNT]:
            done = True
            # Simplified final reward logic until Phase 3 integrated Graders
            if a_type == ActionType.APPROVE and self.task_id == "task1_easy":
                breakdown.correct_final_decision += 0.30
                message = "Correct Final Decision: Approved clean customer."
            elif a_type in [ActionType.ESCALATE, ActionType.FREEZE_ACCOUNT] and self.task_id == "task3_hard":
                breakdown.correct_final_decision += 0.40
                message = "Correct Final Decision: Synthetic ID halted."
            else:
                breakdown.penalty -= 0.50
                message = "Incorrect Final Decision. Protocol breach."
            
            # Efficient investigation logic
            if self._state.step < self._state.max_steps * 0.7:
                breakdown.efficient_investigation += 0.10

        # Terminate if max steps exceeded
        if self._state.step >= self._state.max_steps and not done:
            done = True
            message = "Max steps reached without decision."
            breakdown.penalty -= 0.20

        # Compute total
        step_total = sum([
            breakdown.correct_sequencing,
            breakdown.efficient_investigation,
            breakdown.accurate_flag_detection,
            breakdown.professional_interviewing,
            breakdown.correct_final_decision,
            breakdown.penalty
        ])
        
        self._state.cumulative_score += step_total
        self._state.done = done
        self._state.reward_breakdown = breakdown

        obs = self._build_observation(message)
        obs.reward = round(step_total, 4)
        obs.done = done
        obs.metadata = {
            "cumulative_score": round(self._state.cumulative_score, 4),
            "breakdown": breakdown.model_dump(),
            "flags": self._state.investigation_flags_found
        }
        return obs

    def _build_observation(self, message: str) -> Observation:
        config = TASK_CONFIG[self.task_id]
        
        # Enumerate actions dynamically for UI rendering
        avail = [a.value for a in ActionType]

        return Observation(
            task_id=self.task_id,
            episode_id=self._state.episode_id,
            step=self._state.step,
            max_steps=self._state.max_steps,
            customer=self._state.customer,
            available_actions=avail,
            completed_actions=self._state.actions_taken,
            task_description=config["description"],
            message=message,
            done=self._state.done,
            reward=0.0,
            metadata={}
        )

    @property
    def state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Environment not initialized.")
        return self._state
        
    def grade(self) -> float:
        """Call the deterministic grader"""
        return grade_episode(self._state)