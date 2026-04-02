from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openenv.core import Action as BaseAction, Observation as BaseObservation, State as BaseState

class ActionType(str, Enum):
    REQUEST_ADDITIONAL_DOCUMENTS = "request_additional_documents"
    VERIFY_DOCUMENT_AUTHENTICITY = "verify_document_authenticity"
    ANALYZE_TRANSACTION_PATTERNS = "analyze_transaction_patterns"
    CHECK_WATCHLISTS = "check_watchlists"
    INTERVIEW_CUSTOMER = "interview_customer"
    PERFORM_RISK_SCORING = "perform_risk_scoring"
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    FREEZE_ACCOUNT = "freeze_account"

class CustomerProfile(BaseModel):
    customer_id: str
    personal_info: Dict[str, Any]
    documents: List[Dict[str, Any]]
    transaction_history: List[Dict[str, Any]]
    device_signals: Dict[str, Any]
    behavioral_signals: Dict[str, Any]
    interview_log: List[str] = Field(default_factory=list)
    watchlist_report: Optional[str] = None
    red_flags: List[str] = Field(default_factory=list)

class RewardBreakdown(BaseModel):
    correct_sequencing: float = 0.0
    efficient_investigation: float = 0.0
    accurate_flag_detection: float = 0.0
    professional_interviewing: float = 0.0
    correct_final_decision: float = 0.0
    penalty: float = 0.0
    total: float = 0.0
    reasoning: str = ""

class Observation(BaseObservation):
    task_id: str
    episode_id: str
    step: int
    max_steps: int
    customer: CustomerProfile
    available_actions: List[str]
    completed_actions: List[Dict[str, Any]] = Field(default_factory=list)
    task_description: str
    message: str = ""
    # Inherited fields: done: bool, reward: float | None, metadata: Dict[str, Any]

class Action(BaseAction):
    action_type: ActionType
    target_customer_id: str
    interview_question: Optional[str] = None
    question: Optional[str] = None
    requested_document_type: Optional[str] = None
    risk_score_assigned: Optional[int] = None
    decision_reasoning: Optional[str] = None

class EnvironmentState(BaseState):
    task_id: str
    episode_id: str
    step: int
    max_steps: int
    customer: CustomerProfile
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_score: float = 0.0
    done: bool = False
    reward_breakdown: Optional[RewardBreakdown] = None
    investigation_flags_found: List[str] = Field(default_factory=list)