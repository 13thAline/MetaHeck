from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openenv.core import Action as BaseAction, Observation as BaseObservation, State as BaseState

class ActionType(str, Enum):
    PULL_DOCUMENT_DOSSIER = "pull_document_dossier"
    QUERY_TRANSACTIONS = "query_transactions"
    PULL_DEVICE_SIGNALS = "pull_device_signals"
    CHECK_WATCHLISTS = "check_watchlists"
    INTERVIEW_CUSTOMER = "interview_customer"
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    FREEZE_ACCOUNT = "freeze_account"
    FILE_SAR = "file_sar" 

class CustomerProfile(BaseModel):
    customer_id: str
    status: str = "pending_review"
    personal_info: Dict[str, Any] = Field(default_factory=dict)
    # Extended fields for rich procedural generation
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    transaction_history: List[Dict[str, Any]] = Field(default_factory=list)
    device_signals: Dict[str, Any] = Field(default_factory=dict)
    behavioral_signals: Dict[str, Any] = Field(default_factory=dict)
    watchlist_report: str = "Unchecked"

class RewardBreakdown(BaseModel):
    data_gathering: float = 0.0
    accurate_flagging: float = 0.0
    final_decision: float = 0.0
    penalty: float = 0.0
    total: float = 0.0
    reasoning: str = ""

class Observation(BaseObservation):
    task_id: str
    episode_id: str
    step: int
    max_steps: int
    customer_queue: List[CustomerProfile] 
    investigation_context: str = "" 
    available_actions: List[str]
    completed_actions: List[Dict[str, Any]] = Field(default_factory=list)
    task_description: str
    message: str = ""

class Action(BaseAction):
    action_type: ActionType
    target_customer_id: str
    
    start_date: Optional[str] = None 
    end_date: Optional[str] = None   
    interview_question: Optional[str] = None
    
    decision_reasoning: str 
    flagged_transaction_ids: List[str] = Field(default_factory=list)
    flagged_document_ids: List[str] = Field(default_factory=list)

class EnvironmentState(BaseState):
    task_id: str
    episode_id: str
    step: int
    max_steps: int
    customers: List[CustomerProfile] 
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_score: float = 0.0
    done: bool = False
    reward_breakdown: Optional[RewardBreakdown] = None


# ---------------------------------------------------------------------------
# Episode Manifest — output of the procedural data engine
# ---------------------------------------------------------------------------

class EpisodeManifest(BaseModel):
    """Holds everything the environment needs for one episode.

    - customers: the queue the agent sees (blind — no txns/docs exposed yet)
    - database: keyed by customer_id, contains docs/txns/watchlists/device
      data that the agent can *discover* via step actions.
    - ground_truth: keyed by customer_id, contains expected decisions +
      flagged transaction/document IDs for deterministic grading.
    - network_truth: (task3 only) adjacency dict for entity-linking scoring.
    - evidence_keywords: per-customer keyword sets for reasoning quality scoring.
    """
    task_id: str
    seed: int
    customers: List[CustomerProfile]
    database: Dict[str, Dict[str, Any]]
    ground_truth: Dict[str, Dict[str, Any]]
    network_truth: Dict[str, List[str]] = Field(default_factory=dict)
    evidence_keywords: Dict[str, List[str]] = Field(default_factory=dict)