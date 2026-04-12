from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openenv.core import Action as BaseAction, Observation as BaseObservation, State as BaseState


# ---------------------------------------------------------------------------
# Regulatory Typology — discrete, non-gameable classification codes
# ---------------------------------------------------------------------------

class RegulatoryTypology(str, Enum):
    """Standard FinCEN / FATF typology codes.

    Agents must submit one or more of these discrete codes instead of
    free-text reasoning.  This eliminates keyword-salad exploits.
    """
    STRUCTURING_314A = "STRUCTURING_314A"
    LAYERING_FATF_02 = "LAYERING_FATF_02"
    SHELL_COMPANY_FATF_04 = "SHELL_COMPANY_FATF_04"
    CIRCULAR_TRANSACTION = "CIRCULAR_TRANSACTION"
    SMURFING = "SMURFING"
    TRADE_BASED_ML = "TRADE_BASED_ML"
    RAPID_MOVEMENT = "RAPID_MOVEMENT"
    BENEFICIAL_OWNER_CONCEALMENT = "BENEFICIAL_OWNER_CONCEALMENT"
    SANCTIONS_EVASION = "SANCTIONS_EVASION"
    DEEPFAKE_DOCUMENT = "DEEPFAKE_DOCUMENT"
    ADDRESS_MISMATCH = "ADDRESS_MISMATCH"
    PEP_UNDISCLOSED = "PEP_UNDISCLOSED"
    SYNTHETIC_IDENTITY = "SYNTHETIC_IDENTITY"
    BURST_VELOCITY = "BURST_VELOCITY"
    DORMANT_ACTIVATION = "DORMANT_ACTIVATION"
    CLEAN_PROFILE = "CLEAN_PROFILE"


# ---------------------------------------------------------------------------
# Action types — now includes link_entities and intercept_transaction
# ---------------------------------------------------------------------------

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
    LINK_ENTITIES = "link_entities"
    INTERCEPT_TRANSACTION = "intercept_transaction"


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
    funds_escaped: float = 0.0
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

    # --- Discrete typology codes (replaces free-text decision_reasoning) ---
    regulatory_typology: List[str] = Field(default_factory=list)

    flagged_transaction_ids: List[str] = Field(default_factory=list)
    flagged_document_ids: List[str] = Field(default_factory=list)

    # --- Entity linking (Phase 1 — Task 3 fix) ---
    source_customer_id: Optional[str] = None
    linked_customer_id: Optional[str] = None

    # --- Confidence scoring (Phase 3 — exploit resistance) ---
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Live transaction interception (Phase 2) ---
    transaction_ids_to_intercept: List[str] = Field(default_factory=list)


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
    # --- Live simulation clock (Phase 2) ---
    current_time: str = ""
    escaped_funds: float = 0.0
    intercepted_txn_ids: List[str] = Field(default_factory=list)


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
    - expected_typologies: per-customer list of RegulatoryTypology codes
      that the agent must submit for full credit.
    """
    task_id: str
    seed: int
    customers: List[CustomerProfile]
    database: Dict[str, Dict[str, Any]]
    ground_truth: Dict[str, Dict[str, Any]]
    network_truth: Dict[str, List[str]] = Field(default_factory=dict)
    expected_typologies: Dict[str, List[str]] = Field(default_factory=dict)