from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date
from enum import Enum


class ActionType(str, Enum):
    CLEAR_CUSTOMER = "clear_customer"
    FLAG_FOR_REVIEW = "flag_for_review"
    REQUEST_DOCUMENTS = "request_documents"
    FILE_SAR = "file_sar"
    FREEZE_ACCOUNT = "freeze_account"
    LINK_ENTITIES = "link_entities"
    ADD_NOTE = "add_note"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransactionType(str, Enum):
    CASH_DEPOSIT = "cash_deposit"
    CASH_WITHDRAWAL = "cash_withdrawal"
    WIRE_TRANSFER = "wire_transfer"
    INTERNAL = "internal"
    ATM = "atm"


class Transaction(BaseModel):
    txn_id: str
    date: str  # ISO format string for JSON serialization
    amount_usd: float
    type: TransactionType
    counterparty: Optional[str] = None
    country: Optional[str] = None
    description: Optional[str] = None


class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    dob: str  # ISO format string
    nationality: str
    occupation: str
    annual_income_usd: float
    account_open_date: str  # ISO format string
    documents_present: List[str] = Field(default_factory=list)
    documents_expired: List[str] = Field(default_factory=list)
    documents_missing: List[str] = Field(default_factory=list)
    transactions: List[Transaction] = Field(default_factory=list)
    linked_entity_ids: List[str] = Field(default_factory=list)
    address: str = ""
    phone: str = ""
    pep_flag: bool = False          # Politically Exposed Person
    sanctions_flag: bool = False
    adverse_media: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    task_id: str
    episode_id: str
    step: int
    max_steps: int
    customers: List[CustomerProfile]
    available_actions: List[str]
    completed_actions: List[Dict[str, Any]] = Field(default_factory=list)
    message: Optional[str] = None
    task_description: str = ""


class Action(BaseModel):
    action_type: ActionType
    customer_id: str
    target_customer_id: Optional[str] = None   # for link_entities
    documents_requested: Optional[List[str]] = None
    reason: str = Field(..., min_length=10)     # agent must explain reasoning
    risk_tier: Optional[RiskTier] = None
    sar_details: Optional[Dict[str, Any]] = None  # for file_sar


class RewardBreakdown(BaseModel):
    correct_decision: float = 0.0
    risk_tier_accuracy: float = 0.0
    evidence_quality: float = 0.0
    entity_linking: float = 0.0
    document_handling: float = 0.0
    procedural_compliance: float = 0.0
    penalty: float = 0.0


class Reward(BaseModel):
    step_score: float = Field(ge=0.0, le=1.0)
    cumulative_score: float = Field(ge=0.0, le=1.0)
    breakdown: RewardBreakdown
    feedback: str
    done: bool = False


class EnvironmentState(BaseModel):
    task_id: str
    episode_id: str
    step: int
    max_steps: int
    customers: List[CustomerProfile]
    actions_taken: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    cumulative_score: float
    done: bool