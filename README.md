# KYC Audit Environment

An OpenEnv-compliant environment that simulates a **bank KYC/AML compliance desk**.
AI agents act as fraud analysts — reviewing customer profiles, detecting suspicious
transaction patterns, mapping fraud networks, and filing regulatory reports.

## Why KYC?

Banks employ hundreds of analysts to manually review customers for fraud and money
laundering. This is expensive, slow, and inconsistent. This environment lets you
train and evaluate AI agents to do this work — across three progressively harder tasks.

---

## Tasks

| Task | Difficulty | Description | Max Steps |
|---|---|---|---|
| `task1_doc_check` | Easy | Document completeness & identity verification | 15 |
| `task2_txn_analysis` | Medium | Transaction pattern analysis (structuring, round-trips) | 25 |
| `task3_network_fraud` | Hard | Multi-entity fraud network investigation + SAR filing | 40 |

### Task 1 — Document Check (Easy)
Agent reviews 3 customer profiles. Must identify expired documents, occupation/income
mismatches, and missing verification docs. Actions: `clear_customer`, `flag_for_review`,
`request_documents`.

### Task 2 — Transaction Analysis (Medium)
Agent reviews 5 customers with 30-day transaction histories. Must detect:
- **Structuring/smurfing** — repeated deposits just under $10,000
- **Round-trip transfers** — money sent and returned within 48 hours
- **High-velocity activity** — 50+ transactions from a dormant account

### Task 3 — Network Fraud (Hard)
Agent investigates 10 customers with hidden relationships:
- **Shell company** linked to 3 individuals at a shared address
- **Chain layering** — 3 customers rapidly forwarding funds to cash out
- **PEP flag** — politically exposed person with adverse media
Agent must use `link_entities` to map the network before filing SARs.

---

## Action Space

| Action | Description |
|---|---|
| `clear_customer` | Customer passes KYC — no issues |
| `flag_for_review` | Escalate to senior analyst |
| `request_documents` | Request missing/expired docs from customer |
| `file_sar` | File Suspicious Activity Report |
| `freeze_account` | Block account (use after SAR) |
| `link_entities` | Assert relationship between two customers |
| `add_note` | Add analyst note without final decision |

## Observation Space

Each step returns a full `Observation` containing:
- All customer profiles (documents, transactions, flags, linked entities)
- Current step / max steps
- History of actions taken this episode
- Task description and system messages

## Reward Function

Dense reward signal at every step:

| Component | Weight | Description |
|---|---|---|
| Correct decision | 0.35–0.45 | Right action for this customer |
| Risk tier accuracy | 0.15–0.20 | low/medium/high/critical match |
| Evidence quality | 0.20–0.30 | Reasoning cites relevant red flags |
| Entity linking | 0.0–0.20 | Correct network connections (Task 3) |
| Procedural compliance | variable | SAR before freeze, etc. |
| Penalties | negative | False positives, procedural errors |

Final episode score comes from the task grader (0.0–1.0).

---

## Setup & Usage

### Local development

```bash
git clone <repo>
cd kyc-audit-env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t kyc-audit-env .
docker run -p 7860:7860 kyc-audit-env
```

### API Usage

```bash
# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_doc_check"}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "flag_for_review",
      "customer_id": "CUST-T1-C",
      "reason": "Student occupation with $180k income is suspicious. Missing source of funds documentation.",
      "risk_tier": "high"
    }
  }'

# Get final score
curl http://localhost:7860/grade
```

### Run baseline inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"

python inference.py
```

---

## Baseline Scores (gpt-4o-mini)

| Task | Score |
|---|---|
| task1_doc_check | ~0.72 |
| task2_txn_analysis | ~0.51 |
| task3_network_fraud | ~0.28 |
| **Overall** | **~0.50** |

---

## Project Structure

```
kyc-audit-env/
├── env/
│   ├── models.py            Pydantic models (Observation, Action, Reward)
│   ├── environment.py       KYCEnvironment class (step/reset/state/grade)
│   ├── data_generator.py    Synthetic customer + transaction generator
│   ├── tasks/               Task definitions
│   └── graders/             Deterministic graders for each task
├── main.py                  FastAPI server
├── inference.py             Baseline agent script
├── openenv.yaml             OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```