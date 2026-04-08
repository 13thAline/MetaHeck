---
title: BankKYCAuditEnv
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
short_description: OpenEnv KYC/AML Fraud Detection Environment
---
<div align="center">

# 🏦 BankKYCAuditEnv

### A Procedurally Generated KYC & AML Compliance Environment for AI Agents

> AI agents act as **Senior Fraud Analysts** — investigating customer profiles, querying transaction ledgers, scanning watchlists, and making deterministic compliance decisions against dynamically generated ground truth.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen?style=flat-square)]()
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)]()
[![License](https://img.shields.io/badge/license-MIT-grey?style=flat-square)]()

</div>

---

## 📑 Table of Contents

- [Why BankKYCAuditEnv?](#-why-bankkycauditenv)
- [Architecture](#-architecture)
- [Action Space](#-action-space)
- [Observation Space](#-observation-space)
- [Tasks & Difficulty Progression](#-tasks--difficulty-progression)
- [Reward Function & Grading](#-reward-function--grading)
- [Procedural Data Engine](#-procedural-data-engine)
- [Baseline Inference Agent](#-baseline-inference-agent)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Hackathon Compliance](#-hackathon-compliance)
- [Contributing](#-contributing)

---

## 💡 Why BankKYCAuditEnv?

Financial institutions process **millions** of KYC reviews annually. Current rule-based systems are increasingly bypassed by:

| Threat Vector | Description |
|---|---|
| **Synthetic Identity Fraud** | Frankenstein IDs stitched from real + fabricated data |
| **Structuring** | Deposits kept just under $10,000 to avoid CTR filing |
| **Money Mule Networks** | Chain-layered transfers through shell companies |
| **Circular Transactions** | Funds looping through offshore accounts back to origin |

Human analysts spend **20–40 minutes per case**. AI agents capable of multi-step financial reasoning could dramatically reduce this — but they need **interactive training environments**, not static datasets.

**BankKYCAuditEnv** places the AI directly in the role of a Senior KYC Analyst. The agent receives a queue of customers, must actively investigate each one using discovery actions, and then make a terminal compliance decision backed by flagged evidence.

### 🔑 The Differentiator: Procedural Synthetic Data

Most AI agent benchmarks rely on static, easily memorized datasets. BankKYCAuditEnv solves this entirely:

- Every `reset()` dynamically generates **new, randomized transaction ledgers, customer profiles, and deeply injected fraud patterns** with secretly tracked ground truth
- The environment **cannot be memorized or gamed**
- It serves as an **infinitely reproducible RL/DPO fine-tuning factory** for training and evaluating true financial reasoning in LLMs

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   inference.py                      │
│            (OpenAI client baseline agent)            │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (POST /reset, /step, GET /grade)
                       ▼
┌─────────────────────────────────────────────────────┐
│                  server/app.py                      │
│                 (FastAPI on :7860)                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              env/environment.py                     │
│           BankKYCAuditEnv (OpenEnv core)             │
│                                                     │
│  reset() ──► data_engine.generate_episode()          │
│  step()  ──► queries manifest.database               │
│  grade() ──► passes manifest.ground_truth            │
│              to task-specific grader                 │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   grader1.py    grader2.py    grader3.py
   (F1 score)   (structuring) (network fraud)
```

---

## 🎮 Action Space

The agent navigates a strictly typed `Action` (Pydantic model). Every action requires a `target_customer_id` and `decision_reasoning`.

### Discovery Actions (information gathering)

| Action | Purpose | Required Fields |
|---|---|---|
| `pull_document_dossier` | Retrieve ID documents and address verification | — |
| `query_transactions` | Pull the customer's transaction ledger | `start_date`, `end_date` |
| `check_watchlists` | Query sanctions lists, PEP databases | — |
| `pull_device_signals` | Get IP address, VPN detection, device fingerprint | — |
| `interview_customer` | Ask the customer a direct question | `interview_question` |

### Terminal Actions (compliance decisions)

| Action | When to Use |
|---|---|
| `approve` | Customer is clean — proceed with onboarding |
| `reject` | Customer fails basic KYC — deny account |
| `escalate` | Suspicious indicators warrant Tier 2 review |
| `freeze_account` | Strong fraud signals — immediate account freeze |
| `file_sar` | File a Suspicious Activity Report (network fraud, layering) |

### Evidence Submission

Terminal actions accept two critical evidence fields scored by the grader:

- **`flagged_transaction_ids: List[str]`** — exact transaction IDs the agent believes are fraudulent
- **`flagged_document_ids: List[str]`** — document types with issues (e.g., `"utility_bill"`, `"passport"`)

---

## 👁 Observation Space

The observation is a typed Pydantic model extending `openenv.core.Observation`:

```python
class Observation(BaseObservation):
    task_id: str              # Current task identifier
    episode_id: str           # Unique episode identifier
    step: int                 # Current step number
    max_steps: int            # Maximum allowed steps
    customer_queue: List[CustomerProfile]  # Customers pending review
    investigation_context: str  # Data returned by the last discovery action
    available_actions: List[str]  # Valid action types
    completed_actions: List[Dict]  # History of all actions taken
    task_description: str     # Human-readable task brief
    message: str              # System feedback from the last action
```

### Customer Profile (initial view)

The queue shows only basic info — the agent **must actively query** to see documents, transactions, and device signals:

```python
class CustomerProfile(BaseModel):
    customer_id: str          # e.g., "CUST-3AF27F"
    status: str               # "pending_review" or "processed_*"
    personal_info: Dict       # name, DOB, address, occupation, PEP status
```

> **Design choice:** The observation space is deliberately **blind**. The agent sees names and addresses but must invoke discovery actions to populate the `investigation_context` with actual evidence. This models real-world analyst workflows where data must be actively retrieved from siloed systems.

---

## 📈 Tasks & Difficulty Progression

### Task 1 — Easy (`task1_easy`)

| | |
|---|---|
| **Scenario** | 2–3 customers with minor document discrepancies |
| **Fraud Pattern** | Address mismatch between ID and utility bill (P.O. Box vs. physical address) |
| **Agent Workflow** | Pull documents → spot the mismatch → escalate the flagged customer, approve the clean one |
| **Max Steps** | 15 |
| **Expected Score** | 0.75 – 1.00 |

### Task 2 — Medium (`task2_medium`)

| | |
|---|---|
| **Scenario** | 3–5 customers mixing clean profiles with structuring fraud |
| **Fraud Pattern** | Multiple deposits between $9,000–$9,999 (just under the $10,000 CTR threshold), VPN usage, expired documents, rapid wire transfers |
| **Agent Workflow** | Query transactions → identify structuring pattern → extract exact `TXN-*` IDs → freeze account |
| **Max Steps** | 20 |
| **Expected Score** | 0.50 – 0.85 |

### Task 3 — Hard (`task3_hard`)

| | |
|---|---|
| **Scenario** | 5–8 customers including shell companies, mule chains, and clean accounts |
| **Fraud Patterns** | Shell company fund distribution, chain-layering through intermediaries, circular transactions, spoofed devices, PEP/sanctions hits, deepfake document indicators |
| **Agent Workflow** | Investigate all customers → discover network links via shared IPs/addresses → file SARs on the fraud ring → approve clean customers without false positives |
| **Max Steps** | 30 |
| **Expected Score** | 0.15 – 0.40 |

---

## 🏆 Reward Function & Grading

### Dense Reward Signals

Every `step()` returns a per-step reward (not just end-of-episode):

| Signal | Reward | Trigger |
|---|---|---|
| New data source queried | **+0.05** | First time pulling docs/txns/watchlists/device for a customer |
| Interview conducted | **+0.03** | First interview per customer |
| Correct terminal decision | **+0.10** | Decision matches ground truth |
| Blind decision (no investigation) | **−0.30** | Terminal action without any prior discovery |
| Missing required fields | **−0.05** | e.g., `query_transactions` without dates |
| Max steps exceeded | **−0.20** | Episode ends with unprocessed customers |

### Final Grading (0.0 – 1.0)

Each task uses a dedicated deterministic grader:

<details>
<summary><strong>Grader 1 — Task 1 (Easy)</strong></summary>

| Component | Weight | Metric |
|---|---|---|
| Correct decision | 40% | Exact match (approve / escalate / reject / freeze / file_sar) |
| Evidence quality | 60% | F1 score of flagged transaction & document IDs vs. ground truth |

</details>

<details>
<summary><strong>Grader 2 — Task 2 (Medium)</strong></summary>

| Component | Weight | Metric |
|---|---|---|
| Decision accuracy | 40% | Exact match with partial credit for close calls |
| Risk tier | 20% | Correct risk classification (low / medium / high / critical) |
| Reasoning quality | 30% | Keyword hits in `decision_reasoning` |
| Red flag identification | 10% | Matching known fraud indicators |

</details>

<details>
<summary><strong>Grader 3 — Task 3 (Hard)</strong></summary>

| Component | Weight | Metric |
|---|---|---|
| Decision accuracy | 35% | With partial credit |
| Risk tier | 15% | Correct classification |
| Evidence reasoning | 25% | Keyword analysis |
| Network linking | 20% | F1 of discovered entity links vs. true network |
| SAR filing | 5% | Filed SAR when required |
| False positive penalty | −0.05 | Per clean customer wrongly flagged |

</details>

All graders produce **deterministic, reproducible** scores in the `[0.0, 1.0]` range.

---

## ⚙ Procedural Data Engine

Every `reset()` generates a **completely new episode** via `env/data_engine.py`:

- **Randomized customer profiles** — names, addresses, occupations, DOBs from curated pools
- **Unique transaction IDs** — `TXN-{hex}` format, never repeated across episodes
- **Injected fraud patterns** — structuring amounts, chain transfers, shell company flows
- **Dynamic ground truth** — exact `transaction_ids` marked as fraudulent are secretly logged at generation time and passed to the grader

### Reproducibility

Pass a `seed` parameter or set the `EPISODE_SEED` environment variable for deterministic episodes:

```bash
# Randomized (default) — different customers every time
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task2_medium"}'

# Deterministic — set EPISODE_SEED for identical episodes
export EPISODE_SEED=42
```

### Memory Efficiency

Each episode generates ~50–200 transactions in-memory. No persistent database required. Well within the 8 GB RAM constraint.

---

## 🤖 Baseline Inference Agent

`inference.py` is the official baseline agent:

1. Uses the **OpenAI Python client** to call the configured LLM (default: `gemini-2.5-flash`)
2. Reads credentials from environment variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
3. Runs all 3 tasks sequentially: `task1_easy` → `task2_medium` → `task3_hard`
4. Emits **strict stdout logs** matching the hackathon format:

```
[START] task=task1_easy env=BankKYCAuditEnv model=gemini-2.5-flash
[STEP] step=1 action=pull_document_dossier reward=0.05 done=false error=null
[STEP] step=2 action=escalate reward=0.10 done=true error=null
[END] success=true steps=2 score=0.850 rewards=0.05,0.10
```

> All debug output goes to `stderr` — stdout is reserved exclusively for `[START]`/`[STEP]`/`[END]` lines.

### Verified Baseline Scores

The deterministic graders successfully evaluate the agent against procedurally generated ground truth. Baseline scores demonstrate the intended difficulty curve:

| Task | Score |
|---|---|
| Task 1 (Easy) | **0.667** |
| Task 2 (Medium) | **0.492** |
| Task 3 (Hard) | **0.301** |

**Runtime:** Full 3-task run completes in under 20 minutes on 2 vCPU / 8 GB RAM.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** — fast dependency manager
- **Docker** — for containerized deployment

### Local Development

```bash
# Clone the repository
git clone https://github.com/13thAline/MetaHeck.git
cd MetaHeck

# Install dependencies
uv lock
uv sync

# Validate OpenEnv spec
openenv validate . --verbose

# Start the environment server
uv run python -m server.app
# → Server runs on http://localhost:7860
```

### Run the Baseline Agent

```bash
# In a separate terminal (server must be running)
cp .env.example .env
# Edit .env with your API credentials

python inference.py
```

### Docker Deployment

```bash
# Build the image
docker build -t kyc-audit-env .

# Run the container
docker run -p 7860:7860 --env-file .env kyc-audit-env

# Verify the server is live
curl http://localhost:7860/health

# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_easy"}'
```

### Run Tests

```bash
uv run python test_grader.py
```

This validates:

- **Reproducibility** — same seed → same episode
- **Data integrity** — all ground truth IDs exist in generated data
- **Grader correctness** — perfect agent scores high, sloppy agent scores low

### Pre-Submission Validation

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space
```

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | ✅ | LLM API endpoint (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`) |
| `MODEL_NAME` | ✅ | Model identifier (e.g., `gemini-2.5-flash`) |
| `HF_TOKEN` | ✅ | Hugging Face / API authentication token |
| `EPISODE_SEED` | ❌ | Fixed RNG seed for reproducible episodes (debugging) |
| `TRAJECTORY_LOG_DIR` | ❌ | Directory for state-action-reward JSONL logs (fine-tuning data) |

See [`.env.example`](.env.example) for a ready-to-use template.

---

## 📁 Project Structure

```
MetaHeck/
├── inference.py              # Baseline agent (OpenAI client, strict stdout logs)
├── openenv.yaml              # OpenEnv specification metadata
├── Dockerfile                # Container config (python:3.10-slim + uv)
├── pyproject.toml            # Dependencies and project metadata
├── requirements.txt          # Pip-compatible dependency list
├── .env.example              # Environment variable template
├── validate-submission.sh    # Pre-submission validation script (3-step checker)
├── test_grader.py            # Grader test suite (5 tests across 30 episodes)
│
├── server/
│   ├── __init__.py
│   └── app.py                # FastAPI server (/reset, /step, /state, /grade, /tasks, /health, /close)
│
└── env/
    ├── __init__.py
    ├── models.py              # Pydantic models (Action, Observation, EnvironmentState, EpisodeManifest)
    ├── environment.py         # OpenEnv state machine (BankKYCAuditEnv)
    ├── data_engine.py         # Procedural data generator (seeded RNG, fraud injection)
    └── graders/
        ├── grader1.py         # Task 1: F1 evidence scoring
        ├── grader2.py         # Task 2: Structuring detection + risk tier + reasoning
        └── grader3.py         # Task 3: Network fraud + entity linking + SAR filing
```

---

## 📡 API Reference

All endpoints are served on port **7860**.

### `POST /reset`

Start a new episode. Also accepts `GET` for validator compatibility.

```jsonc
// Request
{ "task_id": "task2_medium", "episode_id": "optional-custom-id" }

// Response
{
  "episode_id": "uuid",
  "observation": { /* Observation model */ }
}
```

### `POST /step`

Submit an action and receive the next observation.

```jsonc
// Request
{
  "episode_id": "uuid",
  "action": {
    "action_type": "query_transactions",
    "target_customer_id": "CUST-3AF27F",
    "decision_reasoning": "Checking for structuring patterns",
    "start_date": "2025-01-01",
    "end_date": "2025-03-31",
    "flagged_transaction_ids": [],
    "flagged_document_ids": []
  }
}

// Response
{
  "observation": { /* Observation model */ },
  "reward": 0.05,
  "done": false,
  "info": {}
}
```

### `GET /grade?episode_id=uuid`

Run the deterministic grader and return the final episode score.

```json
{ "episode_id": "uuid", "score": 0.85 }
```

### `GET /state?episode_id=uuid`

Return the full internal environment state for the given episode.

### `GET /tasks`

List all available tasks with difficulty levels and step limits.

### `GET /health`

Health check endpoint. Returns `{"status": "ok"}`.

### `POST /close`

Clean up and delete a session from memory.

---

## ✅ Hackathon Compliance

| Requirement | Status |
|---|---|
| Real-world task simulation (KYC/AML fraud detection) | ✅ |
| Full OpenEnv spec (`step()` / `reset()` / `state()`, typed models) | ✅ |
| `openenv.yaml` with metadata | ✅ |
| 3 tasks with difficulty progression (easy → medium → hard) | ✅ |
| Deterministic graders producing scores 0.0–1.0 | ✅ |
| Dense reward function with partial progress signals | ✅ |
| Penalties for undesirable behavior (blind decisions, timeouts) | ✅ |
| Baseline `inference.py` using OpenAI client | ✅ |
| Strict `[START]`/`[STEP]`/`[END]` stdout format | ✅ |
| `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars | ✅ |
| Working `Dockerfile` (builds and runs) | ✅ |
| Runs on 2 vCPU / 8 GB RAM, <20 min for 3 tasks | ✅ |
| Deployed to Hugging Face Spaces | ✅ |

---

<div align="center">

**Built for the [OpenEnv Hackathon 2026](https://huggingface.co/) by Team HelloAI** 🚀

</div>