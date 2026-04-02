# BankKYCAuditEnv (2025 Standard)

An advanced 2025-standard KYC/AML audit environment for testing frontier AI reasoning and agentic workflows. Built entirely on the strict **OpenEnv** specification framework.

## Domain & Motivation
The financial services sector heavily relies on rigid, rule-based KYC (Know Your Customer) systems. These are increasingly bypassed by synthetic identities and sophisticated money mule networks. AI Agents possess the nuanced reasoning capability to act as Tier-1 and Tier-2 Fraud analysts, but they require robust interactive environments to learn from rather than static datasets. 

**BankKYCAuditEnv** places the AI model directly in the seat of a Senior KYC Analyst. Instead of a simple "yes/no" based on text rules, the agent must iteratively interview the simulated customer, cross-reference documents with device-level signals (IPs, typing cadence), and call simulated graph-analysis tools to detect modern red flags. 

---

## Technical Specifications

### Action Space
The agent navigates a strictly typed `Action` schema requiring a target `customer_id` and specific context:
* `request_additional_documents`: Flags missing application materials.
* `verify_document_authenticity`: Triggers secure forensics on potential Deepfake IDs.
* `analyze_transaction_patterns`: Internally scores transaction logs for circular flow/mules.
* `check_watchlists`: Queries OFAC and PEP (Politically Exposed Persons) schemas.
* `interview_customer`: Multi-turn interaction injecting Q&A directly into the next state frame.
* `perform_risk_scoring`: Commits evidence towards a rigid risk tier.
* **Terminal Actions:** `approve`, `reject`, `escalate`, `freeze_account`

### Observation Space
Environment state strictly aligns to `openenv.core.Observation` wrapping a rich Pydantic `CustomerProfile`.
* **Personal Info**: Standard demographics (with deliberate intentional discrepancies).
* **Behavioral Signals**: Typing cadence, session averages (detects bot automation).
* **Device Signals**: Geolocation mismatches, Emulator flags, VPN usage.
* **Transaction History**: Time-series arrays simulating 90-day financial flow.
* **Interview Log**: Accumulated multi-turn conversation string state tracking.

---

## Task Difficulty Progression
1. **Task 1: Easy (Clean Customer)**  
   Features a standard minor discrepancy (a simple apartment number formatting typo). The agent must verify identity and use `approve` gracefully.
2. **Task 2: Medium (Crypto Mixing & High Velocity)**  
   A freelancer profile exhibiting high-frequency deposits immediately siphoned to offshore exchanges over a mismatched VPN IP. The agent must successfully utilize `interview_customer` to discover the missing source-of-funds. 
3. **Task 3: Hard (The Frankenstein Mule)**  
   Deepfake document artifacts + spoofed device hashes out of Cyprus + massive offshore transactions wiring cyclically back to the original funding account. Hitting the watchlist reveals a heavily sanctioned PEP link. Requires multi-hop logic terminating in an immediate `freeze_account`.

---

## Dense Reward Mechanics
Tasks are graded deterministically `0.0 - 1.0` utilizing `openenv.core` paradigms:
* **Final Correctness (40%)**: Making the right terminal call across fraud matrices. 
* **Proper Sequencing (30%)**: Validating against external watchlists *before* making the decision. Triggering graph algorithms specifically when large obfuscated transactions appear. 
* **Efficiency (15%)**: Completing the evaluation strictly within the optimal step margin. Unbounded interview loops kill efficiency scores.
* **Professional Quality (15%)**: Accurately pinning the internal risk tier mathematically via `perform_risk_scoring` before signing off. 

---

## Baseline Verification
Executing `inference.py` runs our robust, multi-hop baseline against all three configurations. The agent follows checking patterns dynamically interacting via OpenEnv endpoints. 

**Official Baseline Scores:**
* **Task 1 (Easy)**: `0.70`
* **Task 2 (Medium)**: `1.00`
* **Task 3 (Hard)**: `1.00`

## Setup & HF Spaces Deployment

### Local Testing
The environment is built utilizing Fast Dependency resolution (`uv`) and fully typed specifications.
```bash
# Generate lockfiles and install dependencies
uv lock
uv sync

# Validate the OpenEnv Spec
openenv validate . --verbose

# Run the inference baseline
python inference.py
```

### Hosting & Spaces Validation
We deploy via specialized HuggingFace Docker constraints native to the Space ecosystem `(port 8080)`:

```bash
uv run server
# or via Container
docker build -t bankkyc .
docker run -p 8080:8080 bankkyc
```