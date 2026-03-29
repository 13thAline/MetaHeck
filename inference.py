"""
inference.py — Baseline agent for KYC Audit Environment.
Uses OpenAI client to run an LLM against all 3 tasks.

Required environment variables:
  API_BASE_URL   e.g. https://api.openai.com/v1
  MODEL_NAME     e.g. gpt-4o-mini
  HF_TOKEN       your Hugging Face / API key
"""
import os
import json
import time
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = ["task1_doc_check", "task2_txn_analysis", "task3_network_fraud"]

SYSTEM_PROMPT = """You are a KYC/AML compliance analyst at a bank.
Your job is to review customer profiles and transaction histories, then take actions.

Available actions:
- clear_customer: Customer passes KYC, no issues found
- flag_for_review: Escalate to senior analyst, suspicious but not confirmed
- request_documents: Ask customer to submit specific missing/expired documents
- file_sar: File a Suspicious Activity Report (for confirmed suspicious activity)
- freeze_account: Block the account immediately (use after filing SAR)
- link_entities: Assert a connection between two customers (for network fraud)
- add_note: Add a case note without taking a final decision

Always respond with a valid JSON object in this exact format:
{
  "action_type": "<one of the actions above>",
  "customer_id": "<the customer ID>",
  "target_customer_id": "<only for link_entities>",
  "documents_requested": ["<list>", "<of>", "<docs>"],
  "reason": "<detailed explanation of your decision, min 30 words>",
  "risk_tier": "<low|medium|high|critical>"
}
Do not include any text outside the JSON object.
"""


def call_env(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    if method == "POST":
        resp = requests.post(url, json=payload, timeout=30)
    else:
        resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def format_customer_for_prompt(customer: dict) -> str:
    txn_summary = []
    for t in customer.get("transactions", [])[:10]:  # limit for context
        txn_summary.append(
            f"  - {t['date']}: {t['type']} ${t['amount_usd']:,.0f}"
            + (f" to/from {t['counterparty']}" if t.get('counterparty') else "")
            + (f" ({t['country']})" if t.get('country') else "")
        )

    return f"""
Customer ID: {customer['customer_id']}
Name: {customer['name']} | DOB: {customer['dob']} | Nationality: {customer['nationality']}
Occupation: {customer['occupation']} | Annual Income: ${customer['annual_income_usd']:,.0f}
Account opened: {customer['account_open_date']}
Documents present: {', '.join(customer.get('documents_present', [])) or 'None'}
Documents expired: {', '.join(customer.get('documents_expired', [])) or 'None'}
Documents missing: {', '.join(customer.get('documents_missing', [])) or 'None'}
PEP flag: {customer.get('pep_flag', False)} | Sanctions: {customer.get('sanctions_flag', False)}
Adverse media: {'; '.join(customer.get('adverse_media', [])) or 'None'}
Linked entity IDs: {', '.join(customer.get('linked_entity_ids', [])) or 'None'}
Address: {customer.get('address', 'N/A')}
Transactions ({len(customer.get('transactions', []))} total, showing first 10):
{chr(10).join(txn_summary) if txn_summary else '  No transactions'}
"""


def agent_decide(observation: dict, customer: dict) -> dict:
    """Call the LLM to decide what action to take for a customer."""
    prompt = f"""Task: {observation.get('task_description', '')}

{observation.get('message', '')}

Review this customer and decide what action to take:
{format_customer_for_prompt(customer)}

All customers in this episode: {[c['customer_id'] for c in observation['customers']]}
Actions already taken: {json.dumps([a['customer_id'] + ':' + a['action_type'] 
    for a in observation.get('completed_actions', [])], indent=0)}

What is your KYC decision for customer {customer['customer_id']}?
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


def run_task(task_id: str) -> dict:
    """Run a full episode for a given task. Returns graded result."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print("="*60)

    # Reset environment
    reset_resp = call_env("/reset", "POST", {"task_id": task_id})
    episode_id = reset_resp["episode_id"]
    obs = reset_resp["observation"]

    print(f"Episode ID: {episode_id}")
    print(f"Customers to review: {len(obs['customers'])}")
    print(f"Max steps: {obs['max_steps']}")

    step_count = 0
    done = False
    step_scores = []

    while not done and step_count < obs["max_steps"]:
        # Find next unreviewed customer
        reviewed = {
            a["customer_id"] for a in obs.get("completed_actions", [])
            if a.get("action_type") not in ["link_entities", "add_note"]
        }
        remaining = [c for c in obs["customers"] if c["customer_id"] not in reviewed]

        if not remaining:
            print("All customers reviewed.")
            break

        customer = remaining[0]
        print(f"\nStep {step_count+1}: Reviewing {customer['customer_id']} ({customer['name']})")

        try:
            action_dict = agent_decide(obs, customer)
            print(f"  Action: {action_dict.get('action_type')} | "
                  f"Risk: {action_dict.get('risk_tier')} | "
                  f"Reason: {action_dict.get('reason', '')[:80]}...")

            step_resp = call_env("/step", "POST", {
                "action": action_dict,
                "episode_id": episode_id,
            })
            obs = step_resp["observation"]
            reward = step_resp["reward"]
            done = step_resp["done"]
            step_scores.append(reward["step_score"])

            print(f"  Step score: {reward['step_score']:.3f} | "
                  f"Feedback: {reward['feedback'][:100]}")

        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}. Skipping.")
        except Exception as e:
            print(f"  Error: {e}. Skipping.")

        step_count += 1
        time.sleep(0.5)  # rate limit courtesy

    # Get final graded score
    grade_resp = call_env(f"/grade?episode_id={episode_id}")
    final_score = grade_resp.get("score", 0.0)
    breakdown = grade_resp.get("breakdown", {})
    feedback = grade_resp.get("feedback", "")

    print(f"\nFinal graded score: {final_score:.4f}")
    print(f"Breakdown: {json.dumps(breakdown, indent=2)}")
    print(f"Feedback: {feedback[:200]}")

    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "steps_taken": step_count,
        "final_score": final_score,
        "avg_step_score": round(sum(step_scores) / max(len(step_scores), 1), 4),
        "breakdown": breakdown,
    }


def main():
    print("KYC Audit Environment — Baseline Inference")
    print(f"Model: {MODEL_NAME} | Base URL: {API_BASE_URL}")

    # Wait for server to be ready
    for attempt in range(10):
        try:
            health = call_env("/health")
            print(f"Server ready: {health}")
            break
        except Exception:
            print(f"Waiting for server... ({attempt+1}/10)")
            time.sleep(3)
    else:
        print("Server not available. Exiting.")
        return

    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)
        time.sleep(1)

    # Summary
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['task_id']:30s}  score={r['final_score']:.4f}  steps={r['steps_taken']}")

    overall = sum(r["final_score"] for r in results) / len(results)
    print(f"\nOverall average score: {overall:.4f}")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "overall_score": overall,
        }, f, indent=2)
    print("Results saved to baseline_results.json")


if __name__ == "__main__":
    main()