import os
import json
import re
import requests
from openai import OpenAI

# 1. HACKATHON COMPLIANCE: Using official OpenAI client but routing to free HF endpoint
client = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/v1", 
    api_key=os.environ.get("HF_TOKEN", "hf_placeholder")
)

# You can swap this to mistralai/Mixtral-8x7B-Instruct-v0.1 or whatever the HF free tier supports today
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct:together"

SYSTEM_PROMPT = """You are an elite Anti-Money Laundering (AML) investigator AI.
You must output ONLY valid JSON. No preambles, no explanations, no markdown formatting.

Your output must EXACTLY match this schema:
{
    "action_type": "pull_document_dossier", 
    "target_customer_id": "CUST-001",
    "decision_reasoning": "I need to check the documents before making a decision.",
    "start_date": null,
    "end_date": null,
    "flagged_transaction_ids": [],
    "flagged_document_ids": []
}

Valid action_types: pull_document_dossier, query_transactions, check_watchlists, approve, reject, freeze_account, file_sar.
If using query_transactions, start_date and end_date are REQUIRED (YYYY-MM-DD).
If using terminal actions (freeze_account, file_sar), you MUST provide flagged_transaction_ids or flagged_document_ids.
"""

def extract_json_defensively(raw_text: str) -> dict:
    """Brute-force extracts JSON from hallucinatory LLM outputs."""
    try:
        # Find everything between the first { and last }
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        # Fallback if no regex match but it happens to be valid
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"FATAL: Model completely failed to output JSON. Raw output:\n{raw_text}")

def run_baseline_agent(task_id="task1_easy"):
    # Target the Docker container running locally
    env_url = "http://localhost:8080"
    
    print(f"--- INITIALIZING OPENENV: {task_id} ---")
    reset_res = requests.get(f"{env_url}/reset?task_id={task_id}").json()
    
    episode_id = reset_res["episode_id"]
    obs = reset_res["observation"]
    max_steps = obs.get("max_steps", 15)
    
    for step in range(max_steps):
        print(f"\n[STEP {step + 1}/{max_steps}]")
        
        prompt = f"Current Observation:\n{json.dumps(obs, indent=2)}\n\nWhat is your next action?"
        
        try:
            # 2. THE API CALL: Removed the strict JSON mode that breaks HF endpoints
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1 # Lock it down to prevent creative hallucinations
            )
            
            raw_output = response.choices[0].message.content
            action_payload = extract_json_defensively(raw_output)
            
            print(f"Agent Action: {action_payload.get('action_type')} on {action_payload.get('target_customer_id')}")
            
            # 3. EXECUTE ACTION
            step_res = requests.post(f"{env_url}/step", json={"episode_id": episode_id, "action": action_payload}).json()
            
            if "error" in step_res:
                print(f"Environment Error: {step_res['error']}")
                break
                
            obs = step_res["observation"]
            print(f"Env Message: {obs.get('message')}")
            print(f"Reward: {step_res.get('reward')}")
            
            if step_res.get("done", False) or obs.get("message", "").find("All queue items processed") != -1:
                print("\n[EPISODE COMPLETE]")
                break
                
        except Exception as e:
            print(f"\n[AGENT CRASHED]: {str(e)}")
            break

    # 4. GET FINAL GRADE
    print("\n--- FINAL EVALUATION ---")
    final_score = 0.0
    try:
        grade_res = requests.get(f"{env_url}/grade?episode_id={episode_id}").json()
        final_score = grade_res.get("score", 0.0)
        print(f"Score for {task_id}: {final_score}")
    except Exception as e:
        print(f"Grading Error: {e}")
        
    return final_score

if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN environment variable is not set. Inference might fail.")
        
    print("Starting Hackathon Baseline Evaluation...\n")
    
    score1 = run_baseline_agent("task1_easy")
    score2 = run_baseline_agent("task2_medium")
    score3 = run_baseline_agent("task3_hard")
    
    print("\n==============================")
    print("    BASELINE INFERENCE SCORES   ")
    print("==============================")
    print(f"Task 1 (Easy)  : {score1}")
    print(f"Task 2 (Medium): {score2}")
    print(f"Task 3 (Hard)  : {score3}")
    print("==============================")