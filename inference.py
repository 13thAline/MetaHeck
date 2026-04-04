import os
import json
import re
import requests
import time
from openai import OpenAI

# Google AI Studio routing
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """You are an elite Anti-Money Laundering (AML) investigator AI.
You must output ONLY valid JSON. No preambles, no explanations, no markdown formatting.

CRITICAL INSTRUCTION: You must review your previous actions. DO NOT repeat the exact same discovery action (like pull_document_dossier) on the same customer twice. 
Once you gather the data, immediately use a terminal action to make a decision.

Your output must EXACTLY match this schema:
{
    "action_type": "pull_document_dossier", 
    "target_customer_id": "CUST-001",
    "decision_reasoning": "I need to check the documents before making a decision.",
    "start_date": "2025-01-01",
    "end_date": "2025-02-28",
    "flagged_transaction_ids": [],
    "flagged_document_ids": []
}

Valid action_types: pull_document_dossier, query_transactions, check_watchlists, approve, reject, freeze_account, file_sar.
"""

def extract_json_defensively(raw_text: str) -> dict:
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"FATAL: Model failed to output JSON. Raw:\n{raw_text}")

def run_baseline_agent(task_id="task1_easy"):
    env_url = "http://localhost:8080"
    print(f"\n--- INITIALIZING OPENENV: {task_id} ---")
    reset_res = requests.get(f"{env_url}/reset?task_id={task_id}").json()
    
    episode_id = reset_res["episode_id"]
    obs = reset_res["observation"]
    max_steps = obs.get("max_steps", 15)
    
    # MEMORY FIX: We maintain a running conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for step in range(max_steps):
        print(f"\n[STEP {step + 1}/{max_steps}]")
        
        # Add the current state to the agent's memory
        prompt = f"Current Environment State:\n{json.dumps(obs, indent=2)}\n\nWhat is your next action?"
        messages.append({"role": "user", "content": prompt})
        
        # RATE LIMIT FIX: Automatic Exponential Backoff
        max_retries = 3
        response = None # <--- Initialize this so Python doesn't crash if all retries fail
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=400,
                    temperature=0.1 
                )
                break # Success, break out of retry loop
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    print(f"⚠️ Rate limit hit. Google wants a cooldown. Sleeping for 40 seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(40)
                else:
                    raise e
        
        # If we exhausted all 3 retries and still have no response, the API is hard-blocking us.
        if response is None:
            print(f"🚨 FATAL: Exhausted all retries. The API is hard-blocking your account. Ending episode early.")
            break 
        
        raw_output = response.choices[0].message.content
        action_payload = extract_json_defensively(raw_output)
        
        # Add the agent's action back into its memory so it doesn't loop
        messages.append({"role": "assistant", "content": raw_output})
        
        print(f"Agent Action: {action_payload.get('action_type')} on {action_payload.get('target_customer_id')}")
        
        # Execute Action
        step_res = requests.post(f"{env_url}/step", json={"episode_id": episode_id, "action": action_payload}).json()
        if "error" in step_res:
            print(f"Environment Error: {step_res['error']}")
            break
            
        obs = step_res["observation"]
        print(f"Env Message: {obs.get('message')}")
        print(f"Reward: {step_res.get('reward')}")
        
        if step_res.get("done", False) or obs.get("message", "").find("All queue items processed") != -1:
            print("[EPISODE COMPLETE]")
            break
            
        time.sleep(2) # Small standard buffer

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
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY is not set.")
        
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