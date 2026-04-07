import os
import sys
import json
import re
import urllib.request
import urllib.error
import time
from typing import List, Optional
from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("GEMINI_API_KEY") 

client = OpenAI(api_key=API_KEY or "dummy_key_to_prevent_crash", base_url=API_BASE_URL)
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def debug(msg: str) -> None:
    """Prints to standard error so it doesn't break the hackathon's stdout regex parser."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def _http_request(url: str, method: str = "GET", json_data: Optional[dict] = None) -> dict:
    headers = {"Content-Type": "application/json"} if json_data else {}
    data = json.dumps(json_data).encode("utf-8") if json_data else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:
            raise e

SYSTEM_PROMPT = """You are an elite Anti-Money Laundering (AML) investigator AI.
You must output ONLY valid JSON. No preambles, no explanations, no markdown formatting.

CRITICAL INSTRUCTION: You must review your previous actions. DO NOT repeat the exact same discovery action on the same customer twice. 
Once you gather the data, immediately use a terminal action to make a decision.

IMPORTANT: Use the EXACT customer_id from the queue (e.g. CUST-3AF27F). Do NOT invent IDs.

Your output must EXACTLY match this schema:
{
    "action_type": "pull_document_dossier", 
    "target_customer_id": "<INSERT_ID_FROM_QUEUE>",
    "decision_reasoning": "Checking docs.",
    "start_date": "2025-01-01",
    "end_date": "2025-03-31",
    "flagged_transaction_ids": [],
    "flagged_document_ids": []
}
Valid action_types: pull_document_dossier, query_transactions, check_watchlists, pull_device_signals, interview_customer, approve, reject, escalate, freeze_account, file_sar.
"""

def extract_json_defensively(raw_text: str) -> dict:
    cleaned = re.sub(r'```json|```', '', raw_text).strip()
    try: return json.loads(cleaned)
    except json.JSONDecodeError: pass
    
    try:
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            first_pass = cleaned[start:end+1]
            try: return json.loads(first_pass)
            except json.JSONDecodeError:
                end = cleaned.rfind('}', 0, end)
                if end != -1: return json.loads(cleaned[start:end+1])
    except Exception: pass
    raise ValueError(f"FATAL: Model failed to output JSON. Raw:\n{raw_text}")

def run_baseline_agent(task_id="task1_easy"):
    env_url = os.getenv("ENV_URL", "http://localhost:7860")
    
    log_start(task=task_id, env="BankKYCAuditEnv", model=MODEL_NAME)
    
    try:
        reset_res = _http_request(f"{env_url}/reset?task_id={task_id}", method="POST", json_data={"task_id": task_id})
    except Exception as e:
        debug(f"FATAL: Could not connect to environment at {env_url}. Error: {e}")
        log_step(step=1, action="error", reward=0.0, done=True, error="connection_failed")
        log_end(task=task_id, success=False, steps=0, score=0.0, rewards=[])
        return 
        
    try:
        episode_id = reset_res["episode_id"]
        obs = reset_res["observation"]
        max_steps = obs.get("max_steps", 15)
    except Exception as e:
        debug(f"FATAL: Missing key in reset response - {e}")
        log_step(step=1, action="error", reward=0.0, done=True, error="invalid_reset_response")
        log_end(task=task_id, success=False, steps=0, score=0.0, rewards=[])
        return
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards = []
    
    for step in range(1, max_steps + 1):
        safe_queue = [
            {"customer_id": c.get("customer_id"), "status": c.get("status")}
            for c in obs.get("customer_queue", [])
            if not c.get("status", "").startswith("processed")
        ]
        raw_context = str(obs.get("investigation_context", ""))
        safe_context = raw_context[-3000:] if len(raw_context) > 3000 else raw_context

        trimmed_obs = {
            "queue_remaining": safe_queue,
            "investigation_context": safe_context,
            "message": obs.get("message", "")
        }
        messages.append({"role": "user", "content": f"Update:\n{json.dumps(trimmed_obs)}\nWhat is your next action?"})
        
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=800,
                    temperature=0.1 
                )
                break 
            except Exception as e:
                debug(f"⚠️ API Error: {str(e)}. Sleeping 15s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(15)
        
        if response is None:
            log_step(step=step, action="error", reward=0.0, done=True, error="LLM API Failure Exhausted")
            break 
            
        raw_output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw_output})
        
        try:
            action_payload = extract_json_defensively(raw_output)
            action_str = action_payload.get('action_type', 'unknown')
        except Exception as e:
            debug(f"🚨 JSON PARSE ERROR. Model output was:\n{raw_output}")
            action_payload = {"action_type": "error"}
            action_str = "json_parse_error"
            
        try:
            step_res = _http_request(f"{env_url}/step", method="POST", json_data={"episode_id": episode_id, "action": action_payload})
        except Exception as e:
            debug(f"🚨 Environment Step Error: {e}")
            log_step(step=step, action=action_str, reward=0.0, done=True, error="Environment unreachable")
            break
            
        error_msg = step_res.get('error')
        obs = step_res.get("observation", {})
        
        raw_reward = step_res.get("reward", 0.0)
        reward_val = float(raw_reward) if not isinstance(raw_reward, dict) else float(raw_reward.get("step_score", 0.0))
        
        done_val = step_res.get("done", False) or obs.get("message", "").find("All queue items processed") != -1
        
        rewards.append(reward_val)
        log_step(step=step, action=action_str, reward=reward_val, done=done_val, error=error_msg)
        
        if done_val or error_msg:
            break
            
        time.sleep(2) 

    final_score = 0.0
    try:
        final_score = _http_request(f"{env_url}/grade?episode_id={episode_id}", method="GET").get("score", 0.0)
    except Exception as e:
        debug(f"Grading Error: {e}")
        
    log_end(task=task_id, success=(final_score > 0.0), steps=len(rewards), score=final_score, rewards=rewards)

if __name__ == "__main__":
    if not API_KEY:
        debug("WARNING: API_KEY/HF_TOKEN is missing.")
    
    run_baseline_agent("task1_easy")
    run_baseline_agent("task2_medium")
    run_baseline_agent("task3_hard")