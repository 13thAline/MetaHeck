import os
import sys
import subprocess
import json
import re
import urllib.request
import urllib.error
import time
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    sys.stderr.write("[WARNING] 'openai' missing. Force-installing client dependencies for the external runner...\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.50.0", "python-dotenv", "requests"])
    from openai import OpenAI


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("GEMINI_API_KEY") 
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "kyc-audit-env")

client = OpenAI(api_key=API_KEY or "dummy_key_to_prevent_crash", base_url=API_BASE_URL)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def debug(msg: str) -> None:
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

class RemoteEnv:
    def __init__(self, env_url: str):
        self.env_url = env_url
        self.episode_id = None

    def reset(self, task_id: str):
        res = _http_request(f"{self.env_url}/reset?task_id={task_id}", method="POST", json_data={"task_id": task_id})
        self.episode_id = res.get("episode_id")
        return res

    def step(self, action_payload: dict):
        return _http_request(f"{self.env_url}/step", method="POST", json_data={"episode_id": self.episode_id, "action": action_payload})

    def grade(self):
        return _http_request(f"{self.env_url}/grade?episode_id={self.episode_id}", method="GET")

    def close(self):
        if self.episode_id:
            try:
                _http_request(f"{self.env_url}/close", method="POST", json_data={"episode_id": self.episode_id})
            except Exception as e:
                debug(f"[DEBUG] env.close() error (container cleanup): {e}")

SYSTEM_PROMPT = """You are an AML investigator AI. Output exactly ONE action per turn as a single JSON object.
NEVER output an array. NEVER output multiple actions. No markdown, no explanation.
DO NOT repeat a discovery action on the same customer. Use EXACT customer_ids from the queue.

DISCOVERY (only need action_type + target_customer_id):
{"action_type":"pull_document_dossier","target_customer_id":"CUST-XXXX"}
{"action_type":"query_transactions","target_customer_id":"CUST-XXXX","start_date":"2024-01-01","end_date":"2025-12-31"}
{"action_type":"check_watchlists","target_customer_id":"CUST-XXXX"}
{"action_type":"pull_device_signals","target_customer_id":"CUST-XXXX"}
{"action_type":"interview_customer","target_customer_id":"CUST-XXXX","interview_question":"Explain source of funds"}

TERMINAL (add regulatory_typology + confidence_score):
{"action_type":"approve","target_customer_id":"CUST-XXXX","regulatory_typology":["CLEAN_PROFILE"],"confidence_score":0.85}
{"action_type":"freeze_account","target_customer_id":"CUST-XXXX","regulatory_typology":["STRUCTURING_314A"],"confidence_score":0.9,"flagged_transaction_ids":["TXN-XXXX"]}

Typology codes: STRUCTURING_314A, LAYERING_FATF_02, SHELL_COMPANY_FATF_04, CIRCULAR_TRANSACTION, SMURFING, RAPID_MOVEMENT, ADDRESS_MISMATCH, PEP_UNDISCLOSED, BURST_VELOCITY, DORMANT_ACTIVATION, CLEAN_PROFILE, DEEPFAKE_DOCUMENT, SANCTIONS_EVASION, BENEFICIAL_OWNER_CONCEALMENT, SYNTHETIC_IDENTITY, TRADE_BASED_ML
Terminal action_types: approve, reject, escalate, freeze_account, file_sar
Only include fields relevant to your action. Omit null/empty fields.
"""

def extract_json_defensively(raw_text: str) -> dict:
    import re
    # Remove `<think>...</think>` tags to support models that output reasoning
    text_no_think = re.sub(r'<THINK>.*?</THINK>', '', raw_text, flags=re.IGNORECASE | re.DOTALL)
    text_no_think = re.sub(r'<THINK>.*', '', text_no_think, flags=re.IGNORECASE | re.DOTALL)
    
    cleaned = re.sub(r'```json|```', '', text_no_think).strip()
    # Handle arrays: model sometimes outputs [{...}, {...}] — take the first element
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], dict):
                return parsed[0]
            raise json.JSONDecodeError("Empty array", cleaned, 0)
        return parsed
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

    # --- Truncated JSON repair (model ran out of tokens) ---
    if '{' in cleaned and '}' not in cleaned:
        fragment = cleaned[cleaned.find('{'):]
        # Strip the last incomplete key-value pair
        last_comma = fragment.rfind(',')
        last_colon = fragment.rfind(':')
        if last_comma > 0 and last_comma > last_colon:
            fragment = fragment[:last_comma]
        elif last_colon > 0:
            # Value was being written — strip from the last complete key
            last_complete_comma = fragment.rfind(',', 0, last_colon)
            if last_complete_comma > 0:
                fragment = fragment[:last_complete_comma]
        fragment = fragment.rstrip(' ,\n\r\t') + '}'
        try: return json.loads(fragment)
        except json.JSONDecodeError: pass

    raise ValueError(f"FATAL: Model failed to output JSON. Raw:\n{raw_text}")

def run_baseline_agent(task_id="task1_easy"):
    env_url = os.getenv("ENV_URL", "http://localhost:7860")
    env = RemoteEnv(env_url)
    
    log_start(task=task_id, env="BankKYCAuditEnv", model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    success = False
    final_score = 0.0
    
    try:
        try:
            reset_res = env.reset(task_id)
        except Exception as e:
            debug(f"FATAL: Could not connect to environment at {env_url}. Error: {e}")
            log_step(step=1, action="error", reward=0.0, done=True, error="connection_failed")
            return 
            
        try:
            obs = reset_res["observation"]
            max_steps = obs.get("max_steps", 15)
        except Exception as e:
            debug(f"FATAL: Missing key in reset response - {e}")
            log_step(step=1, action="error", reward=0.0, done=True, error="invalid_reset_response")
            return
        
        
        for step in range(1, max_steps + 1):
            steps_taken = step
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
                "message": obs.get("message", ""),
                "completed_actions_so_far": obs.get("completed_actions", []) 
            }
            
            # 2. Rebuild the prompt completely fresh. Do NOT append.
            step_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Update:\n{json.dumps(trimmed_obs)}\nWhat is your next action?"}
            ]
            
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=step_messages,  # <--- Use the fresh 2-message array
                        response_format={"type": "json_object"},
                        max_tokens=1500,         # <--- Increased token ceiling
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
            
            try:
                action_payload = extract_json_defensively(raw_output)
                action_str = action_payload.get('action_type', 'unknown')
            except Exception as e:
                debug(f"🚨 JSON PARSE ERROR. Model output was:\n{raw_output}")
                action_payload = {"action_type": "error"}
                action_str = "json_parse_error"
                
            try:
                step_res = env.step(action_payload)
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

        try:
            final_score = float(env.grade().get("score", 0.0))
        except Exception as e:
            debug(f"Grading Error: {e}")
            final_score = 0.0
        
        final_score = min(max(final_score, 0.001), 0.999)
        success = (final_score > 0.0)
    
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    if not API_KEY:
        debug("WARNING: API_KEY/HF_TOKEN is missing.")
    
    run_baseline_agent("task1_easy")
    run_baseline_agent("task2_medium")
    run_baseline_agent("task3_hard")