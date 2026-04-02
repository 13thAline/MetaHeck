import os
import json
import requests
import time
from openai import OpenAI
from typing import Dict, Any

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class InferenceAgent:
    def __init__(self, api_base: str, model_name: str, api_key: str):
        self.api_base = api_base
        self.model_name = model_name
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.env_url = "http://localhost:8080"
        
        self.system_prompt = """You are a Senior KYC Analyst and Fraud Investigator for a digital bank.
Your goal is to audit customer onboarding and periodic review cases and assign a final decision.
Use ALL available evidence (documents, device signals, external watchlists, and behavioral signals).
Crucially, you MUST use the following tools in a logical order before making a final decision:
1. `check_watchlists`: First, always check for PEPs and OFAC sanctions.
2. `analyze_transaction_patterns`: Second, detect circular or mule behaviors if transactions look large or suspicious.
3. `interview_customer`: Third, if there is missing source of funds or you need clarification, interview the customer! Provide a tailored "question" to ask them.
4. `perform_risk_scoring`: Fourth, ALWAYS lock in the risk score before taking any terminal actions.

You MUST respond strictly in JSON matching this schema:
{
  "action_type": "string (MUST be one of the available_actions)",
  "target_customer_id": "string",
  "question": "string (REQUIRED ONLY IF action_type is interview_customer)",
  "decision_reasoning": "string"
}
After gathering all evidence and logically traversing the steps above, output your final decision using one of the terminal actions: `approve`, `reject`, `escalate`, or `freeze_account`.
"""

    def start_episode(self, task_id: str) -> Dict[str, Any]:
        print(f"\n--- Starting Task: {task_id} ---")
        response = requests.get(f"{self.env_url}/reset", params={"task_id": task_id})
        response.raise_for_status()
        return response.json()
        
    def step(self, episode_id: str, action: dict) -> Dict[str, Any]:
        response = requests.post(f"{self.env_url}/step", json={"episode_id": episode_id, "action": action})
        response.raise_for_status()
        return response.json()
        
    def grade(self, episode_id: str) -> float:
        response = requests.get(f"{self.env_url}/grade", params={"episode_id": episode_id})
        response.raise_for_status()
        return response.json()["score"]

    def run_task(self, task_id: str):
        try:
            reset_res = self.start_episode(task_id)
        except Exception as e:
            print(f"Failed to reset task {task_id}: {e}")
            return 0.0

        episode_id = reset_res["episode_id"]
        obs = reset_res["observation"]
        
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"New Case Assigned.\nCustomer Profile: {json.dumps(obs['customer'], indent=2)}\nAvailable Actions: {obs['available_actions']}"}
        ]
        
        max_steps = obs.get('max_steps', 20)
        for step_idx in range(max_steps):
            print(f"Step {step_idx+1}: Requesting action from model...")
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    response_format={"type": "json_object"}
                )
                
                content = completion.choices[0].message.content
                action_data = json.loads(content)
                if "target_customer_id" not in action_data:
                    action_data["target_customer_id"] = obs["customer"]["customer_id"]
            except Exception as e:
                print(f"Failed model inference: {e}")
                action_data = {"action_type": "escalate", "target_customer_id": obs["customer"]["customer_id"], "decision_reasoning": "Fallback due to JSON failure"}
                
            print(f" -> Agent chose: {action_data.get('action_type')}")
            history.append({"role": "assistant", "content": json.dumps(action_data)})
            
            try:
                step_res = self.step(episode_id, action_data)
                obs_data = step_res.get("observation", {})
                reward = step_res.get("reward", {})
                done = step_res.get("done", False)
                info = step_res.get("info", {})
                
                message = obs_data.get('message', '')
                flags = info.get('flags', [])
                
                history.append({"role": "user", "content": f"Action result: {message}\nNew flags detected: {flags}"})
                
                if done:
                    print(f" -> Episode Done. Final Message: {message}")
                    break
                    
            except Exception as e:
                print(f"Environment Error: {e}")
                history.append({"role": "user", "content": f"Error executing action: {e}"})
                # Break to avoid infinite loops on systematic action errors
                break
                
        # Final Grade
        time.sleep(1) # Small sleep before grading
        score = self.grade(episode_id)
        print(f"FINAL SCORE for {task_id}: {score}")
        return score

if __name__ == "__main__":
    api_key = OPENAI_API_KEY or os.getenv("HF_TOKEN")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable.")
        exit(1)
        
    agent = InferenceAgent(API_BASE_URL, MODEL_NAME, api_key)
    
    try:
        score1 = agent.run_task("task1_easy")
        score2 = agent.run_task("task2_medium")
        score3 = agent.run_task("task3_hard")
        
        print("\n==============================")
        print("    BASELINE INFERENCE SCORES   ")
        print("==============================")
        print(f"Task 1 (Easy)  : {score1}")
        print(f"Task 2 (Medium): {score2}")
        print(f"Task 3 (Hard)  : {score3}")
        print("==============================")
        
    except requests.exceptions.ConnectionError:
        print("Failed to connect to Environment Server. Is it running on port 8080?")