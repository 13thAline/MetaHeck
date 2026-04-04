"""
Bank KYC Audit Environment — FastAPI server exposing OpenEnv HTTP API.
Endpoints: GET /reset, POST /step, GET /state, GET /grade, GET /health, GET /tasks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

from env.environment import BankKYCAuditEnv, TASK_CONFIG
from env.models import Action
from fastapi import Request, HTTPException

app = FastAPI(
    title="BankKYCAuditEnv",
    description="OpenEnv-compliant KYC/AML fraud detection environment for AI agents (2025 Standards).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
_sessions: Dict[str, BankKYCAuditEnv] = {}
_active_session: Optional[str] = None


class StepRequest(BaseModel):
    action: Action
    episode_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "environment": "bank-kyc-audit-env", "version": "2.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": "easy" if "easy" in tid else "medium" if "medium" in tid else "hard",
                "description": cfg.get("description", "")[:120] + "...",
                "max_steps": cfg.get("max_steps", 20),
            }
            for tid, cfg in TASK_CONFIG.items()
        ]
    }


@app.get("/reset")
@app.post("/reset")
async def reset(request: Request, task_id: str = "task1_easy", episode_id: Optional[str] = None):
    """Start a new episode. Returns initial observation."""
    global _active_session
    
    # Safely handle the POST request from the hackathon validator
    if request.method == "POST":
        try:
            body = await request.json()
            # Override query parameters if they are provided in the JSON body
            task_id = body.get("task_id", task_id)
            episode_id = body.get("episode_id", episode_id)
        except Exception:
            pass # Fallback to defaults if the body is completely empty like '{}'

    try:
        env = BankKYCAuditEnv(task_id=task_id)
        obs = env.reset(episode_id=episode_id)
        current_episode_id = getattr(obs, "episode_id", "default")
        
        _sessions[current_episode_id] = env
        _active_session = current_episode_id
        
        return {
            "episode_id": current_episode_id,
            "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(req: StepRequest):
    """Apply an action. Returns observation, reward, done, info."""
    episode_id = req.episode_id or _active_session
    if not episode_id or episode_id not in _sessions:
        raise HTTPException(status_code=404, detail="No active session. Call /reset first.")

    env = _sessions[episode_id]
    try:
        res = env.step(req.action)
        
        # Handle both OpenEnv direct Observation return or tuple return
        if isinstance(res, tuple):
            obs, reward, done, info = res
        else:
            obs = res
            # Extract reward and done directly from the Observation model if they exist
            # OpenEnv core Observation sets these natively
            reward = {"step_score": getattr(obs, "reward", 0.0), "feedback": getattr(obs, "metadata", {}).get("feedback", "")}
            done = getattr(obs, "done", False)
            info = getattr(obs, "metadata", {})

        return {
            "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state(episode_id: Optional[str] = None):
    """Return full internal state of the current episode."""
    eid = episode_id or _active_session
    if not eid or eid not in _sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    
    env = _sessions[eid]
    return env.state.model_dump() if hasattr(env.state, "model_dump") else env.state


@app.get("/grade")
def grade_episode(episode_id: Optional[str] = None):
    """Run the task grader and return final episode score."""
    eid = episode_id or _active_session
    if not eid or eid not in _sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    
    env = _sessions[eid]
    score = env.grade() if hasattr(env, "grade") else 0.0
    return {"episode_id": eid, "score": score}


@app.get("/")
def root():
    return {
        "name": "BankKYCAuditEnv",
        "openenv": True,
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health"],
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()