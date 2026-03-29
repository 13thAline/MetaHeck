"""
KYC Audit Environment — FastAPI server exposing OpenEnv HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /grade, GET /health
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

from env.environment import KYCEnvironment, TASK_CONFIG
from env.models import Action

app = FastAPI(
    title="KYC Audit Environment",
    description="OpenEnv-compliant KYC/AML fraud detection environment for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (keyed by episode_id for multi-session support)
_sessions: Dict[str, KYCEnvironment] = {}
_active_session: Optional[str] = None


class ResetRequest(BaseModel):
    task_id: str = "task1_doc_check"


class StepRequest(BaseModel):
    action: Action
    episode_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "environment": "kyc-audit-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": "easy" if tid == "task1_doc_check"
                              else "medium" if tid == "task2_txn_analysis"
                              else "hard",
                "description": cfg["description"][:120] + "...",
                "max_steps": cfg["max_steps"],
            }
            for tid, cfg in TASK_CONFIG.items()
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest):
    """Start a new episode. Returns initial observation."""
    global _active_session
    try:
        env = KYCEnvironment(task_id=req.task_id)
        obs = env.reset()
        episode_id = obs.episode_id
        _sessions[episode_id] = env
        _active_session = episode_id
        return {
            "episode_id": episode_id,
            "observation": obs.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Apply an action. Returns observation, reward, done, info."""
    episode_id = req.episode_id or _active_session
    if not episode_id or episode_id not in _sessions:
        raise HTTPException(status_code=404, detail="No active session. Call /reset first.")

    env = _sessions[episode_id]
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state(episode_id: Optional[str] = None):
    """Return full internal state of the current episode."""
    eid = episode_id or _active_session
    if not eid or eid not in _sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    env = _sessions[eid]
    return env.state().model_dump()


@app.get("/grade")
def grade_episode(episode_id: Optional[str] = None):
    """Run the task grader and return final episode score."""
    eid = episode_id or _active_session
    if not eid or eid not in _sessions:
        raise HTTPException(status_code=404, detail="No active session.")
    env = _sessions[eid]
    result = env.grade_episode()
    return {"episode_id": eid, **result}


@app.get("/")
def root():
    return {
        "name": "KYC Audit Environment",
        "openenv": True,
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)