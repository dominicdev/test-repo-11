"""Cloud Run service (Lab 4): POST /runs starts a run, GET /runs/{id} reports status.

State is in Firestore, so the harness, scratchpad and task board are unchanged
from the notebook; only the entry point is new.
"""
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent2027 import config
from agent2027.store import get_store
from agent2027.subagents import run_system

app = FastAPI(title="agent2027")


class RunRequest(BaseModel):
    goal: str
    n_workers: int = 3
    with_fact_checker: bool = False


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": config.MODEL, "store": config.STORE}


@app.post("/runs")
def start_run(req: RunRequest):
    run_id = config.new_run_id()
    store = get_store()
    store.update_run(run_id, goal=req.goal, status="queued")
    # Work continues after the response: deploy with instance-based billing
    # (--no-cpu-throttling) or the CPU is throttled once this returns.
    threading.Thread(
        target=lambda: run_system(req.goal, run_id, min(req.n_workers, 5), req.with_fact_checker, store),
        daemon=True,
    ).start()
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    run = get_store().get_run(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    tasks = get_store().tasks(run_id)
    return {
        "run_id": run_id,
        "status": run.get("status"),
        "result": run.get("result"),
        "tasks": [{"id": t["id"], "role": t["role"], "status": t["status"], "verified": t["verified"]} for t in tasks],
    }
