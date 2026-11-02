"""Append one JSON line per tool call, so every run leaves a full trajectory."""
import json
import threading
import time

from . import config

_lock = threading.Lock()


def log_step(run_id, agent_id, step, tool, args, result, tokens=0):
    if not run_id:
        return
    path = config.WORKDIR / "trajectories" / f"{run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "agent_id": agent_id,
        "step": step,
        "tool": tool,
        "args": args,
        "result_preview": str(result)[:300],
        "tokens": tokens,
        "ts": time.time(),
    }
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
