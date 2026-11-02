"""Lab 3 as a plain script: orchestrator + workers coordinating through shared state.

    python examples/lab3_subagents.py
    python examples/lab3_subagents.py --fact-checker
    python examples/lab3_subagents.py --workers 4 "Your goal here"
"""
import sys
import threading
import time

from agent2027 import config
from agent2027.store import get_store
from agent2027.subagents import run_system

GOAL = "Compare three popular vector databases for a small startup: strengths, weaknesses, pricing model."


def board(store, run_id):
    print(f"{'id':10} {'role':13} {'status':10} {'worker':12} verified")
    for t in store.tasks(run_id):
        print(f"{t['id']:10} {t['role']:13} {t['status']:10} {str(t['worker_id']):12} {t['verified']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    fact_checker = "--fact-checker" in args
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 3
    rest = [a for i, a in enumerate(args)
            if not a.startswith("--") and not (i and args[i - 1] == "--workers")]
    goal = rest[0] if rest else GOAL

    store, run_id, out = get_store(), config.new_run_id(), {}
    print(f"run={run_id} workers={workers} fact_checker={fact_checker} model={config.MODEL}")
    t = threading.Thread(target=lambda: out.update(
        answer=run_system(goal, run_id, n_workers=workers, with_fact_checker=fact_checker)))
    t.start()
    while t.is_alive():  # the "task board": poll shared state while workers run
        time.sleep(10)
        board(store, run_id)
        print()
    t.join()
    board(store, run_id)
    print("\nANSWER:\n", out.get("answer"))
