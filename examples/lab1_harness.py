"""Lab 1 as a plain script: the agent loop with DuckDuckGo search.

    python examples/lab1_harness.py
    python examples/lab1_harness.py "Research X and write notes.md"
"""
import sys

from agent2027 import config
from agent2027.harness import run_agent
from agent2027.tools import core_tools, read_file

TASK = (
    "Research how prompt injection attacks work against LLM agents. Search the web, read at least "
    "three good sources, and write a summary with source URLs to notes.md."
)

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else TASK
    print(f"model={config.MODEL} search={config.SEARCH_MODE} workdir={config.WORKDIR}")
    answer = run_agent(task, core_tools(), run_id="lab1", agent_id="lab1")
    print("\nFINAL:", answer)
    try:
        print("\n--- notes.md ---\n" + read_file("notes.md"))
    except Exception as e:  # noqa: BLE001
        print("no notes.md written:", e)
