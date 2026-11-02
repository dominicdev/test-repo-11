# Building Agents for 2027: workshop code

A framework-free agent harness on the Gemini API, built up across four labs.

| Lab | Notebook | Package code |
|---|---|---|
| 1. Harness from scratch (DuckDuckGo search) | `labs/lab1_harness.ipynb` | `agent2027/harness.py`, `tools.py` |
| 2. Scratchpad and compaction | `labs/lab2_scratchpad.ipynb` | `agent2027/scratchpad.py`, `store.py` |
| 3. Subagents through shared state | `labs/lab3_subagents.ipynb` | `agent2027/subagents.py` |
| 4. Serve on Cloud Run | `labs/lab4_deploy.ipynb` | `deploy/cloudrun/` |

## Run locally

```bash
pip install -e ".[serve]"
export GEMINI_API_KEY=...          # AI Studio key
python tests/test_offline.py       # fake client, no key or network needed
```

## Before the event

- Set `REPO_URL` in `scripts/make_notebooks.py`, then `python scripts/make_notebooks.py`.
- Set `AGENT_MODEL` (default `gemini-flash-latest`) and test it on the free tier.
- Pre-run the workshop queries with `SEARCH_MODE=live` and commit `workdir/.search_cache.json`
  as `agent2027/data/search_cache.json`, so `SEARCH_MODE=cached` works if DuckDuckGo throttles the room.

## Not built yet

`memory.py` (Sessions + Memory Bank), `armor.py` (Model Armor), the Agent Runtime deployment,
`liveview/`, and the poisoned-site security demo.
