"""Offline tests: a scripted fake Gemini client, no API key or network needed.

    python tests/test_offline.py        (or: pytest tests)
"""
import json
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from google.genai import types

from agent2027 import config, scratchpad, store as store_mod, tools
from agent2027.harness import run_agent
from agent2027.subagents import run_system

config.WORKDIR = Path(tempfile.mkdtemp(prefix="agent2027-test-"))
config.SEARCH_MODE = "cached"


def resp(text=None, calls=None, tokens=100):
    parts = ([types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
             if calls else [types.Part(text=text)])
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts))],
        function_calls=[p.function_call for p in parts if p.function_call] or None,
        text=None if calls else text,
        usage_metadata=SimpleNamespace(prompt_token_count=tokens),
    )


class FakeModels:
    """Agent calls (with tools) use one tool then answer; other calls get canned text."""

    def __init__(self, tokens=100):
        self.tokens, self.calls = tokens, 0

    def generate_content(self, model, contents, config=None):
        self.calls += 1
        tools_cfg = getattr(config, "tools", None)
        if tools_cfg:
            system = config.system_instruction or ""
            used = sum(1 for c in contents if c.role == "user" and any(p.function_response for p in c.parts or []))
            if "fact checker" in system:
                return resp("VERDICT: PASS\nlooks fine")
            if used == 0:
                return resp(calls=[("list_files", {"path": "."})], tokens=self.tokens)
            return resp("worker answer with https://example.com", tokens=self.tokens)
        if getattr(config, "response_mime_type", None) == "application/json":
            return resp(json.dumps(["subtask a", "subtask b", "subtask c"]))
        return resp("final synthesis / summary")


def fake_client(**kw):
    c = SimpleNamespace(models=FakeModels(**kw))
    config.get_client = lambda: c
    return c


def test_tools_and_safety():
    assert "wrote" in tools.write_file("notes.md", "hi")
    assert tools.read_file("notes.md") == "hi"
    for bad in ("../secret.txt", "/etc/passwd"):
        try:
            tools.read_file(bad)
            raise AssertionError("path escape not blocked")
        except ValueError:
            pass
    for url in ("http://127.0.0.1/", "http://169.254.169.254/computeMetadata/v1/", "file:///etc/passwd"):
        try:
            tools._assert_public(url)
            raise AssertionError(f"{url} not blocked")
        except ValueError:
            pass
    assert "No cached results" in tools.web_search("never cached query xyz")
    tools._cache_store("cached q", [{"title": "T", "url": "https://u", "snippet": "S"}])
    assert "https://u" in tools.web_search("Cached  Q")
    assert "unknown tool" in tools.dispatch({}, SimpleNamespace(name="nope", args={}))
    assert "ERROR" in tools.dispatch({t.name: t for t in tools.core_tools()},
                                     SimpleNamespace(name="read_file", args={"path": "missing.txt"}))
    for t in tools.core_tools():  # every schema builds
        assert t.declaration().name == t.name
    assert tools.core_tools()[4].declaration().parameters.required == []  # list_files: path optional


def test_loop_and_trajectory():
    fake_client()
    out = run_agent("do it", [t for t in tools.core_tools() if t.name == "list_files"], run_id="t1", verbose=False)
    assert out.startswith("worker answer")
    rows = (config.WORKDIR / "trajectories" / "t1.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["tool"] == "list_files"
    assert run_agent("x", [], max_steps=1, verbose=False) is not None


def test_compaction():
    fake_client(tokens=50_000)  # always over the threshold
    st = store_mod.MemoryStore()
    st.update_run("r1", goal="g")
    comp = scratchpad.make_compactor("the task", st, "r1", threshold=1000)
    new = comp([types.Content(role="user", parts=[types.Part(text="a")])] * 5, 50_000)
    assert len(new) == 1 and "scratchpad" in new[0].parts[1].text.lower()
    assert st.get_run("r1")["last_summary"]
    t = {x.name: x for x in scratchpad.scratch_tools(st, "r1")}
    t["scratch_write"].fn("findings", "f1")
    t["scratch_write"].fn("findings", "f2")
    assert st.get_run("r1")["findings"] == ["f1", "f2"]


def test_no_double_claims():
    st = store_mod.MemoryStore()
    ids = {st.add_task("r", "researcher", f"p{i}") for i in range(30)}
    got, lock = [], threading.Lock()

    def grab(w):
        while (t := st.claim_task("r", "researcher", w)):
            with lock:
                got.append(t["id"])

    ts = [threading.Thread(target=grab, args=(f"w{i}",)) for i in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(got) == sorted(ids)


def test_full_system():
    fake_client()
    st = store_mod.MemoryStore()
    ans = run_system("compare things", "sys1", n_workers=3, with_fact_checker=True, store=st, timeout=60)
    assert ans == "final synthesis / summary"
    ts = st.tasks("sys1")
    assert len(ts) == 3 and all(t["status"] == "done" and t["verified"] for t in ts)
    assert st.get_run("sys1")["status"] == "done"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    shutil.rmtree(config.WORKDIR, ignore_errors=True)
