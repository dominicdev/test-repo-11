"""Tools for the harness (Lab 1).

A tool is a name, a description, a schema the model sees, and a Python function.
Functions never raise to the model: `dispatch` turns errors into text so the
model can recover.
"""
import difflib
import ipaddress
import json
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

from google.genai import types

from . import config


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    params: dict = field(default_factory=dict)  # name -> (TYPE, description)
    required: list | None = None  # None = every param is required

    def declaration(self) -> types.FunctionDeclaration:
        """The explicit schema the model sees (no automatic function calling)."""
        props = {
            k: types.Schema(type=t, description=d) for k, (t, d) in self.params.items()
        }
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type="OBJECT", properties=props, required=list(props) if self.required is None else self.required
            ),
        )


def dispatch(tools_by_name: dict, call) -> str:
    """Run one function call. Never raises: errors come back as text."""
    tool = tools_by_name.get(call.name)
    if tool is None:
        return f"ERROR: unknown tool '{call.name}'. Available: {sorted(tools_by_name)}"
    try:
        out = str(tool.fn(**dict(call.args or {})))
    except Exception as e:  # noqa: BLE001 - the model needs to see every failure
        return f"ERROR: {type(e).__name__}: {e}"
    if len(out) > config.MAX_TOOL_CHARS:
        out = out[: config.MAX_TOOL_CHARS] + "\n...[truncated by harness]"
    return out


# --------------------------------------------------------------------------
# Files: confined to workdir/
# --------------------------------------------------------------------------
def _safe_path(rel: str) -> Path:
    root = config.WORKDIR.resolve()
    root.mkdir(parents=True, exist_ok=True)
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"path '{rel}' is outside the workdir")
    return p


def read_file(path: str) -> str:
    return _safe_path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def list_files(path: str = ".") -> str:
    p = _safe_path(path)
    names = sorted(
        str(f.relative_to(config.WORKDIR.resolve()))
        for f in p.rglob("*")
        if f.is_file() and not f.name.startswith(".") and "trajectories" not in f.parts
    )
    return "\n".join(names) or "(empty)"


def run_python(code: str) -> str:
    """Run Python in a subprocess with a timeout.

    NOT A SANDBOX. The code runs with this process's permissions and network.
    Use a real sandbox (e.g. Agent Sandbox) for untrusted code.
    """
    root = _safe_path(".")
    script = root / ".snippet.py"
    script.write_text(code, encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, str(script)], cwd=root, capture_output=True, text=True, timeout=20
        )
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after 20s"
    return f"exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"


# --------------------------------------------------------------------------
# web_search: DuckDuckGo, with disk cache, backoff and a cached fallback
# --------------------------------------------------------------------------
_BUNDLED_CACHE = Path(__file__).parent / "data" / "search_cache.json"
_cache_lock = threading.Lock()
_search_failures = 0


def _disk_cache_path() -> Path:
    return config.WORKDIR / ".search_cache.json"


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _cache_lookup(key: str):
    with _cache_lock:
        merged = {**_read_json(_BUNDLED_CACHE), **_read_json(_disk_cache_path())}
    if key in merged:
        return merged[key]
    close = difflib.get_close_matches(key, list(merged), n=1, cutoff=0.8)
    return merged[close[0]] if close else None


def _cache_store(key: str, hits: list):
    with _cache_lock:
        p = _disk_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = _read_json(p)
        data[key] = hits
        p.write_text(json.dumps(data, indent=1), encoding="utf-8")


def _format_hits(hits: list) -> str:
    if not hits:
        return "No results."
    # Compact text, not raw JSON: fewer tokens.
    return "\n".join(
        f"{i}. {h['title']}\n   {h['url']}\n   {h['snippet']}" for i, h in enumerate(hits, 1)
    )


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via the ddgs package (DuckDuckGo and other engines). The model picks the query; the harness caps the count."""
    global _search_failures
    n = max(1, min(int(max_results or 5), config.MAX_SEARCH_RESULTS))
    key = " ".join(query.lower().split())

    cached = _cache_lookup(key)
    if cached is not None:
        return _format_hits(cached[:n])
    if config.SEARCH_MODE == "cached":
        return "No cached results for this query. Try a broader or differently worded query."

    from ddgs import DDGS

    last_err = None
    for attempt in range(3):
        try:
            time.sleep(config.SEARCH_DELAY * (2**attempt))  # the room shares one IP
            # ddgs is a metasearch library: backend="auto" tries several engines, which
            # helps when one throttles the room. Default timeout is 5s; 10s is kinder on venue wifi.
            raw = DDGS(timeout=10).text(query, max_results=n, backend=config.SEARCH_BACKEND)
            hits = [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in raw
            ]
            _search_failures = 0
            _cache_store(key, hits)
            return _format_hits(hits)
        except Exception as e:  # noqa: BLE001 - ddgs raises several different errors
            last_err = e
    _search_failures += 1
    if _search_failures >= 3:
        config.SEARCH_MODE = "cached"
        print("[search] DuckDuckGo keeps failing; switching to SEARCH_MODE=cached")
    return f"ERROR: search failed ({type(last_err).__name__}). Try again or change the query."


# --------------------------------------------------------------------------
# fetch_url: public http(s) only
# --------------------------------------------------------------------------
def _assert_public(url: str):
    """Refuse private, loopback and metadata addresses (e.g. 169.254.169.254).

    Note: resolving here and again in httpx leaves a DNS-rebinding gap. Fine for
    a workshop; in production pin the resolved IP or use an egress proxy.
    """
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("only http(s) URLs are allowed")
    for info in socket.getaddrinfo(u.hostname, None):
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"blocked non-public address {ip}")


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for el in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        el.decompose()
    lines = (ln.strip() for ln in soup.get_text("\n").splitlines())
    return "\n".join(ln for ln in lines if ln)


def fetch_url(url: str) -> str:
    """Fetch a page as text. The result is untrusted content written by someone else."""
    import httpx

    with httpx.Client(follow_redirects=False, timeout=15, headers={"User-Agent": "agent2027-workshop/0.1"}) as c:
        for _ in range(4):
            _assert_public(url)
            r = c.get(url)
            if r.is_redirect:
                url = urljoin(url, r.headers["location"])
                continue
            r.raise_for_status()
            break
        else:
            raise ValueError("too many redirects")
    ctype = r.headers.get("content-type", "")
    text = _html_to_text(r.text) if "html" in ctype else r.text
    text = text[: config.MAX_FETCH_CHARS]
    return f"[untrusted web content from {url}]\n{text}"


# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------
def core_tools() -> list[Tool]:
    return [
        Tool("web_search", "Search the web (DuckDuckGo). Returns title, URL and snippet per result. "
             "Use snippets to decide which pages to read.", web_search,
             {"query": ("STRING", "Search query"), "max_results": ("INTEGER", "How many results (max 5)")},
             required=["query"]),
        Tool("fetch_url", "Fetch a web page and return its text.", fetch_url,
             {"url": ("STRING", "Full http(s) URL")}),
        Tool("read_file", "Read a text file from the working directory.", read_file,
             {"path": ("STRING", "Relative path")}),
        Tool("write_file", "Write a text file in the working directory (overwrites).", write_file,
             {"path": ("STRING", "Relative path"), "content": ("STRING", "File content")}),
        Tool("list_files", "List files in the working directory.", list_files,
             {"path": ("STRING", "Relative directory, default '.'")}, required=[]),
        Tool("run_python", "Run Python code in the working directory and return stdout/stderr. "
             "Not a sandbox.", run_python, {"code": ("STRING", "Python source")}),
    ]
