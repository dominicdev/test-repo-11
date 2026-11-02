"""Working state outside the context window, plus compaction (Lab 2)."""
from google.genai import types

from . import config
from .harness import user
from .tools import Tool

SECTIONS = ("plan", "findings", "notes", "status")


def snapshot(store, run_id) -> str:
    run = store.get_run(run_id)
    findings = "\n".join(f"- {f}" for f in run.get("findings", [])) or "(none yet)"
    return (
        f"GOAL: {run.get('goal', '')}\n"
        f"PLAN: {run.get('plan', '(none yet)')}\n"
        f"NOTES: {run.get('notes', '')}\n"
        f"FINDINGS SO FAR:\n{findings}"
    )


def scratch_tools(store, run_id) -> list[Tool]:
    def scratch_write(section: str, content: str) -> str:
        if section not in SECTIONS:
            return f"ERROR: section must be one of {SECTIONS}"
        if section == "findings":
            store.append_run(run_id, "findings", content)  # findings accumulate
        else:
            store.update_run(run_id, **{section: content})  # others are overwritten
        return f"saved to {section}"

    def scratch_read() -> str:
        return snapshot(store, run_id)

    return [
        Tool("scratch_write",
             "Save progress OUTSIDE your context window. section: plan (overwrite), "
             "findings (append one finding with its source URL), notes (overwrite), status (overwrite). "
             "Save every finding as soon as you learn it.",
             scratch_write, {"section": ("STRING", "plan | findings | notes | status"),
                             "content": ("STRING", "Text to save")}),
        Tool("scratch_read", "Read the full scratchpad: goal, plan, notes and all findings so far.",
             scratch_read, {}, required=[]),
    ]


def make_compactor(task, store, run_id, threshold=None):
    """Build an on_step hook: past the token threshold, summarise, reset the
    history and rebuild context from the task + scratchpad + summary."""

    def compact(contents, tokens):
        limit = threshold or config.COMPACT_AT_TOKENS
        if tokens <= limit:
            return contents
        ask = user("Summarise your progress so far in under 200 words: what you have done, "
                   "what remains, and what to do next. Do not call tools.")
        resp = config.get_client().models.generate_content(
            model=config.MODEL, contents=contents + [ask])
        summary = resp.text or ""
        print(f"[compaction] {tokens} tokens > {limit}: history reset, rebuilt from scratchpad")
        store.update_run(run_id, last_summary=summary)
        return [types.Content(role="user", parts=[
            types.Part(text=task),
            types.Part(text=("Your earlier context was compacted. Current scratchpad:\n"
                             f"{snapshot(store, run_id)}\n\nYour summary of progress:\n{summary}\n\n"
                             "Continue the task from here. Save new findings with scratch_write.")),
        ])]

    return compact
