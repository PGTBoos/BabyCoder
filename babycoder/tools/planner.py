"""
babycoder.tools.planner

AGENT PLANNER toolkit: the shared todo board. NOT private per agent: it is
the handoff channel between an architect and a coder in separate processes.
"""

import json
import os

from .. import core
from ..core import P, tool

TODO_OWNERS = ("architect", "coder")
TODO_STATUSES = ("pending", "done")


@tool("Read the current shared todo list as JSON.")
def todo_read():
    # Read core.TODO_PATH at call time, not once at import, so it can be
    # pointed elsewhere (a test, a second project) without reloading anything.
    if not os.path.exists(core.TODO_PATH):
        return "[]"
    with open(core.TODO_PATH, "r", encoding="utf-8") as f:
        return f.read()


@tool('Replace the whole todo list. Each item must be exactly {"description": "...", '
      '"owner": "architect" | "coder", "status": "pending" | "done"}.',
      items=P("array", items={"type": "object"}))
def todo_write(items):
    # Checked here rather than hoped for: a board a small model wrote with
    # bare strings or a missing owner used to be saved as-is, and the room
    # then had to guess what it meant. Rejecting it with the exact problem
    # lets the model fix it on its next turn instead.
    if not isinstance(items, list):
        return "ERROR: items must be a list of todo objects"
    problems = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append(f"item {i} is not an object")
            continue
        if not str(item.get("description", "")).strip():
            problems.append(f"item {i} has no description")
        if item.get("owner") not in TODO_OWNERS:
            problems.append(f"item {i} owner must be one of {TODO_OWNERS}")
        if item.get("status") not in TODO_STATUSES:
            problems.append(f"item {i} status must be one of {TODO_STATUSES}")
    if problems:
        return "ERROR: todo list not saved: " + "; ".join(problems)
    # Write then rename, so a second process never reads a half-written board.
    tmp = core.TODO_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    os.replace(tmp, core.TODO_PATH)
    return f"OK: saved {len(items)} todo items"
