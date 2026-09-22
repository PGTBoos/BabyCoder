"""
agent_room.py

Third driver: a room that lets an architect and a coder hand a task back
and forth without a human running each script by hand. Coder and
architect stay exactly as they are - this file only adds a decision
loop on top of them.

Deliberately NOT how you might first picture "two agents talking": there
is no ping-pong chat between them, and "who goes next" is not decided by
asking a third LLM to read the room. It's decided by plain Python reading
the shared todo list (todo_read/todo_write, already in agent_toolkit.py),
same file both agents already know how to use. That keeps the one thing
that actually needs judgement (what to plan, what to write) on the LLM,
and the one thing that doesn't (whose turn is it) off of it - an LLM
call to decide "whose turn" would just be one more thing that can be
misread by a smaller model, for state a few lines of Python can check
for free.

Why this can plug into non-coding rooms later without a rewrite: nothing
below this docstring assumes the todo items are about code. The only
coding-specific things in this file are the two persona prompts and the
starter file the demo seeds at the bottom - swap those (and the toolkit
each persona is allowed to use) for a different domain's personas/tools
and the round loop, the todo schema, and the stop conditions all still
apply as-is.

Run:
    pip install requests
    python agent_room.py
"""

import json

from agent_toolkit import configure_workspace, run_agent, todo_read, todo_write, write_file
from architect_agent import PERSONA as ARCHITECT_BASE_PERSONA, ALLOWED as ARCHITECT_ALLOWED

# The coder has no reusable PERSONA constant to import - coder_agent.py's
# __main__ block sends run_agent a one-off task string with no
# persona_prompt at all, because that file is a fixed demo, not a role.
# The room needs the coder to behave as a standing role instead (pick up
# whatever's assigned to it, not one hardcoded task), so that role
# description is written fresh here rather than pulled from a file that
# doesn't define it.
CODER_BASE_PERSONA = (
    "You are the CODER agent in a multi-agent room. You do not plan; you "
    "implement whatever todo items are assigned to you. Use read_symbol / "
    "update_symbol / check_syntax and the rest of your tools as normal."
)

# Both roles read and write the same JSON list via todo_read/todo_write.
# todo_write REPLACES the whole file, it does not merge - so this schema
# note has to tell both personas to read the full list first, keep every
# item they're not touching, and write the full list back. Skipping that
# is the most likely way for one role to silently erase the other's
# entries.
TODO_SCHEMA_NOTE = (
    "\n\nThe shared todo list (todo_read / todo_write) is a JSON array. "
    "Every item must have exactly these fields:\n"
    '  {"description": "...", "owner": "architect" | "coder", "status": "pending" | "done"}\n'
    "todo_write REPLACES THE ENTIRE LIST. Always todo_read first, keep every "
    "existing item you are not changing, and write the complete list back - "
    "never just the item you added or changed.\n"
    "When you finish implementing an item, set its status to \"done\" and "
    "write the list back rather than deleting it, so the room can see it "
    "was completed rather than never having existed."
)

ARCHITECT_PERSONA = ARCHITECT_BASE_PERSONA + TODO_SCHEMA_NOTE
CODER_PERSONA = CODER_BASE_PERSONA + TODO_SCHEMA_NOTE


def _read_todos() -> list:
    """Best-effort parse of the shared todo file. Malformed or missing
    content is treated as an empty board rather than raised, since a
    smaller model occasionally writes something that isn't quite the
    schema - the room should keep going and let the next architect turn
    clean it up, not crash on it."""
    try:
        items = json.loads(todo_read())
    except json.JSONDecodeError:
        return []
    return items if isinstance(items, list) else []


def _pending(items: list, owner: str) -> list:
    return [
        t for t in items
        if isinstance(t, dict) and t.get("owner") == owner and t.get("status") != "done"
    ]


def decide_next(items: list) -> str:
    """Deterministic turn selection - the whole point of doing this in
    Python instead of another LLM call. Rules, in order:
      1. Nothing planned yet -> architect goes first.
      2. Anything pending for the coder -> coder's turn.
      3. Nothing pending for the coder, but something pending that isn't
         the coder's (i.e. the architect's own) -> architect's turn.
      4. Nothing pending anywhere -> None, the room considers this done.
    """
    if not items:
        return "architect"
    if _pending(items, "coder"):
        return "coder"
    if any(isinstance(t, dict) and t.get("status") != "done" for t in items):
        return "architect"
    return None


def run_room(task: str, max_rounds: int = 6, interactive: bool = True) -> None:
    """Alternate architect/coder turns, driven entirely by the shared
    todo list's state, until either nothing is pending (done), two
    rounds in a row leave the todo list unchanged (stuck - handed back to
    a human rather than looped on forever), or max_rounds is hit (same
    reasoning as run_agent's own max_steps: an explicit ceiling beats a
    silent infinite loop).
    """
    stale_rounds = 0

    for round_no in range(1, max_rounds + 1):
        items = _read_todos()
        speaker = decide_next(items)

        if speaker is None:
            print("[ROOM] no pending todos for either role - looks done.")
            return

        print(f"\n[ROOM] round {round_no}: {speaker}'s turn "
              f"({len(items)} todo item(s) on the board)")

        if interactive:
            note = input(
                "[ROOM] press Enter to continue, type a note to hand to "
                f"{speaker} first, or 'stop': "
            ).strip()
            if note.lower() == "stop":
                print("[ROOM] stopped by user.")
                return
        else:
            note = ""

        state_before = todo_read()

        # Re-point the shared workspace/backup/transcript globals at this
        # role's own folder right before its turn. Importing
        # architect_agent above already called configure_workspace once,
        # but whichever role ran last is whatever's active now - this
        # call is what actually decides where THIS turn's tools land, not
        # the import.
        if speaker == "architect":
            configure_workspace("architect")
            prompt = (
                f"Overall task: {task}\n\n"
                "Check todo_read first. If nothing is planned yet, break the "
                "task into a small number of clear, actionable todos owned by "
                "\"coder\". If coder todos already exist and are all done, "
                "either add any remaining follow-up work or, if the task is "
                "genuinely complete, make no changes and say so."
            )
        else:
            configure_workspace("sandbox")
            prompt = (
                f"Overall task: {task}\n\n"
                "Check todo_read first, find the pending todo item(s) with "
                "owner \"coder\", and implement the highest-priority one. "
                "Mark it done in the shared todo list when finished."
            )

        if note:
            prompt += f"\n\nNote from the person running this room: {note}"

        persona = ARCHITECT_PERSONA if speaker == "architect" else CODER_PERSONA
        allowed = ARCHITECT_ALLOWED if speaker == "architect" else None
        result = run_agent(prompt, persona_prompt=persona, allowed_tools=allowed)
        print(f"[ROOM] {speaker} finished: {result}")

        if todo_read() == state_before:
            stale_rounds += 1
            if stale_rounds >= 2:
                print("[ROOM] two rounds in a row with no change to the todo "
                      "list - stopping so a human can look, rather than "
                      "looping on nothing.")
                return
        else:
            stale_rounds = 0

    print(f"[ROOM] hit max_rounds ({max_rounds}) without finishing - "
          "stopping so a human can look.")


if __name__ == "__main__":
    # Same starting point coder_agent.py uses on its own, so this demo is
    # directly comparable to running the coder standalone. Explicitly
    # reset to the coder's sandbox first: importing architect_agent above
    # already left the shared workspace pointed at sandbox_architect.
    configure_workspace("sandbox")
    write_file(
        "graph_tools.py",
        "def add_edge(graph, a, b):\n"
        "    graph.setdefault(a, []).append(b)\n"
        "    graph.setdefault(b, []).append(a)\n"
        "    return graph\n"
        "\n"
        "\n"
        "def shortest_path(graph, start, end):\n"
        "    # TODO: not implemented yet.\n"
        "    return None\n",
    )
    todo_write([])  # clear any todos left over from a previous run

    run_room(
        "In graph_tools.py, implement shortest_path(graph, start, end) as a "
        "real breadth-first search over `graph` (a dict mapping each node "
        "to a list of neighbors, built by add_edge). It should return the "
        "list of nodes on a shortest path from start to end, inclusive, or "
        "None if no path exists."
    )
