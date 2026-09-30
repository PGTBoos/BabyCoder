"""
agent_room.py

Third driver: a room that lets an architect and a coder hand a task back
and forth without a human running each script by hand. Coder and
architect stay exactly as they are - this file only adds a decision
loop on top of them.

Deliberately NOT how you might first picture "two agents talking": there
is no ping-pong chat between them, and "who goes next" is not decided by
asking a third LLM to read the room. It's decided by plain Python reading
the shared todo list (todo_read/todo_write, in the babycoder package),
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
import sys

from babycoder import AGENT_CODER, coding, planner, run_agent, using
from agent_common import ensure_utf8_console, print_usage, read_task
from architect_agent import (PERSONA as ARCHITECT_PERSONA, ALLOWED as ARCHITECT_ALLOWED,
                             WORKSPACE as ARCHITECT_WORKSPACE)
from coder_agent import PERSONA as CODER_PERSONA, DEMO_SEED, WORKSPACE as CODER_WORKSPACE

# Both personas, and the todo schema note baked into them, now come from the
# role files themselves rather than being restated here. The room used to
# define the coder's role inline because coder_agent.py was a fixed demo with
# no reusable role to import; now that the coder runs standalone it has one,
# and a room that redefined it would just be a second copy free to drift from
# the role the coder actually plays when run on its own.


def _read_todos() -> list:
    """Best-effort parse of the shared todo file. Malformed or missing
    content is treated as an empty board rather than raised, since a
    smaller model occasionally writes something that isn't quite the
    schema - the room should keep going and let the next architect turn
    clean it up, not crash on it."""
    try:
        items = json.loads(planner.todo_read())
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
    if not any(isinstance(t, dict) for t in items):
        # The board is non-empty but nothing on it is even shaped like an
        # item - a smaller model wrote bare strings instead of objects, say.
        # That is NOT done; without this the room congratulates itself and
        # exits on a task nobody started. Hand it to the architect to
        # rewrite, which is what _read_todos' docstring already promises.
        # run_room's stale-round and max_rounds guards keep that bounded.
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

        state_before = planner.todo_read()

        # Each role's turn runs in that role's own Workspace, handed to
        # run_agent explicitly. Nothing is reconfigured between turns, so
        # there is no "whichever role ran last" state to get wrong.
        if speaker == "architect":
            workspace = ARCHITECT_WORKSPACE
            prompt = (
                f"Overall task: {task}\n\n"
                "Check todo_read first. If nothing is planned yet, break the "
                "task into a small number of clear, actionable todos owned by "
                "\"coder\". If coder todos already exist and are all done, "
                "read the coder's work under @coder/ to check it, then either "
                "add follow-up todos for what is missing or wrong, or, if the "
                "task is genuinely complete, make no changes and say so."
            )
        else:
            workspace = CODER_WORKSPACE
            prompt = (
                f"Overall task: {task}\n\n"
                "Check todo_read first, find the pending todo item(s) with "
                "owner \"coder\", and implement the highest-priority one. "
                "Mark it done in the shared todo list when finished."
            )

        if note:
            prompt += f"\n\nNote from the person running this room: {note}"

        persona = ARCHITECT_PERSONA if speaker == "architect" else CODER_PERSONA
        allowed = ARCHITECT_ALLOWED if speaker == "architect" else AGENT_CODER
        result = run_agent(prompt, persona_prompt=persona, allowed_tools=allowed,
                           workspace=workspace)
        print(f"[ROOM] {speaker} finished: {result}")

        if planner.todo_read() == state_before:
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


DEMO_TASK = (
    "In graph_tools.py, implement shortest_path(graph, start, end) as a "
    "real breadth-first search over `graph` (a dict mapping each node "
    "to a list of neighbors, built by add_edge). It should return the "
    "list of nodes on a shortest path from start to end, inclusive, or "
    "None if no path exists."
)


if __name__ == "__main__":
    ensure_utf8_console()
    task, mode = read_task(
        prompt="What should the room work on? ",
        demo_task=DEMO_TASK,
    )

    if mode == "help":
        print_usage("agent_room.py")
        sys.exit(0)
    if mode == "quit":
        sys.exit(0)
    if mode == "todos":
        # A room with no overall task has nothing to hand the architect. The
        # coder alone can work an existing board, so say that rather than
        # spinning up two roles to discover there is nothing to plan.
        print("The room needs an overall task to work on.\n")
        print("To just work through todos that already exist, run:")
        print("    python coder_agent.py --todos\n")
        print_usage("agent_room.py")
        sys.exit(0)

    if mode == "demo":
        # Only the demo seeds a starting file and wipes the board. A real task
        # must not have a stray graph_tools.py dropped into the sandbox, and
        # must not silently discard todos someone left there on purpose.
        with using(CODER_WORKSPACE):
            coding.write_file("graph_tools.py", DEMO_SEED)
        planner.todo_write([])

    run_room(task)
