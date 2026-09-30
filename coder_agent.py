"""
coder_agent.py

The CODER role, runnable on its own. Tell it what to build and it builds it,
in its own sandbox, using babycoder's shared coding-agent core.

Three ways to give it work, and the difference matters:

  python coder_agent.py                     asks you what to build
  python coder_agent.py "add a retry loop"  same thing, from the command line
  python coder_agent.py --file spec.md      same thing, from a file
  python coder_agent.py --todos             no new task: work the shared
                                            todo list an architect left
  python coder_agent.py --demo              the old fixed BFS demo

The --todos mode is the handoff. architect_agent.py, run separately and
possibly hours earlier, writes items owned by "coder" to the shared todo
list; this picks them up. Neither process has to be running at the same time
as the other, and neither can see the other's sandbox - the todo list is the
only thing they share, which is exactly why todo_read/todo_write are not
sandboxed per agent (see TODO_PATH in babycoder/core.py).

PERSONA is exported rather than inlined into the run so agent_room.py can
drive this same role inside a room without a second, drifting copy of the
role description living there.

Run:
    pip install requests
    python coder_agent.py
"""

import sys

from babycoder import AGENT_CODER, coding, configure_workspace, planner, run_agent
from agent_common import ensure_utf8_console, WORK_THE_BOARD, print_usage, read_task

# "sandbox" (not "coder") on purpose: configure_workspace special-cases that
# name to the plain <SCRIPT_DIR>/sandbox folder, which is where this driver
# has always worked and, more importantly, the folder agent_room.py points a
# coder turn at. Renaming it would give the standalone coder and the room's
# coder two different folders, so work started in the room could not be
# picked up by a later standalone run - which is the whole point of the
# handoff below.
WORKSPACE = configure_workspace("sandbox")

# Role only. How to edit code well (look before changing, one change at a
# time, check_syntax afterwards) is the coding toolset's usage note in
# babycoder, and arrives with the grant.
PERSONA = (
    "You are the CODER agent. You implement; you do not plan. If a task is "
    "ambiguous, implement the reading you can defend and say which reading "
    "you took rather than stopping to ask."
)

# The original fixed demo, kept because it is a genuinely useful smoke test:
# it checks whether a model can write correct BFS, not merely whether the
# tool plumbing runs.
DEMO_SEED = (
    "def add_edge(graph, a, b):\n"
    "    graph.setdefault(a, []).append(b)\n"
    "    graph.setdefault(b, []).append(a)\n"
    "    return graph\n"
    "\n"
    "\n"
    "def shortest_path(graph, start, end):\n"
    "    # TODO: not implemented yet.\n"
    "    return None\n"
)

DEMO_TASK = (
    "In graph_tools.py, list the symbols, then read shortest_path to see its "
    "current unfinished state. Then use update_symbol to rewrite "
    "shortest_path so it does a real breadth-first search over `graph` (a "
    "dict mapping each node to a list of neighbors, as built by add_edge) and "
    "returns the list of nodes on a shortest path from start to end, "
    "inclusive, or None if no path exists. Run check_syntax afterwards. "
    "Finally propose (but do not assume approval for) a `python -c` command "
    "that builds a small graph with add_edge and calls shortest_path on it to "
    "sanity check the result."
)


if __name__ == "__main__":
    ensure_utf8_console()
    task, mode = read_task(
        prompt="What should the coder build? (blank = work the todo list): ",
        demo_task=DEMO_TASK,
    )

    if mode == "help":
        print_usage("coder_agent.py")
        sys.exit(0)
    if mode == "quit":
        sys.exit(0)

    if mode == "demo":
        # Only the demo seeds a file. A real task must never have a stray
        # graph_tools.py dropped into its sandbox - the old __main__ wrote
        # this unconditionally, which was fine when the file WAS the demo and
        # is wrong now that it is one mode among several.
        coding.write_file("graph_tools.py", DEMO_SEED)

    if mode == "todos":
        board = planner.todo_read()
        if not board.strip() or board.strip() in ("[]", "[]\n"):
            print("Nothing on the shared todo list, and no task given.")
            print("Give the coder something to do, or run architect_agent.py "
                  "first to plan some work.\n")
            print_usage("coder_agent.py")
            sys.exit(0)
        print("Working the shared todo list:")
        print(board)
        print()
        task = WORK_THE_BOARD

    # AGENT_CODER is CODER_TOOLS | TODO_TOOLS | GUIDANCE_TOOLS. Passing it
    # explicitly rather than leaving allowed_tools=None (which grants
    # everything) means the coder does not silently gain memory or research
    # just because those now exist in the toolkit.
    print(run_agent(task, persona_prompt=PERSONA, allowed_tools=AGENT_CODER, workspace=WORKSPACE))
