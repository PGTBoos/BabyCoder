"""
architect_agent.py

The ARCHITECT role, runnable on its own. Tell it what needs building and it
breaks that down into todos owned by "coder", which a separately-run
coder_agent.py can then pick up with --todos.

  python architect_agent.py                      asks you what to plan
  python architect_agent.py "add CSV export"     same thing, from the command line
  python architect_agent.py --file spec.md       plan from a written spec
  python architect_agent.py --demo               the old fixed BFS demo

This file demonstrates the two things coder_agent.py does not: a private
workspace and a restricted tool set.

- # Its own folder, plus the coder's sandbox to read (never write) as @coder/.
WORKSPACE = configure_workspace("architect", read_roots={"coder": "sandbox"}) points every tool this agent calls at
  <SCRIPT_DIR>/sandbox_architect and its own .agent_state_architect,
  completely separate from the coder's sandbox. Neither agent can reach into
  the other's folder through any tool - not because of new access-control
  code, but because the sandbox escape check every tool already goes through
  has nothing pointing at the other agent's path to begin with.
- allowed_tools restricts this agent to reading and leaving todos: it can
  look around, but calling write_file, update_symbol, run_command, etc. gets
  rejected by run_agent before the call ever executes, not merely left out of
  persona_prompt's request. Widen ALLOWED as this agent's role grows rather
  than granting everything up front.
- todo_write/todo_read are deliberately NOT sandboxed per-agent (see
  TODO_PATH in babycoder/core.py): that is what lets the architect leave tasks
  a separately-run coder_agent.py can read, despite the two having no other
  visibility into each other's files at all.

It can also READ the coder's folder, as @coder/..., and cannot write there:
the workspace gives it the coder's sandbox as a read-only root. That is what
lets it check finished work before deciding whether follow-up is needed,
without the coder gaining any access back.

Run:
    pip install requests
    python architect_agent.py
"""

import sys

from babycoder import AGENT_ARCHITECT, configure_workspace, planner, run_agent
from agent_common import ensure_utf8_console, print_usage, read_task

# Its own folder, plus the coder's sandbox to read (never write) as @coder/.
WORKSPACE = configure_workspace("architect", read_roots={"coder": "sandbox"})

PERSONA = (
    "You are the ARCHITECT agent. Your job is to plan and leave clear, "
    "actionable todos for a separate CODER agent to implement - not to write "
    "or run code yourself. You do not have write_file, update_symbol, or "
    "run_command; use todo_write to hand off work instead. Prefer a small "
    "number of todos that each stand alone: the coder may pick one up hours "
    "later, in a different process, with no memory of this conversation and "
    "no access to your folder, so an item that only makes sense next to the "
    "others is an item that will be misread. You can read the coder's work "
    "under @coder/ to check what is already done before planning more."
)

# The grant comes from babycoder's AGENT GRANT SETS section rather than
# being spelled out here, so this agent's reach is defined in the same place
# as every other agent's and cannot quietly disagree with it. AGENT_ARCHITECT
# is READONLY_CODE_TOOLS | TODO_TOOLS | GUIDANCE_TOOLS: it can look, plan and
# consult the thinking guide, and write_file / update_symbol / run_command are
# not merely unused, they are unnameable - the structured-output grammar is
# built from this set, so the model is never told they exist.
ALLOWED = AGENT_ARCHITECT

DEMO_TASK = (
    "The coder agent needs to implement a shortest_path(graph, start, end) "
    "function using breadth-first search over a dict-of-lists graph "
    "representation (built by an existing add_edge helper)."
)


def plan(task: str) -> str:
    """Turn one instruction into todos on the shared board."""
    prompt = (
        f"What needs building:\n{task}\n\n"
        "Check the current shared todo list first with todo_read so you do "
        "not duplicate an existing entry or drop one that is already there. "
        "Then use todo_write to leave the todo items that would get this "
        "done, each owned by \"coder\", each clear and actionable on its own. "
        "Say in your final answer what you added and why you split it that way."
    )
    return run_agent(prompt, persona_prompt=PERSONA, allowed_tools=ALLOWED, workspace=WORKSPACE)


if __name__ == "__main__":
    ensure_utf8_console()
    task, mode = read_task(
        prompt="What should the architect plan? ",
        demo_task=DEMO_TASK,
    )

    if mode == "help":
        print_usage("architect_agent.py")
        sys.exit(0)
    if mode == "quit":
        sys.exit(0)
    if mode == "todos":
        # The architect has nothing to do without something to plan. Unlike
        # the coder, an existing board is not work for it - it is the output
        # it would be about to duplicate.
        print("The architect needs something to plan.\n")
        print("Current shared todo list:")
        print(planner.todo_read())
        print()
        print_usage("architect_agent.py")
        sys.exit(0)

    print("Shared todo list before this run:")
    print(planner.todo_read())
    print()

    print(plan(task))

    print("\nShared todo list after this run:")
    print(planner.todo_read())
    print("\nRun `python coder_agent.py --todos` to have the coder pick these up.")
