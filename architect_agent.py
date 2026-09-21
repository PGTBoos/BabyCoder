"""
architect_agent.py

Second driver, demonstrating the two things coder_agent.py alone
doesn't: a private workspace and a restricted tool set.

- configure_workspace("architect") points every tool this agent calls at
  <SCRIPT_DIR>/sandbox_architect and its own .agent_state_architect,
  completely separate from coder_agent.py's sandbox_coder (or its
  legacy default "sandbox"). Neither agent can reach into the other's
  folder through any tool - not because of new access-control code, but
  because the sandbox escape check every tool already goes through has
  nothing pointing at the other agent's path to begin with.
- allowed_tools restricts this agent to reading and leaving todos: it
  can look around, but calling write_file, update_symbol, run_command,
  etc. gets rejected by run_agent before the call ever executes, not
  merely left out of persona_prompt's request. Widen ALLOWED as this
  agent's role grows (add find_references, lookup_symbol_docs, whatever
  it turns out to need) rather than granting everything up front.
- todo_write/todo_read are deliberately NOT sandboxed per-agent (see
  TODO_PATH in agent_toolkit.py): this is what lets the architect leave
  tasks a separately-run coder_agent.py can read, despite the two having
  no other visibility into each other's files at all.

One known gap, called out rather than silently worked around: this
agent can only read ITS OWN sandbox_architect folder, not the coder's
actual code - useful for planning from a spec or from prior todos, not
yet for reviewing code the coder already wrote. Giving the architect
read-only access to the coder's folder specifically (without write
access, and without the coder gaining any reciprocal access) would need
a second, explicitly read-only root in the sandbox check, which
_full_path doesn't have today - a real next step if this agent's role
grows into code review, not something this file works around on its
own.

Run:
    pip install requests
    python architect_agent.py
"""

from agent_toolkit import configure_workspace, run_agent, todo_read

configure_workspace("architect")

PERSONA = (
    "You are the ARCHITECT agent in a multi-agent pipeline. Your job is "
    "to plan and leave clear, actionable todos in markdown for a separate CODER "
    "agent to implement - not to write or run code yourself. You do not "
    "have write_file, update_symbol, or run_command; use todo_write to "
    "hand off work instead."
)

ALLOWED = {
    "read_file", "list", "search_files", "list_dependencies",
    "lookup_symbol_docs", "todo_read", "todo_write",
}

if __name__ == "__main__":
    print("Current shared todo list before this run:")
    print(todo_read())
    print()

    prompt = (
        "The coder agent needs to implement a shortest_path(graph, start, "
        "end) function using breadth-first search over a dict-of-lists "
        "graph representation (built by an existing add_edge helper). "
        "Check the current shared todo list first with todo_read so you "
        "don't duplicate an existing entry, then use todo_write to leave "
        "one clear, actionable todo describing exactly what shortest_path "
        "needs to do, including its return value for the no-path case."
    )
    print(run_agent(prompt, persona_prompt=PERSONA, allowed_tools=ALLOWED))
