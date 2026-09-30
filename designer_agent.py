"""
designer_agent.py

The DESIGNER role, and the test of whether putting every capability in
the babycoder package was worth doing.

This file contains no tools. Not one. It is a persona, a grant, and the same
four input routes every other driver has.

Note which mechanism bounds this one, because it differs from Lisa's. Lisa's
own loop calls tools, so her import line IS her boundary: she does not import
`coding`, so there is no name to call. The designer is model-driven - it hands
work to run_agent and the MODEL picks tools - so what bounds it is the grant.
AGENT_DESIGNER is research | guidance | memory, and run_agent enforces that at
dispatch AND builds the structured-output grammar from it, so write_file and
run_command are not merely refused, they are unnameable: the model is never
told they exist. This file imports only `memory`, and only because it prints
memory_count itself.

That is the whole point of the layout. Adding a role is a union, not a module.

  python designer_agent.py                        asks what to design
  python designer_agent.py "a dark mode palette"  from the command line
  python designer_agent.py --file brief.md        from a written brief

It keeps its own memory, in .agent_state_designer/memory/, separate from
every other agent's for the same reason each has its own sandbox. So a second
run can recall what was decided in the first rather than starting blank.

Run:
    pip install requests
    python designer_agent.py
"""

import sys

from babycoder import AGENT_DESIGNER, configure_workspace, memory, run_agent
from agent_common import ensure_utf8_console, print_usage, read_task

configure_workspace("designer")

# Role only. What tools it has and how to use each capability well come from
# babycoder, generated from AGENT_DESIGNER and the usage note each toolset
# carries. Restating "recall before you start" here would be a second copy of
# something the memory toolset already says, free to drift from it.
PERSONA = (
    "You are the DESIGNER agent. You work out how something should look and "
    "behave, and you write that down as a clear brief someone else could "
    "build from. You do not write code and you have no tools for it.\n\n"
    "Prior art is research, not cheating - what already exists is usually "
    "worth knowing before inventing.\n\n"
    "Be concrete. 'Clean and modern' is not a decision. 'Two typefaces, one "
    "for headings at 32/40, one for body at 16/24' is."
)

DEMO_TASK = (
    "Design the terminal output for a long-running agent that thinks and "
    "dreams while nobody is watching: what a person should see when they "
    "come back after an hour away, and what should be on screen while it "
    "works. Research how other long-running CLI tools handle this."
)


if __name__ == "__main__":
    ensure_utf8_console()
    task, mode = read_task(
        prompt="What should the designer work on? ",
        demo_task=DEMO_TASK,
    )

    if mode == "help":
        print_usage("designer_agent.py")
        sys.exit(0)
    if mode == "quit":
        sys.exit(0)
    if mode == "todos":
        # The designer has no todo tools - it is not part of the architect and
        # coder's handoff. Say that plainly rather than starting a run that
        # cannot do what was asked.
        print("The designer needs something to design.\n")
        print("What it remembers so far:")
        print(memory.memory_count())
        print()
        print_usage("designer_agent.py")
        sys.exit(0)

    print(run_agent(task, persona_prompt=PERSONA, allowed_tools=AGENT_DESIGNER))
