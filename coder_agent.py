"""
coder_agent.py

A single-purpose driver for agent_toolkit's shared coding-agent core: a
general-purpose implementer with no special persona layered on top (see
agent_toolkit._build_system_content for what persona_prompt is for, if
a future driver - a reviewer, a tester - wants to add one). This is the
same demo task that used to live in tool_calling.py's own __main__
block; only the file it lives in changed, not what it does.

Run:
    pip install requests
    python coder_agent.py
"""

from agent_toolkit import write_file, run_agent

if __name__ == "__main__":
    # Always reset to a known-clean starting file before each run (rather
    # than "write it only if missing"), since the previous demo file kept
    # its mutations from run to run and drifted across several unrelated
    # tasks until it no longer matched any of them. write_file backs up
    # whatever was there before overwriting it, so nothing is lost.
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

    # A real problem to solve, not just a formula to rename: this checks
    # whether the model can actually write correct BFS, not only whether
    # the tool plumbing works.
    prompt = (
        "In graph_tools.py, list the symbols, then read shortest_path to "
        "see its current unfinished state. Then use update_symbol to "
        "rewrite shortest_path so it does a real breadth-first search over "
        "`graph` (a dict mapping each node to a list of neighbors, as built "
        "by add_edge) and returns the list of nodes on a shortest path from "
        "start to end, inclusive, or None if no path exists. Run "
        "check_syntax afterwards. Finally propose (but do not assume "
        "approval for) a `python -c` command that builds a small graph "
        "with add_edge and calls shortest_path on it to sanity check the "
        "result."
    )
    print(run_agent(prompt))
