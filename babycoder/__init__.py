"""
babycoder - a sandboxed tool layer for small local LLM agents.

An agent takes capabilities by importing toolsets:

    from babycoder import memory, research, guidance, run_agent

and, when a model picks the tools, by granting them:

    run_agent(task, allowed_tools=memory | research | guidance)

Layout:
    core.py        settings, console hooks, Workspace, Tool, sandbox helpers
    backends.py    per-language symbol backends (Python real, others stubs)
    advisories.py  post-edit diagnostics folded into tool results
    model.py       streaming model calls, thinking control, interruption
    toolsets.py    Toolset, ToolBox (one agent's view), grant sets
    loop.py        run_agent
    tools/         one module per capability: coding, planner, research,
                   guidance, memory, dreams
"""

from . import model as _model
from .core import (CHARACTER_TEMPERATURE, CODING_TEMPERATURE, SCRIPT_DIR, TODO_PATH, P, Tool,
                   Workspace, configure_model, configure_workspace, console, notice, set_console,
                   settings, tool, using, ws)
from .loop import (INTERRUPTED, MAX_TOOL_RESULT_CHARS, NO_ANSWER_MARKER, UNREACHABLE, gave_no_answer,
                   model_unreachable, run_agent, was_interrupted)
from .model import (INTERRUPTED_NOTFOUND, MAX_TOKENS_CEILING, MAX_TOKENS_DEFAULT,
                    RUNAWAY_THINKING_CHARS, ask_model, call_model)
from .toolsets import (AGENT_ARCHITECT, AGENT_CODER, AGENT_DESIGNER, AGENT_GRANTS, AGENT_LISA,
                       AGENT_LISA_AWAKE, CODER_TOOLS, DISPATCH, DREAM_TOOLS, GUIDANCE_TOOLS, HELP,
                       MEMORY_TOOLS, READONLY_CODE_TOOLS, REGISTRY, RESEARCH_TOOLS, TODO_TOOLS,
                       TOOL_BY_NAME, TOOLS, TOOLSETS, ToolBox, Toolset, coding, dreams, guidance,
                       help_tool, memory, planner, reading, research)
from .tools.memory import DREAM_KEEP, MAX_GOALS, MAX_MEMORIES, MemorySystem, active_memory


def reset_model_probe(result=None):
    """Forget which thinking-off keys work, or set them. A mock server in a
    test passes {} so the probe does not eat its first scripted reply."""
    _model._quiet_cache = result
