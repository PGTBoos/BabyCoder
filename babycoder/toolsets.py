"""
babycoder.toolsets

Tools grouped into capabilities, and the ToolBox that is one agent's view of
them.

    from babycoder import memory, research, guidance

That import line is what an agent can do. Each toolset also carries a
`usage` note; an agent's system prompt gets the notes for what it was granted.
"""

from .core import P, Tool, _fuzzy_hint, _soft_not_found
from .tools import coding as _c
from .tools import dreams as _d
from .tools import guidance as _g
from .tools import memory as _m
from .tools import planner as _p
from .tools import research as _r


class Toolset:
    """A named group of Tools, importable as one capability."""

    def __init__(self, name, doc, *tools, usage=""):
        self.name = name
        self.__doc__ = doc
        self.usage = usage
        self._tools = {t.name: t for t in tools}
        for t in tools:
            setattr(self, t.name, t)
            setattr(self, t.fn.__name__, t)   # coding.list_things as well as coding.list

    def names(self):
        return set(self._tools)

    def tools(self):
        return list(self._tools.values())

    def __or__(self, other):
        return self.names() | (other.names() if isinstance(other, Toolset) else set(other))

    __ror__ = __or__

    def __sub__(self, other):
        return self.names() - (other.names() if isinstance(other, Toolset) else set(other))

    def __contains__(self, item):
        return item in self._tools

    def __iter__(self):
        return iter(self._tools)

    def __len__(self):
        return len(self._tools)

    def __repr__(self):
        return f"<toolset {self.name}: {', '.join(sorted(self._tools))}>"


coding = Toolset(
    "coding", "Editing code in a sandbox, with backups.",
    _c.read_file, _c.write_file, _c.list_things, _c.search_files, _c.replace_in_file,
    _c.replace_in_project, _c.read_symbol, _c.update_symbol, _c.insert_symbol, _c.delete_symbol,
    _c.rename_symbol, _c.find_references, _c.add_import, _c.remove_import, _c.check_syntax,
    _c.format_file, _c.add_dependency, _c.lookup_symbol_docs, _c.restore_backup, _c.run_command,
    usage="Look at what is there before you change it, change one thing at a time, and run "
          "check_syntax after an edit rather than assuming it worked.")

reading = Toolset(
    "reading", "Looking at code without touching it.",
    _c.read_file, _c.list_things, _c.search_files, _c.read_symbol, _c.find_references,
    _c.lookup_symbol_docs)

planner = Toolset(
    "planner", "The shared todo board, the handoff between architect and coder.",
    _p.todo_read, _p.todo_write,
    usage="todo_read before you todo_write: the write replaces the whole list, so write back "
          "every item, not just the one you touched. Finishing an item means setting its status "
          "to \"done\", not deleting it. Whoever picks an item up may do so hours later with none "
          "of your context, so each one has to stand on its own.")

research = Toolset(
    "research", "Looking things up on Wikipedia and the web.",
    _r.search_wikipedia, _r.search_web, _r.research_topic,
    usage="Look something up rather than guessing at it. Say what you found and where, and say "
          "plainly when a search turned up nothing useful instead of filling the gap yourself.")

guidance = Toolset(
    "guidance", "The structured-thinking guide.",
    _g.read_guide,
    usage="Reach for read_guide when you are stuck or going in circles, rather than pushing "
          "harder at the same approach.")

memory = Toolset(
    "memory", "Long-term memory and what is on your mind.",
    _m.memory_count, _m.memory_at, _m.memory_recent, _m.memory_oldest, _m.memory_random,
    _m.memory_find, _m.remember, _m.consolidate, _m.goals_list, _m.goal_add, _m.goal_close,
    _m.dreams_recent, _m.about_person,
    usage="Look before you answer something you may already have been told. Reach for "
          "memory_oldest and memory_random as well as recent. Keep what you would want to know "
          "next time, not what you can work out again. Put something on your mind with goal_add, "
          "and let it go with goal_close once it is answered or stops mattering. about_person is "
          "for people you have talked to: what they have told you about themselves, which no "
          "lookup anywhere could have told you.")

dreams = Toolset(
    "dreams", "Dreaming and tidying memory.",
    _d.dream, _d.tidy_memories,
    usage="Dreaming is not reasoning and does not have to make sense. Dream first, then tidy.")

TOOLSETS = {t.name: t for t in (coding, reading, planner, research, guidance, memory, dreams)}

# Every tool, by the name a model calls it by. Built from the toolsets, so it
# cannot fall out of step with them.
REGISTRY = {}
for _set in TOOLSETS.values():
    REGISTRY.update(_set._tools)
del _set


###############################################################################
# TOOLBOX - one agent's view of the registry
#
# Built by run_agent from allowed_tools. Its help, reference, schema, grammar
# and dispatch all contain only the granted tools. An ungranted tool is
# answered exactly like one that does not exist, so help cannot be used to
# discover the rest of the catalogue.
###############################################################################

HELP = Tool(lambda tool_name=None: None, "help",
            "Look up exactly how to call a tool. With no tool_name, lists every tool you "
            "have. Call this before guessing at a tool's arguments.",
            {"tool_name": P("string", optional=True)})


class ToolBox:
    """The tools one agent may use, plus the help generated from them."""

    def __init__(self, allowed=None):
        names = set(REGISTRY) if allowed is None else set(allowed) - {"help"}
        unknown = names - set(REGISTRY)
        if unknown:
            raise ValueError(f"granted tools that do not exist: {sorted(unknown)}")
        self.tools = {n: REGISTRY[n] for n in sorted(names)}

    def names(self):
        """Granted names plus help, which every agent always has: one that
        cannot ask what it is allowed to do cannot work within a restriction."""
        return set(self.tools) | {"help"}

    def get(self, name):
        return HELP if name == "help" else self.tools.get(name)

    def help(self, tool_name=None):
        if tool_name:
            t = self.get(tool_name)
            if t is None:
                return _soft_not_found("tool", tool_name, _fuzzy_hint(tool_name, self.names()))
            return f"{t.signature()}\n{t.usage()}"
        # Generic help, derived from each granted tool, grouped by what it is
        # for. Toolsets granted whole come first, so an architect's read tools
        # are listed under "reading" rather than under the "coding" set it lacks.
        lines, shown = ["Your tools:"], set()
        granted = set(self.tools)
        for ts in sorted(TOOLSETS.values(), key=lambda ts: not ts.names() <= granted):
            mine = [t for t in ts.tools() if t.name in granted and t.name not in shown]
            if not mine:
                continue
            lines.append(f"\n{ts.name} - {ts.__doc__}")
            for t in mine:
                lines.append(f"  {t.signature()} - {t.short}")
                shown.add(t.name)
        lines.append("\nCall help(tool_name='...') for the full description and arguments of one tool.")
        return "\n".join(lines)

    def usage(self, name):
        t = self.get(name)
        return t.usage() if t else ""

    def reference(self):
        """One line per tool for the system prompt."""
        return "\n".join(f"{t.signature()}: {t.help}" for t in list(self.tools.values()) + [HELP])

    def usage_notes(self):
        return [ts.usage for ts in TOOLSETS.values() if ts.usage and ts.names() & set(self.tools)]

    def schemas(self):
        return [t.schema() for t in self.tools.values()] + [HELP.schema()]

    def response_format(self):
        return {"type": "json_schema",
                "json_schema": {"name": "tool_call_batch", "strict": True,
                                "schema": tool_call_schema(self.names())}}

    def call(self, name, args):
        if name == "help":
            return self.help(**args)
        return self.tools[name](**args)


def tool_call_schema(names):
    """Grammar for structured mode. Tool names are an enum, so a model cannot
    emit a call to a tool it was not granted."""
    return {
        "type": "object",
        "properties": {
            "notes": {"type": ["string", "null"]},
            "calls": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": sorted(names)},
                               "arguments": {"type": "object"}},
                "required": ["name", "arguments"]}},
            "final_answer": {"type": ["string", "null"]},
        },
        "required": ["notes", "calls", "final_answer"],
    }


def help_tool(tool_name=None, allowed_tools=None):
    """Help as an agent with this grant would see it."""
    return ToolBox(allowed_tools).help(tool_name)


# Compatibility views for older code and tests.
DISPATCH = dict(REGISTRY, help=help_tool)
TOOLS = [t.schema() for t in REGISTRY.values()] + [HELP.schema()]
TOOL_BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}


###############################################################################
# AGENT GRANT SETS
#
# Two ways an agent is bounded:
#   1. Its import line, for an agent whose own loop calls tools (Lisa's sleep).
#   2. Its grant, for an agent whose MODEL picks tools through run_agent. The
#      grant builds the ToolBox, so an ungranted tool is not merely refused,
#      it is unnameable.
###############################################################################

CODER_TOOLS = coding.names()
READONLY_CODE_TOOLS = reading.names()
TODO_TOOLS = planner.names()
RESEARCH_TOOLS = research.names()
GUIDANCE_TOOLS = guidance.names()
MEMORY_TOOLS = memory.names()
DREAM_TOOLS = dreams.names()

AGENT_CODER = coding | planner | guidance
AGENT_ARCHITECT = reading | planner | guidance
AGENT_LISA = memory | dreams | research | guidance
# Dreaming happens TO her when she is tired; it is not a move she can pick.
# consolidate is withheld too: invariant 3, only sleep merges memory.
AGENT_LISA_AWAKE = (memory | research | guidance) - {"consolidate"}
AGENT_DESIGNER = research | guidance | memory

AGENT_GRANTS = {"coder": AGENT_CODER, "architect": AGENT_ARCHITECT, "lisa": AGENT_LISA,
                "lisa_awake": AGENT_LISA_AWAKE, "designer": AGENT_DESIGNER}


def _assert_grants_are_real():
    """Fail at import, not at run time, on a grant naming a missing tool."""
    for agent, granted in AGENT_GRANTS.items():
        unknown = set(granted) - set(REGISTRY)
        if unknown:
            raise RuntimeError(f"agent {agent!r} is granted tools that do not exist: {sorted(unknown)}")


_assert_grants_are_real()
