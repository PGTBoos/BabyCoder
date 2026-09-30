"""
babycoder.core

What every tool is built on:

  settings   - where the model is. Environment variables override it.
  console    - hooks for an agent with a person at the terminal.
  Workspace  - the folders a tool call reads and writes, including folders
               it may only read (the architect's view of the coder's work).
  Tool / P   - a callable that knows how to describe itself.
  sandbox    - path confinement, backups, transcript.

Nothing in here imports another babycoder module, so every other module
can import from it without a cycle.
"""

import contextlib
import difflib
import functools
import json
import os
import re
import shutil
import time

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
# The repo folder, where the agent scripts, sandboxes and state live. The
# package moved one level down; the folders on disk did not.
SCRIPT_DIR = os.path.dirname(PACKAGE_DIR)

MAX_READ_BYTES = 256 * 1024

# 0.2 for editing code, where there is one right answer. 0.8 for a character,
# where 0.2 gives flat, clipped sentences that repeat between turns.
CODING_TEMPERATURE = 0.2
CHARACTER_TEMPERATURE = 0.8


###############################################################################
# SETTINGS
###############################################################################

class Settings:
    """Where the model is and how to talk to it.

    Read from the environment so nobody has to edit code to point BabyCoder
    at their own server:
      BABYCODER_URL       default http://localhost:1234/v1/chat/completions
      BABYCODER_MODEL     default local-model
      BABYCODER_THINKING  1 to let reasoning models think (default: ask them not to)
    Or in code: configure_model(url=..., model=..., thinking=...).
    """

    def __init__(self):
        self.url = os.environ.get("BABYCODER_URL", "http://localhost:1234/v1/chat/completions")
        self.model = os.environ.get("BABYCODER_MODEL", "local-model")
        self.thinking = os.environ.get("BABYCODER_THINKING", "").lower() in ("1", "true", "yes")


settings = Settings()


def configure_model(url=None, model=None, thinking=None):
    """Change the model settings for this process. Returns the settings."""
    if url is not None:
        settings.url = url
    if model is not None:
        settings.model = model
    if thinking is not None:
        settings.thinking = bool(thinking)
    return settings


###############################################################################
# CONSOLE HOOKS
#
# An agent with a person at the terminal (Lisa) registers its console once.
# Every long call then answers to it: notices and progress go through the
# console's gated writer instead of print(), a generation is abandoned the
# moment the person starts typing, and run_command asks through it.
#
# One mutable object rather than module globals, so every module that did
# `from .core import console` sees the same hooks.
###############################################################################

class Console:
    def __init__(self):
        self.takes_turn = None   # () -> True when the person wants the turn
        self.progress = None     # (mark) -> show a progress mark inline
        self.on_tool = None      # (name) -> a tool is about to run
        self.on_result = None    # (name, args, result) -> a tool has run
        self.write = None        # (text) -> show a whole line
        self.approve = None      # (command, cwd) -> True to let it run


console = Console()


def set_console(takes_turn=None, progress=None, on_tool=None, write=None, approve=None,
                on_result=None):
    """Hand the toolkit the person's console. Called once, at startup.
    Calling it with no arguments detaches it again."""
    console.takes_turn, console.progress = takes_turn, progress
    console.on_tool, console.write, console.approve = on_tool, write, approve
    console.on_result = on_result


def notice(text, verbose=True):
    """Every [NOTICE] goes through here. A bare print() from deep inside a
    model call is what used to land in the middle of a line someone was
    typing at Lisa."""
    if not verbose:
        return
    if console.write:
        console.write(text)
    else:
        print(text)


###############################################################################
# WORKSPACE
###############################################################################

# Deliberately NOT per agent: the todo list is the handoff channel between an
# architect and a coder in different processes.
TODO_PATH = os.path.join(SCRIPT_DIR, "shared_todo.json")


class Workspace:
    """The folders one agent works in.

    workdir     - its private sandbox. Every file tool is confined to it.
    read_roots  - other agents' sandboxes it may READ, addressed as
                  "@label/path". Writing there is refused. This is how the
                  architect reviews what the coder wrote without being able
                  to touch it, and without the coder gaining anything back.
    state_dir   - backups, transcript and memory. Outside every workdir, so
                  no file tool can reach them.
    """

    def __init__(self, name="sandbox", read_roots=None):
        self.name = name
        self.workdir = self.folder_for(name)
        os.makedirs(self.workdir, exist_ok=True)
        self.state_dir = os.path.join(SCRIPT_DIR, ".agent_state" if name == "sandbox" else f".agent_state_{name}")
        os.makedirs(self.state_dir, exist_ok=True)
        self.backup_dir = os.path.join(self.state_dir, "backups")
        self.backup_manifest = os.path.join(self.backup_dir, "manifest.jsonl")
        self.transcript_path = os.path.join(self.state_dir, "transcript.jsonl")
        # A folder of markdown files (memories.md, goals.md, dreams.md, state.md).
        self.memory_path = os.path.join(self.state_dir, "memory")
        # An agent may have its own guide (lisa_thinking_guide.md is written
        # to her personally); everyone else shares thinking_guide.md.
        own_guide = os.path.join(SCRIPT_DIR, f"{name}_thinking_guide.md")
        self.guide_path = own_guide if os.path.exists(own_guide) else os.path.join(SCRIPT_DIR, "thinking_guide.md")
        self.read_roots = {}
        for label, other in (read_roots or {}).items():
            folder = self.folder_for(other)
            os.makedirs(folder, exist_ok=True)
            self.read_roots[label] = folder

    @staticmethod
    def folder_for(name):
        """<SCRIPT_DIR>/sandbox for the coder, <SCRIPT_DIR>/sandbox_<name> for everyone else."""
        return os.path.realpath(os.path.join(SCRIPT_DIR, name if name == "sandbox" else f"sandbox_{name}"))

    def __repr__(self):
        extra = f", reads {sorted(self.read_roots)}" if self.read_roots else ""
        return f"<workspace {self.name}{extra}>"

    def resolve(self, path, write=False):
        """Turn a tool's path into a real path, refusing anything outside the
        workdir, or outside a read root, or any write into a read root."""
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        base, rel = self.workdir, path
        if path.startswith("@"):
            label, _, rel = path[1:].partition("/")
            if label not in self.read_roots:
                known = ", ".join(f"@{l}" for l in sorted(self.read_roots)) or "none"
                raise ValueError(f"there is no folder called @{label} (available: {known})")
            if write:
                raise ValueError(f"@{label} is read-only for this agent")
            base, rel = self.read_roots[label], rel or "."
        if os.path.isabs(rel):
            raise ValueError(f"absolute paths are not allowed: {path}")
        full = os.path.realpath(os.path.join(base, rel))
        if full != base and not full.startswith(base + os.sep):
            raise ValueError(f"path escapes sandbox: {path}")
        return full

    def display(self, full):
        """The path a model should use for `full`: plain inside the workdir,
        @label/... inside a read root."""
        for label, root in self.read_roots.items():
            if full == root or full.startswith(root + os.sep):
                return f"@{label}/" + os.path.relpath(full, root).replace(os.sep, "/")
        return os.path.relpath(full, self.workdir).replace(os.sep, "/")


_current = None


def ws():
    """The workspace tools act in right now. Created on first use, so merely
    importing babycoder no longer makes folders on disk."""
    global _current
    if _current is None:
        _current = Workspace("sandbox")
    return _current


def configure_workspace(name="sandbox", read_roots=None):
    """Make agent `name`'s folders the default for this process. Returns the
    Workspace, which can also be handed to run_agent(workspace=...)."""
    global _current
    _current = Workspace(name, read_roots)
    return _current


@contextlib.contextmanager
def using(workspace):
    """Run a block in `workspace`, then put the previous one back. run_agent
    does this for its whole run, so a room can hand each turn its own
    workspace without anyone reconfiguring globals between turns."""
    global _current
    previous = _current
    _current = workspace
    try:
        yield workspace
    finally:
        _current = previous


###############################################################################
# TOOL OBJECTS
###############################################################################

def P(type_="string", desc="", optional=False, enum=None, items=None):
    """One argument of a tool: its JSON type, and whether it is required."""
    spec = {"type": type_}
    if desc:
        spec["description"] = desc
    if enum:
        spec["enum"] = list(enum)
    if items:
        spec["items"] = items
    return spec, not optional


class Tool:
    """A callable tool that knows how to describe itself.

    Calling it calls the function. Everything a model is told about the tool
    (schema, reference line, help, the hint after a bad call) comes from
    `help` and `params`, so there is one place to change either.
    """

    def __init__(self, fn, name, help, params):
        self.fn = fn
        self.name = name
        self.help = " ".join(help.split())
        self.params = params            # arg name -> (json spec, required)
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"<tool {self.signature()}>"

    @property
    def short(self):
        """First sentence of the help, for listings."""
        # Split at a sentence end followed by a capital, so "e.g. 'x'" survives.
        first = re.split(r"(?<=[.!?])\s+(?=[A-Z])", self.help, maxsplit=1)[0]
        return first if len(first) <= 90 else first[:87].rsplit(" ", 1)[0] + "..."

    def signature(self):
        """read_file(path) / list(target, path?) - '?' marks optional."""
        args = [a if req else f"{a}?" for a, (_, req) in self.params.items()]
        return f"{self.name}({', '.join(args)})"

    def usage(self):
        """Full help plus the arguments, as help(tool_name=...) answers."""
        parts = []
        for arg, (spec, req) in self.params.items():
            text = arg if req else f"{arg} (optional)"
            if spec.get("enum"):
                text += f" [one of: {', '.join(spec['enum'])}]"
            parts.append(text)
        return f"{self.help} Expected arguments: {', '.join(parts) or '(no arguments)'}."

    def schema(self):
        """The OpenAI function-calling schema, for native tool mode."""
        return {"type": "function", "function": {
            "name": self.name,
            "description": self.help,
            "parameters": {"type": "object",
                           "properties": {a: s for a, (s, _) in self.params.items()},
                           "required": [a for a, (_, r) in self.params.items() if r]}}}


def tool(help, tool_name=None, **params):
    """Decorator: turn a function into a Tool with this help and these args.
    tool_name overrides the function name (it is not called `name`, because
    several tools have an argument called name)."""
    def wrap(fn):
        return Tool(fn, tool_name or fn.__name__, help, params)
    return wrap


###############################################################################
# SANDBOX AND BOOKKEEPING
###############################################################################

def _full_path(path, write=False):
    return ws().resolve(path, write)


def _backup(path):
    """Snapshot a file before it is overwritten. The original path goes in a
    manifest rather than the backup's filename, so none can be misread."""
    w = ws()
    full = w.resolve(path, write=True)
    if not os.path.isfile(full):
        return
    os.makedirs(w.backup_dir, exist_ok=True)
    backup_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}.bak"
    shutil.copy2(full, os.path.join(w.backup_dir, backup_name))
    with open(w.backup_manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps({"backup": backup_name, "original": path}) + "\n")


def _read_source(path):
    full = _full_path(path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"{path} does not exist")
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def _write_source(path, content):
    full = _full_path(path, write=True)
    _backup(path)
    os.makedirs(os.path.dirname(full) or ws().workdir, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def _match_indent(code, indent):
    """Dedent `code` to its common leading level, then re-indent by `indent`."""
    lines = code.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return code
    common = min(len(l) - len(l.lstrip()) for l in non_empty)
    return "\n".join(indent + l[common:] if l.strip() else l for l in lines)


def _log_event(event):
    try:
        with open(ws().transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **event}) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Soft "not found" results
#
# A target not being where it was expected is often benign (already applied,
# wrong file). NOTFOUND: results do not abort a batch the way ERROR: does,
# and carry a fuzzy-match hint so the model can tell a typo from an absence.
# ---------------------------------------------------------------------------

def _fuzzy_hint(target, candidates):
    candidates = sorted({c for c in candidates if c})
    if not candidates:
        return "There's nothing else of that kind here either, so this probably isn't a typo."
    close = difflib.get_close_matches(target, candidates, n=3, cutoff=0.6)
    if close:
        return f"Close matches that do exist: {', '.join(close)}. This might be a typo for one of those."
    return ("Nothing similar exists here either, so this is likely not a typo, just genuinely "
            "absent, unused, or already handled.")


def _closest_line_hint(source, target):
    first_line = next((l for l in target.splitlines() if l.strip()), target).strip()
    lines = [l.strip() for l in source.splitlines() if l.strip()]
    if not lines:
        return "The file is empty, so nothing could match."
    close = difflib.get_close_matches(first_line, lines, n=1, cutoff=0.6)
    if close:
        return f"Closest existing line: {close[0]!r}. This might be a typo, or a whitespace/indentation mismatch."
    return ("Nothing similar exists in the file either, so this is likely not a typo, just "
            "already applied, the wrong file, or the wrong target.")


def _soft_not_found(subject, target, hint):
    return " ".join(
        f"NOTFOUND: {subject} '{target}' was not found. {hint} This is a soft result, not "
        "necessarily a mistake, so if it wasn't a typo it's fine to move on to the rest of "
        "the task rather than treating it as a failure.".split())
