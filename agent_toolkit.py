"""
agent_toolkit.py

Shared core for coding agents: sandboxed file tools, AST-based symbol
tools behind a per-language dispatch table, the post-edit advisory
checks (duplicate definitions, unexplained loops, missing docstrings,
test-diversity gating), and the run_agent loop that drives an LLM
through them. Only Python has a real language backend; csharp/typescript
are registered as stubs so the hook is already in place for Roslyn / the
TS compiler API. Language is picked from file extension, or passed
explicitly, with no fallback guess when the extension is unknown, the
file type decides.

This module is meant to be imported, not run directly - it has no
__main__ block of its own. A per-agent driver file (e.g. coder_agent.py)
imports run_agent and the tools it needs, supplies its own task and,
optionally, its own persona_prompt (see _build_system_content /
run_agent) layered on top of the shared tool-usage rules below, and
calls run_agent(...) from its own if __name__ == "__main__": block.
Every driver that imports this module shares the same WORKDIR,
.agent_state, backups, and transcript - anchored to *this* file's own
location (SCRIPT_DIR, from __file__), not the driver's - so multiple
agents (a coder, later maybe a reviewer or tester) operate on the same
sandboxed project by default, which is the natural setup for a
multi-agent pipeline where different roles collaborate on one codebase
rather than each getting an isolated copy of it.

Safety features:
  * Path resolution is sandboxed (no absolute paths, no traversal, no
    symlink escape).
  * Every write is backed up first, into .agent_state/backups/ next to
    this module, outside the sandbox the model can see. Tracked via a
    manifest so the original path is never reconstructed by parsing the
    backup filename. Because it sits outside WORKDIR, no tool call can
    read, list, or search into it, the model has no path that leads
    there.
  * Every edit is re-parsed before it is committed; invalid edits roll back.
  * Renames are token-based, so strings and comments are not touched.
  * run_command requires explicit human approval and shows the cwd.
  * All tool calls and results are written to a JSONL transcript.

Use from a driver file:
    from agent_toolkit import run_agent
    print(run_agent("your task here"))
"""

import ast
import difflib
import fnmatch
import functools
import io
import json
import keyword
import os
import platform
import pydoc
import re
import shutil
import subprocess
import time
import tokenize

import requests

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MAX_READ_BYTES = 256 * 1024
MAX_MESSAGES_IN_CONTEXT = 40
MAX_TOKENS_DEFAULT = 2048

# WORKDIR and friends start pointed at a default "sandbox" - single-agent
# scripts that never call configure_workspace() behave exactly as before.
# A multi-agent setup calls configure_workspace(name) once, before any
# tool is used, to point this same shared module at an agent-specific
# folder instead. Every tool below reads these by name at call time
# (Python resolves a bare global name against the module's namespace on
# each call, not once at def time), so reconfiguring these six names is
# enough to redirect every existing tool - no per-function changes
# needed, and none of the ~2000 lines below this point had to change to
# get per-agent sandboxes.
WORKDIR = None
STATE_DIR = None
BACKUP_DIR = None
BACKUP_MANIFEST = None
TRANSCRIPT_PATH = None


def configure_workspace(name: str = "sandbox") -> None:
    """Point this toolkit's sandbox, state, backups, and transcript at
    <SCRIPT_DIR>/<name> and <SCRIPT_DIR>/.agent_state_<name> instead of
    the defaults. Call this once, before any tool function runs, from a
    driver file that wants its own private folder rather than sharing
    the default "sandbox" with every other driver that imports this
    module.

    Each agent's WORKDIR is genuinely private to it: _full_path's
    existing escape check (no absolute paths, no traversal, no symlink
    escape) already refuses anything outside the *currently configured*
    WORKDIR, so once two drivers call this with different names, neither
    can reach the other's folder through any tool - not because of new
    access-control code, just because the sandbox check that already
    existed has nothing pointing at the other agent's path to begin
    with. Backups and the transcript become per-agent alongside WORKDIR
    for the same reason a coder's backups mean nothing in an architect's
    context, and vice versa.

    TODO_PATH is deliberately NOT reconfigured here and stays fixed
    directly under SCRIPT_DIR regardless of which workspace is active -
    see its own definition for why: it's the intended handoff channel
    between agents (an architect leaving todos a coder can read), so it
    has to be the one thing that stays shared even while everything else
    becomes private.
    """
    global WORKDIR, STATE_DIR, BACKUP_DIR, BACKUP_MANIFEST, TRANSCRIPT_PATH
    folder = name if name == "sandbox" else f"sandbox_{name}"
    WORKDIR = os.path.realpath(os.path.join(SCRIPT_DIR, folder))
    os.makedirs(WORKDIR, exist_ok=True)
    STATE_DIR = os.path.join(SCRIPT_DIR, f".agent_state_{name}" if name != "sandbox" else ".agent_state")
    os.makedirs(STATE_DIR, exist_ok=True)
    BACKUP_DIR = os.path.join(STATE_DIR, "backups")
    BACKUP_MANIFEST = os.path.join(BACKUP_DIR, "manifest.jsonl")
    TRANSCRIPT_PATH = os.path.join(STATE_DIR, "transcript.jsonl")


# Backups, the manifest, and the transcript are orchestration bookkeeping,
# not part of the project the model is working on. They live as a sibling
# of the sandbox, not inside it, so no tool call can ever reach them, the
# same _full_path sandbox check that blocks path traversal blocks this too.
# The model never sees these paths in a list_dir or search_files result
# because they are not under WORKDIR at all, not because of a filter.
configure_workspace("sandbox")


# ---------------------------------------------------------------------------
# Sandbox-safe filesystem helpers
# ---------------------------------------------------------------------------

def _full_path(path: str) -> str:
    """Resolve `path` against WORKDIR and refuse anything that escapes it."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if os.path.isabs(path):
        raise ValueError(f"absolute paths are not allowed: {path}")
    full = os.path.realpath(os.path.join(WORKDIR, path))
    if full != WORKDIR and not full.startswith(WORKDIR + os.sep):
        raise ValueError(f"path escapes sandbox: {path}")
    return full


def _backup(path: str) -> None:
    """Snapshot an existing file into BACKUP_DIR before it is overwritten.

    The original path is recorded in a manifest rather than encoded into
    the backup filename, so a path containing a literal double underscore
    or an unusual character can never be misread on restore.
    """
    full = _full_path(path)
    if not os.path.exists(full) or not os.path.isfile(full):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    ns = time.time_ns() % 1_000_000_000
    backup_name = f"{ts}-{ns:09d}.bak"
    shutil.copy2(full, os.path.join(BACKUP_DIR, backup_name))
    with open(BACKUP_MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps({"backup": backup_name, "original": path}) + "\n")


def _read_source(path: str) -> str:
    full = _full_path(path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"{path} does not exist")
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def _write_source(path: str, content: str) -> None:
    _backup(path)
    full = _full_path(path)
    os.makedirs(os.path.dirname(full) or WORKDIR, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def _match_indent(code: str, indent: str) -> str:
    """Dedent `code` to its common leading level, then re-indent by `indent`."""
    lines = code.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return code
    common = min(len(l) - len(l.lstrip()) for l in non_empty)
    out = []
    for l in lines:
        if l.strip():
            out.append(indent + l[common:])
        else:
            out.append(l)
    return "\n".join(out)


def _log_event(event: dict) -> None:
    try:
        with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **event}) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Soft "not found" results
#
# A target (symbol, import, anchor, literal text) not being where it was
# expected is very often benign: already applied, wrong file, wrong name,
# not a typo at all. Treating it the same as a broken assumption (the
# ERROR: prefix, which aborts the rest of a batched set of tool calls in
# run_agent) punishes that common case. These NOTFOUND: results are
# non-aborting on purpose, and carry a fuzzy-match hint so the model can
# tell "this looks like a typo" from "nothing like this exists here" for
# itself instead of guessing.
# ---------------------------------------------------------------------------

def _fuzzy_hint(target: str, candidates) -> str:
    """Compare `target` against known `candidates` and say in natural
    language whether this looks like a typo or a genuine absence."""
    candidates = sorted({c for c in candidates if c})
    if not candidates:
        return "There's nothing else of that kind here either, so this probably isn't a typo."
    close = difflib.get_close_matches(target, candidates, n=3, cutoff=0.6)
    if close:
        return f"Close matches that do exist: {', '.join(close)}. This might be a typo for one of those."
    return "Nothing similar exists here either, so this is likely not a typo, just genuinely absent, unused, or already handled."


def _closest_line_hint(source: str, target: str) -> str:
    """Same idea as _fuzzy_hint, but for an arbitrary literal text search
    (replace_in_file) rather than a known identifier: compare against the
    file's own lines instead of a fixed candidate list."""
    first_line = next((l for l in target.splitlines() if l.strip()), target).strip()
    lines = [l.strip() for l in source.splitlines() if l.strip()]
    if not lines:
        return "The file is empty, so nothing could match."
    close = difflib.get_close_matches(first_line, lines, n=1, cutoff=0.6)
    if close:
        return f"Closest existing line: {close[0]!r}. This might be a typo, or a whitespace/indentation mismatch."
    return "Nothing similar exists in the file either, so this is likely not a typo, just already applied, the wrong file, or the wrong target."


def _soft_not_found(subject: str, target: str, hint: str) -> str:
    return (
        f"NOTFOUND: {subject} '{target}' was not found. {hint} "
        "This is a soft result, not necessarily a mistake, so if it wasn't a "
        "typo it's fine to move on to the rest of the task rather than "
        "treating it as a failure."
    )


# ---------------------------------------------------------------------------
# 1. Language backends
# ---------------------------------------------------------------------------

class LanguageBackend:
    """Interface every language backend must implement."""

    def list_symbols(self, path: str) -> str:
        raise NotImplementedError

    def read_symbol(self, path: str, name: str, scope: str = None) -> str:
        raise NotImplementedError

    def update_symbol(self, path: str, name: str, new_code: str, scope: str = None) -> str:
        raise NotImplementedError

    def insert_symbol(self, path: str, code: str, after: str = None, scope: str = None) -> str:
        raise NotImplementedError

    def list_imports(self, path: str) -> str:
        raise NotImplementedError

    def add_import(self, path: str, statement: str) -> str:
        raise NotImplementedError

    def remove_import(self, path: str, name: str) -> str:
        raise NotImplementedError

    def rename_symbol(self, path: str, old_name: str, new_name: str) -> str:
        raise NotImplementedError

    def delete_symbol(self, path: str, name: str, scope: str = None) -> str:
        raise NotImplementedError

    def find_references(self, path: str, name: str, scope: str = None) -> str:
        raise NotImplementedError

    def check_syntax(self, path: str) -> str:
        raise NotImplementedError


class PythonBackend(LanguageBackend):
    """Real implementation built on the standard library ast module."""

    def _find_node(self, tree: ast.Module, name: str, scope: str = None):
        if scope:
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == scope:
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
                            return child
            return None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return node
        return None

    def _symbol_names(self, tree: ast.Module) -> list:
        """Every function/class/method name defined in the file, for
        suggesting near-matches when a lookup misses."""
        names = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            names.append(child.name)
        return names

    def _import_names(self, tree: ast.Module) -> list:
        """Every name a file's imports would be looked up by (module names
        and imported members), for the same fuzzy-match purpose."""
        names = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.append(node.module)
                names.extend(a.name for a in node.names)
        return names

    # -- read-only -----------------------------------------------------------

    def list_symbols(self, path: str) -> str:
        tree = ast.parse(_read_source(path))
        symbols = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append({"kind": "function", "name": node.name, "line": node.lineno})
            elif isinstance(node, ast.ClassDef):
                symbols.append({"kind": "class", "name": node.name, "line": node.lineno})
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append({
                            "kind": "method", "name": child.name,
                            "scope": node.name, "line": child.lineno,
                        })
        return json.dumps(symbols, indent=2)

    def read_symbol(self, path: str, name: str, scope: str = None) -> str:
        source = _read_source(path)
        tree = ast.parse(source)
        node = self._find_node(tree, name, scope)
        if node is None:
            subject = f"symbol{f' in scope {scope}' if scope else ''}"
            hint = _fuzzy_hint(name, self._symbol_names(tree))
            return _soft_not_found(subject, name, hint)
        return ast.get_source_segment(source, node)

    def list_imports(self, path: str) -> str:
        tree = ast.parse(_read_source(path))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.append("import " + ", ".join(a.name for a in node.names))
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"from {node.module} import " + ", ".join(a.name for a in node.names))
        return "\n".join(imports) if imports else "(no imports)"

    def check_syntax(self, path: str) -> str:
        source = _read_source(path)
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return f"ERROR: syntax error at line {exc.lineno}, col {exc.offset}: {exc.msg}"
        return f"OK: {path} parses cleanly"

    # -- mutating ------------------------------------------------------------

    def update_symbol(self, path: str, name: str, new_code: str, scope: str = None) -> str:
        source = _read_source(path)
        tree = ast.parse(source)
        node = self._find_node(tree, name, scope)
        if node is None:
            subject = f"symbol{f' in scope {scope}' if scope else ''}"
            hint = _fuzzy_hint(name, self._symbol_names(tree))
            return _soft_not_found(subject, name, hint)
        indent = " " * (node.col_offset or 0)
        new_code = _match_indent(new_code, indent)
        lines = source.splitlines(keepends=True)

        # If new_code itself leads with a comment, the model is re-
        # supplying this symbol's explanation. A leading comment lives
        # outside the node's own span (node.lineno is the 'def' line
        # itself, never a comment sitting above it), so without this,
        # each re-edit's new explanation lands just above whatever
        # explanation a *previous* edit left there, rather than
        # replacing it - reproduced directly, three edits in a row
        # stacked three near-identical comments above the same function.
        # Absorb any existing, contiguous run of comment-only lines
        # immediately above the function into the replaced range, so a
        # fresh explanation replaces the stale one instead of piling on
        # top of it. Stops at the first non-comment or blank line, so an
        # unrelated section header further up, separated by a blank
        # line, is left untouched.
        start_line = node.lineno - 1
        if new_code.lstrip().startswith("#"):
            while start_line > 0 and lines[start_line - 1].strip().startswith("#"):
                start_line -= 1

        new_lines = lines[:start_line] + [new_code.rstrip("\n") + "\n"] + lines[node.end_lineno:]
        new_source = "".join(new_lines)
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: edit would produce invalid syntax at line {exc.lineno}: {exc.msg}"
        _write_source(path, new_source)
        return f"OK: updated {name} in {path}"

    def insert_symbol(self, path: str, code: str, after: str = None, scope: str = None) -> str:
        source = _read_source(path)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)

        # Guard against silently creating a duplicate definition: figure
        # out what name the inserted code defines, and refuse if that
        # name already exists in the target scope. Without this, calling
        # insert_symbol twice (e.g. across two separate runs against the
        # same file, which the model has no memory of) just keeps
        # appending copies rather than erroring.
        inserted_name = None
        try:
            for node in ast.parse(code).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    inserted_name = node.name
                    break
        except SyntaxError:
            pass  # the full-file parse below will catch genuinely bad code

        if inserted_name:
            existing = self._find_node(tree, inserted_name, scope)
            if existing is not None:
                where = f" in scope {scope}" if scope else ""
                return (
                    f"ERROR: '{inserted_name}' already exists{where} in {path} "
                    f"at line {existing.lineno}. Use update_symbol to replace it, "
                    f"or delete_symbol first if you intend to reinsert it."
                )

        indent = ""
        if after:
            node = self._find_node(tree, after, scope)
            if node is None:
                hint = _fuzzy_hint(after, self._symbol_names(tree))
                return _soft_not_found("anchor symbol", after, hint)
            insert_at = node.end_lineno
            indent = " " * (node.col_offset or 0)
        elif scope:
            node = self._find_node(tree, scope, None)
            if node is None:
                class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
                hint = _fuzzy_hint(scope, class_names)
                return _soft_not_found("scope", scope, hint)
            insert_at = node.end_lineno
            child_indent = 4
            for child in getattr(node, "body", []):
                if hasattr(child, "col_offset"):
                    child_indent = child.col_offset
                    break
            indent = " " * child_indent
        else:
            insert_at = len(lines)

        code = _match_indent(code, indent)
        new_lines = lines[:insert_at] + ["\n", code.rstrip("\n") + "\n"] + lines[insert_at:]
        new_source = "".join(new_lines)
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: insert would produce invalid syntax at line {exc.lineno}: {exc.msg}"
        _write_source(path, new_source)
        anchor = after or (f"end of {scope}" if scope else "end of file")
        return f"OK: inserted new symbol after {anchor} in {path}"

    def add_import(self, path: str, statement: str) -> str:
        source = _read_source(path)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)

        # Same duplicate guard as insert_symbol: don't silently pile up
        # the same import line a second or third time across re-runs.
        normalized = statement.strip()
        if any(line.strip() == normalized for line in lines):
            return f"ERROR: import '{normalized}' is already present in {path}"

        insert_index = 0
        if lines and lines[0].startswith("#!"):
            insert_index = 1
        if len(lines) > insert_index and re.match(r"^#.*coding[:=]", lines[insert_index]):
            insert_index += 1
        body = tree.body
        if (body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            insert_index = max(insert_index, body[0].end_lineno)
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                insert_index = max(insert_index, node.end_lineno)

        new_lines = lines[:insert_index] + [statement.rstrip("\n") + "\n"] + lines[insert_index:]
        new_source = "".join(new_lines)
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: adding import would produce invalid syntax: {exc.msg}"
        _write_source(path, new_source)
        return f"OK: added '{statement}' to {path}"

    def remove_import(self, path: str, name: str) -> str:
        source = _read_source(path)
        tree = ast.parse(source)
        line_starts = [0]
        for line in source.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))

        def alias_text(alias):
            return f"{alias.name} as {alias.asname}" if alias.asname else alias.name

        # node -> replacement text, or None to delete the statement
        # entirely. `import os, sys` naming "os" must keep "sys", not
        # delete the whole line - same for `from os import path, sep`
        # naming "path". Only a whole-module match on an ImportFrom
        # (naming "os" for `from os import path`) removes the full
        # statement, since there the imported module itself is what was
        # named, not one of several names sharing a line.
        replacements = {}
        removed = False

        for node in tree.body:
            if isinstance(node, ast.Import):
                if any(a.name == name for a in node.names):
                    removed = True
                    kept = [a for a in node.names if a.name != name]
                    replacements[node] = (
                        "import " + ", ".join(alias_text(a) for a in kept) if kept else None
                    )
            elif isinstance(node, ast.ImportFrom):
                whole_module_match = node.module == name
                alias_match = any(a.name == name for a in node.names)
                if whole_module_match or alias_match:
                    removed = True
                    if whole_module_match:
                        replacements[node] = None
                    else:
                        kept = [a for a in node.names if a.name != name]
                        if kept:
                            dots = "." * node.level
                            replacements[node] = (
                                f"from {dots}{node.module or ''} import "
                                + ", ".join(alias_text(a) for a in kept)
                            )
                        else:
                            replacements[node] = None

        if not removed:
            hint = _fuzzy_hint(name, self._import_names(tree))
            return _soft_not_found("import", name, hint)

        # Apply from the bottom of the file up, so offsets computed
        # against the original source stay valid for statements above
        # ones already rewritten.
        new_source = source
        for node in sorted(replacements, key=lambda n: n.lineno, reverse=True):
            start = line_starts[node.lineno - 1]
            end = line_starts[node.end_lineno]
            text = replacements[node]
            new_source = new_source[:start] + (text + "\n" if text is not None else "") + new_source[end:]

        _write_source(path, new_source)
        return f"OK: removed import {name} from {path}"

    def rename_symbol(self, path: str, old_name: str, new_name: str) -> str:
        if not new_name.isidentifier():
            return f"ERROR: {new_name!r} is not a valid Python identifier"
        source = _read_source(path)
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except tokenize.TokenError as exc:
            return f"ERROR: could not tokenize {path}: {exc}"

        line_starts = [0]
        for line in source.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))

        edits = []
        for tok in tokens:
            if tok.type == tokenize.NAME and tok.string == old_name:
                (sr, sc), (er, ec) = tok.start, tok.end
                start = line_starts[sr - 1] + sc
                end = line_starts[er - 1] + ec
                edits.append((start, end))

        if not edits:
            known_names = {
                tok.string for tok in tokens
                if tok.type == tokenize.NAME and not keyword.iskeyword(tok.string)
            }
            hint = _fuzzy_hint(old_name, known_names)
            return _soft_not_found("name", old_name, hint)

        new_source = source
        for start, end in reversed(edits):
            new_source = new_source[:start] + new_name + new_source[end:]

        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: rename would produce invalid syntax: {exc.msg}"

        _write_source(path, new_source)
        return f"OK: renamed {len(edits)} occurrence(s) of {old_name} to {new_name} in {path}"

    def delete_symbol(self, path: str, name: str, scope: str = None) -> str:
        source = _read_source(path)
        tree = ast.parse(source)
        node = self._find_node(tree, name, scope)
        if node is None:
            subject = f"symbol{f' in scope {scope}' if scope else ''}"
            hint = _fuzzy_hint(name, self._symbol_names(tree))
            return _soft_not_found(subject, name, hint)
        lines = source.splitlines(keepends=True)
        new_lines = lines[: node.lineno - 1] + lines[node.end_lineno:]
        new_source = "".join(new_lines)
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: delete would produce invalid syntax at line {exc.lineno}: {exc.msg}"
        _write_source(path, new_source)
        return f"OK: deleted {name} from {path}"

    def find_references(self, path: str, name: str, scope: str = None) -> str:
        """AST-aware, unlike search_files: matches actual name/attribute
        usage, not text that merely contains the string (a docstring
        mention, an unrelated same-named variable in a comment)."""
        source = _read_source(path)
        tree = ast.parse(source)
        lines = source.splitlines()
        seen = set()
        refs = []
        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", None)
            if lineno is None:
                continue
            hit = (isinstance(node, ast.Name) and node.id == name) or (
                isinstance(node, ast.Attribute) and node.attr == name
            )
            if hit and (lineno, name) not in seen:
                seen.add((lineno, name))
                text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
                refs.append(f"{path}:{lineno}: {text}")
        if not refs:
            return f"(no references to {name} found in {path})"
        return "\n".join(refs)


class NotImplementedBackend(LanguageBackend):
    """Placeholder so the dispatch table already lists every planned language."""

    def __init__(self, language_name: str):
        self.language_name = language_name

    def _not_ready(self):
        return f"ERROR: {self.language_name} backend not implemented yet"

    def list_symbols(self, path): return self._not_ready()
    def read_symbol(self, path, name, scope=None): return self._not_ready()
    def update_symbol(self, path, name, new_code, scope=None): return self._not_ready()
    def insert_symbol(self, path, code, after=None, scope=None): return self._not_ready()
    def list_imports(self, path): return self._not_ready()
    def add_import(self, path, statement): return self._not_ready()
    def remove_import(self, path, name): return self._not_ready()
    def rename_symbol(self, path, old_name, new_name): return self._not_ready()
    def delete_symbol(self, path, name, scope=None): return self._not_ready()
    def find_references(self, path, name, scope=None): return self._not_ready()
    def check_syntax(self, path): return self._not_ready()


LANGUAGE_BACKENDS = {
    "python": PythonBackend(),
    "csharp": NotImplementedBackend("csharp"),
    "typescript": NotImplementedBackend("typescript"),
}

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".cs": "csharp",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def _resolve_backend(path: str, language: str = None) -> LanguageBackend:
    # The file type decides, there is no python fallback for an unknown
    # extension, only a genuinely unrecognized language returns a stub.
    if language is None:
        ext = os.path.splitext(path)[1].lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)
        if language is None:
            return NotImplementedBackend(f"unknown extension {ext!r}")
    backend = LANGUAGE_BACKENDS.get(language)
    if backend is None:
        return NotImplementedBackend(language)
    return backend


# ---------------------------------------------------------------------------
# 2. Language-agnostic tools
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    full = _full_path(path)
    if not os.path.exists(full):
        return f"ERROR: {path} does not exist"
    if not os.path.isfile(full):
        return f"ERROR: {path} is not a file"
    size = os.path.getsize(full)
    # Read and cap by raw bytes first, then decode, so a multi-byte UTF-8
    # file cannot slip more actual bytes through than MAX_READ_BYTES.
    with open(full, "rb") as f:
        raw = f.read(MAX_READ_BYTES)
    content = raw.decode("utf-8", errors="replace")
    if size > MAX_READ_BYTES:
        content += f"\n\n[TRUNCATED: file is {size} bytes, showing first {MAX_READ_BYTES}]"
    return content


def write_file(path: str, content: str) -> str:
    _write_source(path, content)
    return f"OK: wrote {len(content)} chars to {path}"


def list_dir(path: str = ".") -> str:
    full = _full_path(path)
    if not os.path.exists(full):
        return f"ERROR: {path} does not exist"
    if not os.path.isdir(full):
        return f"ERROR: {path} is not a directory"
    entries = sorted(e for e in os.listdir(full) if not e.startswith("."))
    return "\n".join(entries) if entries else "(empty)"


def search_files(pattern: str, path: str = ".", regex: bool = False, max_results: int = 100) -> str:
    root = _full_path(path)
    if not os.path.exists(root):
        return f"ERROR: {path} does not exist"
    try:
        matcher = re.compile(pattern if regex else re.escape(pattern))
    except re.error as exc:
        return f"ERROR: bad pattern: {exc}"
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if matcher.search(line):
                            rel = os.path.relpath(fpath, WORKDIR)
                            results.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(results) >= max_results:
                                return "\n".join(results) + f"\n... (truncated at {max_results})"
            except OSError:
                continue
    return "\n".join(results) if results else "(no matches)"


def replace_in_file(path: str, old_str: str, new_str: str, count: int = -1) -> str:
    """Literal text find-replace within one file, not symbol-aware. Useful
    for updating a specific call pattern (e.g. a function's new parameter)
    where the text is written the same way each time it appears. If the
    file is Python, the result is syntax-checked before it is committed."""
    source = _read_source(path)
    occurrences = source.count(old_str)
    if occurrences == 0:
        hint = _closest_line_hint(source, old_str)
        return _soft_not_found(f"text in {path}", old_str, hint)

    if count is not None and count >= 0:
        new_source = source.replace(old_str, new_str, count)
        replaced = min(count, occurrences)
    else:
        new_source = source.replace(old_str, new_str)
        replaced = occurrences

    ext = os.path.splitext(path)[1].lower()
    if EXTENSION_TO_LANGUAGE.get(ext) == "python":
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: replacement would produce invalid syntax at line {exc.lineno}: {exc.msg}"

    _write_source(path, new_source)
    return f"OK: replaced {replaced} occurrence(s) in {path}"


def replace_in_project(old_str: str, new_str: str, path: str = ".",
                        file_glob: str = None, count_per_file: int = -1) -> str:
    """The same literal find-replace as replace_in_file, applied across
    every matching file under `path` (default: the whole project). A file
    is skipped, not partially written, if the replacement would break its
    Python syntax; other languages are replaced without a syntax check
    since there is no backend for them yet."""
    root = _full_path(path)
    if not os.path.exists(root):
        return f"ERROR: {path} does not exist"

    results = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            if file_glob and not fnmatch.fnmatch(fname, file_glob):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, WORKDIR)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    source = f.read()
            except (OSError, UnicodeDecodeError):
                continue

            occurrences = source.count(old_str)
            if occurrences == 0:
                continue

            if count_per_file is not None and count_per_file >= 0:
                new_source = source.replace(old_str, new_str, count_per_file)
                replaced = min(count_per_file, occurrences)
            else:
                new_source = source.replace(old_str, new_str)
                replaced = occurrences

            ext = os.path.splitext(fname)[1].lower()
            if EXTENSION_TO_LANGUAGE.get(ext) == "python":
                try:
                    ast.parse(new_source)
                except SyntaxError as exc:
                    results.append(f"{rel}: SKIPPED, would break syntax at line {exc.lineno}: {exc.msg}")
                    continue

            _write_source(rel, new_source)
            total += replaced
            results.append(f"{rel}: replaced {replaced}")

    if not results:
        return f"(no matches for '{old_str}' under {path})"
    return f"OK: {total} total replacement(s) across {len(results)} file(s)\n" + "\n".join(results)


def format_file(path: str) -> str:
    """Deterministic formatting, not model judgement, same principle as
    rename_symbol. Python only for now, via black if it's installed."""
    if not os.path.exists(_full_path(path)):
        return f"ERROR: {path} does not exist"
    ext = os.path.splitext(path)[1].lower()
    if EXTENSION_TO_LANGUAGE.get(ext) != "python":
        return f"ERROR: formatting for {ext or 'this file type'} not implemented yet"
    try:
        import black
    except ImportError:
        return "ERROR: black is not installed (pip install black) to enable formatting"
    source = _read_source(path)
    try:
        formatted = black.format_str(source, mode=black.Mode())
    except Exception as exc:
        return f"ERROR: could not format {path}: {type(exc).__name__}: {exc}"
    if formatted == source:
        return f"OK: {path} already formatted"
    _write_source(path, formatted)
    return f"OK: formatted {path}"


REQUIREMENTS_PATH = "requirements.txt"


def list_dependencies(language: str = None) -> str:
    language = language or "python"
    if language != "python":
        return f"ERROR: dependency listing for {language} not implemented yet"
    full = _full_path(REQUIREMENTS_PATH)
    if not os.path.exists(full):
        return "(no requirements.txt found)"
    return _read_source(REQUIREMENTS_PATH)


def add_dependency(name: str, version: str = None, language: str = None) -> str:
    language = language or "python"
    if language != "python":
        return f"ERROR: dependency management for {language} not implemented yet"
    existing = _read_source(REQUIREMENTS_PATH) if os.path.exists(_full_path(REQUIREMENTS_PATH)) else ""
    for line in existing.splitlines():
        pkg = re.split(r"[=<>!~\[]", line.strip())[0]
        if pkg == name:
            return f"ERROR: {name} already listed in requirements.txt (as '{line.strip()}')"
    entry = f"{name}=={version}" if version else name
    new_content = existing
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    new_content += entry + "\n"
    _write_source(REQUIREMENTS_PATH, new_content)
    return f"OK: added '{entry}' to requirements.txt"


def lookup_symbol_docs(symbol: str, language: str = None) -> str:
    """Offline documentation lookup, no network call. Works for stdlib and
    any package already installed in this environment, e.g. 'os.path.join'
    or 'json.dumps'. Not a substitute for live docs on a package that
    isn't installed here."""
    language = language or "python"
    if language != "python":
        return f"ERROR: docs lookup for {language} not implemented yet"
    try:
        text = pydoc.render_doc(symbol, renderer=pydoc.plaintext)
    except Exception as exc:
        return f"ERROR: could not find docs for '{symbol}': {type(exc).__name__}: {exc}"
    max_len = 3000
    if len(text) > max_len:
        text = text[:max_len] + "\n... (truncated)"
    return text


def run_command(command: str) -> str:
    print(f"\n[AGENT WANTS TO RUN]: {command}")
    print(f"[WORKING DIRECTORY]: {WORKDIR}")
    approved = input("Allow this command to run? [y/N]: ").strip().lower()
    if approved != "y":
        return "DECLINED: user did not approve running this command"
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=60,
        )
        output = f"exit_code={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        missing = re.search(r"No module named '([\w.]+)'", result.stderr)
        if missing:
            output += (
                f"\n\nSUGGESTION: '{missing.group(1)}' isn't installed. Use "
                f"add_dependency to add it to requirements.txt, then "
                f"'pip install -r requirements.txt' (with approval) before "
                f"retrying this command."
            )
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60s"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# Deliberately NOT under STATE_DIR, and not reconfigured by
# configure_workspace: every other piece of state (WORKDIR, backups, the
# transcript) is meant to be private to whichever agent is running, but
# the todo list is the one thing meant to cross agents on purpose - an
# architect agent leaving tasks a coder agent (a different process, a
# different configured workspace) can still read. Fixed directly under
# SCRIPT_DIR so it means the same file no matter which workspace is active.
TODO_PATH = os.path.join(SCRIPT_DIR, "shared_todo.json")


def todo_read() -> str:
    if not os.path.exists(TODO_PATH):
        return "[]"
    with open(TODO_PATH, "r", encoding="utf-8") as f:
        return f.read()


def todo_write(items: list) -> str:
    with open(TODO_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    return f"OK: saved {len(items)} todo items"


def list_backups() -> str:
    if not os.path.exists(BACKUP_MANIFEST):
        return "(no backups)"
    with open(BACKUP_MANIFEST, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return "(no backups)"
    out = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            out.append(f"{entry['backup']}  ->  {entry['original']}")
        except (json.JSONDecodeError, KeyError):
            continue
    return "\n".join(out) if out else "(no backups)"


def restore_backup(name: str) -> str:
    if not name or os.sep in name or "/" in name or "\\" in name or name in (".", ".."):
        return f"ERROR: invalid backup name: {name}"
    if not os.path.exists(BACKUP_MANIFEST):
        return f"ERROR: no backups recorded"
    original = None
    with open(BACKUP_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("backup") == name:
                original = entry.get("original")
    if original is None:
        return f"ERROR: backup {name} not found in manifest"
    src = os.path.join(BACKUP_DIR, name)
    if not os.path.isfile(src):
        return f"ERROR: backup file {name} is missing on disk"
    try:
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        _write_source(original, content)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    return f"OK: restored {original} from {name}"


# ---------------------------------------------------------------------------
# 3. Language-routed tools
# ---------------------------------------------------------------------------

def list_symbols(path: str, language: str = None) -> str:
    return _resolve_backend(path, language).list_symbols(path)


def read_symbol(path: str, name: str, scope: str = None, language: str = None) -> str:
    return _resolve_backend(path, language).read_symbol(path, name, scope)


def update_symbol(path: str, name: str, new_code: str, scope: str = None, language: str = None) -> str:
    return _resolve_backend(path, language).update_symbol(path, name, new_code, scope)


def insert_symbol(path: str, code: str, after: str = None, scope: str = None, language: str = None) -> str:
    return _resolve_backend(path, language).insert_symbol(path, code, after, scope)


def list_imports(path: str, language: str = None) -> str:
    return _resolve_backend(path, language).list_imports(path)


def add_import(path: str, statement: str, language: str = None) -> str:
    return _resolve_backend(path, language).add_import(path, statement)


def remove_import(path: str, name: str, language: str = None) -> str:
    return _resolve_backend(path, language).remove_import(path, name)


def rename_symbol(path: str, old_name: str, new_name: str, language: str = None) -> str:
    return _resolve_backend(path, language).rename_symbol(path, old_name, new_name)


def delete_symbol(path: str, name: str, scope: str = None, language: str = None) -> str:
    return _resolve_backend(path, language).delete_symbol(path, name, scope)


def find_references(path: str, name: str, scope: str = None, language: str = None) -> str:
    return _resolve_backend(path, language).find_references(path, name, scope)


def check_syntax(path: str, language: str = None) -> str:
    return _resolve_backend(path, language).check_syntax(path)


# ---------------------------------------------------------------------------
# 3a. Generic post-write advisories. Rather than teaching every writer
#     function individually to notice these, checkers run after any of
#     them succeeds and fold a note into that tool's own result string,
#     the exact channel the model already reads its results from.
#       - Duplicate definitions: this is what would have caught
#         calculate_power ending up defined three times. insert_symbol's
#         own duplicate guard only stops IT from creating a new
#         duplicate, it says nothing about a file that already has one
#         from before the guard existed, or one introduced some other
#         way (a raw write_file, a future non-symbol tool).
#       - Loop with no assert: a nudge, not a bug finder, towards
#         stating a loop's invariant as a real runtime check instead of
#         leaving it implicit, so a violation has a chance to surface
#         the next time the code actually runs.
#     Both are Python-only for now, same limit as the rest of this file;
#     other languages pass through unchanged rather than being silently
#     skipped without saying so.
# ---------------------------------------------------------------------------

def _find_duplicate_symbols_python(path: str) -> list:
    """Scan a python file's AST for any function/class name defined more
    than once at the same scope (module level, or within the same class).
    Returns a list of {"name", "scope", "lines"} dicts, empty if clean."""
    source = _read_source(path)
    tree = ast.parse(source)
    results = []

    def scan(body, scope_name):
        seen = {}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seen.setdefault(node.name, []).append(node.lineno)
        for name, lines in seen.items():
            if len(lines) > 1:
                results.append({"name": name, "scope": scope_name, "lines": lines})
        for node in body:
            if isinstance(node, ast.ClassDef):
                scan(node.body, node.name)

    scan(tree.body, None)
    return results


def _comment_lines(source: str) -> set:
    """Line numbers carrying a '#' comment. ast strips comments entirely,
    so this uses tokenize instead, purely to check whether a nearby line
    of human-readable explanation exists, not to read its content."""
    lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                lines.add(tok.start[0])
    except tokenize.TokenError:
        pass
    return lines


def _find_unexplained_loop_functions_python(path: str) -> list:
    """Scan for functions/methods with a for/while loop whose invariant
    isn't written down anywhere in the code. Not a bug by itself, just a
    nudge: a loop almost always exists to maintain some invariant (each
    item handled once, an index stays in bounds, a queue never re-admits
    the same node), and that's exactly the kind of thing a model gets
    subtly wrong without noticing.

    "Written down" means a comment, specifically, not just something the
    model said in this turn's notes: notes are ephemeral (truncated out of
    context on a long run, gone the moment the conversation moves on),
    while a comment sits in the file and is still there the next time
    this function gets read, edited, or reviewed by a human. Satisfied
    either by an assert with a comment near it (the invariant, stated),
    or a comment right above the loop itself explaining why none applies,
    since "no invariant needed" is itself a claim worth being able to
    check later, not just having been said once. An assert with no
    comment doesn't satisfy this: an unexplained assert is barely better
    than an unstated one, since a future reader (model or human) still
    can't tell whether it encodes the loop's real invariant or is
    decorative. Returns a list of {"name", "scope"} dicts, empty if every
    loop-bearing function already has its reasoning on the page."""
    source = _read_source(path)
    tree = ast.parse(source)
    comment_lines = _comment_lines(source)

    def explained_at(lineno):
        return lineno in comment_lines or (lineno - 1) in comment_lines

    def loop_is_explained(loop_node):
        # Right above the loop, on the loop's own line, or as the first
        # line of the loop's own body ("while queue:\n    # invariant...\n
        # node = ...") are all the same idiomatic spot for this comment;
        # checking only the first two missed the third, which is exactly
        # where a model actually put it in practice.
        if explained_at(loop_node.lineno):
            return True
        first_stmt = loop_node.body[0] if loop_node.body else None
        return first_stmt is not None and explained_at(first_stmt.lineno)

    results = []

    def scan(body, scope_name):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                loops = [n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While))]
                if not loops:
                    continue
                asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
                explained = (
                    any(explained_at(n.lineno) for n in asserts)
                    or any(loop_is_explained(n) for n in loops)
                )
                if not explained:
                    results.append({"name": node.name, "scope": scope_name, "lineno": node.lineno})
            elif isinstance(node, ast.ClassDef):
                scan(node.body, node.name)

    scan(tree.body, None)
    return results


def _has_testable_logic_python(path: str) -> bool:
    """True if this file has at least one non-trivial function with a loop
    or a conditional branch — code where one hand-picked example can pass
    while a real edge case (an untaken branch, a duplicate-enqueue path,
    an off-by-one at a boundary) goes unexercised. Same triviality filter
    as the docstring check: a name starting with '_' or a body of one
    statement or fewer doesn't count, for the same reason it doesn't
    there. Used to decide whether a file needs more than one sanity-check
    run before a final answer is accepted; see _update_test_diversity_tracking
    and its use in run_agent."""
    try:
        tree = ast.parse(_read_source(path))
    except (SyntaxError, FileNotFoundError):
        return False

    def scan(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                trivial = node.name.startswith("_") or len(node.body) <= 1
                if trivial:
                    continue
                if any(isinstance(n, (ast.For, ast.While, ast.If)) for n in ast.walk(node)):
                    return True
            elif isinstance(node, ast.ClassDef):
                if scan(node.body):
                    return True
        return False

    return scan(tree.body)



def _find_undocumented_functions_python(path: str) -> list:
    """Scan for functions/methods with no docstring, skipping the two
    cases where nagging about it would just be noise: a name starting
    with '_' (the private/helper convention, where a docstring is rarely
    expected) and a body of one statement or fewer (a one-line wrapper or
    a stub, where 'what does this do' is already answered by reading it).
    Same limit as the loop-invariant check: this only sees whether a
    docstring exists, not whether it's accurate, so it can't catch a
    docstring left stale after the function's behavior changed underneath
    it. That's a real gap, not a design choice; there's no cheap static
    way to tell 'still true' from 'used to be true', so the best this can
    do is make the model's own next read of the function surface the
    mismatch, by requiring the model to have looked at the docstring
    recently enough to have written one in the first place. Returns a
    list of {"name", "scope"} dicts, empty if every non-trivial function
    already documents itself."""
    tree = ast.parse(_read_source(path))
    results = []

    def scan(body, scope_name):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                trivial = node.name.startswith("_") or len(node.body) <= 1
                if not trivial and ast.get_docstring(node) is None:
                    results.append({"name": node.name, "scope": scope_name, "lineno": node.lineno})
            elif isinstance(node, ast.ClassDef):
                scan(node.body, node.name)

    scan(tree.body, None)
    return results



def _with_post_edit_advisories(func):
    """Wrap a mutating tool: after it succeeds, scan the file it touched
    for three things and append a note to its own return string for each
    that fires: duplicate definitions, a loop-bearing function whose
    invariant isn't explained anywhere in the code, and a non-trivial
    function with no docstring (see _find_unexplained_loop_functions_python
    and _find_undocumented_functions_python for the reasoning and the
    known limits of each). None is proof of a bug, and none starts with
    ERROR or NOTFOUND, so none trips the batch-abort logic in run_agent.

    The loop and docstring checks share a second thing worth noting: a
    hedged "consider doing X, this is just advisory" paragraph was found
    (by testing) to be reliably ignored by a smaller model, and even a
    directive paragraph ("do X before finishing") did better but still
    read as one more piece of chat text to skim. So both are rendered as
    a compact `path:line: CODE message` block, styled like a real linter
    or compiler diagnostic (flake8/pydocstyle-shaped) rather than prose,
    with an explicit one-line disclaimer that the codes (PROJ-INV001,
    PROJ-DOC001) are project-specific, not real Python or flake8 rules —
    so the shape reads as "an error to fix," the disclaimer stops that
    from being mistaken for an actual language error to look up or a
    real tool to install. A no-op for non-python files or a result that
    wasn't a success in the first place."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, str) or not result.startswith("OK"):
            return result

        path = kwargs.get("path", args[0] if args else None)
        if not path:
            return result
        language = kwargs.get("language")
        ext = os.path.splitext(path)[1].lower()
        if (language or EXTENSION_TO_LANGUAGE.get(ext)) != "python":
            return result

        try:
            dups = _find_duplicate_symbols_python(path)
            bare_loops = _find_unexplained_loop_functions_python(path)
            undocumented = _find_undocumented_functions_python(path)
        except (SyntaxError, OSError, ValueError):
            # OSError covers FileNotFoundError/PermissionError/
            # IsADirectoryError; ValueError covers _full_path rejecting a
            # bad path. None of these are this check's job to handle -
            # letting one escape would turn an already-successful edit
            # into a propagated exception that run_agent's generic
            # handler then reports as an ERROR, wrongly aborting the
            # rest of the batch over an advisory-only failure.
            return result

        if dups:
            details = "; ".join(
                f"'{d['name']}' defined {len(d['lines'])} times at lines {d['lines']}"
                + (f" inside class {d['scope']}" if d["scope"] else "")
                for d in dups
            )
            result += (
                f"\n[WARNING] {path} now contains duplicate definitions: {details}. "
                f"This is a real bug, not a style nit: Python silently keeps only the "
                f"last definition and the earlier ones become dead code. Use "
                f"delete_symbol to remove the extra copies, keeping one."
            )

        style_lines = []
        fixes = []

        if bare_loops:
            for f in bare_loops:
                scope = f" (class {f['scope']})" if f["scope"] else ""
                style_lines.append(
                    f"{path}:{f['lineno']}: PROJ-INV001 loop in "
                    f"'{f['name']}'{scope} has no comment/assert stating its "
                    f"invariant"
                )
            fixes.append(
                "PROJ-INV001: via update_symbol, either add a runtime assert "
                "for the invariant with a comment stating it (e.g. "
                "'# invariant: no node is enqueued twice'), or add a comment "
                "above the loop saying there is none and why."
            )

        if undocumented:
            for f in undocumented:
                scope = f" (class {f['scope']})" if f["scope"] else ""
                style_lines.append(
                    f"{path}:{f['lineno']}: PROJ-DOC001 '{f['name']}'{scope} "
                    f"has no docstring"
                )
            fixes.append(
                "PROJ-DOC001: via update_symbol, add a one- or two-sentence "
                "docstring describing role, parameters, and return value. If "
                "a docstring already exists and your edit changed behavior, "
                "verify it's still accurate and fix it if not."
            )

        if style_lines:
            result += (
                "\n[style check: project-specific rules, not real Python or "
                "flake8 errors, but treat them the same way: fix before "
                "running a sanity check or giving a final answer]\n"
                + "\n".join(style_lines) + "\n"
                + "\n".join(fixes)
            )
        return result
    return wrapper


for _name in (
    "write_file", "insert_symbol", "update_symbol", "delete_symbol",
    "rename_symbol", "add_import", "remove_import", "replace_in_file",
    "format_file",
    # check_syntax deliberately excluded: it's a pure read, but its
    # success message also starts with "OK:", so wrapping it re-fires
    # every advisory a second (or third) time with no intervening edit -
    # confirmed to happen in practice, not just in theory.
):
    globals()[_name] = _with_post_edit_advisories(globals()[_name])
del _name


# ---------------------------------------------------------------------------
# 3b. The "list" meta-tool: collapses list_symbols/list_imports/list_dir/
#     list_dependencies/list_backups into one tool with a `target` enum.
#     This cuts five schemas (resent on every single request, per the
#     earlier context-cost discussion) down to one, and picking a value
#     from a constrained enum is generally an easier, more reliably
#     enforceable choice for a model than picking the right function name
#     out of a long list of similarly-prefixed ones.
# ---------------------------------------------------------------------------

def _list_symbols_target(path=None, language=None):
    if not path:
        return "ERROR: target 'symbols' requires 'path'. Usage: list(target='symbols', path='file.py')"
    return list_symbols(path, language)


def _list_imports_target(path=None, language=None):
    if not path:
        return "ERROR: target 'imports' requires 'path'. Usage: list(target='imports', path='file.py')"
    return list_imports(path, language)


def _list_dir_target(path=None, language=None):
    return list_dir(path or ".")


def _list_dependencies_target(path=None, language=None):
    return list_dependencies(language)


def _list_backups_target(path=None, language=None):
    return list_backups()


LIST_TARGETS = {
    "symbols": {
        "fn": _list_symbols_target,
        "usage": "list(target='symbols', path='file.py') -> function/class/method names and kinds in a file",
    },
    "imports": {
        "fn": _list_imports_target,
        "usage": "list(target='imports', path='file.py') -> import statements in a file",
    },
    "dir": {
        "fn": _list_dir_target,
        "usage": "list(target='dir', path='.') -> files in a directory, default sandbox root",
    },
    "dependencies": {
        "fn": _list_dependencies_target,
        "usage": "list(target='dependencies') -> declared project dependencies",
    },
    "backups": {
        "fn": _list_backups_target,
        "usage": "list(target='backups') -> available file backups, newest first",
    },
}


def list_things(target: str, path: str = None, language: str = None) -> str:
    entry = LIST_TARGETS.get(target)
    if entry is None:
        options = "\n".join(f"  - {v['usage']}" for v in LIST_TARGETS.values())
        return (
            f"ERROR: unknown list target '{target}'. Valid targets: "
            f"{', '.join(sorted(LIST_TARGETS))}.\nUsage:\n{options}"
        )
    return entry["fn"](path, language)


# ---------------------------------------------------------------------------
# 4. Tool schemas
# ---------------------------------------------------------------------------

def _tool(name, description, properties, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


_LANG_PROP = {"language": {
    "type": "string",
    "description": "Optional override, e.g. python, csharp, typescript",
}}

TOOLS = [
    _tool("read_file", "Read a whole file's raw content (capped at 256 KB).",
          {"path": {"type": "string"}}, ["path"]),
    _tool("write_file", "Create or overwrite a file with given content (a backup is taken first).",
          {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _tool("list", "List something: symbols, imports, dir, dependencies, or backups. One tool "
                 "covering all five, pick the target and supply path when the target needs one.",
          {"target": {"type": "string", "enum": sorted(LIST_TARGETS.keys())},
           "path": {"type": "string"}, **_LANG_PROP},
          ["target"]),
    _tool("search_files", "Search file contents for a substring or regex.",
          {"pattern": {"type": "string"}, "path": {"type": "string"},
           "regex": {"type": "boolean"}, "max_results": {"type": "integer"}},
          ["pattern"]),
    _tool("replace_in_file", "Literal find-replace within one file, not symbol-aware. "
                             "For updating a call pattern written the same way each time "
                             "it appears. Python files are syntax-checked before committing.",
          {"path": {"type": "string"}, "old_str": {"type": "string"},
           "new_str": {"type": "string"},
           "count": {"type": "integer", "description": "max replacements, omit for all"}},
          ["path", "old_str", "new_str"]),
    _tool("replace_in_project", "The same literal find-replace applied across every matching "
                                "file under a folder (default: whole project). Use when a "
                                "function signature or call pattern changed and every call "
                                "site needs the same textual update. A file is skipped, not "
                                "partially written, if it would break Python syntax.",
          {"old_str": {"type": "string"}, "new_str": {"type": "string"},
           "path": {"type": "string"},
           "file_glob": {"type": "string", "description": "e.g. *.py to restrict which files are touched"},
           "count_per_file": {"type": "integer", "description": "max replacements per file, omit for all"}},
          ["old_str", "new_str"]),
    _tool("read_symbol", "Read the full source of one named function, class, or method.",
          {"path": {"type": "string"}, "name": {"type": "string"},
           "scope": {"type": "string"}, **_LANG_PROP}, ["path", "name"]),
    _tool("update_symbol", "Replace a function/class/method's body by name. The edit is "
                           "validated before it is committed; an invalid edit is rolled back.",
          {"path": {"type": "string"}, "name": {"type": "string"},
           "new_code": {"type": "string"}, "scope": {"type": "string"}, **_LANG_PROP},
          ["path", "name", "new_code"]),
    _tool("insert_symbol", "Insert a new function/class after a named anchor symbol, "
                           "into a class, or at end of file.",
          {"path": {"type": "string"}, "code": {"type": "string"},
           "after": {"type": "string"}, "scope": {"type": "string"}, **_LANG_PROP},
          ["path", "code"]),
    _tool("add_import", "Add an import statement after the shebang, encoding cookie, "
                        "module docstring, and any existing imports.",
          {"path": {"type": "string"}, "statement": {"type": "string"}, **_LANG_PROP},
          ["path", "statement"]),
    _tool("remove_import", "Remove an import (by module or imported name) from a file.",
          {"path": {"type": "string"}, "name": {"type": "string"}, **_LANG_PROP},
          ["path", "name"]),
    _tool("rename_symbol", "Rename every identifier occurrence in a file. Token-aware: "
                           "strings and comments are left alone. Prefer this over editing "
                           "text by hand for renames.",
          {"path": {"type": "string"}, "old_name": {"type": "string"},
           "new_name": {"type": "string"}, **_LANG_PROP},
          ["path", "old_name", "new_name"]),
    _tool("delete_symbol", "Remove a function, class, or method by name.",
          {"path": {"type": "string"}, "name": {"type": "string"},
           "scope": {"type": "string"}, **_LANG_PROP}, ["path", "name"]),
    _tool("find_references", "Find actual usages of a name (AST-aware), unlike search_files "
                             "which matches any occurrence of the text, comments included. "
                             "Run this before rename_symbol or delete_symbol on anything you "
                             "did not just write yourself.",
          {"path": {"type": "string"}, "name": {"type": "string"},
           "scope": {"type": "string"}, **_LANG_PROP}, ["path", "name"]),
    _tool("check_syntax", "Parse a file and report syntax errors, if any. Run this after "
                          "any edit to confirm the file still parses.",
          {"path": {"type": "string"}, **_LANG_PROP}, ["path"]),
    _tool("format_file", "Apply deterministic code formatting (black, for Python).",
          {"path": {"type": "string"}}, ["path"]),
    _tool("add_dependency", "Add a dependency to the project's dependency file.",
          {"name": {"type": "string"}, "version": {"type": "string"},
           "language": {"type": "string"}}, ["name"]),
    _tool("lookup_symbol_docs", "Offline documentation lookup for a stdlib or installed "
                                "symbol, e.g. 'os.path.join'. No network call. Use this "
                                "before guessing the signature of an unfamiliar stdlib or "
                                "library call.",
          {"symbol": {"type": "string"}, "language": {"type": "string"}}, ["symbol"]),
    _tool("restore_backup", "Restore a file from a backup name returned by list(target='backups').",
          {"name": {"type": "string"}}, ["name"]),
    _tool("run_command", "Propose a shell command. Requires explicit human approval before "
                         "it runs; never assume it ran just because you called this.",
          {"command": {"type": "string"}}, ["command"]),
    _tool("todo_read", "Read the current todo list as JSON.", {}),
    _tool("todo_write", "Overwrite the todo list. Each item: {id, task, status}.",
          {"items": {"type": "array", "items": {"type": "object"}}}, ["items"]),
    _tool("help", "Look up exactly how to call a tool. With no tool_name, lists every "
                 "available tool with a one-line description. Call this before guessing "
                 "at a tool's arguments if you are unsure.",
          {"tool_name": {"type": "string"}}),
]

DISPATCH = {
    "read_file": read_file,
    "write_file": write_file,
    "list": list_things,
    "search_files": search_files,
    "replace_in_file": replace_in_file,
    "replace_in_project": replace_in_project,
    "read_symbol": read_symbol,
    "update_symbol": update_symbol,
    "insert_symbol": insert_symbol,
    "add_import": add_import,
    "remove_import": remove_import,
    "rename_symbol": rename_symbol,
    "delete_symbol": delete_symbol,
    "find_references": find_references,
    "check_syntax": check_syntax,
    "format_file": format_file,
    "add_dependency": add_dependency,
    "lookup_symbol_docs": lookup_symbol_docs,
    "restore_backup": restore_backup,
    "run_command": run_command,
    "todo_read": todo_read,
    "todo_write": todo_write,
}

# Name -> schema lookup, used to turn a wrong tool call into a usage hint
# the model can act on immediately, instead of a bare error it has to
# guess how to fix.
TOOL_BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}


def _tool_usage_hint(name: str) -> str:
    schema = TOOL_BY_NAME.get(name)
    if not schema:
        return ""
    props = schema["parameters"].get("properties", {})
    required = set(schema["parameters"].get("required", []))
    parts = []
    for pname, pspec in props.items():
        marker = "" if pname in required else " (optional)"
        enum = pspec.get("enum")
        type_hint = f" [one of: {', '.join(enum)}]" if enum else ""
        parts.append(f"{pname}{marker}{type_hint}")
    args_desc = ", ".join(parts) if parts else "(no arguments)"
    return f"{schema['description']} Expected arguments: {args_desc}."


def help_tool(tool_name: str = None) -> str:
    """Callable version of the same usage hint that already fires on a
    wrong call, so a model can ask up front instead of guessing and
    getting corrected after the fact."""
    if not tool_name:
        active = _active_tool_names()
        lines = []
        for name in sorted(DISPATCH.keys()):
            if name not in active:
                continue
            schema = TOOL_BY_NAME.get(name)
            desc = schema["description"] if schema else ""
            short = desc if len(desc) <= 80 else desc[:77].rsplit(" ", 1)[0] + "..."
            lines.append(f"{name}: {short}")
        return (
            "Available tools:\n" + "\n".join(lines)
            + "\n\nCall help(tool_name='...') for full usage on any one of them."
        )
    if tool_name not in TOOL_BY_NAME:
        return f"ERROR: unknown tool '{tool_name}'. Valid tools: {', '.join(sorted(DISPATCH.keys()))}"
    return _tool_usage_hint(tool_name)


DISPATCH["help"] = help_tool


# Set by run_agent for the duration of one call, from its allowed_tools
# parameter; None means unrestricted (every driver that doesn't pass
# allowed_tools behaves exactly as before this existed). help is always
# reachable regardless, restricted or not, since an agent that can't even
# ask what it's allowed to do can't productively work within a
# restriction - the restriction is on what it can DO, not on whether it
# can find out what it can do.
_ALLOWED_TOOLS = None


def _active_tool_names() -> set:
    return set(DISPATCH) if _ALLOWED_TOOLS is None else (_ALLOWED_TOOLS | {"help"})


# ---------------------------------------------------------------------------
# 4b. Structured output mode: an alternative to relying on the backend's
#     native tool_calls routing (which is what broke against Qwen2.5-Coder,
#     the intent was there, it just never reached the tool_calls field).
#     Grammar-constrained JSON forces the outer shape at the token level
#     regardless of chat template quality. Only the outer shape is
#     constrained (a list of {name, arguments} calls, name locked to a
#     real tool name via enum); arguments stays a generic object rather
#     than a 22-way oneOf, which would make for a large, fragile grammar.
#     Existing argument validation in the dispatch loop already covers
#     the rest. This mode also drops the full per-tool JSON schemas from
#     the request entirely, replaced by one compact reference line per
#     tool in the system prompt, cheaper per turn, not just different.
# ---------------------------------------------------------------------------

def _tool_call_schema(active_names) -> dict:
    return {
        "type": "object",
        "properties": {
            "notes": {"type": ["string", "null"]},
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": sorted(active_names)},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
            "final_answer": {"type": ["string", "null"]},
        },
        "required": ["notes", "calls", "final_answer"],
    }


# The full, unrestricted schema - what an agent with no allowed_tools
# restriction gets, and the shape referenced elsewhere by name.
TOOL_CALL_SCHEMA = _tool_call_schema(DISPATCH.keys())


def _structured_response_format() -> dict:
    # Rebuilt per call rather than reusing TOOL_CALL_SCHEMA directly: a
    # restricted agent's grammar-constrained output should make a
    # disallowed tool name literally unselectable, on top of (not
    # instead of) the dispatch-time rejection below - defense in depth,
    # not relying on either check alone.
    schema = TOOL_CALL_SCHEMA if _ALLOWED_TOOLS is None else _tool_call_schema(_active_tool_names())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tool_call_batch",
            "strict": True,
            "schema": schema,
        },
    }


def _compact_tool_reference() -> str:
    """One line per tool: name(args) - description. Replaces the full
    nested JSON tool schemas as the model's guidance in structured mode,
    the schema no longer needs to teach the model the tool names or
    required arguments, just the enum; this is what teaches it the rest.
    Only lists the currently active (allowed) tools, so a restricted
    agent isn't even told about tools it can't use, rather than being
    told and then separately blocked."""
    active = _active_tool_names()
    lines = []
    for name in sorted(TOOL_BY_NAME.keys()):
        if name not in active:
            continue
        schema = TOOL_BY_NAME[name]
        props = schema["parameters"].get("properties", {})
        required = set(schema["parameters"].get("required", []))
        arg_parts = [p if p in required else f"{p}?" for p in props]
        lines.append(f"{name}({', '.join(arg_parts)}): {schema['description']}")
    return "\n".join(lines)


STRUCTURED_OUTPUT_FORMAT_NOTE = (
    "# Response format\nYou must respond with exactly one JSON object shaped like: "
    '{"notes": <string or null>, "calls": [{"name": "<tool name>", "arguments": {...}}, ...], '
    '"final_answer": <string or null>}. Use "notes" for a brief one- or two-sentence '
    'rationale for what you are about to do or why, this is your only channel for plain '
    'language in this mode, use it. Put one or more calls in "calls" to act. Only batch '
    'more than one call together when each one\'s correctness does not depend on what an '
    'earlier call in the SAME batch will return, e.g. do not both look something up and '
    'act on an assumption about what it contains in the same batch; look it up alone '
    'first, then act on the next turn once you can see the real result. When the task is '
    'done, leave "calls" as an empty array and put your answer in "final_answer". Never leave '
    'both empty unless you are genuinely stuck.'
)


def _apply_structured_output(message: dict, verbose: bool = True) -> dict:
    """Parse a structured-output response's content (which grammar
    constraints guarantee is valid JSON matching TOOL_CALL_SCHEMA) into
    the same message shape the rest of run_agent already expects, so
    nothing downstream needs to know which mode produced it."""
    content = (message.get("content") or "").strip()
    if not content:
        return message
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        if verbose:
            print(f"[NOTICE] structured output was not valid JSON despite the schema: {exc}")
        return message
    if not isinstance(parsed, dict):
        return message

    calls = parsed.get("calls") or []
    final_answer = parsed.get("final_answer")
    notes = parsed.get("notes")
    new_tool_calls = []
    for i, c in enumerate(calls):
        if not isinstance(c, dict) or "name" not in c:
            continue
        new_tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})},
        })
    message["tool_calls"] = new_tool_calls
    message["content"] = final_answer or ""
    # Surface as reasoning_content purely for display/logging consistency
    # with native mode; it is never sent back to the server as history
    # (run_agent's history_message only keeps role/content/tool_calls).
    message["reasoning_content"] = notes or ""
    return message


# ---------------------------------------------------------------------------
# 5. Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a coding assistant with tools for reading and editing code by "
    "symbol name (functions, classes, methods, imports), not by line number. "
    "Each tool listed under '# Your tools' below describes exactly what it "
    "does and what arguments it takes; call help(tool_name='...') any time "
    "mid-task if you want that same description again instead of guessing. "
    "A tool returning an error that a language backend is not implemented "
    "yet is a hard stop for that operation on that file, not something to "
    "retry."
)

# The model has no other way to know what OS/shell run_command actually
# executes on, and defaults to Unix-style quoting (single quotes) because
# that's overwhelmingly what its training data looks like. On Windows,
# shell=True runs through cmd.exe, which does not treat single quotes as
# string delimiters at all, so a `python -c '...'` one-liner fails with a
# syntax error rather than a "command not found"-style hint that the
# quoting itself was the problem. Stating the actual platform up front
# lets the model pick the right quoting the first time instead of
# discovering it by trial and error (or repeating the same broken call).
_HOST_OS = platform.system()  # 'Windows', 'Linux', 'Darwin', ...
if _HOST_OS == "Windows":
    _SHELL_NOTE = (
        "run_command executes on Windows via cmd.exe, not a POSIX shell: "
        "use double quotes for shell-level quoting, e.g. "
        'python -c "code here", not python -c \'code here\'. '
        "cmd.exe does not treat single quotes as string delimiters at all, "
        "so single-quoted arguments will fail with a syntax error inside "
        "whatever you're invoking, not a shell-level error."
    )
else:
    _SHELL_NOTE = (
        f"run_command executes on {_HOST_OS} via a POSIX shell (sh-compatible): "
        "both single and double quotes work as usual."
    )


def _build_system_content(use_structured_output: bool, persona_prompt: str = "") -> str:
    """Assemble the system message as clearly labelled sections instead of
    one long paragraph: an optional per-agent persona section first (see
    below), the short core prompt, the full tool reference (previously
    only shown to the model in structured-output mode, now shown in
    both, since relying on a local model's chat template to surface the
    native tool schemas well is exactly the kind of assumption that
    broke before), and the OS/shell note. The workflow tips that used to
    live in SYSTEM_PROMPT's prose (check_syntax after an edit,
    find_references before a rename/delete, etc.) now live on the
    individual tool descriptions instead, so they show up here AND in
    help(tool_name='...') from one place, rather than two copies to keep
    in sync.

    persona_prompt is where a specific agent's role, in a multi-agent
    setup, belongs - e.g. a reviewer agent saying it should read and
    critique but never call write_file, or a tester agent saying its job
    is to write and run tests, not implement features. It goes first, as
    its own section, so it can add constraints or framing on top of the
    shared tool rules without needing to restate or duplicate them -
    every driver gets the same tool-usage rules and tool reference
    either way; persona_prompt is additive, not a replacement for them.
    Empty by default: a driver with no particular persona (a single
    general-purpose coding agent) just gets the shared rules alone,
    unchanged from before this parameter existed."""
    sections = []
    if persona_prompt.strip():
        sections.append(f"# Your role\n{persona_prompt.strip()}")
    sections += [
        SYSTEM_PROMPT,
        "# Your tools\n" + _compact_tool_reference()
        + "\n\n(Call help(tool_name='...') any time you want the full "
          "description and arguments for just one of these again.)",
        f"# Your operating system\n{_SHELL_NOTE}",
    ]
    if use_structured_output:
        sections.append(STRUCTURED_OUTPUT_FORMAT_NOTE)
    return "\n\n".join(sections)


def _normalize_assistant(message: dict) -> dict:
    if "content" not in message or message["content"] is None:
        message["content"] = ""
    return message


def _truncate_messages(messages: list, max_messages: int = MAX_MESSAGES_IN_CONTEXT) -> list:
    if len(messages) <= max_messages:
        return messages
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    # The first non-system message is the original task (user_prompt).
    # Some nudges re-embed it when they fire (_nudge_message does), but
    # not all of them do - the test-diversity [BEFORE YOU FINISH] message
    # doesn't - so without pinning it here, a long enough run can
    # truncate away the only place the actual task was ever stated.
    task = rest[:1]
    tail = rest[1:]

    slots = max_messages - len(system) - len(task)
    # tail[-0:] is tail[0:], the whole list - a non-positive slot count
    # must not silently mean "keep everything", the opposite of what
    # truncation is for.
    keep = tail[-slots:] if slots > 0 else []
    while keep and keep[0].get("role") == "tool":
        keep.pop(0)
    return system + task + keep


def _consume_stream(lines, on_token=None):
    """Parse an OpenAI-style SSE stream into one accumulated message dict.
    Separated from the network call so it can be unit tested against
    synthetic lines with no server involved."""
    content = ""
    reasoning = ""
    tool_calls_acc = {}
    finish_reason = None

    for line in lines:
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}

        piece = delta.get("content")
        if piece:
            content += piece
            if on_token:
                on_token(piece, False)

        rpiece = delta.get("reasoning_content")
        if rpiece:
            reasoning += rpiece
            if on_token:
                on_token(rpiece, True)

        for tc in (delta.get("tool_calls") or []):
            idx = tc.get("index", 0)
            entry = tool_calls_acc.setdefault(idx, {"id": None, "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                entry["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                entry["function"]["arguments"] += fn["arguments"]

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    tool_calls = []
    for idx in sorted(tool_calls_acc):
        tc = tool_calls_acc[idx]
        if not tc["id"]:
            tc["id"] = f"call_{idx}"
        tool_calls.append({"id": tc["id"], "type": "function", "function": tc["function"]})

    message = {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
    }
    return message, finish_reason


def call_model(messages: list, max_retries: int = 3, max_tokens: int = MAX_TOKENS_DEFAULT,
                verbose: bool = True, use_structured_output: bool = False):
    last_exc = None
    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if use_structured_output:
                payload["response_format"] = _structured_response_format()
            else:
                active = _active_tool_names()
                payload["tools"] = [t for t in TOOLS if t["function"]["name"] in active]
                payload["tool_choice"] = "auto"

            response = requests.post(LM_STUDIO_URL, json=payload, timeout=300, stream=True)
            response.raise_for_status()

            def on_token(piece, is_reasoning):
                # Print live so a long generation is visibly alive in
                # this console too, not just in LM Studio's own log.
                print(piece, end="", flush=True)

            message, finish_reason = _consume_stream(
                response.iter_lines(decode_unicode=True),
                on_token=on_token if verbose else None,
            )
            if verbose and (message["content"] or message["reasoning_content"]):
                print()  # newline after the streamed text
            if finish_reason == "length":
                print(f"[NOTICE] generation hit max_tokens ({max_tokens}) and was cut off.")

            if use_structured_output:
                message = _apply_structured_output(message, verbose=verbose)

            return message
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"model call failed after {max_retries} attempts: {last_exc}")


def _extract_fake_tool_calls(content: str) -> list:
    """Fallback for a model that writes its tool call out as JSON text in
    plain content instead of the API's structured tool_calls field, a real
    gap seen with some local backends/templates even for models genuinely
    fine-tuned for tool use. Finds ```json {"name": ..., "arguments": ...}```
    blocks (or bare JSON objects shaped the same way) and converts them
    into the same shape a real tool_calls entry would have, so intent
    that was clearly there does not just get discarded as inert prose."""
    calls = []
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidates = fenced or re.findall(r"\{[^{}]*\"name\"[^{}]*\}", content, re.DOTALL)
    for i, blob in enumerate(candidates):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append({
                "id": f"fake_call_{i}",
                "type": "function",
                "function": {
                    "name": obj["name"],
                    "arguments": json.dumps(obj.get("arguments", {})),
                },
            })
    return calls


def _tool_call_signature(tool_calls) -> tuple:
    """A comparable fingerprint of a batch of tool calls, used to detect
    the model repeating the exact same call instead of making progress."""
    return tuple(sorted(
        (c["function"]["name"], c["function"].get("arguments", "")) for c in tool_calls
    ))


def _nudge_message(user_prompt: str, reason: str) -> str:
    """A corrective reminder injected mid-conversation, not a fresh system
    prompt, since not every backend renders a second system message the
    same way. Restates the actual task and lists every tool by name so
    the model has something concrete to act on instead of drifting."""
    menu = help_tool()
    return (
        f"[REMINDER] {reason}\n"
        f"Your task is: {user_prompt}\n"
        f"Use one of the tools below to make progress, or give your final "
        f"answer directly in plain text if the task is already done:\n{menu}"
    )


def _update_test_diversity_tracking(edit_tracker: dict, name: str, args: dict, result) -> None:
    """Keep edit_tracker (path -> set of distinct run_command argument
    strings tried since that path's last edit) in sync with what a tool
    call just did.

    A successful edit to a file that has non-trivial loop/branch logic
    (re)starts tracking for that path at zero: whatever was tested before
    this edit says nothing about the code as it now stands, the same way
    a passing test suite from before a change doesn't vouch for the
    change. A successful run_command counts toward every currently-
    tracked path whose module name (the file's stem) appears in the
    command text — a plain textual link, not real import-graph analysis,
    but enough for the common shape this is meant to catch: a
    `python -c "from module import ...; ..."` sanity check. A command
    that's textually identical to one already counted for that path adds
    nothing (it's a set); this deliberately can't tell whether two
    textually different commands actually exercise different code paths,
    only that the model tried something other than repeating itself,
    which is the concrete, cheap, hard-to-fake floor this is aiming for,
    not a guarantee of real coverage."""
    ok = isinstance(result, str) and result.startswith("OK")
    if name in ("write_file", "insert_symbol", "update_symbol", "replace_in_file"):
        if not ok:
            return
        path = args.get("path")
        if not path:
            return
        try:
            # "foo.py", "./foo.py" and "foo/../foo.py" all resolve to the
            # same file via _full_path but would otherwise become three
            # separate tracker entries, keyed on whatever string the
            # model happened to type this time.
            path = os.path.relpath(_full_path(path), WORKDIR)
        except ValueError:
            pass  # fall back to the raw string; shouldn't happen given the call already succeeded
        ext = os.path.splitext(path)[1].lower()
        if EXTENSION_TO_LANGUAGE.get(ext) != "python":
            return
        if _has_testable_logic_python(path):
            edit_tracker[path] = set()
        else:
            edit_tracker.pop(path, None)
    elif name == "run_command":
        if not (isinstance(result, str) and not result.startswith("DECLINED")):
            return
        command_text = (args.get("command") or "").strip()
        if not command_text:
            return
        for tracked_path in edit_tracker:
            stem = os.path.splitext(os.path.basename(tracked_path))[0]
            if stem and re.search(rf"\b{re.escape(stem)}\b", command_text):
                edit_tracker[tracked_path].add(command_text)


def _run_agent_inner(
    user_prompt: str,
    max_steps: int = 10,
    max_consecutive_empty: int = 2,
    max_consecutive_repeats: int = 2,
    max_test_diversity_nudges: int = 2,
    max_tokens: int = MAX_TOKENS_DEFAULT,
    verbose: bool = True,
    use_structured_output: bool = True,
    persona_prompt: str = "",
) -> str:
    system_content = _build_system_content(use_structured_output, persona_prompt)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    empty_count = 0
    repeat_count = 0
    last_signature = None
    edit_tracker = {}  # path -> set of distinct run_command texts tried since its last edit
    test_diversity_nudges = 0

    for step in range(1, max_steps + 1):
        messages = _truncate_messages(messages)
        message = call_model(messages, max_tokens=max_tokens, verbose=verbose,
                              use_structured_output=use_structured_output)
        message = _normalize_assistant(message)

        # Only role/content/tool_calls belong on an assistant message sent
        # back to the API as history. reasoning_content is an output-only
        # field some backends add to the streamed delta (Qwen3's thinking
        # mode, for instance); echoing it back on the next request's
        # messages array is exactly what triggered "Invalid 'messages' in
        # payload", the server didn't recognize the extra key.
        history_message = {"role": "assistant", "content": message.get("content", "")}
        if message.get("tool_calls"):
            history_message["tool_calls"] = message["tool_calls"]
        messages.append(history_message)

        tool_calls = message.get("tool_calls")
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")

        if verbose:
            print(f"\n[STEP {step}] tool_calls={len(tool_calls or [])}, "
                  f"content_len={len(content)}, reasoning_len={len(reasoning)}")
            if reasoning.strip():
                print(f"[NOTES] {reasoning}")

        if not tool_calls and content.strip():
            fake_calls = _extract_fake_tool_calls(content)
            if fake_calls:
                if verbose:
                    print(f"[NOTICE] model wrote {len(fake_calls)} tool call(s) as plain "
                          f"text instead of using the tool-calling API, executing them "
                          f"anyway: {[c['function']['name'] for c in fake_calls]}")
                tool_calls = fake_calls
                message["tool_calls"] = fake_calls  # keep history consistent for next turn

        if not tool_calls:
            if content.strip():
                under_tested = {p: c for p, c in edit_tracker.items() if len(c) < 2}
                if under_tested and test_diversity_nudges < max_test_diversity_nudges:
                    test_diversity_nudges += 1
                    details = "; ".join(
                        f"{p} (tested {len(c)} distinct way(s) so far)"
                        for p, c in under_tested.items()
                    )
                    if verbose:
                        print(f"[NOTICE] holding final answer, insufficient test "
                              f"diversity ({test_diversity_nudges}/{max_test_diversity_nudges}): {details}")
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[BEFORE YOU FINISH] {details}. A single hand-picked "
                            f"example run_command isn't enough evidence for code with "
                            f"a loop or branch in it — it can pass on the one input "
                            f"you tried and still be wrong on others. Run at least "
                            f"one more run_command with meaningfully different "
                            f"arguments (not the same command again) before giving "
                            f"your final answer."
                        ),
                    })
                    continue
                return content

            # No tool call, no content: gave up or got confused. Nudge it
            # back on track with the task and tool menu, up to a limit,
            # rather than ending the run on the first blank response.
            empty_count += 1
            if verbose:
                print(f"[NOTICE] empty response ({empty_count}/{max_consecutive_empty})"
                      + (f", reasoning was: {reasoning[:200]}" if reasoning.strip() else ""))
            if empty_count > max_consecutive_empty:
                warning = (
                    f"[WARNING] Model returned no tool calls and no content "
                    f"{empty_count} times in a row, giving up."
                )
                print(warning)
                return warning

            nudge = _nudge_message(
                user_prompt,
                "You returned nothing usable, no tool call and no answer.",
            )
            messages.append({"role": "user", "content": nudge})
            continue

        # Got at least one tool call: reset the empty-response counter,
        # then check whether it is the exact same call as last time.
        empty_count = 0
        signature = _tool_call_signature(tool_calls)
        if signature == last_signature:
            repeat_count += 1
            if verbose:
                print(f"[NOTICE] repeated identical tool call "
                      f"({repeat_count}/{max_consecutive_repeats})")
            if repeat_count > max_consecutive_repeats:
                warning = (
                    "[WARNING] Model repeated the exact same tool call "
                    f"{repeat_count} times in a row without progress, giving up."
                )
                print(warning)
                return warning
        else:
            repeat_count = 0
        last_signature = signature

        for i, call in enumerate(tool_calls):
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments") or "{}"
            if verbose:
                print(f"  -> calling {name}({raw_args})")
            _log_event({"event": "tool_call", "name": name, "raw_args": raw_args})

            args = None
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                hint = _tool_usage_hint(name)
                result = f"ERROR: could not parse tool arguments: {exc}. {hint}"
            else:
                func = DISPATCH.get(name)
                if func is None:
                    result = (
                        f"ERROR: unknown tool '{name}'. Valid tools: "
                        f"{', '.join(sorted(DISPATCH.keys()))}"
                    )
                elif _ALLOWED_TOOLS is not None and name not in _active_tool_names():
                    # Real enforcement, not just omitting it from the tool
                    # listing: a persona_prompt telling an agent not to use
                    # something is a request the model can still ignore;
                    # this can't be, since the call never reaches func at
                    # all.
                    result = (
                        f"ERROR: '{name}' is not available to this agent. "
                        f"Available tools: {', '.join(sorted(_active_tool_names()))}"
                    )
                else:
                    try:
                        result = func(**args)
                    except TypeError as exc:
                        hint = _tool_usage_hint(name)
                        result = f"ERROR: bad arguments for {name}: {exc}. {hint}"
                    except Exception as exc:
                        result = f"ERROR: {type(exc).__name__}: {exc}"

            if args is not None:
                _update_test_diversity_tracking(edit_tracker, name, args, result)

            if verbose:
                shown = str(result)
                print(f"  <- {shown[:200]}{'...' if len(shown) > 200 else ''}")
            _log_event({"event": "tool_result", "name": name, "result": str(result)[:2000]})
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": str(result),
            })

            # A whole batch was planned in one generation pass, before any
            # of it had actually run, so it encodes assumptions about the
            # current state rather than observations of it. The moment
            # one of those assumptions is wrong, every call queued after
            # it in this same batch is suspect too (it was planned against
            # the same stale picture), so stop here rather than mechanically
            # working through a plan already known to be broken, and let
            # the model re-plan on fresh information next turn instead.
            #
            # This deliberately only fires on ERROR, not on NOTFOUND: a
            # missing symbol/import/anchor/text is very often benign
            # (already applied, wrong file, wrong target) rather than a
            # sign the whole batch's assumptions were wrong, so it gets a
            # result the model can read and keep going on, not a halt.
            if isinstance(result, str) and result.startswith("ERROR"):
                remaining = tool_calls[i + 1:]
                if remaining:
                    skipped_names = [c["function"]["name"] for c in remaining]
                    # The assistant message that started this batch already
                    # declared a tool_call_id for each of these; a strict
                    # OpenAI-compatible backend expects every declared
                    # tool_call_id to get a matching tool-role response
                    # before any other role message follows. Synthesize one
                    # for each call this abort is skipping, rather than
                    # leaving it unanswered.
                    for skipped in remaining:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": skipped["id"],
                            "content": (
                                "SKIPPED: batch aborted because an earlier call "
                                "in this same batch failed; this call was never run."
                            ),
                        })
                    notice = (
                        f"[NOTICE] Skipped {len(remaining)} remaining queued call(s) "
                        f"({', '.join(skipped_names)}): the call above failed, which "
                        f"means this batch's assumptions about the current file state "
                        f"were wrong. Re-check the actual current state (e.g. "
                        f"list(target='symbols') or read_file) before planning your "
                        f"next steps, rather than continuing the original plan."
                    )
                    if verbose:
                        print(notice)
                    messages.append({"role": "user", "content": notice})
                break

        # Moved here (after every tool_call_id in this batch has its
        # matching tool-role response) rather than issued before the loop
        # above: inserting a user-role message between an assistant's
        # declared tool_calls and their tool responses is itself a
        # message-ordering violation a strict backend can reject.
        if repeat_count > 0:
            nudge = _nudge_message(
                user_prompt,
                "You just repeated the same tool call as last time without new progress.",
            )
            messages.append({"role": "user", "content": nudge})

    warning = f"[WARNING] Stopped: max_steps ({max_steps}) reached without a final answer."
    print(warning)
    return warning


def run_agent(
    user_prompt: str,
    max_steps: int = 10,
    max_consecutive_empty: int = 2,
    max_consecutive_repeats: int = 2,
    max_test_diversity_nudges: int = 2,
    max_tokens: int = MAX_TOKENS_DEFAULT,
    verbose: bool = True,
    use_structured_output: bool = True,
    persona_prompt: str = "",
    allowed_tools=None,
) -> str:
    """Thin wrapper around _run_agent_inner that manages the lifecycle of
    _ALLOWED_TOOLS: set it for the duration of this call (None means no
    restriction, the default, identical to every driver that doesn't
    pass allowed_tools at all), then clear it again once this call
    returns or raises, so a restriction from one run_agent call can
    never linger and affect an unrelated direct tool call made
    afterward. allowed_tools is a set/list of tool names to enforce -
    see _run_agent_inner's dispatch loop for where a call outside that
    set actually gets rejected, and _active_tool_names/
    _compact_tool_reference/help_tool/_structured_response_format for
    where the restriction also keeps the model from being told about, or
    grammar-permitted to pick, a tool it can't use in the first place.
    """
    global _ALLOWED_TOOLS
    _ALLOWED_TOOLS = set(allowed_tools) if allowed_tools is not None else None
    try:
        return _run_agent_inner(
            user_prompt,
            max_steps=max_steps,
            max_consecutive_empty=max_consecutive_empty,
            max_consecutive_repeats=max_consecutive_repeats,
            max_test_diversity_nudges=max_test_diversity_nudges,
            max_tokens=max_tokens,
            verbose=verbose,
            use_structured_output=use_structured_output,
            persona_prompt=persona_prompt,
        )
    finally:
        _ALLOWED_TOOLS = None


