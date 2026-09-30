"""
babycoder.tools.coding

AGENT CODER toolkit: everything that changes code, plus everything that reads
it. Confined to the calling agent's Workspace, backed up before every write;
run_command asks a human first.

Paths starting with @label/ reach a read-only folder of another agent (see
Workspace.read_roots). Every write resolves with write=True, so a write there
is refused before anything is touched.
"""

import ast
import fnmatch
import json
import os
import pydoc
import re
import subprocess

from ..advisories import advised
from ..backends import _is_python, _resolve_backend
from ..core import (MAX_READ_BYTES, P, _closest_line_hint, _full_path, _read_source,
                    _soft_not_found, _write_source, console, tool, ws)

_PATH = P("string")
_OPT_PATH = P("string", optional=True)
_LANG = P("string", "Optional override, e.g. python, csharp, typescript", optional=True)
_SCOPE = P("string", "Class name, for a method", optional=True)


def _inside(full, root):
    return full == root or full.startswith(root + os.sep)


def _walk_files(root):
    """Every non-hidden file under root, as (real path, display path). A
    symlink pointing outside root is skipped: os.walk does not follow
    directory links, but it happily lists a file link to /etc/passwd, and
    search_files used to open it and return the contents."""
    w = ws()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            real = os.path.realpath(os.path.join(dirpath, fname))
            if _inside(real, root):
                yield real, w.display(real)


@tool("Read a whole file's raw content (capped at 256 KB).", path=_PATH)
def read_file(path):
    full = _full_path(path)
    if not os.path.exists(full):
        return f"ERROR: {path} does not exist"
    if not os.path.isfile(full):
        return f"ERROR: {path} is not a file"
    size = os.path.getsize(full)
    # Cap by bytes before decoding, so multi-byte text cannot slip past it.
    with open(full, "rb") as f:
        content = f.read(MAX_READ_BYTES).decode("utf-8", errors="replace")
    if size > MAX_READ_BYTES:
        content += f"\n\n[TRUNCATED: file is {size} bytes, showing first {MAX_READ_BYTES}]"
    return content


@tool("Create or overwrite a file with given content (a backup is taken first).",
      path=_PATH, content=P("string"))
@advised
def write_file(path, content):
    _write_source(path, content)
    return f"OK: wrote {len(content)} chars to {path}"


def list_dir(path="."):
    full = _full_path(path or ".")
    if not os.path.exists(full):
        return f"ERROR: {path} does not exist"
    if not os.path.isdir(full):
        return f"ERROR: {path} is not a directory"
    entries = sorted(e for e in os.listdir(full) if not e.startswith("."))
    return "\n".join(entries) if entries else "(empty)"


@tool("Search file contents for a substring or regex.",
      pattern=P("string"), path=_OPT_PATH,
      regex=P("boolean", optional=True), max_results=P("integer", optional=True))
def search_files(pattern, path=".", regex=False, max_results=100):
    root = _full_path(path or ".")
    if not os.path.exists(root):
        return f"ERROR: {path} does not exist"
    try:
        matcher = re.compile(pattern if regex else re.escape(pattern))
    except re.error as exc:
        return f"ERROR: bad pattern: {exc}"
    results = []
    for real, shown in _walk_files(root):
        try:
            with open(real, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if matcher.search(line):
                        results.append(f"{shown}:{i}: {line.rstrip()}")
                        if len(results) >= max_results:
                            return "\n".join(results) + f"\n... (truncated at {max_results})"
        except OSError:
            continue
    return "\n".join(results) if results else "(no matches)"


def _replace(source, old_str, new_str, count):
    occurrences = source.count(old_str)
    if count is not None and count >= 0:
        return source.replace(old_str, new_str, count), min(count, occurrences)
    return source.replace(old_str, new_str), occurrences


@tool("Literal find-replace within one file, not symbol-aware. For updating a call "
      "pattern written the same way each time it appears. Python files are "
      "syntax-checked before committing.",
      path=_PATH, old_str=P("string"), new_str=P("string"),
      count=P("integer", "max replacements, omit for all", optional=True))
@advised
def replace_in_file(path, old_str, new_str, count=-1):
    _full_path(path, write=True)   # refuse a read-only folder before reading it
    source = _read_source(path)
    if old_str not in source:
        return _soft_not_found(f"text in {path}", old_str, _closest_line_hint(source, old_str))
    new_source, replaced = _replace(source, old_str, new_str, count)
    if _is_python(path):
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: replacement would produce invalid syntax at line {exc.lineno}: {exc.msg}"
    _write_source(path, new_source)
    return f"OK: replaced {replaced} occurrence(s) in {path}"


@tool("The same literal find-replace applied across every matching file under a "
      "folder (default: whole project). Use when a signature or call pattern changed "
      "and every call site needs the same update. A file is skipped, not partially "
      "written, if it would break Python syntax.",
      old_str=P("string"), new_str=P("string"), path=_OPT_PATH,
      file_glob=P("string", "e.g. *.py to restrict which files are touched", optional=True),
      count_per_file=P("integer", "max replacements per file, omit for all", optional=True))
def replace_in_project(old_str, new_str, path=".", file_glob=None, count_per_file=-1):
    root = _full_path(path or ".", write=True)
    if not os.path.exists(root):
        return f"ERROR: {path} does not exist"
    results, total = [], 0
    for real, rel in _walk_files(root):
        if file_glob and not fnmatch.fnmatch(os.path.basename(real), file_glob):
            continue
        try:
            source = _read_source(rel)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if old_str not in source:
            continue
        new_source, replaced = _replace(source, old_str, new_str, count_per_file)
        if _is_python(rel):
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


@tool("Apply deterministic code formatting (black, for Python).", path=_PATH)
@advised
def format_file(path):
    if not os.path.exists(_full_path(path, write=True)):
        return f"ERROR: {path} does not exist"
    if not _is_python(path):
        return f"ERROR: formatting for {os.path.splitext(path)[1] or 'this file type'} not implemented yet"
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


def list_dependencies(language=None):
    if (language or "python") != "python":
        return f"ERROR: dependency listing for {language} not implemented yet"
    if not os.path.exists(_full_path(REQUIREMENTS_PATH)):
        return "(no requirements.txt found)"
    return _read_source(REQUIREMENTS_PATH)


@tool("Add a dependency to the project's dependency file.",
      name=P("string"), version=P("string", optional=True), language=_LANG)
def add_dependency(name, version=None, language=None):
    if (language or "python") != "python":
        return f"ERROR: dependency management for {language} not implemented yet"
    existing = _read_source(REQUIREMENTS_PATH) if os.path.exists(_full_path(REQUIREMENTS_PATH)) else ""
    for line in existing.splitlines():
        if re.split(r"[=<>!~\[]", line.strip())[0] == name:
            return f"ERROR: {name} already listed in requirements.txt (as '{line.strip()}')"
    entry = f"{name}=={version}" if version else name
    if existing and not existing.endswith("\n"):
        existing += "\n"
    _write_source(REQUIREMENTS_PATH, existing + entry + "\n")
    return f"OK: added '{entry}' to requirements.txt"


@tool("Offline documentation lookup for a stdlib or installed symbol, e.g. "
      "'os.path.join'. No network call. Use this before guessing the signature of "
      "an unfamiliar call.",
      symbol=P("string"), language=_LANG)
def lookup_symbol_docs(symbol, language=None):
    if (language or "python") != "python":
        return f"ERROR: docs lookup for {language} not implemented yet"
    try:
        text = pydoc.render_doc(symbol, renderer=pydoc.plaintext)
    except Exception as exc:
        return f"ERROR: could not find docs for '{symbol}': {type(exc).__name__}: {exc}"
    return text if len(text) <= 3000 else text[:3000] + "\n... (truncated)"


def _ask_at_terminal(command, cwd):
    print(f"\n[AGENT WANTS TO RUN]: {command}")
    print(f"[WORKING DIRECTORY]: {cwd}")
    try:
        return input("Allow this command to run? [y/N]: ").strip().lower() == "y"
    except EOFError:
        return False   # nobody at the terminal: fail closed


@tool("Propose a shell command. Requires explicit human approval before it runs; "
      "never assume it ran just because you called this.",
      command=P("string"))
def run_command(command):
    cwd = ws().workdir
    # The only tool that has to ask a person something. It goes through the
    # console's approve hook when one is registered, so a room, a UI or a
    # test decides how that question is asked instead of this tool calling
    # input() itself.
    approve = console.approve or _ask_at_terminal
    if not approve(command, cwd):
        return "DECLINED: user did not approve running this command"
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60s"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    output = f"exit_code={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    missing = re.search(r"No module named '([\w.]+)'", result.stderr)
    if missing:
        output += (f"\n\nSUGGESTION: '{missing.group(1)}' isn't installed. Use add_dependency, "
                   "then 'pip install -r requirements.txt' (with approval) before retrying.")
    return output


def list_backups():
    manifest = ws().backup_manifest
    if not os.path.exists(manifest):
        return "(no backups)"
    out = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in reversed([l.strip() for l in f if l.strip()]):
            try:
                entry = json.loads(line)
                out.append(f"{entry['backup']}  ->  {entry['original']}")
            except (json.JSONDecodeError, KeyError):
                continue
    return "\n".join(out) if out else "(no backups)"


@tool("Restore a file from a backup name returned by list(target='backups').", name=P("string"))
def restore_backup(name):
    w = ws()
    if not name or any(s in name for s in (os.sep, "/", "\\")) or name in (".", ".."):
        return f"ERROR: invalid backup name: {name}"
    if not os.path.exists(w.backup_manifest):
        return "ERROR: no backups recorded"
    original = None
    with open(w.backup_manifest, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("backup") == name:
                original = entry.get("original")
    if original is None:
        return f"ERROR: backup {name} not found in manifest"
    src = os.path.join(w.backup_dir, name)
    if not os.path.isfile(src):
        return f"ERROR: backup file {name} is missing on disk"
    try:
        with open(src, "r", encoding="utf-8") as f:
            _write_source(original, f.read())
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    return f"OK: restored {original} from {name}"


# -- language-routed tools --------------------------------------------------

def list_symbols(path, language=None):
    return _resolve_backend(path, language).list_symbols(path)


def list_imports(path, language=None):
    return _resolve_backend(path, language).list_imports(path)


@tool("Read the full source of one named function, class, or method.",
      path=_PATH, name=P("string"), scope=_SCOPE, language=_LANG)
def read_symbol(path, name, scope=None, language=None):
    return _resolve_backend(path, language).read_symbol(path, name, scope)


@tool("Replace a function/class/method by name with new code. Validated before it "
      "is committed; an invalid edit is rolled back.",
      path=_PATH, name=P("string"), new_code=P("string"), scope=_SCOPE, language=_LANG)
@advised
def update_symbol(path, name, new_code, scope=None, language=None):
    return _resolve_backend(path, language).update_symbol(path, name, new_code, scope)


@tool("Insert a new function/class after a named anchor symbol, into a class, or at end of file.",
      path=_PATH, code=P("string"), after=P("string", optional=True), scope=_SCOPE, language=_LANG)
@advised
def insert_symbol(path, code, after=None, scope=None, language=None):
    return _resolve_backend(path, language).insert_symbol(path, code, after, scope)


@tool("Add an import statement after the shebang, encoding cookie, module docstring, "
      "and any existing imports.",
      path=_PATH, statement=P("string"), language=_LANG)
@advised
def add_import(path, statement, language=None):
    return _resolve_backend(path, language).add_import(path, statement)


@tool("Remove an import (by module or imported name) from a file.",
      path=_PATH, name=P("string"), language=_LANG)
@advised
def remove_import(path, name, language=None):
    return _resolve_backend(path, language).remove_import(path, name)


@tool("Rename every identifier occurrence in a file. Token-aware: strings and "
      "comments are left alone. Prefer this over editing text by hand for renames.",
      path=_PATH, old_name=P("string"), new_name=P("string"), language=_LANG)
@advised
def rename_symbol(path, old_name, new_name, language=None):
    return _resolve_backend(path, language).rename_symbol(path, old_name, new_name)


@tool("Remove a function, class, or method by name.",
      path=_PATH, name=P("string"), scope=_SCOPE, language=_LANG)
@advised
def delete_symbol(path, name, scope=None, language=None):
    return _resolve_backend(path, language).delete_symbol(path, name, scope)


@tool("Find actual usages of a name (AST-aware), unlike search_files which matches "
      "any text, comments included. Run this before rename_symbol or delete_symbol "
      "on anything you did not just write yourself.",
      path=_PATH, name=P("string"), scope=_SCOPE, language=_LANG)
def find_references(path, name, scope=None, language=None):
    return _resolve_backend(path, language).find_references(path, name, scope)


# Deliberately not @advised: a pure read whose success also starts with OK.
@tool("Parse a file and report syntax errors, if any. Run this after any edit.",
      path=_PATH, language=_LANG)
def check_syntax(path, language=None):
    return _resolve_backend(path, language).check_syntax(path)


# -- the "list" meta-tool: five listings behind one enum ---------------------

def _need_path(target, path, fn, language):
    if not path:
        return f"ERROR: target '{target}' requires 'path'. Usage: list(target='{target}', path='file.py')"
    return fn(path, language)


LIST_TARGETS = {
    "symbols": lambda path, lang: _need_path("symbols", path, list_symbols, lang),
    "imports": lambda path, lang: _need_path("imports", path, list_imports, lang),
    "dir": lambda path, lang: list_dir(path or "."),
    "dependencies": lambda path, lang: list_dependencies(lang),
    "backups": lambda path, lang: list_backups(),
}


@tool("List something: symbols or imports of a file, files in a dir, declared "
      "dependencies, or backups. Supply path for symbols/imports/dir.",
      tool_name="list",
      target=P("string", enum=sorted(LIST_TARGETS)), path=_OPT_PATH, language=_LANG)
def list_things(target, path=None, language=None):
    fn = LIST_TARGETS.get(target)
    if fn is None:
        return f"ERROR: unknown list target '{target}'. Valid targets: {', '.join(sorted(LIST_TARGETS))}."
    return fn(path, language)
