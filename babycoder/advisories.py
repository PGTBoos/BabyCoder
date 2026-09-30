"""
babycoder.advisories

Run after any mutating tool succeeds, folded into that tool's own result:
duplicate definitions (a real bug), loops with no stated invariant, and
non-trivial functions with no docstring. Rendered as linter-style lines
because a small model skims a polite paragraph and fixes a diagnostic.
None start with ERROR/NOTFOUND, so none abort a batch.
"""

import ast
import functools
import io
import tokenize

from .backends import _is_python
from .core import _read_source


def _find_duplicate_symbols_python(path) -> list:
    tree = ast.parse(_read_source(path))
    results = []

    def scan(body, scope_name):
        seen = {}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seen.setdefault(node.name, []).append(node.lineno)
        results.extend({"name": n, "scope": scope_name, "lines": l} for n, l in seen.items() if len(l) > 1)
        for node in body:
            if isinstance(node, ast.ClassDef):
                scan(node.body, node.name)

    scan(tree.body, None)
    return results


def _comment_lines(source) -> set:
    lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                lines.add(tok.start[0])
    except tokenize.TokenError:
        pass
    return lines


def _find_unexplained_loop_functions_python(path) -> list:
    """Functions with a loop whose invariant is not written down as a comment
    (next to an assert, above the loop, or on the loop body's first line).
    A comment, not the model's notes: notes vanish, a comment stays."""
    source = _read_source(path)
    tree = ast.parse(source)
    comments = _comment_lines(source)

    def explained_at(lineno):
        return lineno in comments or (lineno - 1) in comments

    def loop_is_explained(loop):
        return explained_at(loop.lineno) or (bool(loop.body) and explained_at(loop.body[0].lineno))

    results = []

    def scan(body, scope_name):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                loops = [n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While))]
                if not loops:
                    continue
                asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
                if not (any(explained_at(a.lineno) for a in asserts) or any(loop_is_explained(l) for l in loops)):
                    results.append({"name": node.name, "scope": scope_name, "lineno": node.lineno})
            elif isinstance(node, ast.ClassDef):
                scan(node.body, node.name)

    scan(tree.body, None)
    return results


def _is_trivial(node) -> bool:
    return node.name.startswith("_") or len(node.body) <= 1


def _has_testable_logic_python(path) -> bool:
    """True if a non-trivial function has a loop or branch, i.e. code where
    one hand-picked example can pass while an edge case goes untested."""
    try:
        tree = ast.parse(_read_source(path))
    except (SyntaxError, FileNotFoundError, ValueError):
        return False

    def scan(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _is_trivial(node) and any(isinstance(n, (ast.For, ast.While, ast.If)) for n in ast.walk(node)):
                    return True
            elif isinstance(node, ast.ClassDef) and scan(node.body):
                return True
        return False

    return scan(tree.body)


def _find_undocumented_functions_python(path) -> list:
    tree = ast.parse(_read_source(path))
    results = []

    def scan(body, scope_name):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _is_trivial(node) and ast.get_docstring(node) is None:
                    results.append({"name": node.name, "scope": scope_name, "lineno": node.lineno})
            elif isinstance(node, ast.ClassDef):
                scan(node.body, node.name)

    scan(tree.body, None)
    return results


def advised(func):
    """Decorator for mutating tools: after success, append advisories for the
    file that was touched. A no-op for non-python files and failed calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, str) or not result.startswith("OK"):
            return result
        path = kwargs.get("path", args[0] if args else None)
        if not path or not _is_python(path, kwargs.get("language")):
            return result
        try:
            dups = _find_duplicate_symbols_python(path)
            bare_loops = _find_unexplained_loop_functions_python(path)
            undocumented = _find_undocumented_functions_python(path)
        except (SyntaxError, OSError, ValueError):
            # An advisory must never turn a successful edit into an ERROR.
            return result

        if dups:
            details = "; ".join(
                f"'{d['name']}' defined {len(d['lines'])} times at lines {d['lines']}"
                + (f" inside class {d['scope']}" if d["scope"] else "") for d in dups)
            result += (f"\n[WARNING] {path} now contains duplicate definitions: {details}. "
                       "This is a real bug, not a style nit: Python keeps only the last "
                       "definition. Use delete_symbol to remove the extra copies, keeping one.")

        style, fixes = [], []
        for f in bare_loops:
            scope = f" (class {f['scope']})" if f["scope"] else ""
            style.append(f"{path}:{f['lineno']}: PROJ-INV001 loop in '{f['name']}'{scope} "
                         "has no comment/assert stating its invariant")
        if bare_loops:
            fixes.append("PROJ-INV001: via update_symbol, either add an assert for the invariant "
                         "with a comment stating it (e.g. '# invariant: no node is enqueued twice'), "
                         "or a comment above the loop saying there is none and why.")
        for f in undocumented:
            scope = f" (class {f['scope']})" if f["scope"] else ""
            style.append(f"{path}:{f['lineno']}: PROJ-DOC001 '{f['name']}'{scope} has no docstring")
        if undocumented:
            fixes.append("PROJ-DOC001: via update_symbol, add a one- or two-sentence docstring "
                         "describing role, parameters and return value.")
        if style:
            result += ("\n[style check: project-specific rules, not real Python or flake8 errors, "
                       "but fix them before a sanity check or final answer]\n"
                       + "\n".join(style) + "\n" + "\n".join(fixes))
        return result
    return wrapper
