"""
babycoder.backends

Language backends for the symbol tools. Only Python is real; C# and
TypeScript are registered stubs so the hook for Roslyn / the TS compiler API
is already in place. The file extension decides, with no python fallback.
"""

import ast
import io
import json
import keyword
import os
import re
import tokenize

from .core import _fuzzy_hint, _match_indent, _read_source, _soft_not_found, _write_source


class LanguageBackend:
    """Interface every language backend must implement."""

    def list_symbols(self, path): raise NotImplementedError
    def read_symbol(self, path, name, scope=None): raise NotImplementedError
    def update_symbol(self, path, name, new_code, scope=None): raise NotImplementedError
    def insert_symbol(self, path, code, after=None, scope=None): raise NotImplementedError
    def list_imports(self, path): raise NotImplementedError
    def add_import(self, path, statement): raise NotImplementedError
    def remove_import(self, path, name): raise NotImplementedError
    def rename_symbol(self, path, old_name, new_name): raise NotImplementedError
    def delete_symbol(self, path, name, scope=None): raise NotImplementedError
    def find_references(self, path, name, scope=None): raise NotImplementedError
    def check_syntax(self, path): raise NotImplementedError


class PythonBackend(LanguageBackend):
    """Real implementation on the standard library ast module."""

    def _find_node(self, tree, name, scope=None):
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

    def _symbol_names(self, tree) -> list:
        names = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
                if isinstance(node, ast.ClassDef):
                    names += [c.name for c in node.body
                              if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return names

    def _import_names(self, tree) -> list:
        names = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.append(node.module)
                names.extend(a.name for a in node.names)
        return names

    def _missing_symbol(self, tree, name, scope):
        subject = f"symbol{f' in scope {scope}' if scope else ''}"
        return _soft_not_found(subject, name, _fuzzy_hint(name, self._symbol_names(tree)))

    # -- read-only -----------------------------------------------------------

    def list_symbols(self, path):
        tree = ast.parse(_read_source(path))
        symbols = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append({"kind": "function", "name": node.name, "line": node.lineno})
            elif isinstance(node, ast.ClassDef):
                symbols.append({"kind": "class", "name": node.name, "line": node.lineno})
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append({"kind": "method", "name": child.name,
                                        "scope": node.name, "line": child.lineno})
        return json.dumps(symbols, indent=2)

    def read_symbol(self, path, name, scope=None):
        source = _read_source(path)
        tree = ast.parse(source)
        node = self._find_node(tree, name, scope)
        if node is None:
            return self._missing_symbol(tree, name, scope)
        return ast.get_source_segment(source, node)

    def list_imports(self, path):
        tree = ast.parse(_read_source(path))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.append("import " + ", ".join(a.name for a in node.names))
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"from {node.module} import " + ", ".join(a.name for a in node.names))
        return "\n".join(imports) if imports else "(no imports)"

    def check_syntax(self, path):
        try:
            ast.parse(_read_source(path))
        except SyntaxError as exc:
            return f"ERROR: syntax error at line {exc.lineno}, col {exc.offset}: {exc.msg}"
        return f"OK: {path} parses cleanly"

    # -- mutating ------------------------------------------------------------

    def update_symbol(self, path, name, new_code, scope=None):
        source = _read_source(path)
        tree = ast.parse(source)
        node = self._find_node(tree, name, scope)
        if node is None:
            return self._missing_symbol(tree, name, scope)
        new_code = _match_indent(new_code, " " * (node.col_offset or 0))
        lines = source.splitlines(keepends=True)

        # A leading comment in new_code is a fresh explanation of this symbol.
        # It sits outside the node's span, so without absorbing the existing
        # comment block above the def, each re-edit stacked another copy on
        # top of the last. Stops at the first blank or non-comment line.
        start_line = node.lineno - 1
        if new_code.lstrip().startswith("#"):
            while start_line > 0 and lines[start_line - 1].strip().startswith("#"):
                start_line -= 1

        new_source = "".join(lines[:start_line] + [new_code.rstrip("\n") + "\n"] + lines[node.end_lineno:])
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: edit would produce invalid syntax at line {exc.lineno}: {exc.msg}"
        _write_source(path, new_source)
        return f"OK: updated {name} in {path}"

    def insert_symbol(self, path, code, after=None, scope=None):
        source = _read_source(path)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)

        # Refuse to create a duplicate: across separate runs the model has no
        # memory of having inserted this already.
        inserted_name = None
        try:
            for node in ast.parse(code).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    inserted_name = node.name
                    break
        except SyntaxError:
            pass  # the full-file parse below reports genuinely bad code
        if inserted_name:
            existing = self._find_node(tree, inserted_name, scope)
            if existing is not None:
                where = f" in scope {scope}" if scope else ""
                return (f"ERROR: '{inserted_name}' already exists{where} in {path} "
                        f"at line {existing.lineno}. Use update_symbol to replace it, "
                        f"or delete_symbol first if you intend to reinsert it.")

        indent = ""
        if after:
            node = self._find_node(tree, after, scope)
            if node is None:
                return _soft_not_found("anchor symbol", after, _fuzzy_hint(after, self._symbol_names(tree)))
            insert_at = node.end_lineno
            indent = " " * (node.col_offset or 0)
        elif scope:
            node = self._find_node(tree, scope, None)
            if node is None:
                class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
                return _soft_not_found("scope", scope, _fuzzy_hint(scope, class_names))
            insert_at = node.end_lineno
            child_indent = next((c.col_offset for c in node.body if hasattr(c, "col_offset")), 4)
            indent = " " * child_indent
        else:
            insert_at = len(lines)

        code = _match_indent(code, indent)
        new_source = "".join(lines[:insert_at] + ["\n", code.rstrip("\n") + "\n"] + lines[insert_at:])
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: insert would produce invalid syntax at line {exc.lineno}: {exc.msg}"
        _write_source(path, new_source)
        anchor = after or (f"end of {scope}" if scope else "end of file")
        return f"OK: inserted new symbol after {anchor} in {path}"

    def add_import(self, path, statement):
        source = _read_source(path)
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)

        normalized = statement.strip()
        if any(line.strip() == normalized for line in lines):
            return f"ERROR: import '{normalized}' is already present in {path}"

        insert_index = 0
        if lines and lines[0].startswith("#!"):
            insert_index = 1
        if len(lines) > insert_index and re.match(r"^#.*coding[:=]", lines[insert_index]):
            insert_index += 1
        body = tree.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            insert_index = max(insert_index, body[0].end_lineno)
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                insert_index = max(insert_index, node.end_lineno)

        new_source = "".join(lines[:insert_index] + [statement.rstrip("\n") + "\n"] + lines[insert_index:])
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: adding import would produce invalid syntax: {exc.msg}"
        _write_source(path, new_source)
        return f"OK: added '{statement}' to {path}"

    def remove_import(self, path, name):
        source = _read_source(path)
        tree = ast.parse(source)
        line_starts = [0]
        for line in source.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))

        def alias_text(alias):
            return f"{alias.name} as {alias.asname}" if alias.asname else alias.name

        # node -> replacement text, or None to delete the whole statement.
        # `import os, sys` naming "os" keeps "sys"; only naming the module of
        # a from-import removes the whole statement.
        replacements = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                if any(a.name == name for a in node.names):
                    kept = [a for a in node.names if a.name != name]
                    replacements[node] = "import " + ", ".join(alias_text(a) for a in kept) if kept else None
            elif isinstance(node, ast.ImportFrom):
                if node.module == name:
                    replacements[node] = None
                elif any(a.name == name for a in node.names):
                    kept = [a for a in node.names if a.name != name]
                    replacements[node] = (
                        f"from {'.' * node.level}{node.module or ''} import "
                        + ", ".join(alias_text(a) for a in kept)) if kept else None

        if not replacements:
            return _soft_not_found("import", name, _fuzzy_hint(name, self._import_names(tree)))

        # Bottom up, so offsets computed against the original stay valid.
        new_source = source
        for node in sorted(replacements, key=lambda n: n.lineno, reverse=True):
            start, end = line_starts[node.lineno - 1], line_starts[node.end_lineno]
            text = replacements[node]
            new_source = new_source[:start] + (text + "\n" if text is not None else "") + new_source[end:]
        _write_source(path, new_source)
        return f"OK: removed import {name} from {path}"

    def rename_symbol(self, path, old_name, new_name):
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

        edits = [(line_starts[t.start[0] - 1] + t.start[1], line_starts[t.end[0] - 1] + t.end[1])
                 for t in tokens if t.type == tokenize.NAME and t.string == old_name]
        if not edits:
            known = {t.string for t in tokens if t.type == tokenize.NAME and not keyword.iskeyword(t.string)}
            return _soft_not_found("name", old_name, _fuzzy_hint(old_name, known))

        new_source = source
        for start, end in reversed(edits):
            new_source = new_source[:start] + new_name + new_source[end:]
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: rename would produce invalid syntax: {exc.msg}"
        _write_source(path, new_source)
        return f"OK: renamed {len(edits)} occurrence(s) of {old_name} to {new_name} in {path}"

    def delete_symbol(self, path, name, scope=None):
        source = _read_source(path)
        tree = ast.parse(source)
        node = self._find_node(tree, name, scope)
        if node is None:
            return self._missing_symbol(tree, name, scope)
        lines = source.splitlines(keepends=True)
        new_source = "".join(lines[: node.lineno - 1] + lines[node.end_lineno:])
        try:
            ast.parse(new_source)
        except SyntaxError as exc:
            return f"ERROR: delete would produce invalid syntax at line {exc.lineno}: {exc.msg}"
        _write_source(path, new_source)
        return f"OK: deleted {name} from {path}"

    def find_references(self, path, name, scope=None):
        source = _read_source(path)
        tree = ast.parse(source)
        lines = source.splitlines()
        seen, refs = set(), []
        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", None)
            if lineno is None:
                continue
            hit = ((isinstance(node, ast.Name) and node.id == name)
                   or (isinstance(node, ast.Attribute) and node.attr == name))
            if hit and lineno not in seen:
                seen.add(lineno)
                text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
                refs.append(f"{path}:{lineno}: {text}")
        return "\n".join(sorted(refs, key=lambda r: int(r.split(":")[1]))) if refs \
            else f"(no references to {name} found in {path})"


class NotImplementedBackend(LanguageBackend):
    """Placeholder so the dispatch table already lists every planned language."""

    def __init__(self, language_name):
        self.language_name = language_name

    def _not_ready(self, *args, **kwargs):
        return f"ERROR: {self.language_name} backend not implemented yet"

    list_symbols = read_symbol = update_symbol = insert_symbol = _not_ready
    list_imports = add_import = remove_import = rename_symbol = _not_ready
    delete_symbol = find_references = check_syntax = _not_ready


LANGUAGE_BACKENDS = {
    "python": PythonBackend(),
    "csharp": NotImplementedBackend("csharp"),
    "typescript": NotImplementedBackend("typescript"),
}

EXTENSION_TO_LANGUAGE = {".py": "python", ".cs": "csharp", ".ts": "typescript", ".tsx": "typescript"}


def _resolve_backend(path, language=None) -> LanguageBackend:
    # The file type decides. No python fallback for an unknown extension.
    if language is None:
        ext = os.path.splitext(path)[1].lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)
        if language is None:
            return NotImplementedBackend(f"unknown extension {ext!r}")
    return LANGUAGE_BACKENDS.get(language) or NotImplementedBackend(language)


def _is_python(path, language=None) -> bool:
    return (language or EXTENSION_TO_LANGUAGE.get(os.path.splitext(path)[1].lower())) == "python"
