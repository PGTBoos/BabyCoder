"""
babycoder.tools.guidance

AGENT GUIDANCE toolkit: the thinking guide, on request rather than pasted
into the prompt. A small model given ten patterns up front recites them; one
that asks when it is stuck uses them. Each agent reads its own
<name>_thinking_guide.md if there is one, else the shared thinking_guide.md.
"""

from ..core import P, _fuzzy_hint, _soft_not_found, tool, ws


@tool("Read the structured-thinking guide. Call with no section to see what is in "
      "it, then with a section name when you are stuck.",
      section=P("string", optional=True))
def read_guide(section=None):
    try:
        with open(ws().guide_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError as exc:
        return f"NOTFOUND: no thinking guide at {ws().guide_path} ({exc})"
    headings = [l for l in lines if l.startswith("#")]
    if not section:
        return ("Sections:\n" + "\n".join(f"  {h.lstrip('# ')}" for h in headings[:20])
                + "\n\nCall read_guide(section='...') with one of these.")

    target = section.lower().strip()
    out, capturing, level = [], False, 0
    for line in lines:
        if line.startswith("#"):
            this_level = len(line) - len(line.lstrip("#"))
            if capturing and this_level <= level:
                break
            if not capturing and target in line.lower():
                capturing, level = True, this_level
        if capturing:
            out.append(line)
    if not out:
        return _soft_not_found("guide section", section, _fuzzy_hint(section, [h.lstrip("# ") for h in headings]))
    return "\n".join(out[:120])
