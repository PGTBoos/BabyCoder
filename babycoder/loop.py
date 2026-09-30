"""
babycoder.loop

run_agent: drive a model through its granted tools until it answers.
"""

import json
import os
import platform
import re

from .backends import _is_python
from .advisories import _has_testable_logic_python
from .core import CODING_TEMPERATURE, _full_path, _log_event, console, notice, using, ws
from .model import UNREACHABLE_NOTFOUND, ModelUnreachable, INTERRUPTED_NOTFOUND, MAX_TOKENS_DEFAULT, call_model
from .toolsets import ToolBox

MAX_MESSAGES_IN_CONTEXT = 40

# The most of any one tool result kept in the conversation. _truncate_messages
# counts messages, not size, so a single 256 KB read_file used to push the
# task itself out of an 8k context without anything saying so. The full
# result still goes to the transcript.
MAX_TOOL_RESULT_CHARS = 8000

SYSTEM_PROMPT = (
    "You are a coding assistant with tools for reading and editing code by symbol name "
    "(functions, classes, methods, imports), not by line number. A tool returning an error "
    "that a language backend is not implemented yet is a hard stop for that file, not "
    "something to retry.")

CODE_READING_SYSTEM_PROMPT = (
    "You work with code, reading it by symbol name rather than by line number. You cannot "
    "change it: you have no tools that write, and that is deliberate, so describe what "
    "should change rather than attempting it.")

NON_CODING_SYSTEM_PROMPT = (
    "Use the tools rather than guessing at something you could look up or recall, and do "
    "not describe calling a tool instead of calling it. If no tool fits, say what you "
    "think and why.")

CODE_EDIT_TRIGGERS = {"write_file", "update_symbol", "insert_symbol", "delete_symbol",
                      "rename_symbol", "replace_in_file", "replace_in_project",
                      "format_file", "run_command"}
CODE_READ_TRIGGERS = {"read_symbol", "read_file", "find_references", "check_syntax", "lookup_symbol_docs"}

if platform.system() == "Windows":
    _SHELL_NOTE = ("run_command executes on Windows via cmd.exe, not a POSIX shell: use double "
                   "quotes for shell-level quoting, e.g. python -c \"code here\". Single quotes "
                   "are not string delimiters in cmd.exe.")
else:
    _SHELL_NOTE = (f"run_command executes on {platform.system()} via a POSIX shell: single and "
                   "double quotes both work.")

STRUCTURED_OUTPUT_FORMAT_NOTE = (
    "# Response format\nRespond with exactly one JSON object: "
    '{"notes": <string or null>, "calls": [{"name": "<tool>", "arguments": {...}}, ...], '
    '"final_answer": <string or null>}. "notes" is a one- or two-sentence rationale. Put one '
    'or more calls in "calls" to act. Only batch calls whose correctness does not depend on '
    "what an earlier call in the same batch returns: look something up alone first, then act "
    'on the next turn. When the task is done, leave "calls" empty and put your answer in '
    '"final_answer".')

# What run_agent returns instead of an answer. Machinery, never speech: a
# caller that speaks its result must be able to tell the difference.
NO_ANSWER_MARKER = "[WARNING]"
INTERRUPTED = f"{NO_ANSWER_MARKER} Interrupted."
UNREACHABLE = f"{NO_ANSWER_MARKER} Model unreachable."


def gave_no_answer(result):
    """True when run_agent stopped without producing something sayable."""
    return not result or result.startswith(NO_ANSWER_MARKER)


def was_interrupted(result):
    return result in (INTERRUPTED, INTERRUPTED_NOTFOUND)


def model_unreachable(result):
    """True when a call returned because the server could not be reached.
    Not a failure of the model at the task, so a caller should wait and try
    again later rather than count it against anything."""
    return bool(result) and (result.startswith(UNREACHABLE) or result.startswith(UNREACHABLE_NOTFOUND))


def _build_system_content(box, workspace, use_structured_output, persona_prompt=""):
    """Role first, then the core prompt that fits what was granted, the tool
    reference, the usage notes for the granted capabilities, any read-only
    folders, and the OS note only for an agent that can run commands."""
    names = box.names()
    sections = []
    if persona_prompt.strip():
        sections.append(f"# Your role\n{persona_prompt.strip()}")
    if names & CODE_EDIT_TRIGGERS:
        sections.append(SYSTEM_PROMPT)
    elif names & CODE_READ_TRIGGERS:
        sections.append(CODE_READING_SYSTEM_PROMPT)
    else:
        sections.append(NON_CODING_SYSTEM_PROMPT)
    sections.append("# Your tools\n" + box.reference())
    notes = box.usage_notes()
    if notes:
        sections.append("# Using these well\n" + "\n".join(f"- {n}" for n in notes))
    if workspace.read_roots and names & CODE_READ_TRIGGERS:
        lines = [f"  @{label}/  e.g. read_file(path='@{label}/file.py'), list(target='dir', path='@{label}')"
                 for label in sorted(workspace.read_roots)]
        sections.append("# Other folders\nYou can read, but not change, these folders by starting a "
                        "path with their name. Plain paths are your own folder.\n" + "\n".join(lines))
    if "run_command" in names:
        sections.append(f"# Your operating system\n{_SHELL_NOTE}")
    if use_structured_output:
        sections.append(STRUCTURED_OUTPUT_FORMAT_NOTE)
    return "\n\n".join(sections)


def _truncate_messages(messages, max_messages=MAX_MESSAGES_IN_CONTEXT):
    """Keep system, the original task, and the most recent tail. A tail must
    not start with orphaned tool results."""
    if len(messages) <= max_messages:
        return messages
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    task, tail = rest[:1], rest[1:]
    slots = max_messages - len(system) - len(task)
    keep = tail[-slots:] if slots > 0 else []
    while keep and keep[0].get("role") == "tool":
        keep.pop(0)
    return system + task + keep


def _cap(result):
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    return (result[:MAX_TOOL_RESULT_CHARS]
            + f"\n\n[TRUNCATED: {len(result)} chars, showing the first {MAX_TOOL_RESULT_CHARS}. "
              "Use read_symbol for one function, or search_files for one thing, instead of "
              "reading everything.]")


def _extract_fake_tool_calls(content):
    """Native mode only: a model that writes its tool call as JSON text in
    content instead of the tool_calls field still gets it executed."""
    calls = []
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidates = fenced or re.findall(r"\{[^{}]*\"name\"[^{}]*\}", content, re.DOTALL)
    for i, blob in enumerate(candidates):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append({"id": f"fake_call_{i}", "type": "function",
                          "function": {"name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}))}})
    return calls


def _tool_call_signature(tool_calls):
    return tuple(sorted((c["function"]["name"], c["function"].get("arguments", "")) for c in tool_calls))


def _nudge_message(box, user_prompt, reason):
    return (f"[REMINDER] {reason}\nYour task is: {user_prompt}\n"
            f"Use one of your tools to make progress, or give your final answer if the task "
            f"is already done.\n\n{box.help()}")


def _update_test_diversity_tracking(edit_tracker, name, args, result):
    """edit_tracker: path -> set of distinct run_command texts tried since
    that file was last edited. An edit to a file with branching logic resets
    it; a run_command naming the module counts toward it. A cheap, textual
    floor against "tested once with one hand-picked input", not coverage."""
    ok = isinstance(result, str) and result.startswith("OK")
    if name in ("write_file", "insert_symbol", "update_symbol", "replace_in_file"):
        path = args.get("path")
        if not ok or not path:
            return
        try:
            path = os.path.relpath(_full_path(path), ws().workdir)
        except ValueError:
            pass
        if not _is_python(path):
            return
        if _has_testable_logic_python(path):
            edit_tracker[path] = set()
        else:
            edit_tracker.pop(path, None)
    elif name == "run_command":
        command_text = (args.get("command") or "").strip()
        if not isinstance(result, str) or result.startswith("DECLINED") or not command_text:
            return
        for tracked_path, tried in edit_tracker.items():
            stem = os.path.splitext(os.path.basename(tracked_path))[0]
            if stem and re.search(rf"\b{re.escape(stem)}\b", command_text):
                tried.add(command_text)


def _dispatch(box, name, raw_args):
    """Run one call. Returns (args or None, result string)."""
    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"ERROR: could not parse tool arguments: {exc}. {box.usage(name)}"
    if name not in box.names():
        return args, f"ERROR: unknown tool '{name}'. Your tools: {', '.join(sorted(box.names()))}"
    if console.on_tool:
        console.on_tool(name)
    try:
        result = box.call(name, args)
    except TypeError as exc:
        return args, f"ERROR: bad arguments for {name}: {exc}. {box.usage(name)}"
    except Exception as exc:
        return args, f"ERROR: {type(exc).__name__}: {exc}"
    # Only a call that really ran reaches the watcher, never a refused one.
    if console.on_result:
        console.on_result(name, args, result)
    return args, result


def run_agent(user_prompt, max_steps=10, max_consecutive_empty=2, max_consecutive_repeats=2,
              max_test_diversity_nudges=2, max_tokens=MAX_TOKENS_DEFAULT, verbose=True,
              use_structured_output=True, persona_prompt="", allowed_tools=None,
              temperature=CODING_TEMPERATURE, should_stop=None, workspace=None):
    """Drive the model through its granted tools until it answers.

    workspace: the folders this run works in. Defaults to the current one
    (configure_workspace). Passing it is how a room gives each role its own
    folders without reconfiguring anything between turns; the previous
    workspace is back in place when this returns.

    Returns the answer, or a string starting with NO_ANSWER_MARKER (check
    with gave_no_answer / was_interrupted) when it stopped without one.
    """
    with using(workspace or ws()) as current:
        try:
            return _run(user_prompt, current, ToolBox(allowed_tools), max_steps, max_consecutive_empty,
                        max_consecutive_repeats, max_test_diversity_nudges, max_tokens, verbose,
                        use_structured_output, persona_prompt, temperature, should_stop)
        except ModelUnreachable as exc:
            # Before, this escaped run_agent and killed the agent. The server
            # being down is something to report and wait out, not a crash.
            notice(f"[NOTICE] {exc}", verbose)
            return f"{UNREACHABLE} {exc}"


def _run(user_prompt, workspace, box, max_steps, max_consecutive_empty, max_consecutive_repeats,
         max_test_diversity_nudges, max_tokens, verbose, use_structured_output, persona_prompt,
         temperature, should_stop):
    messages = [{"role": "system",
                 "content": _build_system_content(box, workspace, use_structured_output, persona_prompt)},
                {"role": "user", "content": user_prompt}]
    empty_count = repeat_count = test_nudges = 0
    last_signature = None
    edit_tracker = {}

    def give_up(text):
        if verbose:
            print(text)
        return text

    for step in range(1, max_steps + 1):
        if should_stop is not None and should_stop():
            return INTERRUPTED

        messages = _truncate_messages(messages)
        message = call_model(messages, box=box, max_tokens=max_tokens, verbose=verbose,
                             use_structured_output=use_structured_output,
                             temperature=temperature, should_stop=should_stop)
        if message.get("interrupted"):
            # A half-written reply is never acted on: in structured mode it is
            # half a JSON envelope, and in native mode the fake-call extractor
            # would happily run half a tool call.
            return INTERRUPTED

        tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""

        if verbose:
            print(f"\n[STEP {step}] tool_calls={len(tool_calls)}, content_len={len(content)}, "
                  f"reasoning_len={len(reasoning)}")
            if reasoning.strip():
                print(f"[NOTES] {reasoning}")

        if not tool_calls and content.strip() and not use_structured_output:
            tool_calls = _extract_fake_tool_calls(content)
            if tool_calls:
                notice(f"[NOTICE] model wrote {len(tool_calls)} tool call(s) as plain text; "
                       f"executing them anyway.", verbose)

        # History keeps only what the API accepts back: role/content/tool_calls.
        # reasoning_content echoed back is what caused "Invalid 'messages'".
        history = {"role": "assistant", "content": content}
        if tool_calls:
            history["tool_calls"] = tool_calls
        messages.append(history)

        if not tool_calls:
            if content.strip():
                under_tested = {p: c for p, c in edit_tracker.items() if len(c) < 2}
                if under_tested and test_nudges < max_test_diversity_nudges:
                    test_nudges += 1
                    details = "; ".join(f"{p} (tested {len(c)} distinct way(s) so far)"
                                        for p, c in under_tested.items())
                    notice(f"[NOTICE] holding final answer, insufficient test diversity: {details}", verbose)
                    messages.append({"role": "user", "content": (
                        f"[BEFORE YOU FINISH] {details}. One hand-picked example is not enough for "
                        "code with a loop or branch. Run at least one more run_command with "
                        "meaningfully different arguments before your final answer.")})
                    continue
                return content

            empty_count += 1
            notice(f"[NOTICE] empty response ({empty_count}/{max_consecutive_empty})", verbose)
            if empty_count > max_consecutive_empty:
                return give_up(f"{NO_ANSWER_MARKER} Model returned nothing usable {empty_count} "
                               "times in a row, giving up.")
            messages.append({"role": "user", "content": _nudge_message(
                box, user_prompt, "You returned nothing usable, no tool call and no answer.")})
            continue

        empty_count = 0
        signature = _tool_call_signature(tool_calls)
        repeat_count = repeat_count + 1 if signature == last_signature else 0
        last_signature = signature
        if repeat_count > max_consecutive_repeats:
            return give_up(f"{NO_ANSWER_MARKER} Model repeated the same tool call {repeat_count} "
                           "times without progress, giving up.")

        for i, call in enumerate(tool_calls):
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments") or "{}"
            if verbose:
                print(f"  -> calling {name}({raw_args})")
            _log_event({"event": "tool_call", "name": name, "raw_args": raw_args})

            args, result = _dispatch(box, name, raw_args)
            result = str(result)
            if args is not None:
                _update_test_diversity_tracking(edit_tracker, name, args, result)
            if verbose:
                print(f"  <- {result[:200]}{'...' if len(result) > 200 else ''}")
            _log_event({"event": "tool_result", "name": name, "result": result[:20000]})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": _cap(result)})

            # A batch was planned before any of it ran. Once one call fails,
            # the rest were planned against a picture now known to be wrong:
            # answer each skipped call (strict backends require it) and let
            # the model re-plan. NOTFOUND does not abort; it is often benign.
            if result.startswith("ERROR"):
                remaining = tool_calls[i + 1:]
                for skipped in remaining:
                    messages.append({"role": "tool", "tool_call_id": skipped["id"],
                                     "content": "SKIPPED: an earlier call in this batch failed; "
                                                "this call was never run."})
                if remaining:
                    messages.append({"role": "user", "content": (
                        f"[NOTICE] Skipped {len(remaining)} queued call(s) because the call above "
                        "failed. Re-check the actual current state before planning your next step.")})
                break

        # After every tool response, never between them: a user message there
        # is itself an ordering violation on a strict backend.
        if repeat_count > 0:
            messages.append({"role": "user", "content": _nudge_message(
                box, user_prompt, "You just repeated the same tool call as last time.")})

    return give_up(f"{NO_ANSWER_MARKER} Stopped: max_steps ({max_steps}) reached without a final answer.")
