"""
babycoder.model

Talking to the model. Two ways in, sharing one streaming reader:
  ask_model  - prompt in, prose out. Dreams, merges, Lisa's spoken replies.
  call_model - one step of the tool-calling agent loop.

Both stream so they can show that something is happening, walk away from a
model that thinks without end, and stop the moment a person at the console
starts typing. Dropping the connection is what cancels the generation at the
server.
"""

import json
import re
import time

import requests

from .core import CODING_TEMPERATURE, console, notice, settings

MAX_TOKENS_DEFAULT = 2048
# How far a budget may grow on its own when a generation is cut off with
# nothing written.
MAX_TOKENS_CEILING = 16384
# Characters of thinking a one-shot prose call sits through, with nothing
# written yet, before giving up (roughly a thousand tokens). The agent loop
# allows three times this.
RUNAWAY_THINKING_CHARS = 4000

INTERRUPTED_NOTFOUND = "NOTFOUND: interrupted"

# What ask_model returns when the server could not be reached at all, as
# opposed to a model that answered badly. Callers treat the two differently:
# an unreachable model is not the model failing at the task.
UNREACHABLE_NOTFOUND = "NOTFOUND: model did not answer"


class ModelUnreachable(RuntimeError):
    """The model server could not be reached, or kept failing, after retries."""

THINKING_TAGS = ("think", "thinking", "reasoning", "thought")


def _split_thinking(content):
    """Pull <think>...</think> blocks out of content. Returns (answer, thought).
    An unclosed opener means it was cut off mid-thought: all of it is thought."""
    if "<" not in content:
        return content, ""
    parts = []

    def take(match):
        parts.append(match.group(1))
        return ""

    for tag in THINKING_TAGS:
        content = re.sub(rf"<{tag}\b[^>]*>(.*?)</{tag}\s*>", take, content, flags=re.DOTALL | re.IGNORECASE)
        open_tag = re.search(rf"<{tag}\b[^>]*>", content, flags=re.IGNORECASE)
        if open_tag:
            parts.append(content[open_tag.end():])
            content = content[:open_tag.start()]
    return content.strip(), "\n".join(p.strip() for p in parts if p.strip())


def _consume_stream(lines, on_token=None, give_up_after=None, should_stop=None):
    """Parse an OpenAI-style SSE stream into (message, finish_reason).

    finish_reason is the server's own, or one of ours:
      "interrupted" - should_stop() said the person wants the turn
      "abandoned"   - more than give_up_after chars of thinking, nothing written
    Separated from the network call so it can be tested on synthetic lines.
    """
    content, reasoning = "", ""
    tool_calls_acc = {}
    finish_reason = None

    for line in lines:
        if should_stop and should_stop():
            finish_reason = "interrupted"
            break
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

        # LM Studio says reasoning_content; other servers say reasoning.
        rpiece = delta.get("reasoning_content") or delta.get("reasoning")
        if rpiece:
            reasoning += rpiece
            if on_token:
                on_token(rpiece, True)
            if give_up_after and not content and len(reasoning) > give_up_after:
                finish_reason = "abandoned"
                break

        for tc in (delta.get("tool_calls") or []):
            entry = tool_calls_acc.setdefault(tc.get("index", 0), {"id": None, "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                entry["id"] = tc["id"]
            fn = tc.get("function") or {}
            entry["function"]["name"] += fn.get("name") or ""
            entry["function"]["arguments"] += fn.get("arguments") or ""

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    # Models without a separate channel put thinking in content between
    # <think> tags. Split after the stream, because the tags arrive broken
    # across chunks.
    content, thought = _split_thinking(content)
    if thought:
        reasoning = f"{reasoning}\n{thought}".strip()

    tool_calls = [{"id": tc["id"] or f"call_{idx}", "type": "function", "function": tc["function"]}
                  for idx, tc in sorted(tool_calls_acc.items())]
    return ({"role": "assistant", "content": content, "reasoning_content": reasoning,
             "tool_calls": tool_calls}, finish_reason)


def _stream_chat(payload, on_token=None, give_up_after=None, should_stop=None, timeout=300):
    """POST one streaming request and read it to the end. Always closes the
    connection - the old code leaked one on every retry and every early exit,
    which kept the server generating into a socket nobody read."""
    payload = dict(payload, stream=True)
    response = requests.post(settings.url, json=payload, timeout=(10, timeout), stream=True)
    try:
        if not response.ok:
            # The server's own explanation is in the body; requests' message
            # alone makes every incompatibility look the same.
            raise requests.HTTPError(f"{response.status_code} {response.reason}: {response.text[:300]}",
                                     response=response)
        # SSE arrives with no charset, and requests then assumes ISO-8859-1:
        # that is what turned curly apostrophes into mojibake.
        response.encoding = "utf-8"
        return _consume_stream(response.iter_lines(decode_unicode=True),
                               on_token=on_token, give_up_after=give_up_after, should_stop=should_stop)
    finally:
        response.close()


class _Marks:
    """Progress marks while a model works with nothing shown: '.' per chunk of
    thinking, ':' per chunk of writing. That answers "what is it busy with"
    without pouring the model's deliberation over the screen."""

    EVERY = 160

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.seen = {True: 0, False: 0}
        self.printed = False

    def __call__(self, piece, is_reasoning):
        if not self.enabled:
            return
        self.seen[is_reasoning] += len(piece)
        if self.seen[is_reasoning] < self.EVERY:
            return
        self.seen[is_reasoning] = 0
        mark = "." if is_reasoning else ":"
        if console.progress:
            console.progress(mark)
        elif console.takes_turn is None:
            print(mark, end="", flush=True)
            self.printed = True

    def done(self):
        # A registered console ends its own inline line when it next writes.
        if self.printed:
            print(flush=True)
            self.printed = False


# Ways of asking a model to stop thinking, best first. Servers spell it
# differently and do not say when they ignore it (LM Studio falls back to
# thinking ON for an unsupported value), so only the effect can be probed.
THINKING_OFF_CANDIDATES = [
    {"reasoning_effort": "off", "chat_template_kwargs": {"enable_thinking": False}},
    {"reasoning_effort": "off"},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"reasoning_effort": "none"},
    {"reasoning_effort": "low"},
    {"think": False},
    {},
]

_quiet_cache = None   # None = not probed yet


def _quiet_keys(verbose=False):
    """The payload keys that actually stop this model thinking. Probed once
    per run with a one-word request per candidate."""
    global _quiet_cache
    if settings.thinking:
        return {}
    if _quiet_cache is not None:
        return _quiet_cache
    for candidate in THINKING_OFF_CANDIDATES:
        payload = {"model": settings.model, "temperature": 0, "max_tokens": 64,
                   "messages": [{"role": "user", "content": "Reply with the one word: ready"}], **candidate}
        try:
            message, _ = _stream_chat(payload, timeout=60)
        except requests.HTTPError:
            # This server rejects that spelling outright (400). That is an
            # answer about the candidate, not about the server: try the next.
            continue
        except requests.RequestException:
            return {}   # no server at all; do not cache a verdict reached offline
        if not message["reasoning_content"].strip():
            if candidate:
                notice(f"[NOTICE] thinking is quietened by {candidate}", verbose)
            _quiet_cache = candidate
            return candidate
    notice("[NOTICE] this model thinks whatever it is asked; relying on the runaway cutoff instead.", verbose)
    _quiet_cache = {}
    return {}


def _unwrap_envelope(text):
    """A model fresh from structured turns may hand back the JSON envelope
    even when asked for prose. Return final_answer from it if so."""
    stripped = text.strip()
    if not (stripped.startswith("{") and "final_answer" in stripped):
        return text
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return text
    answer = parsed.get("final_answer") if isinstance(parsed, dict) else None
    return answer.strip() if isinstance(answer, str) and answer.strip() else text


def ask_model(prompt, system_prompt="", max_tokens=1024, temperature=0.7, verbose=True):
    """One prompt in, prose out. Never raises: an unreachable model returns a
    NOTFOUND string the caller can skip, same as every other soft miss."""
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
        + [{"role": "user", "content": prompt}]
    grew = False
    while True:
        payload = {"model": settings.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens, **_quiet_keys(verbose)}
        marks = _Marks(verbose)
        try:
            message, finish = _stream_chat(payload, on_token=marks,
                                           give_up_after=None if settings.thinking else RUNAWAY_THINKING_CHARS,
                                           should_stop=console.takes_turn)
        except requests.RequestException as exc:
            return f"NOTFOUND: model did not answer ({exc})"
        finally:
            marks.done()

        if finish == "interrupted":
            return INTERRUPTED_NOTFOUND
        answer = message["content"]
        if not answer.strip() and finish == "length" and not grew and max_tokens * 4 <= MAX_TOKENS_CEILING:
            # Spent the whole budget planning. One bigger try; a third is a
            # hang with a progress bar.
            notice(f"[NOTICE] {max_tokens} tokens went on thinking with nothing written; "
                    f"one retry at {max_tokens * 4}.", verbose)
            max_tokens, grew = max_tokens * 4, True
            continue
        if finish == "length" and answer.strip():
            notice(f"[NOTICE] answer hit max_tokens ({max_tokens}) and stops mid-sentence.", verbose)
        answer = _unwrap_envelope(answer)
        if not answer.strip():
            if finish == "abandoned":
                return "NOTFOUND: the model was still thinking about it and never started writing"
            return "NOTFOUND: the model wrote no answer"
        return answer


def _apply_structured_output(message, verbose=True):
    """Turn the grammar-constrained JSON reply into the same message shape
    native tool mode produces, so nothing downstream cares which mode ran."""
    content = (message.get("content") or "").strip()
    if not content:
        return message
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        notice(f"[NOTICE] structured output was not valid JSON despite the schema: {exc}", verbose)
        return message
    if not isinstance(parsed, dict):
        return message
    message["tool_calls"] = [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}
        for i, c in enumerate(parsed.get("calls") or []) if isinstance(c, dict) and "name" in c]
    message["content"] = parsed.get("final_answer") or ""
    message["reasoning_content"] = parsed.get("notes") or ""
    return message




def call_model(messages, box=None, max_retries=3, max_tokens=MAX_TOKENS_DEFAULT, verbose=True,
               use_structured_output=False, temperature=CODING_TEMPERATURE, should_stop=None):
    """One step of the agent loop. Returns the assistant message.

    message["interrupted"] is set when the person took the turn mid-generation;
    the loop must stop then rather than act on a half-written reply.
    """
    if box is None:
        from .toolsets import ToolBox
        box = ToolBox()
    stop = should_stop or console.takes_turn
    attempt, last_exc = 0, None
    while attempt < max_retries:
        payload = {"model": settings.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, **_quiet_keys(verbose)}
        if use_structured_output:
            payload["response_format"] = box.response_format()
        else:
            payload["tools"] = box.schemas()
            payload["tool_choice"] = "auto"

        if verbose:
            def on_token(piece, is_reasoning):
                print(piece, end="", flush=True)
            marks = None
        else:
            # Quiet agents still show that something is happening, through
            # the console if one is registered.
            marks = on_token = _Marks(enabled=console.progress is not None)

        try:
            message, finish = _stream_chat(payload, on_token=on_token, should_stop=stop,
                                           give_up_after=None if settings.thinking else RUNAWAY_THINKING_CHARS * 3)
        except requests.RequestException as exc:
            last_exc = exc
            attempt += 1
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            continue
        finally:
            if marks:
                marks.done()

        if finish == "interrupted":
            message["interrupted"] = True
            return message
        if verbose and (message["content"] or message["reasoning_content"]):
            print()
        empty = not message["content"].strip() and not message["tool_calls"]
        if finish == "length" and empty and max_tokens * 4 <= MAX_TOKENS_CEILING and attempt < max_retries - 1:
            notice(f"[NOTICE] cut off at {max_tokens} tokens with nothing written yet; "
                    f"retrying at {max_tokens * 4}.", verbose)
            max_tokens *= 4
            attempt += 1
            continue
        if finish == "length":
            notice(f"[NOTICE] generation hit max_tokens ({max_tokens}) and was cut off.", verbose)
        elif finish == "abandoned":
            notice("[NOTICE] the model kept thinking without writing anything; dropped that attempt.", verbose)
        if use_structured_output:
            message = _apply_structured_output(message, verbose)
        return message
    raise ModelUnreachable(f"model call failed after {max_retries} attempts: {last_exc}")
