"""
emotion_agent.py

A turn-based character agent whose internal state (emotions, scene,
thought, goal, text) is enforced by a tool-calling schema rather than
hoped for as free-text JSON. The model doesn't write JSON into its
reply and hope it parses, it calls a tool whose arguments ARE the
schema, so the API itself constrains the shape of what comes back. That
is the same reason babycoder's coding tools work reliably on
small local models where raw text output does not, structure enforced
at the calling layer beats structure requested in a prompt.

State fields:
  persona  - the system prompt. Set at startup, changed only via the
             /persona command, never by the model itself each turn.
             Letting a model rewrite its own system prompt turn to turn
             is exactly the kind of thing that drifts fast on a small
             model, so it is deliberately not part of the per-turn tool.
  emotions - a name -> 0..1 intensity dict. The label set is whatever
             you put in STARTUP_SCHEMA at the top of this file, nothing
             hardcoded past that. Each turn, every existing value decays
             a little toward zero BEFORE the model's update is merged
             in, in Python, not by asking the model to remember to do
             it, so a feeling doesn't linger forever just because
             nothing revisited it. Values are clamped to 0..1 in Python
             regardless of what the model sends.
  scene    - short filmic description of the current situation/action.
  thought  - internal monologue, not spoken aloud, shown dim.
  goal     - the character's current driving objective.
  text     - the actual line of dialogue shown to the user.

Only "text" is required in the tool schema, everything else is
optional per turn, Python merges whatever the model sent into the
persistent state rather than requiring the model to restate fields
that have not changed.

Direct overrides: /persona, /emotion <name> <0-1>, /goal, /scene,
/state, /quit let you (or another script) alter any field without
going through the model at all, same "python code updates the schema,
and it will use it" idea, the model always sees the current state fed
back to it fresh each turn, it never has to remember its own history
of a value.

Memory: the last MEMORY_TURNS turns (user line + agent's spoken line)
are kept in MEMORY_FILE as plain markdown, one entry per turn,
separated by "---". The file is not an ever-growing transcript, on
every write it is trimmed back down to MEMORY_TURNS entries, so its
size stays constant regardless of how long a session runs. It is
reloaded at startup, so a new run continues from where the last one
left off rather than starting blank.

Falls back gracefully rather than crashing if a backend does not honor
tool_choice, or a model ignores the tool entirely - same NOTFOUND
philosophy as babycoder's other tools: a bad turn is a warning,
not a stack trace.

Run:
    pip install requests
    python emotion_agent.py
"""

import json
import re

import requests

# Where the model is comes from the same place as every other agent:
# babycoder.settings, set by BABYCODER_URL / BABYCODER_MODEL or
# configure_model(). This agent keeps its own request code on purpose - one
# non-streaming call with a single required tool is the whole design - but it
# should not have its own copy of the address to fall out of step.
from babycoder import settings

MAX_TOKENS = 400  # hard cap on generation length - without this, a model
                  # that rambles instead of concluding can run for minutes
                  # and produce nothing usable, rather than failing fast

# Everything project-specific lives here. Change the emotion labels,
# starting persona, goal, or scene for a different character without
# touching any logic below.
STARTUP_SCHEMA = {
    # "/no_think" is Qwen3's own documented switch for turning off its
    # extended "thinking" channel. Without it, some Qwen3 builds (including
    # ones like qwen3.8-14b-instruct-turbo) generate an unbounded internal
    # monologue instead of ever calling a tool or answering - that is what
    # produced the multi-thousand-word "reasoning_content" wall with an
    # empty "content" and "tool_calls": []. Drop this line if you are not
    # running a Qwen3-family model, it is harmless leftover text otherwise.
    "persona": (
        "You are a character in an interactive scene. Stay in character. "
        "Always call the speak_turn tool to respond, never reply in plain "
        "text. /no_think"
    ),
    "emotion_labels": ["joy", "fear", "anger", "sadness", "curiosity"],
    "emotion_decay": 0.85,  # each turn, old values *= this, before merging
    "goal": "",
    "scene": "",
}

MEMORY_FILE = "agent_memory.md"  # markdown log of recent turns, doubles as loaded context
MEMORY_TURNS = 5  # how many recent turns are kept in memory and reloaded into context

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "speak_turn",
            "description": (
                "Report this turn's spoken line and, optionally, any "
                "changes to your internal state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emotions": {
                        "type": "object",
                        "description": (
                            "Only emotions that changed this turn, as "
                            "name: intensity (0 to 1)."
                        ),
                        "additionalProperties": {"type": "number"},
                    },
                    "scene": {
                        "type": "string",
                        "description": "Filmic description of the current situation/action.",
                    },
                    "thought": {
                        "type": "string",
                        "description": "Internal monologue, not spoken aloud.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "Your current driving objective, if it changed.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The actual line of dialogue spoken aloud.",
                    },
                },
                "required": ["text"],
            },
        },
    }
]

COLOR = {
    "emotions": "\033[35m",  # magenta
    "scene": "\033[36m",     # cyan
    "thought": "\033[2;37m", # dim
    "goal": "\033[33m",      # yellow
    "text": "\033[97m",      # bright white
    "warn": "\033[31m",      # red
    "room": "\033[90m",      # gray, for /commands and system notices
    "reset": "\033[0m",
}


def make_initial_state() -> dict:
    return {
        "persona": STARTUP_SCHEMA["persona"],
        "emotions": {name: 0.0 for name in STARTUP_SCHEMA["emotion_labels"]},
        "scene": STARTUP_SCHEMA["scene"],
        "thought": "",
        "goal": STARTUP_SCHEMA["goal"],
        "text": "",
    }


def apply_state_update(state: dict, args: dict) -> None:
    """Merge the model's (partial, untrusted) update into state. Every
    field is validated/clamped here, in Python, the model's output is
    never written straight into state."""
    decay = STARTUP_SCHEMA["emotion_decay"]
    for name in list(state["emotions"]):
        state["emotions"][name] = round(state["emotions"][name] * decay, 3)

    updates = args.get("emotions")
    if isinstance(updates, dict):
        for name, value in updates.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            state["emotions"][name] = round(max(0.0, min(1.0, value)), 3)

    for field in ("scene", "thought", "goal"):
        value = args.get(field)
        if isinstance(value, str) and value.strip():
            state[field] = value.strip()

    text = args.get("text")
    state["text"] = text.strip() if isinstance(text, str) else ""


def print_turn(state: dict) -> None:
    active = {k: v for k, v in state["emotions"].items() if v > 0.01}
    if active:
        summary = ", ".join(f"{k}={v}" for k, v in active.items())
        print(f"{COLOR['emotions']}[emotions] {summary}{COLOR['reset']}")
    if state["scene"]:
        print(f"{COLOR['scene']}[scene] {state['scene']}{COLOR['reset']}")
    if state["thought"]:
        print(f"{COLOR['thought']}[thought] {state['thought']}{COLOR['reset']}")
    if state["goal"]:
        print(f"{COLOR['goal']}[goal] {state['goal']}{COLOR['reset']}")
    print(f"{COLOR['text']}Agent: {state['text']}{COLOR['reset']}")


def load_memory() -> list:
    """Load up to MEMORY_TURNS previous turns from MEMORY_FILE, oldest
    first. Returns a list of {"user": ..., "agent": ...} dicts. A
    missing or empty file just means starting with no memory, not an
    error - same soft-miss philosophy as the rest of this file."""
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return []
    entries = []
    for block in content.split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        user_match = re.search(r"\*\*You:\*\* (.*)", block)
        agent_match = re.search(r"\*\*Agent:\*\* (.*)", block)
        if user_match and agent_match:
            entries.append({"user": user_match.group(1), "agent": agent_match.group(1)})
    return entries[-MEMORY_TURNS:]


def append_memory(entries: list, user_text: str, agent_text: str) -> list:
    """Append one turn to the in-memory list AND to MEMORY_FILE as a
    markdown entry, then trim both down to the last MEMORY_TURNS turns.
    MEMORY_FILE is not an ever-growing transcript, it IS the memory
    window, its whole content is always just the last MEMORY_TURNS
    entries, in plain readable markdown. Returns the trimmed list so
    the caller's variable stays in sync with what's on disk."""
    entries = entries + [{"user": user_text, "agent": agent_text}]
    entries = entries[-MEMORY_TURNS:]
    blocks = [f"**You:** {e['user']}\n**Agent:** {e['agent']}" for e in entries]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(blocks) + "\n")
    return entries


def build_messages(state: dict, memory_entries: list, user_input: str) -> list:
    # The state briefing is rebuilt fresh every call rather than left to
    # accumulate in memory - the model should see where things stand
    # NOW, not a growing log of every value it has ever held. Dialogue
    # continuity (what was actually said) comes from memory_entries
    # instead, loaded from MEMORY_FILE and capped at MEMORY_TURNS.
    state_brief = "Current internal state (for continuity, do not print verbatim):\n" + json.dumps(
        {
            "emotions": state["emotions"],
            "scene": state["scene"],
            "thought": state["thought"],
            "goal": state["goal"],
        }
    )
    messages = [
        {"role": "system", "content": state["persona"] + "\n\n" + state_brief},
    ]
    for entry in memory_entries:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["agent"]})
    messages.append({"role": "user", "content": user_input})
    return messages


def call_llm(messages: list, tool_choice) -> dict:
    payload = {
        "model": settings.model,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.8,
        "max_tokens": MAX_TOKENS,
        # The template-level Qwen3 switch for thinking mode - more
        # reliable than the /no_think text hint, if LM Studio's backend
        # passes it through to the chat template renderer.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    response = requests.post(settings.url, json=payload, timeout=120)
    if not response.ok:
        # requests' own "400 Client Error" text does not include LM
        # Studio's actual explanation for the rejection - that's in the
        # response body. Without this, every backend incompatibility
        # looks identical from the outside.
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {response.text}",
            response=response,
        )
    return response.json()["choices"][0]["message"]


def get_turn(messages: list) -> dict:
    """Returns a dict of whatever fields the model actually gave us.
    Never raises on a malformed or missing tool call - a bad turn just
    means "text" falls back to whatever plain content the model wrote,
    same soft-miss idea as babycoder's NOTFOUND results."""
    try:
        message = call_llm(messages, tool_choice="required")
    except requests.RequestException as e:
        print(f"{COLOR['warn']}[warn] tool_choice=\"required\" failed ({e}), retrying without it.{COLOR['reset']}")
        try:
            message = call_llm(messages, tool_choice=None)
        except requests.RequestException as e2:
            print(f"{COLOR['warn']}[warn] request failed entirely: {e2}{COLOR['reset']}")
            return {"text": f"(no response - {e2})"}

    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        content = message.get("content") or ""
        reasoning = (message.get("reasoning_content") or "").strip()
        if not content and reasoning:
            # Non-whitespace reasoning_content with empty content and no
            # tool call means the model spent the whole turn thinking and
            # never actually answered. Printing a snippet here means you
            # never have to go dig through LM Studio's own log for this.
            print(f"{COLOR['warn']}[warn] model produced only internal "
                  f"reasoning ({len(reasoning)} chars) and never gave an "
                  f"actual answer or tool call this turn. First 200 chars: "
                  f"{reasoning[:200]!r}{COLOR['reset']}")
            return {"text": "(model produced no answer, only internal reasoning)"}
        print(f"{COLOR['warn']}[warn] model did not use the tool this turn, "
              f"falling back to its plain text as the spoken line.{COLOR['reset']}")
        return {"text": content}

    raw_args = tool_calls[0]["function"]["arguments"]
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        print(f"{COLOR['warn']}[warn] model's tool arguments were not valid JSON, "
              f"ignoring structured fields this turn.{COLOR['reset']}")
        return {"text": message.get("content") or ""}
    print(f"{COLOR['room']}[debug] model used the tool this turn, state will update.{COLOR['reset']}")
    return args


def handle_command(raw: str, state: dict) -> bool:
    """Direct, model-bypassing edits to the state. Returns False to quit."""
    parts = raw[1:].split(maxsplit=2)
    if not parts:
        return True
    cmd = parts[0].lower()

    if cmd in ("quit", "exit"):
        return False
    if cmd == "state":
        print(f"{COLOR['room']}{json.dumps(state, indent=2)}{COLOR['reset']}")
    elif cmd == "persona" and len(parts) > 1:
        state["persona"] = raw.split(maxsplit=1)[1]
        print(f"{COLOR['room']}[room] persona updated.{COLOR['reset']}")
    elif cmd == "goal" and len(parts) > 1:
        state["goal"] = raw.split(maxsplit=1)[1]
        print(f"{COLOR['room']}[room] goal updated.{COLOR['reset']}")
    elif cmd == "scene" and len(parts) > 1:
        state["scene"] = raw.split(maxsplit=1)[1]
        print(f"{COLOR['room']}[room] scene updated.{COLOR['reset']}")
    elif cmd == "emotion" and len(parts) == 3:
        name, value = parts[1], parts[2]
        try:
            state["emotions"][name] = round(max(0.0, min(1.0, float(value))), 3)
            print(f"{COLOR['room']}[room] emotions[{name}] set to "
                  f"{state['emotions'][name]} directly, bypassing the model.{COLOR['reset']}")
        except ValueError:
            print(f"{COLOR['warn']}[room] usage: /emotion <name> <0..1>{COLOR['reset']}")
    else:
        print(f"{COLOR['room']}[room] commands: /persona <text>, /emotion <name> <0-1>, "
              f"/goal <text>, /scene <text>, /state, /quit{COLOR['reset']}")
    return True


def main() -> None:
    state = make_initial_state()
    memory_entries = load_memory()
    if memory_entries:
        print(f"{COLOR['room']}[room] loaded {len(memory_entries)} turn(s) of memory "
              f"from {MEMORY_FILE}.{COLOR['reset']}")
    print(f"{COLOR['room']}[room] emotion_agent ready. /state to inspect, /quit to leave.{COLOR['reset']}")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{COLOR['room']}[room] bye.{COLOR['reset']}")
            break

        if not user_input:
            continue
        if user_input.startswith("/"):
            if not handle_command(user_input, state):
                break
            continue

        messages = build_messages(state, memory_entries, user_input)
        args = get_turn(messages)
        apply_state_update(state, args)
        print_turn(state)

        memory_entries = append_memory(memory_entries, user_input, state["text"])


if __name__ == "__main__":
    main()