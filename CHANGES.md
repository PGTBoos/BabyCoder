# Refactor notes

Unzip over the repo. `agent_toolkit.py` is now a two-line shim; the code lives
in the `babycoder/` package. Run `python tests/check_babycoder.py` first: 85
offline checks against a mock LM Studio.

## Structure

- **Tools are objects.** `@tool("help text", arg=P("string"))` on the
  function. The schema, the system prompt line, `help(tool_name=...)` and the
  bad-call hint are generated from that one declaration. A tool is still
  called like the function.
- **ToolBox per run.** `run_agent` builds one from the grant. Generic help is
  derived from each granted tool, grouped by toolset. The old module-level
  `_ALLOWED_TOOLS` global is gone.
- **Package split.** `core`, `backends`, `advisories`, `model`, `toolsets`,
  `loop`, and `tools/` with one module per capability. `import agent_toolkit`
  returns the package, so old imports keep working.
- **Workspace objects.** `configure_workspace(name, read_roots=...)` returns a
  `Workspace`; `run_agent(workspace=...)` runs in it and restores the previous
  one. The room passes each role its workspace instead of reconfiguring globals
  between turns. Importing babycoder no longer creates folders on disk.
- **Architect reads the coder's folder** as `@coder/...`, read-only. Its
  prompt says so; the room asks it to check finished work before deciding on
  follow-up. This closes the "known gap" its docstring used to describe.
- **Settings from the environment:** `BABYCODER_URL`, `BABYCODER_MODEL`,
  `BABYCODER_THINKING`, or `configure_model(...)`. `emotion_agent.py` reads the
  same settings; it keeps its own one-call request style on purpose.

## Model calls

- The agent loop now asks the model not to think, like `ask_model` already
  did. Before, a reasoning model spent Lisa's whole budget thinking, was cut
  off with nothing written, and retried at 4x: the invisible busy time.
- The thinking-off probe moves on when the server rejects a candidate with a
  400, instead of giving up and re-probing on every call.
- Every streaming response is closed, including retries and early exits.
- Progress marks while quiet: `.` thinking, `:` writing.
- An interrupted generation returns `INTERRUPTED` and is never acted on.
- The fake-call extractor only runs in native tool mode.
- Tool results are capped at `MAX_TOOL_RESULT_CHARS` in the conversation.

## Lisa

- **Conversation window:** the last four exchanges go into every reply and
  thought, chosen by Python, restored from memory on restart.
- **Plain speech:** tools through the grammar, then the reply as plain prose
  in a second call. `PLAIN_SPEECH = False` to compare on your model.
- **Dreams feed goals:** after a full night she is asked once whether a dream
  left her with a question. At most one new goal per sleep.
- She sleeps only when something happened since her last full sleep, and
  idles in 30 second steps with nothing on her mind.
- A reply you interrupt is dropped, not shown half-finished; your lines are
  answered together.
- GOT IT / MORE / DROP IT parsed per line (a colon in the conclusion used to
  hide the verdict).
- `consolidate` removed from her waking grant (invariant 3).
- Screen: an empty `You: ` prompt is wiped before she writes, progress dots
  are ended before the next line, and all notices go through the console.

## Fixes after the first real run

- Importing an existing memory.json crashed on start: the original toolkit
  gave memories UUID ids and the import expected numbers. It now numbers them
  1, 2, 3..., re-points old dreams at the new numbers and fills in missing
  goal fields. The check now uses exactly the original file shape.
- An unreachable model (LM Studio closed, no model loaded) crashed every
  agent with a RuntimeError. run_agent now returns a marker instead
  (model_unreachable() tells you), and Lisa says so once, keeps running and
  retries every 20 seconds. It does not count as a stall on her goal, and
  nothing is stored as if she had answered.
- She only sleeps with at least two memories to dream from.
- Thinking passes no longer get the recent conversation: a small model
  continued the transcript ("Them: ... Lisa: ...") or drifted off the goal.
  The prompt also says the pass is private, so she stops asking "you"
  questions in her own thoughts.
- A goal is settled after MAX_PASSES (4) passes without a verdict, instead of
  being thought about for ever.
- Arrow keys typed garbage into the line ("àK" on Windows, "[D" elsewhere).
  They are now ignored.
- A dream rests the memories the last two dreams used, while there is enough
  else, and is told not to retell what happened.

## Lisa's console

- Colour on Windows 11 and modern terminals: you cyan, tools yellow, sleep
  dark green, her words default, notices and progress marks grey. `NO_COLOR`
  turns it off; piped output stays plain.
- Every tool call is shown with its arguments and the first lines of its
  result; every merge during sleep shows the originals and what they became.
  `/tools all | outside | off` switches the detail while she runs.
- Escape codes in tool results and model output are stripped before display.

## Studying Lisa's sleep

- `sleep_log.md`: one section per night. Per dream: the memories it drew on,
  why each was picked, their tags, and the dream. Then merges, fading, and
  the question she woke with.
- `merged.md`: the originals of every merge and what replaced them.
- Goals record their origin: `conversation`, `thinking`, `dream`, `by hand`.
- Lookups mark the resulting memory `#reading` with the topic, never the
  page text.
- Dream sampling switch: `uniform` (default) or `residue` (`--residue`),
  weighted 60/25/15 toward since-last-sleep, five to nine days back, older.
- Dreams skip memories about her tools and the guide; a question about her
  own method is not kept as a goal.
- New console hook `on_result(name, args, result)` for anything that wants to
  see what a tool returned.

## Memory

- Plain markdown in `.agent_state_<name>/memory/`: `memories.md`, `goals.md`,
  `dreams.md`, `state.md`. Readable and editable in Notepad++, also while she
  runs: a changed file is reread before the next operation.
- Adding a memory appends to the file instead of rewriting a JSON file.
- Memory numbers are stable. Before, they were list positions that shifted
  after every merge.
- `memory_find` needs every word asked about, forgivingly stemmed ("herons"
  finds "heron").
- Past `MAX_MEMORIES` (2000), sleep lets the oldest conversation lines go;
  her own thinking is never thinned.
- An existing `memory.json` (or `memory.db`, if you ran the short-lived SQLite
  version) is imported on first open and renamed to `*.migrated`.
- `tidy_memories` only merges numbers it actually showed the model.

## Safety and smaller fixes

- `search_files` and `replace_in_project` skip a symlink pointing outside the
  sandbox. Before, search returned the contents of whatever it pointed at.
- `run_command` asks through `set_console(approve=...)` when registered.
- `todo_write` validates the schema and writes atomically.
- `read_guide` uses `<agent>_thinking_guide.md` when it exists.
- `designer_agent.py --todos` called `memory.memory_stats()`, which never existed.

## Your existing tests

Tests that reach into internals will need small updates:

- `agent_toolkit.LM_STUDIO_URL = ...` becomes `configure_model(url=...)`.
- `agent_toolkit._quiet_cache = {}` becomes `reset_model_probe({})`.
- `_ALLOWED_TOOLS`, `_active_tool_names` and `_tool_usage_hint` are gone; use
  `ToolBox(grant)`.
- `WORKDIR` and friends are `ws().workdir` etc.
- Memory numbers are ids, and `mem.memories` is a read-only list of dicts.

`DISPATCH`, `TOOLS`, `TOOL_BY_NAME`, `help_tool` and every tool function still
exist.
