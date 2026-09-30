# The coder

[Back to the overview](../README.md)

AST-based (Abstract Syntax Tree) coding with tool-based, structured edits
instead of raw diffs, for small tool-calling LLMs. Python sits between the
agent and the code: it does the actual edits, checks them, and gives the
model feedback when something goes wrong. Every agent works in its own
sandboxed folder; that is folder-based isolation, though, not a real sandbox.
It is still your PC, so keep that in mind.

## Why I built this

Small, local LLMs can write code, but they lean hard on their own memory to
do it, and that is exactly where they are weakest. Ask one to hold a whole
file in mind while also writing correct code, and it starts forgetting what
it already wrote. It reintroduces bugs it just fixed, drifts from the actual
goal, or goes into loops.

The idea here: stop asking it to remember the code at all. Give the model a
way to operate on code through Python instead, so the file lives in Python,
not in the model's limited context. That frees it from needing the whole
file in memory, and it opens the door to working through a to-do list: one
small task at a time, instead of holding an entire plan in its head for a
long session.

## What it is

Instead of "here is a patch, apply it", the model calls tools, and Python
does the actual edit. A few of them:

| Tool | Does |
|---|---|
| `list("symbols", "file.py")` | see what is in a file before touching it |
| `read_symbol("file.py", "foo")` | pull just one function or class, not the whole file |
| `update_symbol("file.py", "foo", new_code)` | rewrite it; the syntax is checked first and a bad edit is rejected |
| `find_references("file.py", "foo")` | every real usage, before renaming or deleting |
| `rename_symbol(...)` | rename everywhere in a file, leaving strings and comments alone |
| `check_syntax("file.py")` | confirm the file still parses |
| `run_command("pytest")` | runs a shell command, but only after you type "y" |

Twenty coding tools in all. In theory it can rewrite one function in a huge
codebase without the model ever seeing the whole file.

Python also talks back. Asking for a symbol that is not there gets
"NOTFOUND: symbol 'shortst_path' was not found. Close matches that do exist:
shortest_path. This might be a typo for one of those.", not a crash.
After an edit, the result includes linter-style notes: a function defined
twice, a loop with no stated invariant, a function without a docstring. And a
model that declares itself done after testing branching code with only one
input is asked to try another first. This is what keeps a small model on
track instead of derailing.

## Three ways to run it

Every coding script takes work the same way:

    python coder_agent.py                       asks what to build
    python coder_agent.py "add a retry loop"    task on the command line
    python coder_agent.py --file spec.md        task from a file
    python coder_agent.py --demo                built-in demo: implement a BFS shortest_path

**The coder alone.** Tell it what to build, and it builds it.

    python coder_agent.py "add a slugify helper to text_utils.py"

**The architect, then the coder.** The architect turns one instruction into
todos on a shared board; the coder picks them up later, in a separate run.

    python architect_agent.py --file spec.md
    python coder_agent.py --todos

**The room.** Both on one task, taking turns, in one process.

    python agent_room.py "port the parser to streaming"

Who goes next is decided by plain Python reading the todo board, not by
asking a third model: nothing planned means the architect goes first, coder
work pending means the coder's turn, and when the coder is done the architect
reviews the result and adds follow-up or says it is finished. The room stops
when nothing is pending, when two rounds in a row change nothing, or after a
maximum number of rounds, and you can add a note or stop it before each turn.

## Sandboxing

- **Every agent has its own folder.** The coder works in `sandbox/`, the
  architect in `sandbox_architect/`. Every path a tool gets is resolved and
  refused if it points outside the agent's folder, including through `..` or
  a symlink.
- **The architect can read the coder's work, not change it.** It sees the
  coder's folder as `@coder/...`, read-only. The coder cannot see the
  architect's folder at all.
- **The todo board is the only shared thing** (`shared_todo.json`), and a
  malformed board is refused with the exact problem, so the model can fix it.
- **Every edit is backed up first**, and `restore_backup` puts a file back.
- **Shell commands need your "y"**every time. No terminal, no approval.

## How capability works

This is the part of the toolkit every agent shares, Lisa included.

**A tool is one declaration.** Its help and arguments are written once, next
to the function:

```python
@tool("Read the full source of one named function, class, or method.",
      path=P("string"), name=P("string"), scope=P("string", optional=True))
def read_symbol(path, name, scope=None):
    ...
```

The schema the model sees, the line in its system prompt, the answer to
`help(tool_name="read_symbol")` and the hint after a wrong call are all
generated from that.

**Tools come in toolsets** (`coding`, `reading`, `planner`, `research`,
`guidance`, `memory`, `dreams`), one module each in `babycoder/tools/`.

**An agent gets a grant**, a union of toolsets:

```python
AGENT_CODER     = coding | planner | guidance
AGENT_ARCHITECT = reading | planner | guidance
```

The model only ever sees its grant. Its help lists only those tools, its
system prompt describes only those, and the output grammar only allows those
names, so a tool outside the grant is not just refused; it cannot even be
named. Each toolset also brings a short usage note into the system prompt,
so an agent file only has to describe its role. `designer_agent.py` shows
this: it contains no tools at all, only a role and a grant.

## Small models and reasoning models

- **Thinking is switched off where possible.** Every server spells it
  differently and some silently ignore it, so the toolkit tries a few ways
  against the live model and keeps the one that works.
- **A model that thinks without end is stopped.** Past a limit with nothing
  written, the connection is closed, which also stops the generation on the
  server.
- **Structured output.** In the default mode the model answers in a small
  JSON envelope (notes, tool calls, final answer), enforced by a grammar, which
  small models follow far more reliably than free-form tool calling.
- **Long results are cut** in the conversation, with a pointer to the tool
  for reading one symbol, so one big file cannot push the task out of an 8k
  context.
- **A server that is down** is reported, not a crash.

## Not there yet

- Only Python is really supported. C# and TypeScript have a place in the code
  but no implementation yet.
- The architect reviews by reading; it cannot run the coder's tests itself.
- `run_command` needs you at the keyboard, so the room is not unattended.

## Open questions

- How much should the architect plan up front, versus letting the coder
  discover the work?
- Should the room hand over after every todo, or let the coder run until it
  says it is stuck?

Opinions welcome: open an issue or a discussion.
