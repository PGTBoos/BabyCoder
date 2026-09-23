> *AST-based (Abstract Syntax Tree) coding, with tool-based structured edits instead of raw diffs and context length limits.
> Sandboxed by default; a (light weight) Sandboxed Multi-Agent Coding Toolkit.     
> Python as a layer in between the agent and code access, also for automated feedback towards the agent.  
> Written for smaller tool-calling LLMs, improving to keep them on track while writing code.   
> Agents live in (light) folder-based isolation, though it's still your pc, keep that in mind.*  
>   
> **Status: proof of concept, work in progress.**   
> *It works, it's small, and there's an obvious next step that isn't built yet (see below).*  

## Why I built this
Small, local LLMs can write code, but they lean hard on their own content/memory skills to do it.  
And that's exactly where they're weakest.   
Ask one to hold a whole file in mind while also writing correct code, and they start forgetting what it already wrote.  
And they reintroduce bugs it just fixed, drift from the actual goal, or go into loops of thought.  
 
**The idea here:** stop asking it to remember the code at all. 
Give a tool-calling LLM a way to operate on code through a Python add/remove/update chain instead,  
so the file lives in Python, not in the model's limited context.   
That frees it from needing the whole file in memory, and it opens the door to working through a to-do list.  
One small task at a time instead of holding an entire plan in its head for the length of a long session.  
 
## What is it?
 
Instead of "here's a patch, apply it," the model calls tools, and Python code does the actual edit.
Not the LLM trying to do so out of its small llm memory. A few of them:
 
- `list("symbols", "file.py")` - see what's in a file before touching it
- `read_symbol("foo")` - pull just one function or class, not the whole file
- `update_symbol("foo", new_code)` - rewrite it; syntax is checked first, bad edits are rejected
- `find_references("foo")` - see every real usage before renaming or deleting it
- `check_syntax("file.py")` - confirm the file still parses
- `run_command("pytest")` - runs a shell command, but only after you type "y"
- *...etc a lot more of such functions*
In theory, it can rewrite one function in a huge codebase without the LLM ever needing the whole file in its context.

Right now there are two agents:

- **Coder** - given a task, reads a function and rewrites it correctly.  
  Runs end to end today: it takes an unfinished `shortest_path` function and implements it via the AST tools.   
- **Architect** - a restricted, read-mostly agent that plans instead of codes, and leaves a todo for a coder to pick up later.  
  It works, but it's not wired to the coder automatically yet - you run each by hand.  

## What's different about it

- Edits are **symbol-based, not diffs**. The tool understands "this is a function" and "is this still valid Python" - not just text matching.  
- Every agent gets its **own sandboxed folder**, automatically. Two agents literally cannot reach each other's files, because nothing  
  points a path-check at the other one's folder.  
- Any shell command the model wants to run **needs a human "y" first**.
- No AI framework underneath, just one Python file. Easy to read start to finish.

### Why this might be interesting

Most "AI coding agent" demos either give the model raw shell access or wrap it in a big framework. 
This sits in between: small enough to read in one sitting, but with real guardrails (sandboxing, syntax checks, approval-gated commands, backups) 
instead of "trust the model." to do all the coding.
The python part can as well give 'standard' text replies towards the coder (the symbol x was not found, did you mean perhaps ...) etc.
This is helpfull towards keeping smaller llm's on track without derailing of their goal.

### Not there yet

- The two agents don't talk to each other automatically - that's the next real milestone (not that hard from here).
- Only Python is actually supported (other languages are stubbed).
- No tests (my other project includes a test driven development work order, but not included here.
- I am curious of what others think about, at this point this project i still can make big design choises about it.
  Eventually it be some sort of auto coder, for cheaper to run models (12G Vram) or less.

  


### Try it

```bash
pip install requests
# needs a local LM Studio instance running, or point it at another model
python coder_agent.py

# later agent_room will be runner (also architect_agent can be run)
# the idea is the room will follow some python based loop, inbetween agents.
# agent toolkit contains all the commands and tricks, agents can get a set of allowed tools depending their role.
```

## Open question

Should the coder automatically pick up an architect's todos, or should
that be a separate "orchestrator" script that runs both? Not sure yet.
Opinions welcome. 

*I think to keep the coder working till execute, maybe add another command when it's ready then handover to architect for a next todo*
*So agents can do multiple run's, but the switch towards another agent should happen only when (after multiple turns) agent A is ready*
*some user intervention should though as well be possible*. 
***Agent room** will be the place that will bind it together, though coder agent can run on its own for now.*



