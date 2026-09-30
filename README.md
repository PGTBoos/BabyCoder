![BabyCoder in the console](../main/babycoder.png)  
( *this is though all console-based* )

# BabyCoder

A small Python toolbox for running agents on small, local LLMs.
And two very different agents built with it.

The idea behind all of it: small models are good at judgment in the moment
and bad at holding things in mind. So Python holds the structure (the files,
the memory, the loop, the rules) and the model only makes the next decision.
Python also talks back to the model: when a tool call is wrong, it says what
went wrong and what was probably meant, which keeps a small model on track
instead of derailing.

Written for models you can run at home (about 12 GB of VRAM or less), through
LM Studio or any OpenAI-compatible server. No agent framework underneath:
plain Python, readable start to finish.

Status: experimental, but both sides run end to end.

## The two sides of this project

- ### [Lisa - an agent that lives on](../main/docs/Lisa.md)

  Lisa runs continuously. Talk to her, and she answers; leave her alone, and she
  thinks about what is on her mind, looks things up, and after a while she gets
  tired and sleeps. While she sleeps, she dreams about her memories, merges the
  ones that say the same thing, and sometimes wakes up wondering about
  something new. Her whole mind is a folder of markdown files you can open in
  any text editor. You can see her looking up Wikipedia (tool use is yellow)
  Or see her dreaming in green, or see her thinking throughout her days.
  She has a novel idea of dreaming; it isn't mainly cleanup; it drives her thoughts.

  This is where most of my interest is these days. [Read more about Lisa.](../main/docs/Lisa.md)  
  It's is more akin to a research project for me.  

- ### [The coder - AST-based coding for small models](../main/docs/Coder.md)

  Where this project started. Instead of asking a small model to hold a whole
  file in its head and write a diff, it edits code by symbol through Python
  tools: read one function, rewrite it, and Python checks the syntax before
  anything is saved. An architect agent plans work as todos, a coder picks them
  up, and a room runs the two in turn. Everything is folder based sandboxed per agent.
  (though its not a virtual environment, just basic safety here).
  While an agent can take multiple turns to solve something.
  More complex than the Lisa agent, though more targeted towards work.  

  The coder is the most advanced agent for more info: [Read more about the coder.](../main/docs/Coder.md)

-----

## Quick start

    pip install requests
    pip install ddgs              (optional: web search for Lisa)

Start LM Studio with a model loaded, then:

    python lisa_agent.py          talk to Lisa
    python coder_agent.py         give the coder something to build

Pointing it at another server or model needs no code changes:

    set BABYCODER_URL=http://localhost:1234/v1/chat/completions
    set BABYCODER_MODEL=your-model-name

(`export` instead of `set` on Linux and macOS.) It works with any
tool-calling model; reasoning models are handled too (see the coder page).

## What is in the repo

| File | What it is |
|---|---|
| `babycoder/` | The toolkit: every tool, the agent loop, sandboxing, memory, the model layer. |
| `lisa_agent.py` | Lisa. |
| `coder_agent.py`, `architect_agent.py`, `agent_room.py` | The coding agents, and the room that runs them together. |
| `designer_agent.py` | A small example: an agent made of nothing but a role and a set of granted tools. |
| `emotion_agent.py` | An older experiment: a character whose emotions are kept in a tool schema. |
| `LISA_DESIGN.md` | The design Lisa is written against. |
| `tests/` | Offline checks, with a stand-in for the model server. No LM Studio needed. |

Run the checks with:

    python tests/check_babycoder.py

## Feedback welcome

This is a personal research project and I can still make big design choices.
If you are working on something similar, persistent agents, memory that
consolidates while idle, or coding with small models, I would like to hear
from you. Open an issue or a discussion.

MIT licensed.
