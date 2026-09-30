# Lisa

[Back to the overview](../README.md)

Lisa is an agent that exists continuously. She is not a chatbot waiting for a
prompt: she has long-term memory, a few things on her mind, a sense of the
time of day, and a rhythm of waking and sleeping. She runs on a small local
model, and everything she is lives in a folder of plain markdown files.

The mental model is a person sitting in a room with you:

- If you speak, she listens. Nothing outranks that.
- If you are quiet, she gets on with her own thing.
- If you speak while she is thinking, she stops and answers you.
- If she has nothing on her mind, she does nothing. No busywork.
- She gets tired, and sleep is when the day gets filed.

## Why dreams

Most agents that "dream" use it for housekeeping: compress memory, remove
duplicates, fix mistakes. Lisa's sleep does that too, but the part I am
interested in is the other thing dreams do for people: they put unrelated
things side by side, and sometimes that leaves you with a question.

Here is a real example, from her first days. I mentioned in passing that I
am Dutch, that we call autumn "Herfst", and asked what the difference is
between autumn and fall. That night she dreamt:

> It was about these colors, really. Lots of browns and reds. [...] I saw
> this map with names for seasons in Dutch, Winter and Lente and Zomer, and
> then Herfst. [...] Someone kept asking about words, how English had both
> autumn and fall and why the difference mattered. I remember looking at two
> kinds of leaves, one in yellow-gold and one burnt orange, and trying to
> place them on a calendar that seemed to keep spinning backward toward the
> start of things.

She woke up with:

    ~ woke up wondering: If there are multiple ways to say something similar,
      does the difference between those words actually change what the thing
      means?

And the next day, working on that question on her own, she went and found
the actual scholarly name for it:

    > search_wikipedia(query='Sapir-Whorf hypothesis and social constructivism')
      ## Benjamin Lee Whorf
      Benjamin Atwood Lee Whorf was an American linguist [...] best known for
      his advocacy of linguistic relativity.

A throwaway remark, turned into a real line of inquiry, through a dream. That
chain is what this experiment is about.

## How she works

The full design is in [LISA_DESIGN.md](../LISA_DESIGN.md). In short:

**Ten wake cycles, then eight dream cycles. Counted, not clocked.** One
counter, no timers. Talking to her does not make her sleepy; a conversation
is not her spending her own time.

**A wake cycle** takes the goal she has left alone longest, thinks about it
once (looking things up if that helps), and ends in exactly one of: learned
something, concluded it, dropped it, or got nowhere. A goal cannot be circled
for ever: after a few passes she settles it with what she has.

**A dream cycle** picks a few memories, tells a short dream about them, then
looks for memories that say the same thing and merges them. Story first: the
dream puts things side by side, and that is what makes a merge visible. After
a full night she is asked once whether a dream left her wondering about
something, and at most one new goal comes from it.

**Her goals** are few (at most seven) and can be closed. An earlier version
could only add questions and reached 312 open ones.

**The rules that keep her from rotting:**

1. A cycle updates the goal it worked on. It never creates a competing one.
2. Every cycle ends in exactly one outcome. Never "nothing happened, try again".
3. Only sleep prunes or merges memory.
4. Goals are capped and closable.
5. Conversation always comes first.
6. Only her voice and what changed reach the screen (unless you ask to see more).

**What she can and cannot do.** She can remember, look things up on
Wikipedia and the web, and read her thinking guide. She cannot touch files or
run anything: her code does not even import those tools. Her thinking guide
is a method for working on goals, never a topic: she does not dream about it
and does not take questions about it as goals.

**What she reads online** never becomes part of her memory as-is. A memory
made while she looked something up is tagged `#reading` and names the topic,
but the page text stays out. Web pages are long, impersonal and sometimes
written to steer a model, and none of that should become who she is.

## Running her

    python lisa_agent.py            talk to her
    python lisa_agent.py --fast     a whole cycle in seconds, for watching
    python lisa_agent.py --residue  dreams weighted toward the last day

Just type. These work at the prompt:

| Command | Does |
|---|---|
| `/mind` | what is on her mind |
| `/memory` | how much she remembers |
| `/dreams` | her last few dreams |
| `/next` | cycles left before she gets sleepy |
| `/tools all`, `/tools outside`, `/tools off` | how much of her tool use you see |
| `/quit` | say goodbye |

In a colour console (Windows 11, any modern terminal) what you type is cyan,
her tool use is yellow, sleep and dreams are dark green, and her own words are
in your normal colour. Grey dots and colons are progress: `.` means the model
is thinking, `:` means it is writing. `NO_COLOR=1` turns colours off.

With `/tools all` you see every tool call she makes, with what she asked and
the start of what came back, and every merge during sleep. For example:

    > memory_find(about='herfst')
      [14] 2026-09-30 12:01  I said: Herfst is autumn, I think...
    > merged, during sleep:
      [3] They said: the ferry to Texel was late again
      [7] They said: the ferry is late every Friday
      became [12] The Texel ferry is often late, especially on Fridays

She never prints over a line you are typing: she waits until you press Enter.
If the model server is down she says so once, keeps running, and tries again.

## Her mind is a folder

Everything she is lives in `.agent_state_lisa/memory/`, as markdown:

| File | Holds |
|---|---|
| `memories.md` | Everything she remembers, one `## [number] date #tags` section each. |
| `goals.md` | What is on her mind, where each goal came from, and her notes on it. |
| `dreams.md` | Her last ten dreams, and which memories each one came from. |
| `state.md` | Her counters. |
| `merged.md` | Every merge sleep made: the originals and what replaced them. |
| `sleep_log.md` | One section per night. |

Open them in Notepad++ or anything else. You can also edit them while she
runs: she rereads a file when it changes. Add a memory by hand and she can
find it; add an `## open:` line to `goals.md` and she will think about it.

## Studying her sleep

Lisa is built to be observed, not just watched. A night in `sleep_log.md`
looks like this (a shortened example):

    ## Night of 2026-09-29 22:48 - 23:10
    - sampling: residue
    - memories: 42, new since last full sleep: 11
    - dream 1 from [31] recent #conversation, [38] recent #thinking #reading, [12] week #conversation
      > A heron on the ferry deck, not moving, and nobody minded that we were late.
    - merged 2 memories into [43] (41 now)
    - woke up wondering: why would the ferry always be late on Fridays?

What you can look at:

- **Where dreams come from.** Each dream lists the memories it drew on, their
  tags (`#conversation`, `#thinking`, `#reading`) and why each was picked.
- **Whether sleep keeps the meaning.** `merged.md` keeps the originals of
  every merge, so you can judge whether a merge summarised, lost, or bent
  something.
- **Where goals come from.** Every goal says `conversation`, `thinking`,
  `dream` or `by hand`. Do goals from dreams go anywhere different from goals
  from talking?
- **How dreams pick their material.** By default every memory is equally
  likely. With `--residue`, most come from since the last sleep, some from
  about a week back, and a few from long ago. Human dreams seem to draw on
  the day before and again on the week before; the mode is recorded every
  night, so runs with and without it can be compared.

## What I am seeing so far

Early days, but a few things stand out:

- **The dream to question to research chain works**, as in the example
  above, even on small models.
- **Abstraction drift.** Left alone, her thinking tends to climb into ever
  more abstract, self-made vocabulary ("the principle of linguistic
  necessity"), which then finds nothing when she searches for it. The goal
  origins in the logs should show whether this grows.
- **Dreams fill gaps with the most common story shape.** With few memories,
  small models turn dreams into relationship drama by default: who lives
  where, who is leaving whom. Where exactly the confabulation goes is one of
  the more interesting things to watch.
- **Many small models keep thinking** even when asked not to. The toolkit
  tries several ways to switch that off per model and stops a model that
  thinks for too long without writing anything.

## Related work

Others are exploring nearby ideas, mostly with memory maintenance as the goal:

- [Generative Agents](https://github.com/joonspk-research/generative_agents)
  (Stanford, 2023): the memory stream and periodic reflection that most of
  this field builds on.
- [Letta](https://github.com/letta-ai/letta) (formerly MemGPT): sleep-time
  agents that reorganise memory while the main agent is idle.
- [anima](https://github.com/huodebing-alt/anima): an always-on agent on a
  small local model with a multi-phase sleep, very close in spirit.

Lisa's angle is using dreams as a source of curiosity rather than only as
maintenance, and making her mind something you can read and edit by hand.

## Open questions

- Does dream-lag sampling (`--residue`) make dreams and the questions they
  leave more grounded, or just different?
- Should dreams keep her out of the user's personal life, or is where they
  wander exactly what is worth studying?
- How much should a small model think on its own before it needs something
  concrete (a memory, a search that found something) to hold on to?

If you run her, I would love to see your sleep logs.
