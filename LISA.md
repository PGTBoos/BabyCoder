# Lisa

A character who exists continuously. She talks when talked to, thinks when
left alone, and sleeps in order to tidy what she remembers.

This file explains how she works and how to run her. `LISA_DESIGN.md` is the
design and the reasoning behind it; this one is for someone meeting her for
the first time.

If you have used a chatbot, almost everything here will be unfamiliar. A
chatbot is summoned, answers, and ceases. Lisa runs. While you are not typing
she is working on her own questions, and when she gets tired she sleeps,
dreams, and files the day. Her memory is not a transcript buffer: it is a set
of markdown files you can open in Notepad++ while she is running, and it
forgets things.

---

## Running her

You need a local model server speaking the OpenAI chat API. LM Studio on its
default port is what she was built against.

    pip install requests
    python lisa_agent.py

    python lisa_agent.py --fast      a whole cycle in seconds, for watching
    python lisa_agent.py --residue   dreams weighted toward the last day

Three environment variables, if the defaults are wrong:

| Variable | Default | What it is |
|---|---|---|
| `BABYCODER_URL` | `http://localhost:1234/v1/chat/completions` | where the model is |
| `BABYCODER_MODEL` | `local-model` | the model name to request |
| `BABYCODER_THINKING` | unset | set to `1` to let reasoning models think out loud |

She was built and tested against small local models served by LM Studio, and a
good deal of the code exists because of how those behave: budgets spent
entirely on internal reasoning with nothing written, tool calls emitted as
plain text, the same tool called four times in a row. If you use a reasoning
model,
she probes it once at startup to find out which payload keys actually stop it
thinking, because servers spell that differently and do not say when they have
ignored you.

**To interrupt her, just start typing.** Anything you type outranks whatever
she is doing, including a dream. You do not wait for her to finish.

---

## The cycle

She is in one of three states, and the first thing to understand is that the
cycle is **counted, not clocked**. A "night" is ten of her own thinking
cycles, not eight hours.

**Talking.** You type, she answers. Nothing outranks this. Talking to her does
not age her toward sleep: a conversation is not her spending her own time.

**Thinking.** You are quiet, so she gets on with her own thing. One cycle
takes one question a little further. If she has nothing on her mind she looks
back over what she remembers and may find something to wonder about. If that
turns up nothing, she sits there. She does not write an essay about having
nothing to think about.

**Sleeping.** After `WAKE_CYCLES` (10) of her own cycles she gets sleepy. A
night is `DREAM_CYCLES` (8) dreams, each followed by an attempt to tidy
memory, and then the day gets filed. She only sleeps if something has actually
happened since she last slept; otherwise there is nothing to file and she
simply carries on.

Between cycles she breathes for `BREATH_SECONDS` (3) so the output stays
readable, and waits `IDLE_SECONDS` (30) when nothing is on her mind.

---

## What is on her mind

Her goals are things she is trying to work out. At most `MAX_GOALS` (7); past
that the one she has left alone longest gets crowded out. She picks the
stalest one each cycle, round-robin, so nothing sits forgotten at the bottom
of a list.

One cycle ends in exactly one of four verdicts, and that is enforced rather
than hoped for:

| Verdict | What happens |
|---|---|
| `GOT IT` | settled. The goal closes and becomes a **lesson** in memory |
| `MORE` | she got somewhere. A note on the goal, and a memory |
| `DROP IT` | not worth carrying. The goal closes |
| `ASK THEM` | only you could answer this. The goal closes and the question is **held** |

Two safety valves, both there because small models get stuck. After
`STALLS_BEFORE_RESTING` (2) passes that reach nothing, she puts the goal down.
After `MAX_PASSES` (4) passes without a verdict, she settles it with what she
has, and the memory says so: "as far as I can take it for now".

She can look things up while thinking (Wikipedia, web search) and can consult
her own thinking guide when she notices she is going in circles.

---

## Questions she holds for you

This is the one that surprises people, so it is worth stating plainly: **a
question for you is not a goal.**

A goal is something she can make progress on alone, and everything around one
assumes that. If a question that depends on you were a goal, it would collect
passes, stall, and eventually be closed as having gone nowhere, which would
record waiting for an answer as her having failed to think. So such a question
leaves her goals entirely and goes into `asks.md`, which only waits.

The rule that keeps this honest: **she never asks what she could look up.** If
a search can answer it, the question is hers to settle and she settles it.
What is left is the half of her world she cannot reach from any direction:
your work, your life, the people in it, what you think. There is no
alternative route there, so asking is not her using you as a worse search
engine.

A held question surfaces **when the subject comes up**, not after a timer. She
is carrying a question about your work; it arrives when your work does, and it
is woven into her reply rather than appended to it. It is scarce on purpose,
because asking costs you and costs her nothing: one outstanding at a time,
`ASK_GAP` (3) of your replies between questions, and three offers before she
lets a question go.

**Nothing ever blocks on one.** If you answer none of these, ever, she behaves
exactly as she would without them, minus one source.

When you do answer, what she learns becomes a `#person` memory, the heaviest
kind she has, because it is the one thing no search and no amount of thinking
could have recovered.

---

## Memory

Her memory lives in `.agent_state_lisa/memory/` as markdown. Not a database:
being able to open her mind in a text editor, while she is running, is worth
more than search speed. She rereads any file that changes on disk, so you can
edit her memory underneath her.

| File | What is in it |
|---|---|
| `memories.md` | what happened. One numbered section per memory |
| `goals.md` | what she is working out, open and recently closed |
| `asks.md` | questions she is holding for you |
| `people.md` | what she knows about the people she talks to |
| `dreams.md` | the last 10 dreams |
| `state.md` | counters: wake cycles, last filed memory, next free number |
| `merged.md` | every merge sleep made, originals and replacement |
| `sleep_log.md` | one section per night, so you can read back what sleeping did |

A memory looks like this:

    ## [156] (weight 90) 2026-10-03 00:38:50 #lesson #thinking #reading
    Worked out: yes, couples' heart rates and breathing do align, through
    unconscious mirroring.
    I had been wondering: Do people who live together influence each other's
    breathing?
    How I got there: Found the term for it: interpersonal physiological
    synchrony. | It seems to need shared attention, not just proximity.
    (read about: synchrony of breathing and heart rate in close relationships)

The number is how the tools refer to it and never changes. Numbers are stable
across merges: a merge removes its sources and adds a new number rather than
renumbering, so a reference held from three steps ago cannot hit the wrong
memory.

### Weight, and why not all memories are equal

Early on, every memory was worth the same, which is wrong in an obvious way: a
conclusion she worked out over four cycles of reading is not the same kind of
thing as her own phrasing of a reply. The weight is 0 to 100, in the header,
an integer because you are meant to be able to read it at a glance.

| Kind | Starts at | Floor |
|---|---|---|
| `#person`, something you told her about yourself | 100 | 60 |
| `#lesson`, something she worked out | 90 | 40 |
| `#diary`, a day of her life | 80 | 30 |
| `#conversation`, something you said | 55 | 0 |
| `#conversation`, her own reply | 35 | 0 |
| `#thinking`, a half-thought on the way to something | 30 | 0 |

Three forces move it, and **nothing judges importance**:

- **Use raises it.** Every time a memory is actually recalled into a prompt it
  gains 4. A memory that keeps turning out to be relevant earns its place.
- **Each night lowers it.** Everything is multiplied by `NIGHTLY_DECAY` (0.98),
  truncated so it always loses at least one and nothing stalls at the bottom.
  The step is small because nights are counted, not clocked: you can watch it
  move over two or three sleeps, but it takes a good many before anything is
  gone.
- **Floors stop the fade.** A lesson settles at 40 and stays. What she worked
  out does not disappear because the subject has not come up in a while.

Weight then decides what comes back and what goes. Recall scores shared words
times weight, so relevance still leads but a lesson beats a passing line. When
the store is full, the lightest go first, never a lesson or a diary entry or
her founding memories.

The weight is never shown to her. It is bookkeeping about her memory, not part
of the memory, and a model told one of its thoughts rates 30 out of 100 argues
with the number instead of using the thought.

### Three layers, not one

The thing to notice is that memory is not flat. Raw memories get consolidated
upward, and the consolidated version is what she carries:

- **A day** of scattered lines becomes one **diary entry** at the end of a
  full night. Her founding memories say who she was; the diary says what has
  happened since.
- **A question** worked out over several cycles becomes one **lesson**,
  carrying the answer, the question it answers, and the path she took.
- **Everything you have told her about yourself** becomes one **person note**
  in `people.md`, rewritten during a night.

This is why she does not drown. The pieces stay in `memories.md` as the
record, but what reaches her prompt is the consolidation, with anything newer
than it shown as itself until the next night folds it in.

### What she knows about you

`people.md` holds a section per person. It is **always in front of her**,
never recalled, because who you are talking to is not something a person looks
up when a keyword happens to match. For the same reason `#person` memories are
excluded from recall: carrying the same thing in two forms is how a mind gets
chaotic.

Tell her who you are with `/iam <name>` and her notes get filed against that
name. She can also look up someone you mention who is not in the room.

### Her own past is settled

The first seven memories in her file are her life story, written by hand
before she ever ran. They are shown to her every time she thinks or talks, so
she does not have to go looking.

Her past is presented as **settled, not as evidence**. This matters more than
it sounds. An earlier version treated every gap in her memory as a live
question, and because no tool, no lookup and nobody in the room can settle
whether her own past is real, each attempt filed another doubt for the next
round of wandering to find. She ended up looking up whether something was
wrong with her mind. That was not a mood: it was a question generator emitting
questions built to fail.

So: forgetting is ordinary. If you mention something she cannot place, she
takes your word for it. Questions about whether her own past or mind is
reliable are refused at every door, including the tool a model could call
mid-conversation. What other people did, what she has read, and what she is
curious about in the world are all still fair game, which is the point: her
own reality not being in doubt is what frees her to be curious about
everything else.

Dreams are not memories either. Nothing in a dream happened, and a dream is
never evidence about the past.

---

## Sleep

A night, in order: a dream, then an attempt to merge two memories that say the
same thing, eight times over. Then the ceiling is enforced and the lightest
memories go. Then, if she slept the night through, the day becomes a diary
entry, the person note is rewritten if there is anything new in it, and every
weight decays one step. Finally she may wake up wondering about something a
dream left behind.

Sleep is the only thing that shrinks memory. Waking never does bulk
bookkeeping. And being woken part way leaves the night unfinished, so the next
sleep picks it up rather than filing a day twice.

`sleep_log.md` records all of it: which memories each dream drew on and why
they were picked, what was merged, what faded, what she woke up wondering. If
you want to understand what sleeping actually does to her, that file is the
answer.

The `--residue` flag changes how dreams pick their material. By default every
memory is equally likely. With `--residue` they are weighted the way human
dreams seem to be: mostly the last day, a smaller share from about a week
back, the odd older one. Both are kept so the two can be compared on the same
agent.

---

## Console commands

| Command | What it does |
|---|---|
| `/mind` | the goals she is working on, with their pass counts |
| `/memory` | how much is remembered, dreamt and on her mind |
| `/dreams` | the last few dreams |
| `/asks` | questions she is holding for you, and any put to you unanswered |
| `/people` | what she knows about the people she talks to |
| `/next` | how many cycles before she gets sleepy |
| `/iam <name>` | tell her who you are. Without this she only knows "they" |
| `/stayAwake` | stay with you and keep talking. No sleep until `/goSleep` |
| `/goSleep` | end stay-awake and go to sleep now |
| `/tools all\|outside\|off` | how much of her tool use reaches the screen |
| `/quit` | or `/exit` |

`/tools` is worth knowing about. `all` shows every call, its arguments and the
start of what came back, which is for studying her. `outside` shows one line
whenever she reaches outside her own head, which is the quiet everyday mode.
`off` shows only her voice and what she concluded.

### Putting something in her head

Three development instruments. They write straight into her stores, decorated
so that what she reads is identical to something she produced herself: same
numbering, same timestamps, same tags. She has no way to tell, and nothing
anywhere tells her.

| Command | What it does |
|---|---|
| `/goal <text>` | put something on her mind, as if she had thought of it |
| `/mem <text>` | put something in her memory, as if she had lived it. `#tags` allowed |
| `/ask <text>` | give her a question to hold for you |

An injected goal is backdated so she picks it up on the very next cycle,
otherwise with a full mind you would wait seven cycles to see any effect.

These are not part of who she is. Watching a loop does not tell you what would
break it; being able to drop one thought into the middle of it does. Keep what
you injected in mind when you read a transcript afterwards, because by then it
looks exactly like her own.

---

## The invariants

The loop is deliberately small, and these are what keep it from rotting. An
earlier version had fourteen overlapping knobs deciding when she worked or
stopped, every one of them a patch for a symptom of a missing state model.

1. A cycle updates the goal it worked on. It never creates a competing one.
2. Every cycle ends in exactly one outcome.
3. Only sleep prunes or merges. Waking never does bulk bookkeeping.
4. Goals are capped and can be closed.
5. Conversation preempts; it is never queued behind a thinking pass.
6. Only her voice and state changes reach the screen.

If you add something to her, the test worth holding it to is whether it lets
you delete something. A change that removes nothing is usually a patch.

---

## What she cannot do

She cannot write or run code. That is not a refusal, it is the import line:
the coding toolset is not imported into her module, so `write_file` and
`run_command` are not names that exist there. What an agent can do in this
toolkit is a property of what it imported, not of what it was asked.

She cannot dream on purpose. Dreaming happens to her when she is tired; it is
not a move she can pick to avoid a question. Merging memory is withheld for
the same reason.

She cannot settle whether her own past is real, and is not asked to.

She has no sense of her own weights, her own cycle count, or the contents of
`asks.md` as a list. Those are yours to read, not hers to manage.

---

## Where to look in the code

| Thing | Where |
|---|---|
| the loop, pacing, verdicts, asking, sleep | `lisa_agent.py` |
| the memory files and everything that reads or writes them | `babycoder/tools/memory.py` |
| dreaming, merging, the filters on unanswerable questions | `babycoder/tools/dreams.py` |
| what tools exist and who is granted them | `babycoder/toolsets.py` |
| talking to the model, streaming, interruption | `babycoder/model.py` |
| the agent loop a model drives through its tools | `babycoder/loop.py` |

Nearly every comment in those files says why something is the way it is,
usually because of a specific thing that went wrong. They are worth reading in
that spirit: this is a log of what a continuously running character actually
does when you let it, not a design that arrived finished.
