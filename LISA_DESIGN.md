# Agent Lisa - design

The spec `lisa_agent.py` is written against. If the code and this document
disagree, the document is right and the code is a bug.

## Identity

The system prompt is a **transcript frame**, not a role command: *"this is a
transcript of Lisa talking, 25, curious, in her own words."* Small local
models hold that better than "you are Lisa", which they drift out of within a
few turns and answer as "I am Qwen, I cannot feel".

It was a *story* frame first, and that overshot. A model told to write a story
writes one: quoted dialogue, trees turning into pillars of smoke, a closing
image. The frame is only there to stop the model introducing itself as a
model. So it now says outright that there is no narrator and nothing is being
written - a recording, not a story. Same protection, none of the literature.

Context stays deliberately small - the system prompt, the last few exchanges
of the conversation she is in (chosen by Python, not by the model remembering
to look), and the current one. **Memory does the long-term remembering, not
the context window.** That is what makes her runnable on an 8B model
indefinitely.

She speaks in plain prose. Choosing tools goes through the JSON grammar, and
then the reply itself is a separate plain call: a voice written inside
`"final_answer": "..."` comes out stiffer.

## What she has

- **Memories** - what happened. They grow, and only sleep thins them: by
  merging, and past a ceiling by letting the oldest conversation lines go.
- **Dreams** - their own FIFO of the last 10. Not mixed into memories: a dream
  woven from dreams is an echo chamber, and dreams that accumulate as memories
  crowd out lived material.
- **Goals** - what is on her mind. A question, or something unfinished. Capped
  at 7, and they can be **closed**. She is a person with interests, not a
  ticket system.
- **The thinking guide** - a *method*, never a subject. She must not research
  it as a topic, dream about it, or take a question about it as a goal.

## Tools

Memory: `memory_count`, `memory_at`, `memory_recent`, `memory_oldest`,
`memory_random`, `memory_find`, `remember`, `consolidate`,
`goals_list`, `goal_add`, `goal_close`, `dreams_recent`.

Plus internet lookup and the guide.

Several small retrieval tools rather than one clever search, on purpose. A
single "recall the most relevant thing" returns the same handful of entries
every time, so she circles whatever she said most recently. **Oldest and
random access are what give her range.**

## The cycle

**Ten wake cycles, then eight dream cycles. Counted, not clocked.** This one
counter replaces every timer.

**Wake cycle** - take the goal she has left alone longest, think about it once,
look something up if it helps, and record what came of it.

**Dream cycle** - pick random memories, write a short dream about them, *then*
look at whether same-subject memories can be merged into one. Story first: the
fuzziness of having just put unrelated things side by side is what makes the
merge visible. Tidying first would only be deduplication.

**Waking up** - after a full night she is asked once whether a dream left her
wondering about something. At most one new goal per sleep. This is how
sleeping feeds waking: without it, goals only appear when the model happens
to add one mid-conversation, and she mostly idles.

She only sleeps when something happened since she last slept. With nothing
new, there is nothing to file.

**What dreams draw on** - lived material only: conversation, her own
thinking, and what she read (her takeaway, never the page). Never memories
about her tools or the guide: the guide is a method, and dreaming about it
turned the method into a topic. Which memories get picked is a switch:
uniformly, or weighted toward the last day and the week before, the way
human dreams seem to be.

## Observing her

Every night is written to `sleep_log.md`, every merge's originals to
`merged.md`, and every goal records its origin. These are for the person
studying her. She never reads them back.

**You preempt anything, at any point**, dreaming included. Talking to her does
not count toward the ten - a conversation is not her spending her own cycles.

## The invariants

These are the rules that keep it from rotting. Every historical bug in this
agent was one of them being broken.

1. **A cycle updates the goal it worked on. It never creates a competing one.**
   Every multiplying bug came from breaking this.
2. Every cycle ends in exactly one of: *learned*, *concluded*, *dropped*,
   *stalled*. Never "nothing happened, try again identically".
3. **Only sleep prunes or merges.** Waking never does bulk bookkeeping.
4. Goals are capped and closable. Nothing may accumulate without bound.
5. Conversation preempts; it is never queued behind a thinking pass.
6. Only her voice and state changes reach the screen. Prompts and tool calls
   are machinery - unless you ask to see them: `/tools all` shows every tool
   call and merge, in its own colour, for studying how she works.

## What is deliberately not here

- No priority scoring on goals. A score that does not change when she thinks
  about something makes her pick the same goal for ever. Round-robin on
  "least recently touched" instead.
- No time-of-day gating. The hour colours how she *talks* and nothing else.
- No separate "unfinished thread" entity. A stalled goal is a goal with a
  stall count, not a second object competing with itself.
