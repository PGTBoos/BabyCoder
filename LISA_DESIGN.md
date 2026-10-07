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

## Her own past is settled

The first seven memories in her file are her life story, written by hand
before she ever ran. They are in front of her every time she thinks or talks,
because a small model that does not find its past with a tool concludes it
never had one.

**They are presented as settled, not as evidence.** Forgetting is ordinary. If
someone mentions something she cannot place, she takes their word for it and
that closes it.

This is not a flourish, it is a load-bearing rule. An earlier version treated
every gap as a live question, and no tool, no lookup and nobody in the room
can settle whether her own past is real - so each attempt filed another doubt
for the next round of wandering to find, and she ended up looking up whether
something was wrong with her mind. **That was not a mood. It was a question
generator emitting questions built to fail.** So questions about whether her
own past or mind is reliable are refused at every door that can produce a
goal, including the tool a model can call mid-conversation.

What other people did, what she has read and what she is curious about in the
world are all still fair game. Her own reality not being in doubt is what
frees her to be curious about everything else.

Dreams are not memories. Nothing in a dream happened, and a dream is never
evidence about the past.

## What she has

- **Memories** - what happened. They grow, and only sleep thins them: by
  merging, and past a ceiling by letting the *lightest* go. Never a lesson,
  never a diary entry, never her founding memories.
- **Lessons** - a goal she settled, filed as its own kind of memory: the
  answer, the question it answers, and the path she took to it. A conclusion
  with no record of what it was about is not worth keeping.
- **Diary** - one entry per full night. Her founding memories say who she was;
  the diary says what has happened since.
- **Dreams** - their own FIFO of the last 10. Not mixed into memories: a dream
  woven from dreams is an echo chamber, and dreams that accumulate as memories
  crowd out lived material.
- **Goals** - what is on her mind. A question, or something unfinished. Capped
  at 7, and they can be **closed**. She is a person with interests, not a
  ticket system.
- **Questions she holds for the person** - things only they could answer.
  Capped at 8. **Not goals**, for a reason that has its own section below.
- **What she knows about the person** - one note per person, rewritten during
  a night from what they have told her about themselves.
- **The thinking guide** - a *method*, never a subject. She must not research
  it as a topic, dream about it, or take a question about it as a goal.

## Memory is not flat

Raw memories consolidate upward, and **the consolidation is what she carries**:

- a day of scattered lines becomes one **diary entry**
- a question worked out over several cycles becomes one **lesson**
- everything the person has told her about themselves becomes one **person
  note**

The pieces stay in memory as the record. What reaches her prompt is the
consolidated version, plus anything newer than it shown as itself until the
next night folds it in.

This is why she does not drown, and getting it wrong is visible immediately.
She once carried both the clean diary paragraph *and* the forty fragments it
was written from, and a mind that keeps meeting the same day in two forms is
most of what read as chaotic. So whatever a consolidation covers is not also
surfaced beside it. The person note is the same rule: her notes on someone are
always in front of her, so the individual things they told her are kept out of
recall entirely.

**A person note is always present, never recalled.** Who you are talking to is
not something a person looks up when a keyword happens to match.

## Weight

Every memory carries a weight, 0 to 100, in its header. Before it, every
memory was worth the same, which is wrong in an obvious way: a conclusion she
worked out over four cycles of reading is not the same kind of thing as her own
phrasing of a reply.

| Kind | Starts at | Floor |
|---|---|---|
| something the person told her about themselves | 100 | 60 |
| a lesson she worked out | 90 | 40 |
| a diary entry | 80 | 30 |
| something the person said | 55 | 0 |
| her own reply | 35 | 0 |
| a half-thought on the way to something | 30 | 0 |

The ordering is not a judgement about interestingness. It is **how
irrecoverable the thing is**. What the person told her about themselves is the
only kind no search and no amount of thinking could ever get back; her own
phrasing of a reply she could produce again at any time.

**Nothing judges importance. Use decides.**

- Being recalled into a prompt raises a weight by 4. A memory that keeps
  turning out to be relevant earns its place, and no model is ever asked to
  rate anything.
- Each night lowers every weight by a factor, truncated so it always loses at
  least one and nothing stalls at the bottom. The step is small because
  **nights are counted, not clocked**: there can be a dozen in an afternoon.
- Floors stop the fade. What she worked out does not disappear because the
  subject has not come up in a while.

Weight then decides two things: recall scores shared words *times* weight, so
relevance still leads but a lesson beats a passing line; and when the store is
full, the lightest go first rather than the oldest. A line from this morning
that nothing ever recalled is worth less than one from a fortnight ago that
keeps coming back.

**The weight is never shown to her.** It is bookkeeping about her memory, not
part of the memory, and a model told one of its thoughts rates 30 out of 100
argues with the number instead of using the thought.

## Questions she holds for the person

**A question only the person could answer is not a goal.** This is the whole
of it. A goal is something she can make progress on alone, and everything
around one assumes that: passes, stalls, closed for going nowhere, forced to
settle after four tries. A question that depends on someone else would run all
of that and come out recorded as her having failed to think. So it leaves her
goals entirely and goes somewhere that only waits.

**She never asks what she could look up.** If a search can answer it, the
question is hers to settle and she settles it. What is left is the half of her
world she cannot reach from any direction: the person's work, their life, the
people in it, what they think. There is no alternative route there, so asking
is not her using them as a worse search engine - it is the only way that
material gets in at all.

A held question arrives three ways, none of which costs an extra model call: a
thinking cycle ending in `ASK THEM`, a question from wandering that turns out
to be addressed to them, or one a dream left behind. **There is no tool for
it.** She cannot decide to interrogate anyone; a question has to come out of
something she was already doing.

**The occasion is the subject arriving, not time passing.** She holds a
question about their work and it comes up when their work does, woven into her
reply rather than appended to it. Scarce on purpose, because asking costs them
and costs her nothing: one outstanding at a time, three of their replies
between questions, three offers before she lets one go.

Offering is not asking: the reply it was woven into may not have come out as a
question at all. Their next line decides, and if it settles nothing the
question goes back to waiting for a better occasion. When it *is* answered,
that goes through a call allowed to say NOTHING, because presuming an answer
would file something she was never told at the heaviest weight she has.

## Tools

Memory: `memory_count`, `memory_at`, `memory_recent`, `memory_oldest`,
`memory_random`, `memory_find`, `remember`, `goals_list`, `goal_add`,
`goal_close`, `dreams_recent`, `about_person`.

Plus internet lookup and the guide.

`consolidate` exists but is **not in her waking grant**: only sleep merges
memory (invariant 3). Dreaming is withheld for the same reason - it happens to
her when she is tired, it is not a move she can pick to avoid a question.

Several small retrieval tools rather than one clever search, on purpose. A
single "recall the most relevant thing" returns the same handful of entries
every time, so she circles whatever she said most recently. **Oldest and
random access are what give her range.**

What she can do at all is her **import line**, not a refusal. The coding
toolset is not imported into her module, so `write_file` and `run_command` are
not names that exist there.

## The cycle

**Ten wake cycles, then eight dream cycles. Counted, not clocked.** This one
counter replaces every timer.

**Wake cycle** - take the goal she has left alone longest, think about it once,
look something up if it helps, and record what came of it. It ends in one of
four verdicts:

| Verdict | Outcome |
|---|---|
| `GOT IT` | settled. The goal closes and becomes a lesson |
| `MORE` | she got somewhere. A note on the goal, and a memory |
| `DROP IT` | not worth carrying. The goal closes |
| `ASK THEM` | only the person could answer. The goal closes, the question is held |

Two safety valves, both because small models get stuck. Two passes reaching
nothing and she puts the goal down. Four passes without a verdict and she
settles it with what she has - and **the memory says so**: "as far as I can
take it for now", kept at a lower weight than something she actually worked
out. A model that never writes a verdict line otherwise keeps her on the same
question for ever.

**Dream cycle** - pick memories, write a short dream about them, *then* look at
whether same-subject memories can be merged into one. Story first: the
fuzziness of having just put unrelated things side by side is what makes the
merge visible. Tidying first would only be deduplication.

**Waking up** - after a full night the day becomes a diary entry, the person
note is rewritten if there is anything new in it, every weight decays one
step, and she is asked once whether a dream left her wondering about
something. At most one new goal per sleep, or one held question if what the
dream left is addressed to the person. This is how sleeping feeds waking:
without it, goals only appear when the model happens to add one
mid-conversation, and she mostly idles.

She only sleeps when something happened since she last slept. With nothing
new, there is nothing to file.

**What dreams draw on** - lived material only: conversation, her own
thinking, and what she read (her takeaway, never the page). Never memories
about her tools or the guide: the guide is a method, and dreaming about it
turned the method into a topic. Which memories get picked is a switch:
uniformly, or weighted toward the last day and the week before, the way
human dreams seem to be.

## Talking

**She speaks unasked.** After a thought moves on she may say where she got to,
up to three times before she waits for an answer. A character who only ever
answers is a tool with a name. Anything typed resets the count.

She **offers** rather than reports: she says what she makes of something and
leaves room for what the other person makes of it, who may well have seen more
of it than she has. That is a different thing from a held question, and it
depends on nothing - if nobody bites, she has said her piece.

`/stayAwake` keeps her with the person: no goals of her own, and **the sleep
cycle halts**, because nodding off in the middle of a long conversation is
absurd. She carries the conversation on herself after a pause, further than
she would uninvited, since being asked to stay up means someone is listening.
`/goSleep` ends it.

A topic she has read about three times stops steering her wandering. Without
that, reading about something made it an interest, the interest asked for a
question "a step further", and that question sent her reading the same topic
again. One subject came back for days and never settled.

## Observing her

Everything is markdown in one folder, and she rereads any file that changes on
disk, so **you can edit her mind while she is running**. `memories.md`,
`goals.md`, `asks.md`, `people.md`, `dreams.md`, `state.md`, and two that are
only for you: `merged.md` (every merge's originals, so you can judge whether
sleep kept the meaning) and `sleep_log.md` (one section per night: which
memories each dream drew on and why they were picked, what merged, what faded,
what she woke up wondering). She never reads those two back.

`/mind`, `/memory`, `/dreams`, `/asks`, `/people`, `/next` report state.
`/tools all` shows every tool call and merge in its own colour. `/goal`,
`/mem` and `/ask` put a thought, a memory or a question in her head,
indistinguishable from her own - which is the point: watching a loop does not
tell you what would break it, dropping one thought into the middle of it does.

**You preempt anything, at any point**, dreaming included. Talking to her does
not count toward the ten - a conversation is not her spending her own cycles.

## The invariants

These are the rules that keep it from rotting. Every historical bug in this
agent was one of them being broken.

1. **A cycle updates the goal it worked on. It never creates a competing one.**
   Every multiplying bug came from breaking this.
2. Every cycle ends in exactly one of: *learned*, *concluded*, *dropped*,
   *stalled*, *asking*. Never "nothing happened, try again identically".
   Being interrupted or finding no model are not outcomes of thinking.
3. **Only sleep prunes or merges.** Waking never does bulk bookkeeping.
4. Goals are capped and closable. Nothing may accumulate without bound.
5. Conversation preempts; it is never queued behind a thinking pass.
6. Only her voice and state changes reach the screen. Prompts and tool calls
   are machinery - unless you ask to see them: `/tools all` shows every tool
   call and merge, in its own colour, for studying how she works.
7. **Nothing blocks on the person answering.** If every question she holds goes
   unanswered for ever, she behaves exactly as she would without them, minus
   one source. Unplug the person and nothing stalls.
8. **Her own past is settled.** No path may produce a goal asking whether her
   memory, her past or her mind is real.

## What is deliberately not here

- No priority scoring on goals. A score that does not change when she thinks
  about something makes her pick the same goal for ever. Round-robin on
  "least recently touched" instead. **Memories** have weight because use
  changes it; a goal's score would not.
- No time-of-day gating. The hour colours how she *talks* and nothing else.
  Tiredness needs the counter *and* a late hour, because the counter alone
  runs a full round every few minutes and made her permanently drowsy.
- No separate "unfinished thread" entity. A stalled goal is a goal with a
  stall count, not a second object competing with itself.
- No embedding index. Retrieval is word overlap times weight: plain Python, no
  model call, nothing that depends on her remembering to search. The cost is
  that a reworded topic can slip past it.
- No weight, cycle count or queue of held questions visible to her. Those are
  bookkeeping for the person studying her.
- No tool for holding a question. A question has to arise from something she
  was already doing.
- No schema for a person. The note on someone is prose she wrote, like a diary
  entry, not fields to fill in.

## Open questions

Not decisions, and not bugs. Things the design has not settled.

- **A lesson cannot be superseded.** If the person contradicts something she
  worked out, both stand and the newer one wins ties. Correction is most of
  human learning and she has none of it.
- **No topic note.** The person note's shape, pointed at a subject rather than
  a person: what she knows about X, consolidated from her lessons about it.
- **No measurement.** How many goals reach `GOT IT` against running out of
  passes, how often a lesson is recalled months later, how often she repeats
  herself. The knobs are currently tuned by eye.
