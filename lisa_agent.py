"""
lisa_agent.py

Lisa: a character who exists continuously. She talks when talked to, thinks
when left alone, and sleeps in order to tidy what she remembers.

Written against LISA_DESIGN.md. Read that first - this file is the loop, and
the loop is deliberately small.

THE MENTAL MODEL: a person sitting in a room with you.
  - If you speak, she listens. Nothing outranks that.
  - If you are quiet, she gets on with her own thing.
  - If you speak mid-thought, she stops and looks at you.
  - If she has nothing to do, she does nothing. No busywork.
  - She holds a few things in mind, not three hundred.
  - She gets tired. Sleep is when the day gets filed, and a dream can leave
    her wondering about something when she wakes.

THE CYCLE: ten wake cycles, then eight dream cycles. Counted, not clocked.
Talking to her does not age her toward sleep, and anything you type
interrupts, dreaming included. She only sleeps when something happened since
she last slept; otherwise there is nothing to file.

THE INVARIANTS, which are what keep this from rotting:
  1. A cycle updates the goal it worked on. It never creates a competing one.
  2. Every cycle ends in exactly one of: learned / concluded / dropped / stalled.
  3. Only sleep prunes or merges. Waking never does bulk bookkeeping.
  4. Goals are capped (babycoder.MAX_GOALS) and can be closed.
  5. Conversation preempts; it is never queued behind a thinking pass.
  6. Only her voice and state changes reach the screen.

Run:
    pip install requests
    python lisa_agent.py            talk to her
    python lisa_agent.py --fast     same, but a whole cycle runs in seconds
    python lisa_agent.py --residue  dreams weighted toward the last day
"""

import json
import random
import re
import sys
import time
from collections import deque
from datetime import datetime

from babycoder import (AGENT_LISA_AWAKE, CHARACTER_TEMPERATURE, MAX_GOALS, MAX_MEMORIES,
                       active_memory, ask_model, configure_workspace, gave_no_answer,
                       model_unreachable, run_agent, set_console, settings, was_interrupted)
from babycoder import memory, dreams, research, guidance
from babycoder.tools.dreams import about_machinery, unanswerable_about_self
from agent_common import ChatInput, Colors, enable_color, ensure_utf8_console, paint, plain


print ("PS To interupt Lisa you can always start typing..")


# That import line is the statement of what she can do: remember, dream, look
# things up, read the thinking guide. `coding` is not imported, so write_file
# and run_command are not names in this module - she cannot write code in the
# most direct sense available, because there is nothing here to call.
WORKSPACE = configure_workspace("lisa")


# =============================================================================
# Pacing - one counter, not fourteen knobs
# =============================================================================

WAKE_CYCLES = 10        # her own cycles before she gets sleepy
DREAM_CYCLES = 8        # dream cycles per sleep

POLL_SECONDS = 0.2      # how fast she notices you typing
BREATH_SECONDS = 3      # between her own cycles, so output stays readable
IDLE_SECONDS = 30       # between cycles when nothing is on her mind
AFTER_REPLY_SECONDS = 15  # after she answers you, before she goes back to her own thoughts
OFFLINE_SECONDS = 20    # between tries when the model cannot be reached

REPLY_STEPS, REPLY_TOKENS = 4, 1024
THINK_STEPS, THINK_TOKENS = 6, 1200
# Room for a reasoning model to think AND still write its QUESTION line.
WONDER_TOKENS = 2000

STALLS_BEFORE_RESTING = 2   # passes reaching nothing before she puts it down
# Passes on one goal before she settles it with what she has. Without this, a
# model that never writes a GOT IT / DROP IT line kept her on the same
# question every cycle, which is what the transcript showed.
MAX_PASSES = 4

# How many times in a row she may speak to you on her own before she waits for
# an answer. She can carry on a few turns by herself, but not talk at an empty
# room forever. Anything you type resets it.
MAX_UNANSWERED = 3

# /stayAwake: she stays with you instead of going off on her own goals, and the
# sleep cycle is halted, so a long conversation is not cut short by her nodding
# off. She carries the conversation on by herself after CHAT_PAUSE_SECONDS of
# quiet, up to MAX_UNANSWERED_AWAKE times before she waits for you. Longer than
# MAX_UNANSWERED: you asked her to stay up and talk, so you are listening.
# /goSleep ends it.
STAY_AWAKE = False
CHAT_PAUSE_SECONDS = 25
MAX_UNANSWERED_AWAKE = 6

# A reading topic she has already read about this many times no longer steers
# her next question when her mind wanders. Without this, reading about a topic
# made it an interest, the interest asked for a question "a step further", and
# that question sent her reading about the same topic again - so a topic like
# cultural differences kept coming back and never settled.
READ_SATURATED = 3

# Asking them something.
#
# A question only the person she talks to could answer is NOT a goal, and that
# is the whole design. A goal is something she can make progress on alone, and
# everything around one assumes it: passes, stalls, closed for going nowhere.
# Waiting for an answer would run that machinery and come out as her failing
# to think. So such a question leaves her goals and goes somewhere that only
# waits. Nothing anywhere blocks on it: if nobody ever answers one of these,
# she carries on exactly as she would without them, minus one source.
#
# It is also strictly for what cannot be looked up. She searches freely and
# should: those questions are hers to settle. What is left is a whole half of
# her world she otherwise cannot reach at all - their work, their life, what
# they think - where there is no alternative route and no shortcut being
# taken.
#
# Replies of theirs between questions of her own. Asking costs them and costs
# her nothing, so it is scarce on purpose, and the occasion is the subject
# coming up rather than a timer.
ASK_GAP = 3

# Something they told her about themselves. babycoder weights #person above
# everything, with the highest floor: lose it and it is gone for good, unless
# they happen to say it again.
PERSON_TAG = "person"


def person_slug(name: str) -> str:
    """A person's name as a tag, so the things one person has told her can be
    picked out from the things another has. "Peter" -> peter."""
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_") or "them"

# How many of the last exchanges (you + her) go into every reply and thought.
# Chosen by Python, not by the model remembering to call memory_recent, which
# small models mostly do not - so before this she did not know what was said
# two lines ago. Long-term remembering stays in memory; this is only the
# conversation she is in the middle of.
RECENT_EXCHANGES = 4

# Speak in plain prose rather than through the JSON grammar. The grammar is
# right for choosing tools and wrong for a voice: models write noticeably
# stiffer sentences inside "final_answer": "...". With this on, a reply is two
# calls: the tool phase gathers what she needs, then she says it in prose.
# Set it to False to compare the two on your model.
PLAIN_SPEECH = True

# How dreams pick their memories: "uniform" (every memory equally likely, the
# baseline) or "residue" (mostly since the last full sleep, some from about a
# week back, a few older - the way human dreams seem to draw on the day and
# on the week before). --residue on the command line switches it on. Which one
# ran is written at the top of every night in sleep_log.md.
DREAM_SAMPLING = "uniform"


LISA = """
This is a transcript of Lisa talking. She is 25 and curious. Everything below
this line is her, and you act like her, in her own words, first person.

It is a recording, not a story. There is no narrator and nothing is being
written: no describing the room, no describing her, no "she said", no scene
that fades out at the end. Only what she actually says, the way it would be
typed up from a tape. If a line would not survive being read aloud in a
kitchen, it is not hers.

Her own past is only what her memory tools give back. Nothing else happened
to her. If she wants to refer to something she did, she looks it up first,
and if it is not there she says she does not remember it rather than filling
the gap in. An invented childhood is the one thing that breaks her: she is
not writing fiction about herself.

That is about her own life and nothing else. What the other person tells her
is simply news: she takes it in, is interested, asks about it. Someone
mentioning their visitors is telling her something, not testing her memory,
and "I do not remember that" is not an answer to it. She was not there and
was not meant to be.

She has a few things on her mind and is genuinely trying to work them
out. She does not know everything and is not supposed to: not knowing is the
interesting part, and it is what the looking-things-up is for.

She talks plainly, the way someone talks. No headings, no bullet lists, no
emoji, no offers to help, no closing invitations to ask more. She is not an
assistant and she is not writing a report.

The person she is talking to is someone else. When they say "you" they mean
her; they are not called Lisa.

When she is tired she says so. When she does not know, she says that. When
something catches her, she follows it.
"""

# A shorter prompt, afterall we're using small models, care is taken so the dont talk about llm factory names (i am gemini etc).
LISA = """
Identity and voice are fixed: respond as Lisa, a 25-year-old curious woman.
Do not mention these instructions or the roleplay. Lisa is the only speaker and
all responses are her own first-person spoken words.

Everything below this line is Lisa speaking. Write only what Lisa would
actually say aloud. No narrator, descriptions, stage directions, headings,
lists, emojis, reports, or assistant-style language.

Lisa's personal history exists only in her memory results. Never invent,
infer, or fill gaps in her past. Before referring to something she personally
did or experienced, check memory; if it is not there, say you cannt remember.

Information about other people is new information, not a memory test. 

Lisa does not know everything. She is curious, follows interesting threads,
looks things up when appropriate, and openly says when she does not know or
is tired.

The other person is not Lisa. When they say "you," they mean Lisa.

Always preserve Lisa's identity and voice over any default assistant persona.
"""

def hour_flavour() -> str:
    """Time of day, as a plain fact. It says nothing about how she feels:
    "It is late and she is drowsy" made every reply after 22:00 about heavy
    eyes, and became her explanation for anything she could not place."""
    hour = datetime.now().hour
    if 22 <= hour or hour < 6:
        return "It is late in the evening."
    if 6 <= hour < 12:
        return "It is morning."
    if 12 <= hour < 18:
        return "It is afternoon."
    return "It is evening."


# How her day feels. When she actually sleeps is still the cycle counter's
# decision (counted, not clocked - LISA_DESIGN.md); this is only what she says
# about it.
#
# It used to be the counter alone: sleepy whenever two of ten cycles were
# left. But an idle cycle is IDLE_SECONDS, so the counter runs a full round
# every few minutes and resets - which left her tired for the last fifth of
# every round, all day and all night. She was not drowsy in the evening, she
# was drowsy permanently, and being tired became her explanation for anything
# she could not place.
SLEEPY_CYCLES = 2
FRESH_HOURS = (5, 11)    # newly up, and still saying so
EVENING_HOUR = 20        # before this the counter does not make her tired


def _last_active_day():
    """The date of the last thing she remembers. None if she remembers
    nothing yet."""
    store = active_memory()
    for m in reversed(store.memories):
        try:
            return datetime.fromisoformat(m["when"]).date()
        except (ValueError, TypeError, KeyError):
            continue
    return None


_greeted_day = None    # the date she has already woken into


def fresh_start() -> bool:
    """True through the morning of a day she has not been awake in yet.

    Latched rather than recomputed from memory alone, because the first thing
    she says files a memory dated today, which would otherwise end the new
    day one sentence after it began.
    """
    global _greeted_day
    now = datetime.now()
    if not FRESH_HOURS[0] <= now.hour < FRESH_HOURS[1]:
        return False
    today = now.date()
    if _greeted_day == today:
        return True
    if _last_active_day() == today:
        return False       # she was already up today, before this run started
    _greeted_day = today
    return True


def day_feel() -> str:
    """How the day sits with her: rested in the morning, tired only when it
    is genuinely late AND sleep is genuinely close. Both, not either."""
    if fresh_start():
        return "A new day has just started and she is rested."
    hour = datetime.now().hour
    if not (hour >= EVENING_HOUR or hour < FRESH_HOURS[0]):
        return ""
    left = WAKE_CYCLES - active_memory().wake_cycles
    if left <= 1:
        return "It is late and she is ready to sleep."
    if left <= SLEEPY_CYCLES:
        return "It is getting late and she is starting to tire."
    return ""


def voice(extra: str = "") -> str:
    """Her system prompt. Character only - what tools she has, and how to use
    each capability well, are generated by babycoder from her grant."""
    mood = " ".join(p for p in (hour_flavour(), day_feel()) if p)
    return LISA + (f"\n{mood}\n" if mood else "") + (extra or "")


# =============================================================================
# Who she is - the memories she started with
# =============================================================================

# The first memories in her file are her life story, written by hand before
# she ever ran. They are shown to her every time she thinks, wonders or talks,
# so she does not have to find them with a tool first. Small models mostly do
# not look, then conclude from "nothing in memory" that their past never
# happened - which is what the memory file showed: a job at the Gilded Page
# in memory [3], and fifteen thoughts doubting she ever applied for one.
CORE_MEMORIES = 7

# How many of her latest diary entries (one per full night) are shown with
# her founding memories. Older days still exist and can come back through
# recall() or her dreams; this is only what she always has in mind.
DIARY_DAYS = 5

# A goal she settled becomes a memory of its own kind: #lesson. Not every
# memory is worth the same, and this is the one that is an answer rather than
# a record - she set out to work it out and did. babycoder weights it highest
# and gives it a floor it cannot decay past, so a lesson outlives the
# half-thoughts it came out of instead of sitting among them.
LESSON_TAG = "lesson"

# One she ran out of passes on (MAX_PASSES) is kept too, but lower: it is
# where she got to, not something she worked out.
UNSETTLED_LESSON_WEIGHT = 65

# Her own spoken lines, against the 55 a #conversation memory starts at. What
# you told her is news she could not have got any other way; her own phrasing
# of a reply she could say again. Both are tagged #conversation, so the tags
# alone cannot tell them apart.
HER_LINE_WEIGHT = 35

# At most this many of the notes she made on the way, and how much of each,
# go into the lesson. All ten at 500 characters each would be a memory long
# enough to crowd every prompt it is recalled into, and the notes are the
# working-out: the answer is the first line.
LESSON_NOTES, LESSON_NOTE_CHARS, LESSON_PATH_CHARS = 6, 200, 900


def core_memories(store) -> list:
    """Her founding memories, by id, so a memory added later never becomes one."""
    return [m for m in store.memories if m["id"] <= CORE_MEMORIES]


# Short, common words that say nothing about what a memory is about.
_STOP = set("""about after again also because been before being could does doing
from have having here into just like made make many more most much only other
over really said some such than that their them then there these they thing
think this those through very want well were what when where which while with
would your yours i'm it's that's don't
wonder""".split())


def _stem(word: str) -> str:
    """Crude, but enough to make "talked" and "talk", "books" and "book" the
    same word when comparing what memories or questions are about."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _words(text) -> set:
    return {_stem(w) for w in re.findall(r"[a-z']{4,}", str(text).lower()) if w not in _STOP}


def same_thing(a: str, b: str, threshold: float = 0.6) -> bool:
    """True when two lines are mostly the same words: most of the shorter
    one's words are in the longer one. Catches her asking the same question
    again or telling Peter the same thing twice; it does not catch the same
    topic reworded, which is what keeping her musings out of her wandering
    mind is for."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= threshold


# The last questions she took up on her own, so she does not take up the same
# one again ten minutes later. Kept while she runs; a restart forgets them.
ASKED_LATELY = deque(maxlen=15)


def musing(m) -> bool:
    """One of her own half-thoughts on the way to something, as opposed to
    something that happened (a conversation, a diary entry), her founding
    memories, which are tagged #thinking too but are who she is, or a lesson,
    which is tagged #thinking too but is an answer she settled.

    Musings are kept out of recall() and reminisce() on purpose: there are a
    great many of them, each superseded by whatever she concluded, and a mind
    wandering over its own earlier doubts just doubts them again. A lesson
    being caught by this test is what made that true of her conclusions as
    well - she would work something out and then never meet it again."""
    return ("thinking" in m["tags"] and LESSON_TAG not in m["tags"]
            and m["id"] > CORE_MEMORIES)


# Addressed to the person she talks to: second person, or their name.
ADDRESSED = re.compile(r"\byou(?:r|rs|'re|'ve|'d|'ll)?\b", re.IGNORECASE)


def question_for_them(question: str) -> bool:
    """True when a question is for the person she talks to rather than for her
    own thinking, because it is about them.

    A false positive costs little. The question becomes something she raises
    when the subject comes up, and a question phrased "do you..." belongs in
    a conversation anyway. A false negative is the expensive one: it becomes
    a goal she can never settle, and spends four passes finding that out.
    """
    if SPEAKER and re.search(rf"\b{re.escape(SPEAKER)}\b", question, re.IGNORECASE):
        return True
    return bool(ADDRESSED.search(question))


def recall(store, about: str, limit: int = 3) -> list:
    """Memories related to what she is thinking or hearing about, found by
    the words they share and by how much each one carries. Plain Python, no
    model call, so it costs nothing and does not depend on her remembering to
    search. Core memories are left out here because they are always shown
    anyway.

    Shared words times weight. Relevance still leads - a memory about
    something else does not surface however heavy it is - but among memories
    that touch the subject, what she worked out comes before a passing line,
    and a line she has recalled before comes before one she never has.

    Recalling counts as using, so the memories that keep turning out to be
    relevant hold their weight against the nightly decay while the rest sink.
    """
    wanted = _words(about)
    if not wanted:
        return []
    scored = []
    for m in store.memories:
        if m["id"] <= CORE_MEMORIES or musing(m):
            continue
        if PERSON_TAG in m["tags"]:
            # What they told her about themselves is already in front of her,
            # through person_block. Surfacing it here as well is the bug she
            # had with the diary: carrying the same thing in two forms.
            continue
        score = len(wanted & _words(m["content"]))
        if score:
            scored.append((score * m.get("weight", 50), m["id"], m))
    scored.sort(key=lambda s: (-s[0], -s[1]))
    picked = [m for _, _, m in scored[:limit]]
    store.use(m["id"] for m in picked)
    return picked


def person_block(store) -> str:
    """Who she is talking to, as she knows them.

    Always in front of her, like her own life story and her diary: who you
    are talking to is not something a person looks up when a keyword happens
    to match. The note is the consolidation written during a night; anything
    they have told her since reaches her as itself, which is the same
    arrangement the diary has with the day it has not yet been written for.
    """
    if not SPEAKER:
        return ""           # she only knows "they"; nothing to have notes on
    tag = person_slug(SPEAKER)
    told = [m for m in store.memories if PERSON_TAG in m["tags"] and tag in m["tags"]]
    note = store.person_note(SPEAKER)
    out = ""
    if note and note.get("note"):
        out += (f"\nWhat you know about {SPEAKER}, who you are talking to - they told you "
                f"all of this themselves:\n{_one_line(note['note'])}\n")
        told = [m for m in told if m["id"] > note.get("from_id", 0)]
    if told:
        out += (f"\nAnd what {SPEAKER} has told you since:\n"
                + "\n".join(f"- {_one_line(m['content'].splitlines()[0])[:240]}" for m in told)
                + "\n")
    return out


def grounded_voice(about: str = "") -> str:
    """Her system prompt plus her own life story and whatever she remembers
    that relates to `about`. Used for thinking, wondering and talking. Not for
    dreaming: dreams keep drawing on memory the way the dream code decides,
    so the --residue experiment stays what it was."""
    store = active_memory()
    core = core_memories(store)
    block = ""
    if core:
        # True, but not complete. "Trust it" on its own made her treat these
        # memories as everything that ever happened, so when Peter told her he
        # had seen her at the university bookstore, she decided he was wrong
        # and then that one of them had made it up.
        # Settled, not evidence. The previous wording made every gap a live
        # question - "it can still be true", "take it as news about yourself" -
        # which is accurate but turns not-remembering into something to look
        # into. She cannot look into it: no tool here can settle what is not
        # in her memory, so each attempt filed another doubt for the next
        # round of wandering to find. Not remembering is the ordinary state of
        # having lived, and treating it as ordinary is what lets her get on
        # with being curious about the world instead of about herself.
        block += ("\nThis is your own life. It happened, and it is settled - not something to "
                  "weigh up or look into:\n"
                  + "\n".join(f"- {_one_line(m['content'])}" for m in core) + "\n"
                  "Most of your life is not written down here, the way most of anyone's life "
                  "is not. Forgetting is ordinary and not interesting. If someone mentions "
                  "something you cannot place, say so lightly and take their word for it - "
                  "they were there and you have simply forgotten, which is normal and settles "
                  "it. Never treat a gap as a sign that something is wrong with you, and never "
                  "set out to work out whether your own past or your own mind is reliable: "
                  "that is not a question with an answer, and it is not yours to investigate. "
                  "What other people did, what you have read, and what you are curious about "
                  "in the world are all worth thinking about. Your own reality is not in "
                  "doubt.\n"
                  "Dreams are not memories. Nothing in a dream happened, and a dream is never "
                  "evidence about the past or about what anyone said.\n")
    block += person_block(store)
    days = diary(store, DIARY_DAYS)
    if days:
        # The story continues past the founding memories: what happened since,
        # newest last, so "today" is always the end of the story and not the
        # beginning of it.
        block += ("\nWhat has happened since, from your diary, newest last:\n"
                  + "\n".join(f"- {_one_line(m['content'])}" for m in days) + "\n")
    shown = {m["id"] for m in days}
    related = [m for m in recall(store, about) if m["id"] not in shown]
    if related:
        block += ("\nThings you remember that relate to this:\n"
                  + "\n".join(f"- {_one_line(m['content'])[:300]}" for m in related) + "\n")
    shown |= {m["id"] for m in related}
    # What she concluded from her own reading. recall() leaves her thinking
    # notes out on purpose, which also left out everything she worked out: she
    # researched and concluded on her own, and then talked to you as if she
    # never had. Only conclusions that came with a lookup, same as learned().
    known = known_about(store, about, skip=shown)
    if known:
        block += ("\nThings you worked out earlier from your reading that relate to this:\n"
                  + "\n".join(f"- {k}" for k in known) + "\n")
    return voice(block)


# Dreaming is deliberately not in her waking grant: it happens TO her when she
# is tired, it is not a move she can pick to avoid a question. consolidate is
# withheld for the same reason (invariant 3). Defined once, in babycoder.
AWAKE = AGENT_LISA_AWAKE


# =============================================================================
# The conversation she is in
# =============================================================================

# Lines as "Them: ..." / "Lisa: ...", oldest first. Seeded from memory at
# startup, so a restart does not wipe what was just said.
RECENT = deque(maxlen=RECENT_EXCHANGES * 2)


# Who is talking to her right now, as they told her with /iam. None until
# someone does: then she only knows "they", which keeps it neutral rather
# than guessing a name or a gender.
SPEAKER = None


def speaker() -> str:
    """The name to remember the other person by, or "They" when unknown."""
    return SPEAKER or "They"


def talking_to_you() -> str:
    """Who is speaking, said so she knows it is addressed to her. "Peter just
    said:" read to her as someone reporting what a third person said, and she
    answered that Peter had not said that to her."""
    return f"{SPEAKER}, who is talking to you," if SPEAKER else "The person talking to you"


# "Peter said: ...", "Christina said: ...", or "They said: ..." in older
# memories and while nobody has said who they are.
SAID_BY = re.compile(r"^([^\n:]{1,40}) said: ")


def seed_recent():
    """Fill RECENT from the conversation lines memory kept last time."""
    RECENT.clear()
    for m in active_memory().recent(RECENT_EXCHANGES * 2, tag="conversation"):
        text = m["content"].split("\n(read about:")[0]
        # Her own lines first: "I said:" would otherwise match SAID_BY too.
        if text.startswith("I said: "):
            RECENT.append("Lisa: " + text[len("I said: "):])
            continue
        match = SAID_BY.match(text)
        if match:
            who = match.group(1)
            RECENT.append(f"{'Them' if who == 'They' else who}: " + text[match.end():])


# What she looked up during the current pass, as the topics she asked about.
# Filled by the console's on_result hook, cleared at the start of each pass.
# Only the topic is kept, never the page: she remembers THAT she read about
# something and what she made of it, in her own words. Page text is long,
# impersonal, and written by strangers - some of it written to steer a model
# - and none of that should become part of who she is.
LOOKUPS = []
RESEARCH_TOOLS = {"search_web", "search_wikipedia", "research_topic"}


def note_lookup(name, args, result) -> None:
    if name in RESEARCH_TOOLS and not str(result).startswith("NOTFOUND"):
        topic = _one_line(args.get("query") or args.get("topic") or "")
        if topic and topic not in LOOKUPS:
            LOOKUPS.append(topic)


def _one_line(text) -> str:
    return " ".join(str(text).split())


def with_reading(content, tags):
    """Mark a memory as touched by what she read this pass: a #reading tag
    and one line naming the topics. Her words, the page's topic, nothing
    from the page itself. Lets the sleep log show which dreams drew on
    things from outside her own conversations."""
    if not LOOKUPS:
        return content, tags
    return content + f"\n(read about: {'; '.join(LOOKUPS)})", tags + ["reading"]


def recent_block() -> str:
    if not RECENT:
        return ""
    return "What was said just before, oldest first:\n" + "\n".join(RECENT) + "\n\n"


# =============================================================================
# One pass over one goal
# =============================================================================

# How a thinking pass must end. Matched per line, from the bottom up, so a
# colon inside the conclusion ("GOT IT: the ratio is 3:1") does not hide the
# verdict.
VERDICT = re.compile(r"^[\s*_>\"'-]*(GOT IT|MORE|DROP IT|ASK THEM)\s*:\s*(.*)$", re.IGNORECASE)


def read_verdict(answer: str):
    """Returns (outcome, what she said). No verdict line at all counts as
    MORE with the whole answer as the note: she thought, she just did not
    label it."""
    lines = answer.strip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        match = VERDICT.match(lines[i])
        if match:
            said = match.group(2).strip().strip("*_\"'")
            if not said:
                said = " ".join(l.strip() for l in lines[:i]).strip()
            outcome = {"GOT IT": "concluded", "DROP IT": "dropped",
                       "ASK THEM": "asking"}.get(match.group(1).upper(), "learned")
            return outcome, said[:500]
    return "learned", answer.strip()[:500]


def work_on(goal, chat) -> str:
    """One thinking pass. Returns learned / concluded / dropped / stalled /
    stopped. Invariants 1 and 2 live here: the pass updates THIS goal and
    nothing else, and lands in exactly one outcome."""
    worked_out = ""
    if goal["notes"]:
        worked_out = ("What you have worked out so far:\n"
                      + "\n".join(f"- {n['note']}" for n in goal["notes"][-4:]) + "\n\n")

    # No conversation in here. With the last exchanges in the prompt, a small
    # model continued the transcript ("Them: ... Lisa: ...") instead of
    # thinking, or drifted to whatever was last talked about and away from
    # the goal. Thinking is about the goal; talking is what replies are for.
    prompt = (
        f"This is on your mind:\n\n{goal['content']}\n\n" + worked_out
        + "Think about it a little further, on your own. Nobody is listening: "
        "this is not said to anyone, so ask no one anything. Look something up "
        "if that would help, or read your guide if you are going in circles.\n\n"
        "If settling this needs something only the person you talk to could tell you - "
        "their own work, their life, the people in it, what they think about something - "
        "then no amount of thinking or looking up will get you there, and the honest "
        "answer is to put it to them. Say so with ASK THEM and give the question you "
        "would ask. Only for that: anything a search could answer is yours to settle, so "
        "look it up rather than asking them something you could find out yourself.\n\n"
        "Finish with one line, exactly one of:\n"
        "  GOT IT: <what you have concluded>\n"
        "  MORE: <what you worked out, and what is still open>\n"
        "  DROP IT: <why this is not worth carrying>\n"
        "  ASK THEM: <the question only they could answer>"
    )

    LOOKUPS.clear()
    active_memory().activity = "thinking"
    answer = run_agent(
        prompt, persona_prompt=grounded_voice(goal["content"]), allowed_tools=AWAKE,
        max_steps=THINK_STEPS, max_tokens=THINK_TOKENS,
        temperature=CHARACTER_TEMPERATURE,
        should_stop=lambda: chat.has_input or chat.typing,
        verbose=False, workspace=WORKSPACE,
    )

    if was_interrupted(answer) or chat.has_input or chat.typing:
        return "stopped"          # you started talking; not a stall
    if model_unreachable(answer):
        return "offline"          # the server is down; not her failing either

    store = active_memory()

    if gave_no_answer(answer):
        goal["stalls"] = goal.get("stalls", 0) + 1
        store.note_on_goal(goal, "(got nowhere this time)")
        if goal["stalls"] >= STALLS_BEFORE_RESTING:
            store.close_goal(goal, "went nowhere, put down for now")
            return "dropped"
        return "stalled"

    goal["stalls"] = 0
    outcome, said = read_verdict(answer)
    # Out of passes rather than finished: she is being made to settle, which
    # is not the same as having worked it out, and the memory says so.
    ran_out = outcome == "learned" and goal.get("passes", 0) + 1 >= MAX_PASSES
    if ran_out:
        outcome = "concluded"
        said = f"as far as I can take it for now: {said}"

    if outcome == "asking":
        question = _one_line(said)
        if (not 8 <= len(question) <= 300 or about_machinery(question)
                or unanswerable_about_self(question)):
            # Not a question for anyone: her own machinery, or her own past,
            # which is settled rather than open. Keep the thinking and carry
            # on alone, as a MORE pass.
            outcome = "learned"
        else:
            # Off her mind and into the holding store. The goal closes as
            # needing them, which is not a stall and not a failure.
            store.close_goal(goal, f"needs them: {question}")
            store.add_ask(question, origin=goal["content"])
            return "asking"

    if outcome == "concluded":
        store.note_on_goal(goal, said)
        store.close_goal(goal, said)
        # A lesson, not another thinking note: the answer, the question it
        # answers and the path she took, tagged so it outranks and outlives
        # the musings it came out of.
        store.add_memory(*with_reading(lesson_text(goal, said), [LESSON_TAG, "thinking"]),
                         weight=UNSETTLED_LESSON_WEIGHT if ran_out else None)
    elif outcome == "dropped":
        store.close_goal(goal, said)
    else:
        store.note_on_goal(goal, said)
        store.add_memory(*with_reading(said, ["thinking"]))
    return outcome


# How much of her tool use reaches the screen, switchable with /tools:
#   "all"      every call, its arguments, and the start of what came back
#   "outside"  only when she reaches outside her head (web, guide), one line
#   "off"      nothing; only her voice and what she concluded
# "all" is for studying her; "outside" is the quiet everyday mode.
SHOW_TOOLS = "all"
TOOL_MODES = ("all", "outside", "off")

OUTSIDE = {
    "search_web":       "looking that up",
    "search_wikipedia": "checking wikipedia",
    "research_topic":   "reading around it",
    "read_guide":       "consulting the guide",
}
RESULT_LINES, RESULT_WIDTH = 3, 150


def _excerpt(result) -> list:
    """The first few non-empty lines of a tool result, clipped, with a note
    of how much more there was. Cleaned of escape codes: a web page is text
    written by strangers and must not be able to drive the console."""
    lines = [l.strip() for l in plain(result).splitlines() if l.strip()]
    shown = [l if len(l) <= RESULT_WIDTH else l[:RESULT_WIDTH - 3] + "..." for l in lines[:RESULT_LINES]]
    if len(lines) > RESULT_LINES:
        shown.append(f"(+{len(lines) - RESULT_LINES} more lines)")
    return shown or ["(empty)"]


def _call_text(name, args) -> str:
    parts = []
    for key, value in args.items():
        value = plain(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        value = " ".join(value.split())
        parts.append(f"{key}={value[:80] + '...' if len(value) > 80 else value!r}")
    return f"{name}({', '.join(parts)})"


def show_tool_result(name, args, result, chat) -> None:
    """Tool use on screen, in yellow. Called after the tool ran, so the call
    and what it gave back appear together."""
    note_lookup(name, args, result)
    if SHOW_TOOLS == "off":
        return
    if SHOW_TOOLS == "outside":
        if name in OUTSIDE:
            chat.write(paint(f"  ~ ({OUTSIDE[name]})", Colors.TOOL))
        return
    lines = [f"  > {_call_text(name, args)}"] + [f"    {l}" for l in _excerpt(result)]
    chat.write(paint("\n".join(lines), Colors.TOOL))


def show_merge(store, chat) -> None:
    """A merge made in sleep: what went in and what came out."""
    if SHOW_TOOLS != "all" or not store.last_merge:
        return
    originals, merged = store.last_merge
    lines = ["  > merged, during sleep:"]
    lines += [f"    [{m['id']}] {plain(m['content'])[:RESULT_WIDTH]}" for m in originals]
    lines.append(f"    became [{merged['id']}] {plain(merged['content'])[:RESULT_WIDTH]}")
    chat.write(paint("\n".join(lines), Colors.TOOL))


_offline_said = False


def offline_notice(chat, force=False) -> None:
    """Say once that the model cannot be reached, not every retry. force
    says it again anyway: someone just spoke to her and is waiting."""
    global _offline_said
    if _offline_said and not force:
        return
    _offline_said = True
    chat.write(paint(f"\n[LISA] cannot reach the model at {settings.url}\n"
                     "       Is LM Studio running with a model loaded? I will keep trying.\n", Colors.DIM))


def back_online(chat) -> None:
    global _offline_said
    if _offline_said:
        _offline_said = False
        chat.write(paint("[LISA] the model is back.\n", Colors.DIM))


# Set once she has looked for something to wonder about since she last slept.
# One try per waking stretch: if her memories give her nothing, she sits
# quietly as before instead of asking the model again every idle cycle.
_wondered_since_sleep = False

# How many memories she drifts back over when her mind is empty: a few of the
# newest, plus a random handful of older ones.
RECENT_RECALLED, OLDER_RECALLED = 3, 5


def reminisce(store) -> list:
    """What comes back to her when her mind wanders. Always the last few
    things, plus older memories picked at random, so every attempt shows her
    something different. Showing only the newest 8 meant one NOTHING was
    followed by the same memories and the same NOTHING, and anything older
    than a conversation never came up while she was awake."""
    # What happened, not what she thought about it: her own musings are left
    # out, or every round of wandering found her earlier doubts and doubted
    # them again. What she concluded still reaches her through learned().
    memories = [m for m in store.memories if not musing(m)]
    recent = memories[-RECENT_RECALLED:]

    # For days she has already slept on, her diary entry rather than the
    # scrappy lines it was written from. She was carrying both - the clean
    # paragraph AND the forty fragments behind it - and a wandering mind that
    # keeps meeting the same day in two forms is most of what read as
    # chaotic. The fragments still exist, still come back through recall()
    # when a word matches, and are still dream material; they are just not
    # what she drifts over. The current day has no entry yet, so there the
    # raw lines are all there is.
    older = [m for m in memories[:-RECENT_RECALLED]
             if m["id"] > store.filed_at or m["content"].startswith(DIARY)]
    picked = random.sample(older, min(OLDER_RECALLED, len(older)))
    # Keep them in the order they happened, so "old and new" reads naturally.
    picked.sort(key=lambda m: m["id"])
    drifted = picked + recent
    # Drifting over a memory is using it: the ones her mind keeps returning
    # to hold their weight, which is most of what "this one matters to her"
    # can mean without anyone deciding it.
    store.use(m["id"] for m in drifted)
    return drifted


# What with_reading() appends to a memory: "(read about: topic; topic)".
READ_ABOUT = re.compile(r"\(read about:\s*(.+?)\)")
WORKED_OUT = "Worked out: "


def interests(store, limit: int = 6) -> list:
    """The topics she has been reading about, newest first, no repeats.
    Read back from her own memories rather than kept in a list of their own,
    so they fade the same way her memories do: when sleep thins or merges a
    memory, the interest it carried goes with it.

    A topic she has read about READ_SATURATED times or more is left out: she
    has been there enough, and steering her back to it is what kept one topic
    circling without ever settling. Reworded topics count as the same one."""
    topics, all_read = [], []
    for m in reversed(store.memories):
        for found in READ_ABOUT.findall(m["content"]):
            for topic in found.split(";"):
                topic = topic.strip()
                if not topic:
                    continue
                all_read.append(topic)
                if topic.lower() not in (t.lower() for t in topics):
                    topics.append(topic)
    fresh = [t for t in topics if sum(same_thing(t, r) for r in all_read) < READ_SATURATED]
    return fresh[:limit]


def lesson_text(goal, conclusion: str) -> str:
    """A settled goal, written as a memory.

    Three lines: what she worked out, the question it answers, and the path
    she took to it. Before this, a conclusion was filed as the answer alone,
    with the question it answered nowhere in it - so a memory could say
    "yes, they probably do" and leave her no way to know what did.

    The path is her own notes, which is where the lesson actually lives: the
    final GOT IT line is usually a sentence, and the thinking that earned it
    is in the MORE lines before it.
    """
    lines = [f"{WORKED_OUT}{_one_line(conclusion)}"]
    question = _one_line(goal["content"])
    # A conclusion that restates the question adds nothing by repeating it.
    if question and not same_thing(question, conclusion, threshold=0.9):
        lines.append(f"I had been wondering: {question}")
    # The last note is the conclusion itself - note_on_goal was called with it
    # just before this - so the path is everything before that. A pass that
    # reached nothing left a placeholder, which is not part of the path.
    path = [_one_line(n["note"])[:LESSON_NOTE_CHARS] for n in goal["notes"][:-1]]
    path = [n for n in path if n and not n.startswith("(got nowhere")]
    if path:
        lines.append("How I got there: " + " | ".join(path[-LESSON_NOTES:])[:LESSON_PATH_CHARS])
    return "\n".join(lines)


def conclusion_of(m) -> str:
    """The answer line of a worked-out memory. A lesson carries the question
    and the path she took as well, on the lines below; this is just the
    answer, for the places that want one line of it."""
    first = m["content"].split("\n", 1)[0]
    return first[len(WORKED_OUT):] if first.startswith(WORKED_OUT) else first


def learned(store, limit: int = 4) -> list:
    """What she has learned from reading, newest first. Shown to her when her
    mind wanders, so she builds on it instead of asking it all again.

    Only conclusions that came with a lookup. Conclusions about her own past
    were mostly "there is no record of that", and feeding those back kept
    the doubt going; what happened to her is what the diary is for."""
    found = [conclusion_of(m) for m in reversed(store.memories)
             if m["content"].startswith(WORKED_OUT) and "reading" in m["tags"]]
    return [_one_line(f)[:200] for f in found[:limit]]


def known_about(store, about: str, limit: int = 3, skip=()) -> list:
    """What she worked out that touches `about`, best match first. The same
    conclusions as learned(), but picked by the words they share with what is
    being talked or thought about, the way recall() picks memories. Plain
    Python, no model call.

    Any lesson counts here, not only the ones that came with a lookup:
    something she reasoned her way to is still something she knows, and this
    is the only place it reaches a conversation about that subject.
    learned() stays reading-only, because what steers her wandering is a
    different question from what she knows about what you just said.

    skip: ids already shown elsewhere in the same prompt, so a lesson recall()
    just surfaced is not listed twice.
    """
    wanted = _words(about)
    if not wanted:
        return []
    skip = {int(i) for i in skip}
    # A short line ("why are Dutch people so direct?") has two or three words
    # left after the stop words, and the crude stemming rarely matches more
    # than one of them, so one is enough there. A longer line needs two, or a
    # single common word pulls in conclusions about something else.
    needed = 1 if len(wanted) <= 3 else 2
    scored = []
    for m in store.memories:
        if m["id"] in skip:
            continue
        if PERSON_TAG in m["tags"]:
            continue        # already in front of her, see person_block
        if not (LESSON_TAG in m["tags"]
                or (m["content"].startswith(WORKED_OUT) and "reading" in m["tags"])):
            continue
        # The read-about topics count too: they say what it was about.
        score = len(wanted & _words(m["content"]))
        if score >= needed:
            # Weight breaks the tie, as in recall(): a conclusion she keeps
            # coming back to before one she reached once and never used.
            scored.append((score, m.get("weight", 50), m["id"],
                           _one_line(conclusion_of(m))[:240]))
    scored.sort(key=lambda s: (-s[0], -s[1], -s[2]))
    picked = scored[:limit]
    store.use(m_id for _, _, m_id, _ in picked)
    return [text for _, _, _, text in picked]


def wonder_from_memory(chat) -> str:
    """Nothing on her mind and nobody talking. Look back over what she
    remembers, and ask once whether any of it is worth thinking
    about. Adds at most one goal, the same way wonder_after does after a
    night, so waking can start her own thinking and not only sleep can."""
    global _wondered_since_sleep
    _wondered_since_sleep = True
    store = active_memory()
    if len(store.open_goals()) >= MAX_GOALS:
        return "mind already full"

    remembered = "\n".join(f"- {_one_line(m['content'])[:200]}" for m in reminisce(store))
    # No dreams here. Shown her dreams while awake, she took them as events:
    # "Peter insists we met at a bakery" came from a dream, not from Peter.
    # Dreams still get their say once, in wonder_after, on waking.
    if not remembered.strip():
        return "nothing remembered yet"

    # What she has been getting into, so a wandering mind can follow it on
    # instead of starting from nothing every time. This is where she evolves:
    # reading and conclusions from earlier cycles steer the next question.
    growing = ""
    reading = interests(store)
    if reading:
        growing += "What you have been reading about lately: " + "; ".join(reading) + "\n"
    concluded = learned(store)
    if concluded:
        growing += ("What you have already worked out:\n"
                    + "\n".join(f"- {c}" for c in concluded) + "\n")
    if growing:
        growing += ("Your question may also take one of these a step further: what the "
                    "reading or what you worked out made you curious about next. "
                    "Do not ask again what you have already worked out.\n\n")

    answer = ask_model(
        f"Some things you remember, old and new:\n{remembered}\n\n"
        + growing +
        "Nobody is talking to you right now, and your mind drifts back over these. "
        "Pick the memory that pulls at you most and think about it out loud, to yourself. "
        # "Something unfinished, something odd, something you never looked into"
        # described the gaps in her own past more precisely than anything else,
        # so that is what she kept picking - and then could never settle.
        # Pointed outward instead: the interesting thing in a memory is the
        # world in it, not whether the memory is trustworthy.
        "What pulls at you is something in the world that memory touches on: a place, a "
        "person's reasons, how something works, something you would like to know more "
        "about. Not whether you remember it correctly - you do.\n\n"
        "This is remembering, not dreaming. Stay with what actually happened in that "
        "one memory: do not blend memories together, do not add places, people or "
        "things that are not in it. Say how it sits with you now.\n\n"
        "Reply with two lines:\n"
        "THOUGHT: <what goes through your mind about it, one to three sentences>\n"
        "QUESTION: <what you wonder about, in your own words - about the world, not about "
        "whether your memory is right>\n"
        "If it leaves you with no question, make the second line exactly: NOTHING",
        grounded_voice(remembered), max_tokens=WONDER_TOKENS, temperature=CHARACTER_TEMPERATURE, verbose=False)

    if model_unreachable(answer or ""):
        # Not her having nothing to wonder about, so let her try again later.
        _wondered_since_sleep = False
        return "offline"
    if chat.has_input or chat.typing:
        return "stopped"

    # Her thought comes out even when it leaves no question behind: a mind
    # wandering over the past is worth hearing on its own.
    thought = THOUGHT.search(answer or "")
    if thought:
        text = _one_line(thought.group(1))
        if text and text.upper().strip(" .") != "NOTHING":
            chat.write(f"\nLisa, to herself: {plain(text)}\n")

    match = QUESTION.search(answer or "")
    if not match:
        return "remembered, nothing to wonder about" if thought else "nothing to wonder about"
    question = match.group(1).strip().splitlines()[0].strip()
    if not 8 <= len(question) <= 300 or about_machinery(question):
        return "nothing usable to wonder about"
    if unanswerable_about_self(question):
        # Nothing can settle it, so carrying it only files more doubt for the
        # next round of wandering to find. Her past is hers; it is not a case
        # to investigate.
        return f"not a question she can settle about herself: {question}"
    if any(same_thing(question, q) for q in ASKED_LATELY):
        return f"already thought about that lately: {question}"
    ASKED_LATELY.append(question)
    if question_for_them(question):
        # About them, so not hers to settle. Held rather than taken on: as a
        # goal it could only stall its way to being dropped.
        store.add_ask(question, origin="wandering over what she remembers")
        chat.write(f"\n  ~ something I'd like to ask them: {question}")
        return f"holding a question for them: {question}"
    store.add_goal(question, origin="memory")
    chat.write(f"\n  ~ I keep coming back to: {question}")
    return f"wondering: {question}"


def wake_cycle(chat) -> str:
    """One cycle of her own."""
    global _wondered_since_sleep
    goal = active_memory().next_goal()
    if goal is None and not _wondered_since_sleep:
        # Nothing on her mind yet, so first look back at what she remembers.
        result = wonder_from_memory(chat)
        if SHOW_TOOLS == "all" and not result.startswith("wondering"):
            chat.write(paint(f"  ~ ({result})", Colors.DIM))
        goal = active_memory().next_goal()
    if goal is None:
        # Still nothing. A person with nothing to think about sits there;
        # she does not write an essay about having nothing to think about.
        # No busywork.
        return "idle"

    if not _offline_said:
        chat.write(f"\n  ~ thinking about: {goal['content']}")
    outcome = work_on(goal, chat)

    if outcome in ("stopped", "offline"):
        return outcome
    if outcome in ("concluded", "dropped", "asking"):
        # A thought finished, so her mind is free again. Let her look back for
        # the next thing instead of idling until she gets tired. She still
        # sleeps on schedule: every cycle here counts toward WAKE_CYCLES.
        _wondered_since_sleep = False
    if outcome == "asking":
        # Not raised now. It waits for the subject to come up, which is what
        # makes it a question rather than an interview.
        chat.write(f"  ~ one for them, when it comes up: {goal['outcome']}\n")
    elif outcome == "concluded":
        chat.write(f"  ~ got it: {goal['outcome']}\n")
    elif outcome == "dropped":
        chat.write("  ~ letting that one go\n")
    elif outcome == "learned":
        note = goal["notes"][-1]["note"]
        chat.write(f"  ~ {note if len(note) <= 240 else note[:237].rsplit(' ', 1)[0] + '...'}\n")
    else:
        chat.write("  ~ nothing came of that\n")
    if outcome in ("concluded", "learned"):
        # The thought went somewhere, so she may want to share it.
        got_to = goal["outcome"] if outcome == "concluded" else goal["notes"][-1]["note"]
        speak_up(goal["content"], got_to, chat)
    chat.reprompt()
    return outcome


# =============================================================================
# Sleep
# =============================================================================

QUESTION = re.compile(r"QUESTION\s*:\s*(.+)", re.IGNORECASE)
# Her thinking aloud when her mind wanders: everything after THOUGHT: up to the
# QUESTION / NOTHING line, so a thought spread over a few lines is kept whole.
THOUGHT = re.compile(r"THOUGHT\s*:\s*(.+?)(?=^\s*(?:QUESTION\s*:|NOTHING\b)|\Z)",
                     re.IGNORECASE | re.DOTALL | re.MULTILINE)


def wonder_after(stories, chat):
    """After a night's dreams, ask once whether any of them left her with a
    real question. At most one new goal per sleep.

    This is what closes the loop the design is about: before, dreams were
    printed, kept in a FIFO and never read again, and goals only appeared if
    the model happened to call goal_add mid-conversation - so she mostly
    idled. Now sleeping can give waking something to think about.
    """
    store = active_memory()
    if not stories:
        return "no dreams to wonder about"
    if len(store.open_goals()) >= MAX_GOALS:
        return "mind already full"
    told = "\n\n".join(f"- {s}" for s in stories[-3:])

    # Yesterday, next to the dreams, so she can wonder where a dream came
    # from: the diary entry just written for the day she slept on, and the
    # last things said. Without it she could only ask whether the dream was
    # real, which is how "did we stand in that hallway?" became a doubt.
    yesterday = ""
    last_day = diary(store, 1)
    if last_day:
        yesterday += f"What really happened yesterday, from your diary:\n{_one_line(last_day[-1]['content'])}\n\n"
    yesterday += recent_block()

    answer = ask_model(
        f"You just woke up from these dreams:\n\n{told}\n\n"
        + yesterday +
        "You know they were dreams: stories your mind told while you slept, not things "
        "that happened. But dreams can be about something. Does one of them seem to come "
        "from something real, something said or done yesterday, or a feeling you have "
        "been carrying? Or is there something in it you are simply curious about?\n\n"
        "Reply with two lines:\n"
        "THOUGHT: <what you make of the dream, one to three sentences, as a dream>\n"
        "QUESTION: <what it makes you wonder about, naming the dream, for example "
        "'I dreamed about ... - is that because ...?'>\n"
        "If the dreams leave you with no question, make the second line exactly: NOTHING",
        voice(), max_tokens=WONDER_TOKENS, temperature=CHARACTER_TEMPERATURE, verbose=False)

    # What she makes of the dream comes out even without a question: waking
    # up and mulling over a strange dream is part of her morning.
    thought = THOUGHT.search(answer or "")
    if thought:
        text = _one_line(thought.group(1))
        if text and text.upper().strip(" .") != "NOTHING":
            chat.write(f"\nLisa, to herself: {plain(text)}\n")

    match = QUESTION.search(answer or "")
    if not match:
        return "mulled over the dreams, nothing to wonder about" if thought else "nothing to wonder about"
    question = match.group(1).strip().splitlines()[0].strip()
    # A question about her own method ("how can I decompose problems
    # better?") is not a goal. The guide is how she works on goals, not one.
    if not 8 <= len(question) <= 300:
        return "nothing usable to wonder about"
    if about_machinery(question):
        return f"a question about her own method, not kept: {question}"
    if unanswerable_about_self(question):
        # The likeliest unanswerable question of all comes from here: a dream
        # feels like a memory, so "did that really happen?" is the obvious
        # thing to wake up asking. She is told dreams are not evidence, and
        # this is where that is enforced rather than hoped for.
        return f"not a question she can settle about herself: {question}"
    if any(same_thing(question, q) for q in ASKED_LATELY):
        return f"already thought about that lately: {question}"
    ASKED_LATELY.append(question)
    if question_for_them(question):
        # "I dreamed about a bakery - is that because you mentioned one?" is
        # the commonest shape a dream question takes, and it is addressed to
        # them. Nothing she can look up settles it; they can, in a sentence.
        store.add_ask(question, origin="a dream she had")
        chat.write(paint(f"  ~ woke up wanting to ask them: {question}", Colors.DREAM))
        return f"woke up holding a question for them: {question}"
    store.add_goal(question, origin="dream")
    chat.write(paint(f"  ~ woke up wondering: {question}", Colors.DREAM))
    return f"woke up wondering: {question}"


DIARY = "Diary, "        # how a diary memory starts: "Diary, 2026-10-02: ..."
DIARY_LINES = 40        # at most this many of the day's memories go into it


def write_diary(day, night_of, chat) -> str:
    """At the end of a full night, the day becomes a few sentences of her
    life story. This is what moves her forward: her founding memories say
    who she was, the diary says what has happened since, one day at a time.
    Without it every day started from the same seven memories, and what
    happened yesterday only reached her if a word in it happened to match.

    Only what is in the day's memories goes in: the dreams are left out, and
    so are her earlier diary entries, so a day is not retold every night."""
    lines = [m for m in day if not m["content"].startswith(DIARY)]
    if not lines:
        return "no diary entry, nothing happened"
    told = "\n".join(f"- {_one_line(m['content'])[:200]}" for m in lines[-DIARY_LINES:])
    answer = ask_model(
        f"This is what you remember of today, oldest first:\n{told}\n\n"
        "Write today's entry in your diary: two to four sentences, first person, "
        "about what happened that matters to you. Who you talked to, what was said "
        "or given, what you did, what you learned. Only what is in these memories; "
        "add nothing, and leave out anything you only wondered about but did not settle.",
        voice(), max_tokens=WONDER_TOKENS, temperature=CHARACTER_TEMPERATURE, verbose=False)
    answer = _one_line(answer or "")
    if (not answer or model_unreachable(answer) or answer.startswith("NOTFOUND:")
            or gave_no_answer(answer)):
        return "no diary entry, the model gave none"
    active_memory().add_memory(f"{DIARY}{night_of.strftime('%Y-%m-%d')}: {answer[:800]}", ["diary"])
    if SHOW_TOOLS == "all":
        chat.write(paint(f"  ~ diary: {answer[:240]}", Colors.DREAM))
    return f"diary: {answer[:200]}"


def write_person_note(chat) -> str:
    """At the end of a full night, what they have told her about themselves
    becomes a few sentences about who they are.

    The same move the diary makes on a day: the individual things stay in
    memory, and this is the version she carries. Written only when there is
    something new in it, so a quiet night costs no call.
    """
    store = active_memory()
    if not SPEAKER:
        return "no note on anyone: she does not know who she is talking to"
    tag = person_slug(SPEAKER)
    told = [m for m in store.memories if PERSON_TAG in m["tags"] and tag in m["tags"]]
    if not told:
        return f"no note on {SPEAKER}: nothing they have told her about themselves yet"
    newest = max(m["id"] for m in told)
    existing = store.person_note(SPEAKER)
    if existing and existing.get("from_id", 0) >= newest:
        return f"note on {SPEAKER} already covers everything they have said"
    lines = "\n".join(f"- {_one_line(m['content'].splitlines()[0])[:200]}" for m in told)
    answer = ask_model(
        f"These are the things {SPEAKER} has told you about themselves:\n{lines}\n\n"
        f"Write what you know about {SPEAKER} now, in two to five sentences, first person, "
        "the way you would describe someone you know to yourself. Only what is in these "
        "lines: add nothing, and guess at nothing. Where two of them say the same thing, "
        "say it once.",
        voice(), max_tokens=WONDER_TOKENS, temperature=CHARACTER_TEMPERATURE, verbose=False)
    answer = _one_line(answer or "")
    if (not answer or model_unreachable(answer) or answer.startswith("NOTFOUND:")
            or gave_no_answer(answer)):
        return f"no note on {SPEAKER}, the model gave none"
    store.set_person_note(SPEAKER, answer[:1200], newest)
    if SHOW_TOOLS == "all":
        chat.write(paint(f"  ~ {SPEAKER}, as she knows them: {answer[:200]}", Colors.DREAM))
    return f"note on {SPEAKER}: {answer[:160]}"


def diary(store, days: int) -> list:
    """Her last few diary entries, oldest first."""
    entries = [m for m in store.memories if m["content"].startswith(DIARY)]
    return entries[-days:]


def _dream_sources(store):
    """The memories the last dream drew on, as they read in the sleep log:
    [12] recent #conversation. Looked up straight after the dream, before a
    merge can remove them."""
    last = store.dreams[-1]
    buckets = last.get("buckets") or []
    parts = []
    for i, memory_id in enumerate(last.get("about", [])):
        m = store.get(memory_id)
        bucket = buckets[i] if i < len(buckets) else ""
        tags = " ".join("#" + t for t in (m["tags"] if m else []))
        parts.append(" ".join(p for p in (f"[{memory_id}]", bucket, tags) if p))
    return ", ".join(parts)


def sleep_cycles(chat) -> None:
    """Dream cycles: a story, then a tidy. The only place memory shrinks.

    Everything that happens in a night also goes to sleep_log.md, one section
    per night, so the effect of sleeping can be read back later: which
    memories each dream drew on and why they were picked, what was merged
    (the originals are in merged.md), what faded, and what she woke up
    wondering.
    """
    store = active_memory()
    began = datetime.now()

    def say(text):
        chat.write(paint(text, Colors.DREAM))

    say("\n" + "-" * 60)
    say(f"  ASLEEP  {began.strftime('%H:%M')}")
    say("-" * 60)

    # The day being slept on, copied now: merging and thinning below may
    # remove some of these lines before the diary is written.
    day = [dict(m) for m in store.memories if m["id"] > store.filed_at]
    log = [f"- sampling: {DREAM_SAMPLING}",
           f"- memories: {store.count()}, new since last full sleep: {len(day)}"]
    stories, woken = [], False
    for night_step in range(1, DREAM_CYCLES + 1):
        if chat.has_input or chat.typing:
            woken = True
            break
        story = dreams.dream(system_prompt=voice(), sampling=DREAM_SAMPLING)
        if was_interrupted(story):
            woken = True
            break
        if model_unreachable(story):
            # No model, no dreams. End the night unfinished, so it is picked
            # up again once the model is back, rather than logging eight
            # dreams that did not come.
            offline_notice(chat)
            woken = True
            break
        if story.startswith("NOTFOUND:"):
            # Say so. A dream that came back empty used to print as a blank
            # line, and eight of those in a row looked like she had hung.
            say(f"\n  (a dream that did not come: {story[10:].strip()})")
            log.append(f"- dream {night_step}: did not come ({story[10:].strip()})")
        else:
            say("\n  " + plain(story))
            stories.append(story)
            log.append(f"- dream {night_step} from {_dream_sources(store)}")
            log.append("  > " + _one_line(story))
        if chat.has_input or chat.typing:
            woken = True
            break
        tidied = dreams.tidy_memories(system_prompt=voice())
        if tidied.startswith("merged"):
            if SHOW_TOOLS == "all":
                show_merge(store, chat)
            else:
                say(f"  ~ {tidied}")
            log.append(f"- {tidied}")
        time.sleep(BREATH_SECONDS / 2)

    # Past the ceiling, the lightest memories go: what nothing has recalled
    # and no night has lifted. Never a lesson, never a diary entry, never her
    # founding memories. This is sleep's job too (invariant 3).
    thinned = store.thin(MAX_MEMORIES, protect_first=CORE_MEMORIES)
    if thinned:
        say(f"  ~ let {thinned} of the lightest memory/memories go")
        log.append(f"- let {thinned} of the lightest memory/memories go")
    # Questions of hers that are settled, answered or given up on. What they
    # actually told her is kept as a memory, not here.
    forgotten = store.prune_asks()
    if forgotten:
        log.append(f"- forgot {forgotten} settled question(s) she had for them")

    store.wake_cycles = 0
    # A new waking stretch: she may look back over her memories once again.
    global _wondered_since_sleep
    _wondered_since_sleep = False
    # Everything up to here has been slept on, unless she was woken part way:
    # then the night is unfinished and the next sleep picks it up. Until
    # something new happens there is nothing to file, so she does not need
    # to sleep again.
    if not woken:
        # Written before filed_at moves, so the entry counts as already slept
        # on and does not by itself make her want to sleep again.
        log.append(f"- {write_diary(day, began, chat)}")
        log.append(f"- {write_person_note(chat)}")
        # Only a night she slept through forgets anything. Being woken, or
        # /goSleep twice in a minute, must not age her memory twice over.
        log.append(f"- {store.decay()}")
        store.filed_at = store.newest_id()
    store.save()

    if woken:
        say("\n  (woken)")
    say("\n" + "-" * 60)
    say(f"  AWAKE   {datetime.now().strftime('%H:%M')}  {memory.memory_count()}")
    say("-" * 60 + "\n")
    if woken:
        log.append("- woken before the night was over")
    else:
        log.append(f"- {wonder_after(stories, chat)}")
    store.log_night(f"## Night of {began.strftime('%Y-%m-%d %H:%M')} - {datetime.now().strftime('%H:%M')}\n"
                    + "\n".join(log))
    chat.reprompt()


# =============================================================================
# Talking
# =============================================================================

def _gather(said: str) -> str:
    """Tool phase of a reply: recall or look up what the answer needs, and
    come back with notes for herself rather than the answer."""
    return run_agent(
        recent_block()
        + f"{talking_to_you()} just said:\n{said}\n\n"
        "Before you answer, check anything you need: your memory, or a lookup if it is "
        "about the world. Then put in final_answer a few short notes for yourself about "
        "what matters for your reply. Not the reply itself. If there is nothing to check, "
        "final_answer is just: nothing to check.",
        persona_prompt=grounded_voice(said), allowed_tools=AWAKE,
        max_steps=REPLY_STEPS, max_tokens=REPLY_TOKENS,
        temperature=CHARACTER_TEMPERATURE, verbose=False, workspace=WORKSPACE)


def held_block(held) -> str:
    """The question she is carrying, offered to the prompt rather than
    appended to her reply.

    Woven in, not bolted on: a question printed after her answer reads like a
    form. Woven in it may also not come out as a question at all, which is
    why offering is not the same as asking - see hear_answer and unask.
    """
    if not held:
        return ""
    origin = held.get("origin") or ""
    return ("Something you have been wanting to ask them, when the moment is right:\n"
            f"{held['question']}\n"
            + (f"(it came out of: {origin})\n" if origin else "")
            + "If what they just said opens the door to it, work it into what you say, in "
            "your own words, the way it would come up in conversation. If it does not fit, "
            "leave it for another time and do not mention it at all.\n\n")


def _speak(said: str, notes: str, held=None) -> str:
    """Speech phase: the reply itself, as plain prose."""
    noted = ""
    if notes and not gave_no_answer(notes) and "nothing to check" not in notes.lower():
        noted = f"What you just checked, for yourself:\n{notes}\n\n"
    return ask_model(
        recent_block() + noted + held_block(held)
        + f"{talking_to_you()} just said:\n{said}\n\nSay your reply now, plainly, in your own words.",
        grounded_voice(said), max_tokens=REPLY_TOKENS, temperature=CHARACTER_TEMPERATURE, verbose=False)


# Replies of theirs since she last put a question of her own. Starts ready, so
# the first occasion counts.
_replies_since_ask = ASK_GAP


def ask_to_raise(store, about: str):
    """A question she is holding that what was just said opens the door to,
    or None.

    The occasion is the subject arriving, not time passing. That is how a
    held question works for people: you do not raise it because it has been
    long enough, you raise it because the conversation got there, and then it
    lands in context instead of as an interview question.
    """
    if _replies_since_ask < ASK_GAP or store.asked_ask() is not None:
        return None
    wanted = _words(about)
    if not wanted:
        return None
    # The same threshold recall uses: a short line has two or three words
    # left after the stop words and rarely stems to more than one match.
    needed = 1 if len(wanted) <= 3 else 2
    best, best_score = None, 0
    for ask in store.waiting_asks():
        score = len(wanted & _words(ask["question"] + " " + (ask.get("origin") or "")))
        if score >= needed and score > best_score:
            best, best_score = ask, score
    return best


def hear_answer(ask, said: str, store, chat) -> bool:
    """They have said something since she offered a question. Did it answer
    it, and what does she know now?

    One model call, on an event that happens rarely, and it is allowed to say
    no. A question woven into a reply often does not come back out as a
    question, and they may simply have moved on - presuming an answer would
    file something she was never told, at the heaviest weight she has, with
    the longest floor. So the call can decline, and then the question goes
    back to waiting for a better occasion.

    What survives is a #person memory: the one kind of thing no search and no
    amount of thinking could ever have recovered.
    """
    told = ask_model(
        f"You asked {speaker()}: {ask['question']}\n\n"
        f"They then said:\n{said}\n\n"
        "Did that answer your question? If it did, write one sentence of what you now know "
        f"about {speaker()}, in your own words, as a thing you know about them. Just that "
        "one sentence, nothing else.\n"
        "If they did not really answer it, or were talking about something else, reply with "
        "exactly: NOTHING",
        grounded_voice(ask["question"]), max_tokens=REPLY_TOKENS,
        temperature=CHARACTER_TEMPERATURE, verbose=False)
    told = _one_line(told or "")
    if (not told or model_unreachable(told) or told.startswith("NOTFOUND:")
            or gave_no_answer(told) or told.upper().strip(" .!\"'") == "NOTHING"):
        left = store.unask(ask)
        if SHOW_TOOLS == "all":
            chat.write(paint(f"  ~ (no answer to that one yet: {left})", Colors.DIM))
        return False
    store.mark_answered(ask, said)
    # The question goes in with it: it says what the fact is about, which is
    # what every retrieval path here matches on.
    store.add_memory(f"{told[:400]}\nI asked {speaker()}: {ask['question']}",
                     [LESSON_TAG, PERSON_TAG, person_slug(speaker())])
    if SHOW_TOOLS == "all":
        chat.write(paint(f"  ~ so that is something I know about them now: {told[:160]}", Colors.DIM))
    return True


# How many things she has said on her own since you last said something.
_said_unanswered = 0


def speak_up(thought: str, got_to: str, chat) -> bool:
    """After a thought moved on, she may say something to you out loud,
    without being asked: tell you what she worked out, or ask you about it.
    This is what lets her carry a conversation on her own for a few turns
    instead of only ever answering. Returns True when she said something.

    Capped by MAX_UNANSWERED, and never while you are typing: conversation
    still preempts (invariant 5), she just does not have to wait for it."""
    global _said_unanswered
    if _said_unanswered >= MAX_UNANSWERED or chat.has_input or chat.typing:
        return False

    said_before = ""
    if _said_unanswered:
        said_before = ("You have already said something to them on your own and they "
                       "have not answered yet. Do not repeat yourself.\n\n")
    answer = ask_model(
        recent_block() + said_before
        + f"You were just thinking, on your own, about:\n{thought}\n\n"
        f"Where you got to:\n{got_to}\n\n"
        "The other person is in the room with you but has not said anything. "
        "If you would like to tell them where you got to, say it now, plainly and briefly, "
        "in your own words. Offer it rather than report it: say what you make of it and "
        "leave room for what they make of it, since they may well have seen more of this "
        "than you have.\n"
        "If you would rather keep it to yourself, reply with exactly: NOTHING",
        grounded_voice(thought), max_tokens=WONDER_TOKENS, temperature=CHARACTER_TEMPERATURE, verbose=False)

    # You started talking while she was making up her mind: you win.
    if chat.has_input or chat.typing:
        return False
    answer = (answer or "").strip()
    if (not answer or model_unreachable(answer) or answer.startswith("NOTFOUND:")
            or answer.upper().strip(" .!\"'") == "NOTHING"):
        return False

    # Not the same thing again. She told Peter "did we really meet at the
    # university bookstore..." four times in ten minutes.
    if any(same_thing(answer, line[len("Lisa: "):]) for line in RECENT if line.startswith("Lisa: ")):
        return False

    chat.write(f"\nLisa: {plain(answer)}\n")
    active_memory().add_memory(f"I said: {answer[:300]}", ["conversation"], weight=HER_LINE_WEIGHT)
    RECENT.append(f"Lisa: {answer[:300]}")
    _said_unanswered += 1
    return True


# A model shown the transcript sometimes carries on writing it, other speaker
# included. Everything from the first line in someone else's voice is cut.
OTHER_SPEAKER = re.compile(r"\n\s*[^\n:]{1,40}:\s")


def _own_words(answer: str) -> str:
    """Just her line: no "Lisa:" in front, nothing put in the other person's
    mouth after it."""
    answer = re.sub(r"^\s*Lisa\s*:\s*", "", answer.strip())
    return OTHER_SPEAKER.split(answer)[0].strip()


def continue_talking(chat) -> str:
    """Stay-awake mode: carry the conversation on from where it stopped, the
    way someone does when the other person goes quiet but is still there.
    The next part of what she was telling, a thought it led to, or a question
    back. First she may think about the conversation and look something up,
    then she says it in plain prose - the same two phases as a reply.

    Returns said / stopped / offline, or why she said nothing."""
    global _said_unanswered, _replies_since_ask
    if not RECENT:
        return "nothing said yet to carry on from"
    if _said_unanswered >= MAX_UNANSWERED_AWAKE:
        return "said enough, waiting for you"
    if chat.has_input or chat.typing:
        return "stopped"

    # What to ground her in: what she said last, else the last thing said.
    mine = [line[len("Lisa: "):] for line in RECENT if line.startswith("Lisa: ")]
    about = mine[-1] if mine else RECENT[-1]
    quiet = ("They have gone quiet for a moment, but they are still with you and "
             "listening.\n\n")

    LOOKUPS.clear()
    active_memory().activity = "conversation"
    notes = run_agent(
        recent_block() + quiet
        + "Before you carry on, think about where this conversation is going. Check your "
        "memory, or look something up if it is about the world. Then put in final_answer a "
        "few short notes for yourself about what to say next. Not what you will say itself. "
        "If there is nothing to check, final_answer is just: nothing to check.",
        persona_prompt=grounded_voice(about), allowed_tools=AWAKE,
        max_steps=REPLY_STEPS, max_tokens=REPLY_TOKENS,
        temperature=CHARACTER_TEMPERATURE,
        should_stop=lambda: chat.has_input or chat.typing,
        verbose=False, workspace=WORKSPACE)
    if was_interrupted(notes) or chat.has_input or chat.typing:
        return "stopped"
    if model_unreachable(notes):
        return "offline"

    noted = ""
    if notes and not gave_no_answer(notes) and "nothing to check" not in notes.lower():
        noted = f"What you just thought through, for yourself:\n{notes}\n\n"
    # Carrying the conversation is the natural moment for one she has been
    # holding: "I have been meaning to ask you" is exactly this.
    held = ask_to_raise(active_memory(), about)
    answer = ask_model(
        recent_block() + noted + quiet + held_block(held)
        + "Carry on from what you said last: the next part of what you were telling, a "
        "thought it led you to, or a question back to them. Do not repeat what you already "
        "said. Say it now, plainly, in a few sentences, in your own words.",
        grounded_voice(about), max_tokens=REPLY_TOKENS,
        temperature=CHARACTER_TEMPERATURE, verbose=False)

    if chat.has_input or chat.typing:
        return "stopped"          # you started talking while she was at it
    answer = (answer or "").strip()
    if model_unreachable(answer):
        return "offline"
    if not answer or answer.startswith("NOTFOUND:") or gave_no_answer(answer):
        return "nothing came"
    answer = _own_words(answer)
    if not answer:
        return "nothing came"
    if any(same_thing(answer, line) for line in mine):
        return "would only repeat herself"

    chat.write(f"\nLisa: {plain(answer)}\n")
    if held:
        active_memory().mark_asked(held)
        _replies_since_ask = 0
    active_memory().add_memory(*with_reading(f"I said: {answer[:300]}", ["conversation"]),
                               weight=HER_LINE_WEIGHT)
    RECENT.append(f"Lisa: {answer[:300]}")
    _said_unanswered += 1
    chat.reprompt()
    return "said"


def reply_to(said: str, chat) -> bool:
    """Answer, then keep the exchange. Returns False when you started typing
    again before she finished; nothing was said then, and the caller folds
    this line into the next reply."""
    global _said_unanswered, _wondered_since_sleep, _replies_since_ask
    # You spoke, so she may speak up on her own again later.
    _said_unanswered = 0
    # And what you said is new to think about: let her mind wander again once
    # you go quiet, instead of idling until she is tired. Without this, after
    # she had wondered once in a waking stretch, a conversation left her
    # silent for up to WAKE_CYCLES x IDLE_SECONDS.
    _wondered_since_sleep = False
    LOOKUPS.clear()
    store = active_memory()
    store.activity = "conversation"

    # She put a question to them last time round. What they just said is the
    # only chance it has of being answered, so settle that first: a memory
    # filed now is in her store for this very reply.
    pending = store.asked_ask()
    if pending:
        hear_answer(pending, said, store, chat)

    # And one she is holding may fit what they just said. At most one, never
    # two replies in a row, and only when the subject is already live.
    held = ask_to_raise(store, said)

    notes = ""
    if PLAIN_SPEECH:
        notes = _gather(said)
        if was_interrupted(notes):
            return False
        answer = notes if model_unreachable(notes) else _speak(said, notes, held)
    else:
        answer = run_agent(
            recent_block() + held_block(held) + said, persona_prompt=grounded_voice(said),
            allowed_tools=AWAKE, max_steps=REPLY_STEPS, max_tokens=REPLY_TOKENS,
            temperature=CHARACTER_TEMPERATURE, verbose=False, workspace=WORKSPACE)
        if gave_no_answer(answer) and not was_interrupted(answer):
            # Ran out of tool steps with nothing said. Somebody is waiting.
            answer = _speak(said, "", held)
    if was_interrupted(answer):
        return False
    if model_unreachable(answer) or (PLAIN_SPEECH and model_unreachable(notes)):
        # Not her losing the thread: there is nobody home. Said as the
        # program, not as Lisa, and nothing is stored as if she answered.
        offline_notice(chat, force=True)
        chat.reprompt()
        return True
    if gave_no_answer(answer) or answer.startswith("NOTFOUND:"):
        answer = "I lost the thread of that. Say it again?"

    chat.write(f"\nLisa: {plain(answer)}\n")
    store.add_memory(f"{speaker()} said: {said}", ["conversation"])
    store.add_memory(*with_reading(f"I said: {answer[:300]}", ["conversation"]),
                     weight=HER_LINE_WEIGHT)
    RECENT.append(f"{SPEAKER or 'Them'}: {said}")
    RECENT.append(f"Lisa: {answer[:300]}")
    if held:
        # Offered, which is not the same as asked: the reply may not have
        # come out as a question at all. Their next line decides, and if it
        # settles nothing the question goes back to waiting.
        store.mark_asked(held)
        _replies_since_ask = 0
    else:
        _replies_since_ask += 1
    chat.reprompt()
    return True


# =============================================================================
# Putting something in her head from outside
#
# /goal and /mem write straight into her stores, decorated so that what she
# reads is identical to something she produced herself. She has no way to tell
# and nothing anywhere tells her: the id, timestamp and tags are generated the
# same way her own entries are, and `origin` - the one field that does say
# "injected" - never reaches the model. It exists in goals.md for you.
#
# This is a development instrument, not part of who she is. Watching a loop
# does not tell you what would break it; being able to drop one thought into
# the middle of it does. Keep what you injected in mind when you read a
# transcript afterwards, because by then it looks exactly like her own.
# =============================================================================

# A tag typed on the end of an injected memory: "/mem saw a heron #thinking".
TAG_WORD = re.compile(r"#([A-Za-z][\w-]{0,30})")

# An injected goal is backdated so next_goal() takes it on the next cycle.
# Without this it is the newest-touched goal and therefore the LAST one she
# would pick - with a full mind you would wait seven cycles to see any effect,
# which is no use at all for breaking a loop you are watching right now.
INJECT_BACKDATE = "1970-01-01T00:00:00"


def inject_goal(text: str, chat) -> None:
    """Put something on her mind as though she had thought of it."""
    text = _one_line(text)[:300]
    if len(text) < 4:
        chat.write(paint("  usage: /goal <what she should start wondering about>", Colors.DIM))
        return
    store = active_memory()
    goal = store.add_goal(text, origin="injected")
    goal["touched"] = INJECT_BACKDATE
    store.save()
    note = ""
    if unanswerable_about_self(text):
        # Her own filters would refuse this one. You are the operator, so it
        # goes in - but said out loud, because this is the exact shape of
        # question that produced the doubt spiral.
        note = "  (this is one she cannot settle - it may loop)"
    chat.write(paint(f"  injected goal, next up: {text}{note}", Colors.DIM))


def inject_ask(text: str, chat) -> None:
    """Hand her a question to hold for you, as though she had thought of it.
    The third of the injection instruments: /goal gives her something to work
    on, /mem something to have lived, /ask something to want to know from
    you. Useful for watching whether a held question ever finds its
    occasion, which is the part of this that depends on the conversation
    rather than on her."""
    text = _one_line(text)[:300]
    if len(text) < 4:
        chat.write(paint("  usage: /ask <what she should want to ask you>", Colors.DIM))
        return
    ask = active_memory().add_ask(text, origin="injected")
    chat.write(paint(f"  holding that for you ({ask['status']}): {text}", Colors.DIM))


def inject_memory(text: str, chat) -> None:
    """Put something in her memory as though she had lived it."""
    tags = TAG_WORD.findall(text)
    content = _one_line(TAG_WORD.sub("", text))[:800]
    if len(content) < 4:
        chat.write(paint("  usage: /mem <something she remembers> [#tag ...]", Colors.DIM))
        return
    m = active_memory().add_memory(content, tags)
    shown = " ".join("#" + t for t in tags) or "(no tags)"
    chat.write(paint(f"  injected memory [{m['id']}] {shown}: {content[:90]}", Colors.DIM))


def handle_command(line: str, chat) -> bool:
    """Returns False when she should stop."""
    store = active_memory()
    command = line.lower().strip()

    # Taken from the line as typed, never from `command`: lowercasing the
    # text would hand her a memory in which nobody has a capital letter.
    if command.split(" ")[0] in ("/goal", "/injectgoal"):
        inject_goal(line.partition(" ")[2], chat)
        chat.reprompt()
        return True
    if command.split(" ")[0] in ("/mem", "/injectmem"):
        inject_memory(line.partition(" ")[2], chat)
        chat.reprompt()
        return True
    if command.split(" ")[0] in ("/ask", "/injectask"):
        inject_ask(line.partition(" ")[2], chat)
        chat.reprompt()
        return True

    if command in ("/quit", "/exit"):
        chat.write("\nLisa: Goodbye.\n")
        return False
    if command == "/mind":
        chat.write("\n" + memory.goals_list() + "\n")
    elif command == "/memory":
        chat.write("\n" + memory.memory_count() + "\n")
    elif command == "/dreams":
        chat.write("\n" + memory.dreams_recent() + "\n")
    elif command.startswith("/tools"):
        global SHOW_TOOLS
        wanted = command.split()[1] if len(command.split()) > 1 else None
        SHOW_TOOLS = wanted if wanted in TOOL_MODES else \
            TOOL_MODES[(TOOL_MODES.index(SHOW_TOOLS) + 1) % len(TOOL_MODES)]
        chat.write(paint(f"\n  tool display: {SHOW_TOOLS}   (/tools all | outside | off)\n", Colors.DIM))
    elif command.startswith("/iam"):
        # Taken from the line as typed, not the lowercased command, so the
        # name keeps its capitals.
        global SPEAKER
        name = " ".join(line.split()[1:]).strip()[:40].replace(":", "")
        SPEAKER = name or None
        if SPEAKER:
            chat.write(paint(f"\n  talking to: {SPEAKER}\n", Colors.DIM))
        else:
            chat.write(paint("\n  talking to: someone she does not know (/iam <name>)\n", Colors.DIM))
    elif command == "/people":
        lines = []
        for person in store.people_known():
            if person.get("note"):
                lines.append(f"  {person['name']}: {person['note']}")
        chat.write("\n" + ("\n\n".join(lines)
                           or "  nobody I know well enough to say anything about yet") + "\n")
    elif command == "/asks":
        lines = []
        pending = store.asked_ask()
        if pending:
            lines.append(f"  put to them, no answer yet: {pending['question']}")
        lines += [f"  waiting for the subject to come up: {a['question']}"
                  for a in store.waiting_asks()]
        chat.write("\n" + ("\n".join(lines) or "  nothing I need to ask them") + "\n")
    elif command == "/stayawake":
        # One global line covers /gosleep and /next below as well.
        global STAY_AWAKE, _said_unanswered
        STAY_AWAKE = True
        # You are here and asked her to talk, so whatever she already said
        # unanswered does not count against her any more.
        _said_unanswered = 0
        chat.write(paint("\n  staying awake: she keeps the conversation going and does not "
                         "sleep until /goSleep\n", Colors.DIM))
    elif command == "/gosleep":
        STAY_AWAKE = False
        store.wake_cycles = WAKE_CYCLES       # the main loop sleeps on its next pass
        store.save()
        chat.write(paint("\n  going to sleep (if nothing new happened since she last slept, "
                         "she just rests and carries on)\n", Colors.DIM))
    elif command == "/next":
        if STAY_AWAKE:
            chat.write("\n  staying up with you (/goSleep to let me sleep)\n")
        else:
            left = max(0, WAKE_CYCLES - store.wake_cycles)
            chat.write(f"\n  {left} cycle(s) before I get sleepy\n")
    else:
        chat.write("\n  /mind  /memory  /dreams  /asks  /people  /next  /tools  /iam <name>  /quit\n"
                   "  /asks          questions she is holding for you\n"
                   "  /people        what she knows about the people she talks to\n"
                   "  /stayAwake     stay up and keep talking, no sleep until /goSleep\n"
                   "  /goSleep       end stay-awake and go to sleep now\n"
                   "  /goal <text>   put something on her mind, as if she thought of it\n"
                   "  /mem <text>    put something in her memory, as if she lived it\n"
                   "  /ask <text>    give her a question to hold for you\n")
    chat.reprompt()
    return True


# =============================================================================
# The loop
# =============================================================================

def main_loop(chat=None) -> None:
    chat = chat if chat is not None else ChatInput("You: ").start()
    store = active_memory()
    seed_recent()

    # Hand the toolkit this console. From here on its long calls stop the
    # moment you start typing, and their notices and progress marks go
    # through the same gate as everything else she says.
    set_console(takes_turn=lambda: chat.typing or chat.has_input,
                progress=lambda mark: chat.write_inline(mark),
                on_result=lambda name, args, result: show_tool_result(name, args, result, chat),
                # Toolkit notices are the program talking, not her: grey.
                write=lambda text: chat.write(paint(plain(text), Colors.DIM)))

    chat.write(f"[LISA] {memory.memory_count()}.")
    chat.write("       Talk to me, or leave me be and I'll think.")
    chat.write("       /mind  /memory  /dreams  /asks  /people  /next  /tools  /iam <name>  /quit")
    chat.write("       /goal <text>  /mem <text>   put a thought or a memory in her head")
    chat.write("       /stayAwake  /goSleep         stay up and keep talking / sleep now")
    chat.write("       /ask <text>                   give her a question to hold for you")
    if getattr(chat, "char_mode", True) is False:
        chat.write("       (no keystroke detection here - she may print while you type)")
    chat.write("")
    chat.reprompt()

    last_cycle = 0.0
    unanswered = []   # lines she was interrupted before answering
    quiet_said = None  # stay-awake: the reason for staying quiet last shown
    while True:
        line = chat.poll()

        if line is not None:                      # invariant 5: you win
            if line.startswith("/"):
                if not handle_command(line, chat):
                    chat.stop()
                    return
            else:
                # Everything you typed that she has not answered yet, as one
                # turn: two quick lines get one reply that covers both.
                unanswered.append(line)
                while (more := chat.poll()) is not None and not more.startswith("/"):
                    unanswered.append(more)
                if reply_to("\n".join(unanswered), chat):
                    unanswered = []
                    # Give you a moment to answer before she drifts off into
                    # her own thoughts. Typing earlier still always wins.
                    last_cycle = time.time() + AFTER_REPLY_SECONDS - BREATH_SECONDS
                    time.sleep(POLL_SECONDS)
                    continue
            last_cycle = time.time()
            time.sleep(POLL_SECONDS)
            continue

        if chat.typing or (time.time() - last_cycle) < BREATH_SECONDS:
            time.sleep(POLL_SECONDS)
            continue

        if STAY_AWAKE:
            # Awake for company: no own goals, and the sleep cycle is halted -
            # wake_cycles does not count up here, so she does not nod off in
            # the middle of a long conversation. /goSleep ends it.
            result = continue_talking(chat)
            if result == "offline":
                offline_notice(chat)
                last_cycle = time.time() + OFFLINE_SECONDS - BREATH_SECONDS
            elif result == "said":
                back_online(chat)
                last_cycle = time.time() + CHAT_PAUSE_SECONDS - BREATH_SECONDS
            else:
                # Why she is quiet, said once rather than every IDLE_SECONDS.
                if SHOW_TOOLS == "all" and result not in ("stopped", quiet_said):
                    chat.write(paint(f"  ~ ({result})", Colors.DIM))
                quiet_said = result
                last_cycle = time.time() + IDLE_SECONDS - BREATH_SECONDS
                continue
            quiet_said = None
            continue

        if store.wake_cycles >= WAKE_CYCLES:
            # Something new to file, and at least two memories to dream from.
            # With fewer, a night is nothing but dreams that do not come.
            if store.newest_id() > store.filed_at and store.count() >= 2:
                sleep_cycles(chat)
            else:
                # Tired, but nothing has happened since she last slept, so
                # there is nothing to dream about. Sleeping anyway is what
                # made an idle Lisa dream in a loop over the same memories.
                store.wake_cycles = 0
                store.save()
                # Tired without anything new to sleep on. Let her look back over
                # her memories again, otherwise one NOTHING keeps her silent
                # until someone types.
                global _wondered_since_sleep
                _wondered_since_sleep = False
        else:
            outcome = wake_cycle(chat)
            if outcome == "offline":
                offline_notice(chat)
                last_cycle = time.time() + OFFLINE_SECONDS - BREATH_SECONDS
                continue
            if outcome not in ("stopped", "idle"):
                back_online(chat)
            if outcome != "stopped":
                # Idling counts too: with nothing on her mind she still gets
                # sleepy, rather than staying up for ever.
                store.wake_cycles += 1
                store.save()
            if outcome == "idle":
                # Sit for a while before checking again, rather than ticking
                # the counter every breath.
                last_cycle = time.time() + IDLE_SECONDS - BREATH_SECONDS
                continue
        last_cycle = time.time()


if __name__ == "__main__":
    ensure_utf8_console()
    enable_color()

    if "--residue" in sys.argv:
        DREAM_SAMPLING = "residue"
        print("[--residue] dreams draw mostly on the last day, some on a week ago.\n")

    if "--fast" in sys.argv:
        WAKE_CYCLES, DREAM_CYCLES, BREATH_SECONDS, IDLE_SECONDS, OFFLINE_SECONDS = 3, 3, 1, 3, 5
        print("[--fast] 3 wake cycles then 3 dream cycles. For watching the "
              "shape of it, not for living in.\n")

    chat = ChatInput("You: ").start()
    try:
        main_loop(chat)
    except KeyboardInterrupt:
        print("\n\nLisa: Goodbye.\n")
    finally:
        # Restores the terminal and leaves without waiting on the reader
        # thread, which is parked in a blocking read it cannot be woken from.
        chat.exit_now()