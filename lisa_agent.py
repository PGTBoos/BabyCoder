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
import re
import sys
import time
from collections import deque
from datetime import datetime

from babycoder import (AGENT_LISA_AWAKE, CHARACTER_TEMPERATURE, MAX_GOALS, MAX_MEMORIES,
                       active_memory, ask_model, configure_workspace, gave_no_answer,
                       model_unreachable, run_agent, set_console, settings, was_interrupted)
from babycoder import memory, dreams, research, guidance
from babycoder.tools.dreams import about_machinery
from agent_common import ChatInput, Colors, enable_color, ensure_utf8_console, paint, plain

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
OFFLINE_SECONDS = 20    # between tries when the model cannot be reached

REPLY_STEPS, REPLY_TOKENS = 4, 1024
THINK_STEPS, THINK_TOKENS = 6, 1200

STALLS_BEFORE_RESTING = 2   # passes reaching nothing before she puts it down
# Passes on one goal before she settles it with what she has. Without this, a
# model that never writes a GOT IT / DROP IT line kept her on the same
# question every cycle, which is what the transcript showed.
MAX_PASSES = 4

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
this line is her, in her own words, first person.

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


def hour_flavour() -> str:
    """Time of day, as colour on how she talks. Flavour only - it gates
    nothing. Sleep is decided by the cycle counter, not by mood."""
    hour = datetime.now().hour
    if 22 <= hour or hour < 6:
        return "It is late and she is drowsy."
    if 6 <= hour < 9:
        return "It is early and she is not properly awake yet."
    if 14 <= hour < 16:
        return "It is the flat part of the afternoon."
    return ""


def voice(extra: str = "") -> str:
    """Her system prompt. Character only - what tools she has, and how to use
    each capability well, are generated by babycoder from her grant."""
    flavour = hour_flavour()
    return LISA + (f"\n{flavour}\n" if flavour else "") + (extra or "")


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


def seed_recent():
    """Fill RECENT from the conversation lines memory kept last time."""
    RECENT.clear()
    for m in active_memory().recent(RECENT_EXCHANGES * 2, tag="conversation"):
        text = m["content"].split("\n(read about:")[0]
        if text.startswith("They said: "):
            RECENT.append("Them: " + text[len("They said: "):])
        elif text.startswith("I said: "):
            RECENT.append("Lisa: " + text[len("I said: "):])


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
VERDICT = re.compile(r"^[\s*_>\"'-]*(GOT IT|MORE|DROP IT)\s*:\s*(.*)$", re.IGNORECASE)


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
            outcome = {"GOT IT": "concluded", "DROP IT": "dropped"}.get(match.group(1).upper(), "learned")
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
        "Finish with one line, exactly one of:\n"
        "  GOT IT: <what you have concluded>\n"
        "  MORE: <what you worked out, and what is still open>\n"
        "  DROP IT: <why this is not worth carrying>"
    )

    LOOKUPS.clear()
    active_memory().activity = "thinking"
    answer = run_agent(
        prompt, persona_prompt=voice(), allowed_tools=AWAKE,
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
    if outcome == "learned" and goal.get("passes", 0) + 1 >= MAX_PASSES:
        outcome = "concluded"
        said = f"as far as I can take it for now: {said}"

    if outcome == "concluded":
        store.note_on_goal(goal, said)
        store.close_goal(goal, said)
        store.add_memory(*with_reading(f"Worked out: {said}", ["thinking"]))
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


def wake_cycle(chat) -> str:
    """One cycle of her own."""
    goal = active_memory().next_goal()
    if goal is None:
        # Nothing on her mind. A person with nothing to think about sits
        # there; she does not write an essay about having nothing to think
        # about. No busywork.
        return "idle"

    if not _offline_said:
        chat.write(f"\n  ~ thinking about: {goal['content']}")
    outcome = work_on(goal, chat)

    if outcome in ("stopped", "offline"):
        return outcome
    if outcome == "concluded":
        chat.write(f"  ~ got it: {goal['outcome']}\n")
    elif outcome == "dropped":
        chat.write("  ~ letting that one go\n")
    elif outcome == "learned":
        note = goal["notes"][-1]["note"]
        chat.write(f"  ~ {note if len(note) <= 240 else note[:237].rsplit(' ', 1)[0] + '...'}\n")
    else:
        chat.write("  ~ nothing came of that\n")
    chat.reprompt()
    return outcome


# =============================================================================
# Sleep
# =============================================================================

QUESTION = re.compile(r"QUESTION\s*:\s*(.+)", re.IGNORECASE)


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
    answer = ask_model(
        f"These are dreams you just woke up from:\n\n{told}\n\n"
        "Did any of them leave you actually wondering about something, a real question you "
        "would like to think about today? If so, reply with one line:\n"
        "QUESTION: <the question, in your own words>\n"
        "If not, reply with exactly: NOTHING",
        voice(), max_tokens=600, temperature=CHARACTER_TEMPERATURE, verbose=False)
    match = QUESTION.search(answer or "")
    if not match:
        return "nothing to wonder about"
    question = match.group(1).strip().splitlines()[0].strip()
    # A question about her own method ("how can I decompose problems
    # better?") is not a goal. The guide is how she works on goals, not one.
    if not 8 <= len(question) <= 300:
        return "nothing usable to wonder about"
    if about_machinery(question):
        return f"a question about her own method, not kept: {question}"
    store.add_goal(question, origin="dream")
    chat.write(paint(f"  ~ woke up wondering: {question}", Colors.DREAM))
    return f"woke up wondering: {question}"


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

    log = [f"- sampling: {DREAM_SAMPLING}",
           f"- memories: {store.count()}, new since last full sleep: "
           f"{sum(1 for m in store.memories if m['id'] > store.filed_at)}"]
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

    # Past the ceiling, the oldest conversation lines go. Never her own
    # thinking. This is sleep's job too (invariant 3).
    thinned = store.thin(MAX_MEMORIES)
    if thinned:
        say(f"  ~ let {thinned} old conversation line(s) fade")
        log.append(f"- let {thinned} old conversation line(s) fade")

    store.wake_cycles = 0
    # Everything up to here has been slept on, unless she was woken part way:
    # then the night is unfinished and the next sleep picks it up. Until
    # something new happens there is nothing to file, so she does not need
    # to sleep again.
    if not woken:
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
        + f"They just said:\n{said}\n\n"
        "Before you answer, check anything you need: your memory, or a lookup if it is "
        "about the world. Then put in final_answer a few short notes for yourself about "
        "what matters for your reply. Not the reply itself. If there is nothing to check, "
        "final_answer is just: nothing to check.",
        persona_prompt=voice(), allowed_tools=AWAKE,
        max_steps=REPLY_STEPS, max_tokens=REPLY_TOKENS,
        temperature=CHARACTER_TEMPERATURE, verbose=False, workspace=WORKSPACE)


def _speak(said: str, notes: str) -> str:
    """Speech phase: the reply itself, as plain prose."""
    noted = ""
    if notes and not gave_no_answer(notes) and "nothing to check" not in notes.lower():
        noted = f"What you just checked, for yourself:\n{notes}\n\n"
    return ask_model(
        recent_block() + noted
        + f"They just said:\n{said}\n\nSay your reply now, plainly, in your own words.",
        voice(), max_tokens=REPLY_TOKENS, temperature=CHARACTER_TEMPERATURE, verbose=False)


def reply_to(said: str, chat) -> bool:
    """Answer, then keep the exchange. Returns False when you started typing
    again before she finished; nothing was said then, and the caller folds
    this line into the next reply."""
    LOOKUPS.clear()
    active_memory().activity = "conversation"
    notes = ""
    if PLAIN_SPEECH:
        notes = _gather(said)
        if was_interrupted(notes):
            return False
        answer = notes if model_unreachable(notes) else _speak(said, notes)
    else:
        answer = run_agent(
            recent_block() + said, persona_prompt=voice(), allowed_tools=AWAKE,
            max_steps=REPLY_STEPS, max_tokens=REPLY_TOKENS,
            temperature=CHARACTER_TEMPERATURE, verbose=False, workspace=WORKSPACE)
        if gave_no_answer(answer) and not was_interrupted(answer):
            # Ran out of tool steps with nothing said. Somebody is waiting.
            answer = _speak(said, "")
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
    store = active_memory()
    store.add_memory(f"They said: {said}", ["conversation"])
    store.add_memory(*with_reading(f"I said: {answer[:300]}", ["conversation"]))
    RECENT.append(f"Them: {said}")
    RECENT.append(f"Lisa: {answer[:300]}")
    chat.reprompt()
    return True


def handle_command(line: str, chat) -> bool:
    """Returns False when she should stop."""
    store = active_memory()
    command = line.lower().strip()

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
    elif command == "/next":
        left = max(0, WAKE_CYCLES - store.wake_cycles)
        chat.write(f"\n  {left} cycle(s) before I get sleepy\n")
    else:
        chat.write("\n  /mind  /memory  /dreams  /next  /tools  /quit\n")
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
    chat.write("       /mind  /memory  /dreams  /next  /tools  /quit")
    if getattr(chat, "char_mode", True) is False:
        chat.write("       (no keystroke detection here - she may print while you type)")
    chat.write("")
    chat.reprompt()

    last_cycle = 0.0
    unanswered = []   # lines she was interrupted before answering
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
            last_cycle = time.time()
            time.sleep(POLL_SECONDS)
            continue

        if chat.typing or (time.time() - last_cycle) < BREATH_SECONDS:
            time.sleep(POLL_SECONDS)
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
