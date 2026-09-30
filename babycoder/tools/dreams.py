"""
babycoder.tools.dreams

AGENT DREAM toolkit: what an agent does with memory when nobody talks to it.
The story FIRST, then the tidying: putting unrelated memories side by side is
what makes it visible that two of them say the same thing. Dreams weave from
lived memories only, never from other dreams. The only place memory shrinks.
"""

import random
import re
from datetime import datetime

from ..core import CHARACTER_TEMPERATURE, tool
from ..model import ask_model
from .memory import active_memory, consolidate

DREAM_MAX_TOKENS = 1500   # room for a reasoning model to plan before 150 words
DREAM_WORDS = 150

# Memories about HOW she thinks rather than WHAT she thinks about: the guide,
# its patterns, her own tool names. They are real memories and stay in
# memory, but they are not dream material. The guide is a method for working
# on goals, never a subject in itself (LISA_DESIGN.md), and dreaming about it
# turned her method into a topic she then kept returning to.
MACHINERY = re.compile(
    r"\b(thinking guide|the guide|read_guide|solver patterns?|decompos\w*|contradiction test\w*"
    r"|memory_\w+|goal_(?:add|close)|goals_list|dreams_recent|research_topic"
    r"|search_(?:web|wikipedia)|consolidat\w*|tidy_memories|tool calls?)\b",
    re.IGNORECASE)


def about_machinery(text):
    """True when a memory or question is about her own thinking tools."""
    return bool(MACHINERY.search(text or ""))


# How a dream picks its memories.
#   "uniform"  every memory equally likely. The baseline.
#   "residue"  weighted the way human dreams seem to be: mostly the last day
#              (what happened since the last full sleep), a second smaller
#              share from about a week back, and the odd older memory. Sleep
#              research finds both: recent events show up in the next night's
#              dreams, and again around five to seven days later.
# Kept as a switch so the two can be compared on the same agent.
SAMPLINGS = ("uniform", "residue")
RESIDUE_WEIGHTS = {"recent": 0.60, "week": 0.25, "older": 0.15}
WEEK_DAYS = (5, 9)   # a little wider than the 5-7 day lag, for small stores


def _age_days(memory, now):
    try:
        return (now - datetime.fromisoformat(memory["when"])).total_seconds() / 86400
    except (ValueError, TypeError):
        return float("inf")


def pick_memories(pool, n, sampling="uniform", filed_at=0, now=None):
    """Pick n distinct memories for a dream. Returns [(memory, bucket)], in
    memory order, where bucket says why each was picked, for the logs.

    In residue mode each pick first draws a bucket by weight, among the
    buckets that still have memories left, then a memory from it. So with no
    week-old memories yet, their share goes to the other two rather than the
    dream simply getting fewer memories.
    """
    n = min(n, len(pool))
    if sampling != "residue":
        return sorted(((m, "any") for m in random.sample(pool, n)), key=lambda p: p[0]["id"])
    now = now or datetime.now()
    buckets = {"recent": [], "week": [], "older": []}
    for m in pool:
        if m["id"] > filed_at:
            buckets["recent"].append(m)
        elif WEEK_DAYS[0] <= _age_days(m, now) <= WEEK_DAYS[1]:
            buckets["week"].append(m)
        else:
            buckets["older"].append(m)
    picked = []
    while len(picked) < n:
        open_buckets = [b for b in buckets if buckets[b]]
        bucket = random.choices(open_buckets, weights=[RESIDUE_WEIGHTS[b] for b in open_buckets])[0]
        m = buckets[bucket].pop(random.randrange(len(buckets[bucket])))
        picked.append((m, bucket))
    return sorted(picked, key=lambda p: p[0]["id"])


@tool("Dream about a few memories picked at random.")
def dream(system_prompt="", sample=3, sampling="uniform"):
    mem = active_memory()
    pool = [m for m in mem.memories if not about_machinery(m["content"])]
    if len(pool) < 2:
        return "NOTFOUND: not enough lived material to dream about yet"
    # Rest what the last two dreams used, while there is enough else to dream
    # about. With a small memory, uniform picking kept landing on the same
    # few memories, and a night became four versions of one dream.
    just_used = {i for d in mem.dreams[-2:] for i in d.get("about", [])}
    rested = [m for m in pool if m["id"] not in just_used]
    if len(rested) >= 2 * max(2, sample):
        pool = rested
    picked = pick_memories(pool, max(2, sample), sampling, mem.filed_at)
    fragments = "\n".join(f"- {m['content'][:120]}" for m, _ in picked)
    story = ask_model(
        f"You dreamt about these, all mixed up together. Say what the dream was, in at most "
        f"{DREAM_WORDS} words.\n\n{fragments}\n\n"
        # "Write a dream" produced literature. Asking her to SAY what she
        # dreamt gets the breakfast-table version instead.
        "Tell it the way you tell someone a dream the morning after: plain, a bit disjointed, "
        "no scene-setting, no quoted conversation, no ending that ties it off. It does not have "
        "to make sense and you do not explain it. It is about people, places and things, not "
        "about how you think or what you think with.\n\n"
        # The first dream in the transcript retold the day's conversation,
        # with an explanation of what Herfst means. Real dreams almost never
        # replay what happened; they take pieces of it somewhere else.
        "It is not a retelling of what happened. Pieces of these turn up in the wrong place, "
        "mixed with each other and with things that were never there.",
        system_prompt, max_tokens=DREAM_MAX_TOKENS, temperature=CHARACTER_TEMPERATURE)
    if story.startswith("NOTFOUND:"):
        return story
    mem.add_dream(story, about=[m["id"] for m, _ in picked], buckets=[b for _, b in picked])
    return story


@tool("Look for memories saying the same thing and merge them.")
def tidy_memories(system_prompt="", sample=6):
    """One anchor memory against a handful of others, not everything against
    everything. Same subject only: merging unrelated memories loses both."""
    mem = active_memory()
    if mem.count() < 3:
        return "too little to tidy"
    anchor = mem.random(1)[0]
    others = mem.random(sample, exclude=[anchor["id"]])
    listing = "\n".join(f"[{m['id']}] {m['content'][:150]}" for m in [anchor] + others)

    verdict = ask_model(
        "Here are some things I remember.\n\n" + listing + "\n\n"
        f"Are any of them about the SAME thing as [{anchor['id']}], closely enough that one "
        "sentence would remember them all just as well?\n\n"
        "If yes, reply with exactly:\nMERGE: <numbers, comma separated>\n"
        "<the one sentence that replaces them>\n\n"
        "If they are about different things, reply with exactly: KEEP",
        system_prompt, max_tokens=1200, temperature=CHARACTER_TEMPERATURE)

    match = re.search(r"MERGE:\s*([\d,\s\[\]]+)\n(.+)", verdict, flags=re.DOTALL)
    if not match:
        return "nothing worth merging this time"
    # Only numbers that were actually shown. A model inventing [42] used to
    # merge a memory it had never seen.
    shown = {anchor["id"]} | {m["id"] for m in others}
    numbers = {int(n) for n in re.findall(r"\d+", match.group(1))} & shown
    merged = match.group(2).strip()
    if len(numbers) < 2 or anchor["id"] not in numbers or not merged:
        return "nothing worth merging this time"
    return consolidate(sorted(numbers), merged)
