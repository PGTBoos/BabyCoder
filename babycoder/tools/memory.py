"""
babycoder.tools.memory

AGENT MEMORY toolkit: long-term memory, per agent, as plain markdown files in
<state_dir>/memory/ that you can read, and edit, in any text editor:

  memories.md  what happened. One section per memory:
                   ## [12] (weight 55) 2026-09-29 21:40:05 #conversation
                   They said: it is raining
               The [number] is how the tools refer to it. Keep numbers unique
               if you add entries by hand; the next free one is in state.md.
               The weight is 0-100: how much this one carries. Not every
               memory is worth the same, and before it they all were. A
               lesson she worked out starts high and has a floor; her own
               half-thoughts start low and fade. Being used lifts it, each
               night lowers it, and what is left at the bottom is what goes
               first when the store is full. Leave it off an entry you add by
               hand and its tags decide the starting value.
  goals.md     what is on her mind, open and recently closed.
  people.md    what she knows about the people she talks to. One section per
               person, rewritten during a full night from the things they have
               told her, the way a diary entry is written from a day. The
               individual #person memories stay in memories.md; this is the
               consolidation, and it is always in front of her rather than
               waiting to be recalled - who you are talking to is not
               something you look up.
  asks.md      questions she is holding for the person she talks to - things
               no amount of looking up could answer, because they are about
               him. Deliberately NOT goals: a goal is something she can make
               progress on alone, and all the machinery around one (passes,
               stalls, being dropped for going nowhere) would treat waiting
               for an answer as her failing to think. An ask just waits.
  dreams.md    the last DREAM_KEEP dreams.
  state.md     counters: wake cycles, last filed memory, next free number.
  merged.md    every merge sleep made: the originals and what replaced them.
               A merge removes its sources from memories.md, so without this
               there would be no way to judge afterwards whether sleep kept
               the meaning, lost it, or bent it.
  sleep_log.md one section per night: which memories each dream drew on,
               what was merged, what faded, what she woke up wondering.

Why markdown: Lisa is an experiment, and being able to open her mind in
Notepad++ is worth more than a database's search speed. The store stays small
(MAX_MEMORIES) so reading it whole is cheap, and adding a memory appends to
the file rather than rewriting it.

Editing while she runs is fine. Every operation first checks whether a file
changed on disk since it was last read, and rereads it if so. Only save
while she is not in the middle of writing to that same file; the window is a
few milliseconds, but it exists.

Memory numbers are stable: a merge removes its sources and adds a new number,
it never renumbers what is left, so a model holding a number from three
steps ago cannot hit the wrong memory.

An older memory.json or memory.db is imported the first time the folder is
opened, then renamed to *.migrated.

Several small retrieval tools rather than one clever search, on purpose: a
single "most relevant" search returns the same few entries every time, so
the agent circles what it said last. Oldest and random access give it range.
"""

import json
import os
import random
import re
import uuid
from datetime import datetime

from ..core import P, _soft_not_found, tool, ws

DREAM_KEEP = 10
MAX_GOALS = 7
# Questions she is holding for him at once. Small on purpose: she only gets to
# raise one when the subject is already live, so a long queue would just be a
# list of things she never gets round to asking.
MAX_ASKS = 8
# Times she may offer a question before letting it go. Offering is not asking:
# the reply she weaves it into may not come out as a question at all, and he
# may simply move on. Three goes, then it stops being on her mind.
ASK_OFFERS_BEFORE_STALE = 3
# Past this, sleep drops the oldest conversation lines (never her own
# thinking) until it is back under. Chatting adds two memories per exchange
# and merging removes at most a handful per night, so without a ceiling
# memory only ever grows.
MAX_MEMORIES = 2000
CLOSED_GOALS_KEPT = 50

# How much a memory carries, 0-100. An integer, not a fraction, because this
# file is meant to be read and edited by hand: you can see at a glance that 90
# outranks 30, and 0.0014 tells you nothing.
#
# First tag that matches decides, so the order here is the order of priority.
# A lesson is something she set out to work out and settled - the one kind of
# memory that is an answer rather than a record, so it starts highest and is
# the last to go. Her own half-thoughts on the way there start lowest: there
# are a great many of them and each is superseded by whatever she concluded.
WEIGHT_MAX = 100
WEIGHT_DEFAULT = 50
# #person outranks everything: it is something the person she talks to told
# her about himself, which no search and no amount of thinking could recover.
# Lose it and it is gone for good, unless he happens to say it again.
WEIGHT_BY_TAG = (("person", 100), ("lesson", 90), ("diary", 80),
                 ("conversation", 55), ("thinking", 30))

# Below this a memory stops decaying. What she worked out does not fade to
# nothing just because the subject has not come up in a month; a conversation
# line does.
WEIGHT_FLOOR_BY_TAG = (("person", 60), ("lesson", 40), ("diary", 30))
WEIGHT_FLOOR_DEFAULT = 0

# Being used is the only thing that raises a weight: a memory that keeps
# turning out to be relevant earns its place, without anyone judging it.
USE_BUMP = 4
# Per night, applied to everything. Counted, not clocked, like everything else
# about her nights: a night is ten wake cycles, which can pass in minutes, so
# this is deliberately a small step. Watch two or three sleeps and you can see
# it move; it takes a good many before anything is actually gone.
NIGHTLY_DECAY = 0.98


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _one_line(text):
    """Goal and note text lives on one line in goals.md."""
    return " ".join(str(text).split())


def _stem(word):
    """A crude stemmer, enough that "herons" finds "heron" and "feelings"
    finds "feeling". Strips common endings while at least three letters
    remain. Not linguistics, just forgiving search."""
    word = word.lower()
    changed = True
    while changed:
        changed = False
        for suffix in ("ing", "es", "ed", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word, changed = word[:-len(suffix)], True
                break
    return word


# Words too common to say what a memory is about. Without these, every pair
# of memories "shares words" and the merge sampler is back to picking at
# random with extra steps.
_EMPTY_WORDS = set("""a about after again all also am an and any are as at be been
before being but by can could did do doe doing done for from had ha have having
her here him hi how i if in into is it its just like made make many me more most
much my no not now of on one only or other our out over said same she should so
some such than that the their them then there these they thi thing think this
those through to too up very wa want was we well were what when where which while
who why will with would you your
got wonder""".split())

# The last line: words from the labels every lesson carries ("I had been
# wondering: ...", "How I got there: ..."). Without them here, any two lessons
# share two words whatever they are about, and the merge sampler keeps
# offering unrelated pairs. Stemming makes "wondering" into "wonder".


# "Peter said: ", "I said: ", "Diary, 2026-10-04: " - who is talking and when,
# which is not what a memory is ABOUT. Left in, every conversation line shares
# the speaker's name with every other, and that noise outranks the subject:
# a line about Scotland scored the same against an unrelated line of Peter's
# as against the other Scotland line.
_SPEAKER = re.compile(r"^(?:[^\n:]{1,40} said: |Diary, \d{4}-\d{2}-\d{2}: |Worked out: )")


def _content_words(text):
    """The stemmed words in a memory that carry its subject."""
    text = _SPEAKER.sub("", str(text).strip())
    return {w for w in (_stem(x) for x in re.findall(r"[A-Za-z']{3,}", text))
            if w not in _EMPTY_WORDS and len(w) >= 3}


def _default_weight(tags):
    """What a memory of this kind starts at."""
    tags = set(tags or [])
    for tag, weight in WEIGHT_BY_TAG:
        if tag in tags:
            return weight
    return WEIGHT_DEFAULT


def _weight_floor(tags):
    """How low a memory of this kind may decay."""
    tags = set(tags or [])
    for tag, floor in WEIGHT_FLOOR_BY_TAG:
        if tag in tags:
            return floor
    return WEIGHT_FLOOR_DEFAULT


def _clamp_weight(value, tags):
    return max(_weight_floor(tags), min(WEIGHT_MAX, int(round(value))))


def _write_atomic(path, text):
    """Write then rename, so a crash or Ctrl-C never leaves half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# The weight is optional: an entry written before weights existed, or added by
# hand without one, gets the starting value its tags imply.
MEMORY_HEADER = re.compile(r"^## \[(\d+)\]\s*(?:\(weight\s+(\d+)\)\s*)?(.*)$")
GOAL_HEADER = re.compile(r"^## (open|closed): (.*)$")
ASK_HEADER = re.compile(r"^## (waiting|asked|answered|stale): (.*)$")
PERSON_HEADER = re.compile(r"^## (\S.*?)\s*$")
DREAM_HEADER = re.compile(r"^## (\S+ \S+)(?:\s+\(from (.*)\))?\s*$")


class MemorySystem:
    """Persistent memory: what happened, what was dreamt, what is on her mind."""

    def __init__(self, folder=None):
        self.folder = str(folder or ws().memory_path)
        os.makedirs(self.folder, exist_ok=True)
        self.paths = {name: os.path.join(self.folder, f"{name}.md")
                      for name in ("memories", "goals", "asks", "people", "dreams",
                                   "state", "merged", "sleep_log")}
        self._seen = {}               # file -> mtime when we last read or wrote it
        self._memories = []           # dicts: id, when, content, tags; oldest first
        self.goals, self.dreams = [], []
        # Questions held for him. waiting -> asked -> answered, or back to
        # waiting when he does not pick it up, or stale once she has offered
        # it enough times.
        self.asks = []
        # What she knows about each person she talks to: name, the note
        # itself, when it was written and the newest memory it covers.
        self.people = []
        self.wake_cycles = 0
        # Number of the newest memory when sleep last finished. Anything newer
        # has not been slept on yet.
        self.filed_at = 0
        self._next_id = 1
        # What she is doing right now, set by the agent's loop. A goal added
        # while she does it records it as its origin (see add_goal).
        self.activity = "unknown"
        self.last_merge = None
        self._migrate()
        self._sync(force=True)

    # -- keeping in step with the files ----------------------------------------

    def _mtime(self, name):
        try:
            return os.path.getmtime(self.paths[name])
        except OSError:
            return None

    def _mark(self, name):
        self._seen[name] = self._mtime(name)

    def _sync(self, force=False):
        """Reread any file that changed on disk since we last touched it -
        which is what makes editing her memory by hand while she runs work."""
        for name, load in (("memories", self._load_memories), ("goals", self._load_goals),
                           ("asks", self._load_asks), ("people", self._load_people),
                           ("dreams", self._load_dreams), ("state", self._load_state)):
            if force or self._mtime(name) != self._seen.get(name):
                load()
                self._mark(name)
        top = max((m["id"] for m in self._memories), default=0)
        self._next_id = max(self._next_id, top + 1)

    def _read(self, name):
        try:
            with open(self.paths[name], "r", encoding="utf-8") as f:
                return f.read().splitlines()
        except OSError:
            return []

    # -- memories.md -----------------------------------------------------------

    def _load_memories(self):
        memories, current = [], None
        for line in self._read("memories"):
            header = MEMORY_HEADER.match(line)
            if header:
                rest = header.group(3).split()
                tags = [t[1:] for t in rest if t.startswith("#")]
                when = " ".join(t for t in rest if not t.startswith("#")).replace(" ", "T")
                current = {"id": int(header.group(1)), "when": when or _now(),
                           "content": [], "tags": tags,
                           "weight": int(header.group(2)) if header.group(2) else None}
                memories.append(current)
            elif current is not None:
                current["content"].append(line)
        for m in memories:
            m["content"] = "\n".join(m["content"]).strip()
            # No weight in the file: this entry predates weights or was added
            # by hand. Its tags say what kind it is, which is all a starting
            # value needs.
            if m["weight"] is None:
                m["weight"] = _default_weight(m["tags"])
        self._memories = memories

    @staticmethod
    def _format_memory(m):
        tags = " ".join("#" + re.sub(r"\s+", "_", t) for t in m["tags"])
        # A line in the content that looks like a memory header would split
        # the entry in two when read back. Indent it one space.
        body = re.sub(r"^## \[", " ## [", m["content"], flags=re.MULTILINE)
        weight = m.get("weight")
        if weight is None:
            weight = _default_weight(m["tags"])
        return (f"## [{m['id']}] (weight {int(weight)}) {m['when'].replace('T', ' ')} {tags}".rstrip()
                + f"\n{body}\n\n")

    def _memories_preamble(self):
        return ("# Memories\n\n"
                "<!-- One memory per section. The [number] is how the tools refer to it: keep it "
                "unique. Tags follow the date as #tag. (weight 0-100) is how much this memory "
                "carries: it decides what is recalled first and what fades first. Leave it off an "
                "entry you add by hand and the tags decide. You can edit or remove entries while "
                "she runs; she rereads this file when it changes. -->\n\n")

    def _save_memories(self):
        _write_atomic(self.paths["memories"],
                      self._memories_preamble() + "".join(self._format_memory(m) for m in self._memories))
        self._mark("memories")

    # -- goals.md --------------------------------------------------------------

    def _load_goals(self):
        goals, current, in_notes = [], None, False
        for line in self._read("goals"):
            header = GOAL_HEADER.match(line)
            if header:
                content = header.group(2).strip()
                # A goal with no origin line was not written by us, so
                # someone added it to goals.md by hand.
                current = {"id": str(uuid.uuid4()), "key": " ".join(content.lower().split()),
                           "content": content, "status": header.group(1), "outcome": None,
                           "origin": "by hand", "notes": [], "passes": 0, "stalls": 0,
                           "opened": _now(), "touched": _now()}
                goals.append(current)
                in_notes = False
                continue
            if current is None:
                continue
            if in_notes and line.startswith("  - "):
                when, _, note = line[4:].partition(" | ")
                current["notes"].append({"when": when.strip(), "note": note.strip()})
                continue
            if line.startswith("- "):
                key, _, value = line[2:].partition(":")
                key, value = key.strip(), value.strip()
                in_notes = key == "notes"
                if key in ("passes", "stalls"):
                    try:
                        current[key] = int(value)
                    except ValueError:
                        pass
                elif key in ("id", "opened", "touched", "outcome", "origin") and value:
                    current[key] = value
        self.goals = goals

    def _save_goals(self):
        out = ["# On her mind\n",
               "<!-- '## open:' goals are what she thinks about when left alone, oldest-touched "
               "first. Add one by hand the same way. -->\n"]
        for g in self.goals:
            out.append(f"## {g['status']}: {_one_line(g['content'])}")
            out.append(f"- origin: {g.get('origin') or 'unknown'}")
            if g.get("outcome"):
                out.append(f"- outcome: {_one_line(g['outcome'])}")
            out.append(f"- passes: {g.get('passes', 0)}")
            out.append(f"- stalls: {g.get('stalls', 0)}")
            out.append(f"- opened: {g.get('opened', '')}")
            out.append(f"- touched: {g.get('touched', '')}")
            out.append(f"- id: {g['id']}")
            if g.get("notes"):
                out.append("- notes:")
                out += [f"  - {n['when']} | {_one_line(n['note'])}" for n in g["notes"]]
            out.append("")
        _write_atomic(self.paths["goals"], "\n".join(out) + "\n")
        self._mark("goals")

    # -- asks.md ---------------------------------------------------------------

    def _load_asks(self):
        asks, current = [], None
        for line in self._read("asks"):
            header = ASK_HEADER.match(line)
            if header:
                question = header.group(2).strip()
                current = {"id": str(uuid.uuid4()), "key": " ".join(question.lower().split()),
                           "question": question, "status": header.group(1), "origin": "by hand",
                           "answer": None, "offers": 0, "opened": _now(), "touched": _now()}
                asks.append(current)
                continue
            if current is None or not line.startswith("- "):
                continue
            key, _, value = line[2:].partition(":")
            key, value = key.strip(), value.strip()
            if key == "offers":
                try:
                    current["offers"] = int(value)
                except ValueError:
                    pass
            elif key in ("id", "opened", "touched", "origin", "answer") and value:
                current[key] = value
        self.asks = asks

    def _save_asks(self):
        out = ["# Questions for him\n",
               "<!-- Things only the person she talks to could answer, held until the subject "
               "comes up. 'waiting' is on her mind, 'asked' has been put to him and is awaiting "
               "a reply, 'stale' she has given up offering. Add one by hand the same way; she "
               "rereads this file when it changes. Nothing here ever blocks her: if none of "
               "these is ever answered she carries on exactly as she would without them. -->\n"]
        for a in self.asks:
            out.append(f"## {a['status']}: {_one_line(a['question'])}")
            out.append(f"- origin: {a.get('origin') or 'unknown'}")
            if a.get("answer"):
                out.append(f"- answer: {_one_line(a['answer'])}")
            out.append(f"- offers: {a.get('offers', 0)}")
            out.append(f"- opened: {a.get('opened', '')}")
            out.append(f"- touched: {a.get('touched', '')}")
            out.append(f"- id: {a['id']}")
            out.append("")
        _write_atomic(self.paths["asks"], "\n".join(out) + "\n")
        self._mark("asks")

    # -- people.md -------------------------------------------------------------

    def _load_people(self):
        people, current = [], None
        for line in self._read("people"):
            if line.startswith("#") and not line.startswith("## "):
                continue                      # the file's own title
            header = PERSON_HEADER.match(line)
            if header:
                current = {"name": header.group(1).strip(), "note": [],
                           "updated": _now(), "from_id": 0}
                people.append(current)
                continue
            if current is None:
                continue
            if line.startswith("- updated:"):
                current["updated"] = line.partition(":")[2].strip() or _now()
            elif line.startswith("- from_id:"):
                try:
                    current["from_id"] = int(line.partition(":")[2].strip())
                except ValueError:
                    pass
            else:
                current["note"].append(line)
        for person in people:
            person["note"] = "\n".join(person["note"]).strip()
        self.people = people

    def _save_people(self):
        out = ["# People\n",
               "<!-- What she knows about each person she talks to, written during a full "
               "night from the things they have told her. Edit or add a section by hand the "
               "same way; she rereads this file when it changes. from_id is the newest "
               "memory the note covers - anything newer reaches her as itself until the "
               "next night rewrites this. -->\n"]
        for person in self.people:
            out.append(f"## {person['name']}")
            out.append(f"- updated: {person.get('updated', '')}")
            out.append(f"- from_id: {person.get('from_id', 0)}")
            # A body line starting "## " would split the section in two when
            # it is read back. Indent it one space.
            out.append(re.sub(r"^## ", " ## ", person.get("note", ""), flags=re.MULTILINE))
            out.append("")
        _write_atomic(self.paths["people"], "\n".join(out) + "\n")
        self._mark("people")

    def person_note(self, name):
        """What she knows about someone, by name, or None. Case does not
        matter: /iam peter and /iam Peter are the same person."""
        self._sync()
        wanted = " ".join(str(name or "").lower().split())
        if not wanted:
            return None
        found = next((p for p in self.people
                      if " ".join(p["name"].lower().split()) == wanted), None)
        return dict(found) if found else None

    def set_person_note(self, name, note, from_id=0):
        """Replace what she knows about someone. Replaced rather than
        appended: the note is a consolidation of every #person memory, so
        adding to it would say the same things twice."""
        self._sync()
        wanted = " ".join(str(name).lower().split())
        found = next((p for p in self.people
                      if " ".join(p["name"].lower().split()) == wanted), None)
        if found is None:
            found = {"name": str(name), "note": "", "updated": _now(), "from_id": 0}
            self.people.append(found)
        found["note"], found["from_id"], found["updated"] = str(note).strip(), int(from_id), _now()
        self._save_people()
        return dict(found)

    def people_known(self):
        self._sync()
        return [dict(p) for p in self.people]

    # -- dreams.md -------------------------------------------------------------

    def _load_dreams(self):
        dreams, current = [], None
        for line in self._read("dreams"):
            header = DREAM_HEADER.match(line)
            if header:
                about = [int(n) for n in re.findall(r"\d+", header.group(2) or "")]
                current = {"id": str(uuid.uuid4()), "when": header.group(1).replace(" ", "T"),
                           "story": [], "about": about}
                dreams.append(current)
            elif current is not None:
                current["story"].append(line)
        for d in dreams:
            d["story"] = "\n".join(d["story"]).strip()
        self.dreams = dreams[-DREAM_KEEP:]

    def _save_dreams(self):
        out = [f"# Dreams (the last {DREAM_KEEP})\n"]
        for d in self.dreams[-DREAM_KEEP:]:
            buckets = d.get("buckets") or []
            about = ", ".join(f"[{n}]" + (f" {buckets[i]}" if i < len(buckets) else "")
                              for i, n in enumerate(d.get("about", [])))
            out.append(f"## {d['when'].replace('T', ' ')}" + (f" (from {about})" if about else ""))
            out.append(re.sub(r"^## ", " ## ", d["story"], flags=re.MULTILINE))
            out.append("")
        _write_atomic(self.paths["dreams"], "\n".join(out) + "\n")
        self._mark("dreams")

    # -- state.md --------------------------------------------------------------

    def _load_state(self):
        for line in self._read("state"):
            key, _, value = line.partition(":")
            try:
                value = int(value.strip())
            except ValueError:
                continue
            if key.strip() == "wake_cycles":
                self.wake_cycles = value
            elif key.strip() == "filed_at":
                self.filed_at = value
            elif key.strip() == "next_id":
                self._next_id = max(self._next_id, value)

    def _save_state(self):
        _write_atomic(self.paths["state"],
                      "# State\n\n"
                      f"wake_cycles: {self.wake_cycles}\n"
                      f"filed_at: {self.filed_at}\n"
                      f"next_id: {self._next_id}\n")
        self._mark("state")

    def save(self):
        """Write goals, dreams and counters. Memories are written as they are
        added or removed, so this is only the small part."""
        self._save_goals()
        self._save_asks()
        self._save_people()
        self._save_dreams()
        self._save_state()

    def close(self):
        """Nothing to close; kept so callers need not care which store this is."""

    def _append(self, name, title, text):
        new_file = not os.path.exists(self.paths[name])
        with open(self.paths[name], "a", encoding="utf-8") as f:
            if new_file:
                f.write(f"# {title}\n\n")
            f.write(text.rstrip() + "\n\n")

    def archive_merge(self, originals, merged):
        """Record a merge in merged.md: what went in, what came out. Append
        only; nothing reads it back. It is for you, not for her."""
        lines = [f"## {_now().replace('T', ' ')} into [{merged['id']}]"]
        for m in originals:
            tags = " ".join("#" + t for t in m["tags"])
            lines.append(f"- [{m['id']}] {_one_line(m['content'])}" + (f"  {tags}" if tags else ""))
        lines.append(f"- became: {_one_line(merged['content'])}")
        self._append("merged", "Merges made during sleep", "\n".join(lines))
        # The last merge, for a console that wants to show it as it happens.
        self.last_merge = ([dict(m) for m in originals], dict(merged))

    def log_night(self, text):
        """Append one night's section to sleep_log.md."""
        self._append("sleep_log", "Sleep log", text)

    # -- migration from earlier stores ----------------------------------------

    @staticmethod
    def _complete_goal(g):
        """A goal from an older store, with every field this one expects.
        The original memory.json goals had no stalls or origin, for example."""
        content = _one_line(g.get("content", ""))
        return {"id": str(g.get("id") or uuid.uuid4()),
                "key": g.get("key") or content.lower(),
                "content": content,
                "status": g.get("status") if g.get("status") in ("open", "closed") else "open",
                "outcome": g.get("outcome"),
                "origin": g.get("origin") or "unknown",
                "notes": [n for n in (g.get("notes") or []) if isinstance(n, dict) and "note" in n],
                "passes": int(g.get("passes") or 0),
                "stalls": int(g.get("stalls") or 0),
                "opened": str(g.get("opened") or _now()),
                "touched": str(g.get("touched") or _now())}

    def _migrate(self):
        """Bring in memory.db (the short-lived SQLite store) or memory.json,
        once, when this folder has no memories yet."""
        if os.path.exists(self.paths["memories"]):
            return
        state_dir = os.path.dirname(os.path.abspath(self.folder))
        db, legacy = os.path.join(state_dir, "memory.db"), os.path.join(state_dir, "memory.json")
        data, source = None, None
        if os.path.exists(db):
            import sqlite3
            try:
                con = sqlite3.connect(db)
                rows = con.execute("SELECT id, created, content, tags FROM memories ORDER BY id").fetchall()
                mind = con.execute("SELECT value FROM mind WHERE key = 'state'").fetchone()
                con.close()
                data = json.loads(mind[0]) if mind else {}
                data["memories"] = [{"id": r[0], "when": r[1], "content": r[2], "tags": json.loads(r[3])}
                                    for r in rows]
                source = db
            except sqlite3.Error:
                data = None
        if data is None and os.path.exists(legacy):
            try:
                with open(legacy, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["filed_at"] = 0   # counted list positions, not numbers
                source = legacy
            except (json.JSONDecodeError, OSError):
                data = None
        if data is None:
            return
        old = [m for m in data.get("memories", []) if isinstance(m, dict)]
        # memory.db numbered its memories; memory.json gave them UUID strings,
        # and its dreams referred to memories by list position. So keep the
        # numbers only when every one is already a number, and otherwise
        # number them 1, 2, 3... in the order they were kept.
        keep_ids = bool(old) and all(isinstance(m.get("id"), int) for m in old)
        self._memories = [{"id": m["id"] if keep_ids else i + 1,
                           "when": str(m.get("when") or _now()),
                           "content": str(m.get("content", "")),
                           "tags": [str(t) for t in (m.get("tags") or [])],
                           "weight": _default_weight(m.get("tags") or [])}
                          for i, m in enumerate(old)]
        self._next_id = max((m["id"] for m in self._memories), default=0) + 1
        self.goals = [self._complete_goal(g) for g in data.get("goals", []) if isinstance(g, dict)]
        self.dreams = []
        for d in data.get("dreams", [])[-DREAM_KEEP:]:
            if not isinstance(d, dict):
                continue
            about = [a for a in (d.get("about") or []) if isinstance(a, int)]
            if not keep_ids:
                about = [a + 1 for a in about]   # list position -> new number
            self.dreams.append({"id": str(d.get("id") or uuid.uuid4()), "when": str(d.get("when") or _now()),
                                "story": str(d.get("story", "")), "about": about, "buckets": []})
        self.wake_cycles = int(data.get("wake_cycles") or 0)
        self.filed_at = int(data.get("filed_at") or 0) if keep_ids else 0
        self._save_memories()
        self.save()
        os.replace(source, source + ".migrated")

    # -- memories: the interface the tools use ---------------------------------

    def add_memory(self, content, tags=None, weight=None):
        """Keep something. weight is 0-100 and defaults to what this kind of
        memory starts at (WEIGHT_BY_TAG); pass it when the caller knows better
        than the tags do - her own spoken lines are worth less to keep than
        what the other person said, and both are tagged #conversation."""
        self._sync()
        tags = list(tags or [])
        m = {"id": self._next_id, "when": _now(), "content": content.strip(), "tags": tags,
             "weight": _clamp_weight(_default_weight(tags) if weight is None else weight, tags)}
        self._next_id += 1
        self._memories.append(m)
        # Append, not rewrite: adding is the common case and stays cheap.
        new_file = not os.path.exists(self.paths["memories"])
        with open(self.paths["memories"], "a", encoding="utf-8") as f:
            if new_file:
                f.write(self._memories_preamble())
            f.write(self._format_memory(m))
        self._mark("memories")
        self._save_state()
        return dict(m)

    def get(self, memory_id):
        self._sync()
        memory_id = int(memory_id)
        return next((dict(m) for m in self._memories if m["id"] == memory_id), None)

    def count(self):
        self._sync()
        return len(self._memories)

    def newest_id(self):
        self._sync()
        return max((m["id"] for m in self._memories), default=0)

    def recent(self, n=10, tag=None):
        """The last n memories, oldest of them first."""
        self._sync()
        pool = [m for m in self._memories if tag is None or tag in m["tags"]]
        return [dict(m) for m in pool[-n:]] if n > 0 else []

    def oldest(self, n=10):
        self._sync()
        return [dict(m) for m in self._memories[:n]]

    def random(self, n=3, exclude=()):
        self._sync()
        exclude = {int(e) for e in exclude}
        pool = [m for m in self._memories if m["id"] not in exclude]
        return sorted((dict(m) for m in random.sample(pool, min(n, len(pool)))), key=lambda m: m["id"])

    def like(self, memory, n=6):
        """The memories sharing the most words with this one, best first.

        What the merge sampler needs and random() cannot give it. Picking six
        memories at random out of a hundred and fifty and asking whether any
        is about the same thing as a seventh almost always gets "no" - not
        because there is nothing to merge, but because same-subject pairs are
        rare in a random handful. So nothing merged, and the pile grew.

        Plain word overlap, no model call, so it costs nothing. Falls back to
        filling the rest at random: a merge the words did not see is still
        possible, and a sampler that only ever showed obvious pairs would
        never find those.
        """
        self._sync()
        wanted = _content_words(memory["content"])
        scored = []
        for m in self._memories:
            if m["id"] == memory["id"]:
                continue
            shared = len(wanted & _content_words(m["content"]))
            if shared:
                scored.append((shared, m["id"], m))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        picked = [dict(m) for _, _, m in scored[:n]]
        if len(picked) < n:
            seen = {memory["id"]} | {m["id"] for m in picked}
            rest = [m for m in self._memories if m["id"] not in seen]
            picked += [dict(m) for m in random.sample(rest, min(n - len(picked), len(rest)))]
        return sorted(picked, key=lambda m: m["id"])

    def find(self, about, limit=10):
        """Memories containing every word asked about, forgivingly stemmed.
        Newest matches win when there are more than `limit`."""
        self._sync()
        wanted = [_stem(w) for w in re.findall(r"\w+", about or "")]
        if not wanted:
            return []
        hits = []
        for m in reversed(self._memories):
            words = [_stem(w) for w in re.findall(r"\w+", m["content"] + " " + " ".join(m["tags"]))]
            if all(any(w.startswith(s) for w in words) for s in wanted):
                hits.append(dict(m))
                if len(hits) >= limit:
                    break
        return hits[::-1]

    def use(self, ids, bump=USE_BUMP):
        """Mark memories as having just been used - recalled into a prompt,
        dreamt about, written into the diary - and raise their weight a
        little. This is the whole of how a memory earns its keep: nothing
        judges importance, use does.

        Saves only when something actually changed, so a recall that turns up
        memories already at the ceiling costs nothing."""
        self._sync()
        ids = {int(i) for i in ids}
        changed = False
        for m in self._memories:
            if m["id"] not in ids:
                continue
            raised = _clamp_weight(m.get("weight", _default_weight(m["tags"])) + bump, m["tags"])
            if raised != m.get("weight"):
                m["weight"], changed = raised, True
        if changed:
            self._save_memories()
        return changed

    def decay(self, factor=NIGHTLY_DECAY):
        """One night of forgetting: every weight down by `factor`, none below
        its kind's floor. Sleep's job, like merging and thinning (invariant
        3), so a day spent being used lifts a memory and only the night takes
        anything back. Returns a one-line summary for the sleep log."""
        self._sync()
        lowered, floored = 0, 0
        for m in self._memories:
            was = m.get("weight", _default_weight(m["tags"]))
            # Truncated, not rounded. Rounding stalls at the bottom - 10 x
            # 0.98 is 9.8, which rounds back to 10 - so a memory nothing ever
            # recalls would sit there for ever instead of reaching the bottom
            # and being let go. Truncating always takes at least one off.
            now = _clamp_weight(int(was * factor), m["tags"])
            m["weight"] = now
            if now < was:
                lowered += 1
            elif now == was and was == _weight_floor(m["tags"]):
                floored += 1
        self._save_memories()
        self._save_state()
        return (f"weights: {lowered} faded a little, {floored} already at their floor "
                f"(x{factor} a night)")

    def delete(self, ids):
        self._sync()
        ids = {int(i) for i in ids}
        self._memories = [m for m in self._memories if m["id"] not in ids]
        self._save_memories()

    def thin(self, limit=MAX_MEMORIES, protect_first=0):
        """Drop the lightest memories until at most `limit` remain. Returns
        how many went.

        By weight rather than by age, which is the point of having weights: a
        conversation line from this morning that nothing has ever recalled is
        worth less than one from a fortnight ago that keeps coming back. A
        lesson or a diary entry is never dropped here - those are what the
        rest was for - and neither are the first `protect_first` numbers, her
        founding memories, whatever their weight says.

        Oldest first among equal weights, so a tie breaks the way it used to.
        """
        self._sync()
        excess = len(self._memories) - limit
        if excess <= 0:
            return 0
        keep_tags = {"lesson", "diary"}
        droppable = [m for m in self._memories
                     if m["id"] > protect_first and not (keep_tags & set(m["tags"]))]
        droppable.sort(key=lambda m: (m.get("weight", _default_weight(m["tags"])), m["id"]))
        drop = {m["id"] for m in droppable[:excess]}
        if not drop:
            return 0
        self._memories = [m for m in self._memories if m["id"] not in drop]
        self._save_memories()
        return len(drop)

    @property
    def memories(self):
        """Every memory, oldest first, as copies. For inspection and tests."""
        self._sync()
        return [dict(m) for m in self._memories]

    # -- dreams ----------------------------------------------------------------

    def add_dream(self, story, about, buckets=None):
        """buckets, if given, says per memory why it was picked ("recent",
        "week", "older", "any"); it goes into the dreams.md header."""
        self._sync()
        dream = {"id": str(uuid.uuid4()), "when": _now(), "story": story.strip(), "about": list(about),
                 "buckets": list(buckets or [])}
        self.dreams = (self.dreams + [dream])[-DREAM_KEEP:]
        self._save_dreams()
        return dream

    # -- goals -----------------------------------------------------------------

    def _current(self, goal):
        """The live copy of a goal a caller is holding. If goals.md was edited
        and reread since the caller got it, their dict is a stale copy; carry
        what they changed on it (stalls) over to the live one."""
        live = next((g for g in self.goals if g["id"] == goal["id"]), None)
        if live is None:
            return goal
        if live is not goal:
            live["stalls"] = goal.get("stalls", live.get("stalls", 0))
        return live

    def add_goal(self, content, origin=None):
        """Put something on her mind. Past MAX_GOALS the stalest open goal
        is closed to make room.

        origin records where the goal came from: "conversation", "thinking",
        "dream", or "by hand". Without it there is no way to tell afterwards
        whether goals a dream left behind go anywhere different from goals
        that came out of talking."""
        self._sync()
        content = _one_line(content)
        key = content.lower()
        for goal in self.goals:
            if goal["status"] == "open" and goal["key"] == key:
                return goal
        goal = {"id": str(uuid.uuid4()), "key": key, "content": content,
                "status": "open", "outcome": None, "origin": origin or self.activity,
                "notes": [], "passes": 0, "stalls": 0, "opened": _now(), "touched": _now()}
        self.goals.append(goal)
        open_goals = self.open_goals()
        if len(open_goals) > MAX_GOALS:
            self._close(min(open_goals, key=lambda g: g["touched"]), "crowded out by newer interests")
        closed = [g for g in self.goals if g["status"] != "open"]
        if len(closed) > CLOSED_GOALS_KEPT:
            drop = {g["id"] for g in closed[:-CLOSED_GOALS_KEPT]}
            self.goals = [g for g in self.goals if g["id"] not in drop]
        self._save_goals()
        return goal

    def _close(self, goal, outcome):
        goal["status"], goal["outcome"], goal["touched"] = "closed", _one_line(outcome), _now()

    def close_goal(self, goal, outcome):
        self._sync()
        live = self._current(goal)
        self._close(live, outcome)
        if live is not goal:
            goal.update(status=live["status"], outcome=live["outcome"], touched=live["touched"])
        self._save_goals()

    def open_goals(self):
        self._sync()
        return [g for g in self.goals if g["status"] == "open"]

    def next_goal(self):
        """The open goal left alone longest. Round-robin, not a score: a
        score that does not change when she thinks picks the same goal for ever."""
        open_goals = self.open_goals()
        return min(open_goals, key=lambda g: g["touched"]) if open_goals else None

    def note_on_goal(self, goal, note):
        self._sync()
        live = self._current(goal)
        live["notes"] = (live["notes"] + [{"when": _now(), "note": _one_line(note)}])[-10:]
        live["passes"] = live.get("passes", 0) + 1
        live["touched"] = _now()
        if live is not goal:
            goal.update(notes=live["notes"], passes=live["passes"], touched=live["touched"])
        self._save_goals()


    # -- asks: questions held for him ------------------------------------------

    def _current_ask(self, ask):
        """The live copy of an ask a caller is holding, in case asks.md was
        edited and reread since they got it."""
        return next((a for a in self.asks if a["id"] == ask["id"]), ask)

    def add_ask(self, question, origin=None):
        """Hold a question for him. Returns the ask, or the existing one if
        she is already carrying the same question.

        Past MAX_ASKS the stalest waiting one goes: a question she has been
        carrying without an occasion for longer than all the others is the
        one she is least likely to find an occasion for."""
        self._sync()
        question = _one_line(question)
        key = question.lower()
        for ask in self.asks:
            if ask["status"] in ("waiting", "asked") and ask["key"] == key:
                return ask
        ask = {"id": str(uuid.uuid4()), "key": key, "question": question, "status": "waiting",
               "origin": _one_line(origin) if origin else (self.activity or "unknown"),
               "answer": None, "offers": 0, "opened": _now(), "touched": _now()}
        self.asks.append(ask)
        waiting = self.waiting_asks()
        if len(waiting) > MAX_ASKS:
            stalest = min(waiting, key=lambda a: a["touched"])
            stalest["status"], stalest["touched"] = "stale", _now()
        self._save_asks()
        return ask

    def waiting_asks(self):
        self._sync()
        return [a for a in self.asks if a["status"] == "waiting"]

    def asked_ask(self):
        """The question she has put to him and not had an answer to yet, if
        there is one. At most one is ever outstanding."""
        self._sync()
        return next((a for a in self.asks if a["status"] == "asked"), None)

    def mark_asked(self, ask):
        """She has offered this one. Not the same as him having heard it: the
        reply it was woven into may not have come out as a question at all,
        which is what unask is for."""
        self._sync()
        live = self._current_ask(ask)
        live["status"], live["touched"] = "asked", _now()
        live["offers"] = live.get("offers", 0) + 1
        ask.update(status=live["status"], offers=live["offers"], touched=live["touched"])
        self._save_asks()

    def mark_answered(self, ask, answer):
        self._sync()
        live = self._current_ask(ask)
        live["status"], live["answer"], live["touched"] = "answered", _one_line(answer), _now()
        ask.update(status=live["status"], answer=live["answer"], touched=live["touched"])
        self._save_asks()

    def unask(self, ask):
        """He did not pick it up. Back on her mind for another occasion, or
        let go once she has offered it enough times."""
        self._sync()
        live = self._current_ask(ask)
        offered = live.get("offers", 0)
        live["status"] = "stale" if offered >= ASK_OFFERS_BEFORE_STALE else "waiting"
        live["touched"] = _now()
        ask.update(status=live["status"], touched=live["touched"])
        self._save_asks()
        return live["status"]

    def prune_asks(self, keep=30):
        """Forget the oldest settled questions - answered or given up on.
        Sleep's job, like everything else that shrinks (invariant 3). What he
        actually told her is kept as a memory, not here."""
        self._sync()
        settled = [a for a in self.asks if a["status"] in ("answered", "stale")]
        if len(settled) <= keep:
            return 0
        drop = {a["id"] for a in settled[:len(settled) - keep]}
        self.asks = [a for a in self.asks if a["id"] not in drop]
        self._save_asks()
        return len(drop)


_ACTIVE = None


def active_memory():
    """The memory the tools operate on: the current workspace's. Reopened
    when the workspace changes, so an agent never reads another's memory."""
    global _ACTIVE
    folder = ws().memory_path
    if _ACTIVE is None or _ACTIVE.folder != folder:
        _ACTIVE = MemorySystem(folder)
    return _ACTIVE


def _show(rows):
    if not rows:
        return "NOTFOUND: nothing there"
    # The weight is not shown to her. It is bookkeeping about her memory, not
    # part of the memory, and a model told one of its thoughts is worth 30 out
    # of 100 starts arguing with the number instead of using the thought.
    return "\n".join(f"[{m['id']}] {m['when'][:16]}  {m['content']}" for m in rows)


def _clamp(value, default, low, high):
    try:
        return max(low, min(int(value if value is not None else default), high))
    except (TypeError, ValueError):
        return default


# -- the tools ----------------------------------------------------------------

@tool("How much is remembered, dreamt and on your mind.")
def memory_count():
    mem = active_memory()
    return (f"{mem.count()} memories, {len(mem.dreams)} dreams kept, "
            f"{len(mem.open_goals())} things on my mind")


@tool("One specific memory, by the number shown in [brackets] by the other memory tools.",
      index=P("integer"))
def memory_at(index):
    try:
        found = active_memory().get(int(index))
    except (TypeError, ValueError):
        found = None
    return _show([found]) if found else _soft_not_found(
        "memory", str(index), "Numbers are the ones shown in [brackets] by memory_recent and the others.")


@tool("The most recent memories.", count=P("integer", optional=True))
def memory_recent(count=10):
    return _show(active_memory().recent(_clamp(count, 10, 1, 50)))


@tool("The oldest memories - what you are no longer circling.", count=P("integer", optional=True))
def memory_oldest(count=10):
    return _show(active_memory().oldest(_clamp(count, 10, 1, 50)))


@tool("A few memories at random, where unexpected connections come from.", count=P("integer", optional=True))
def memory_random(count=3):
    mem = active_memory()
    if not mem.count():
        return "NOTFOUND: nothing remembered yet"
    return _show(mem.random(_clamp(count, 3, 1, 10)))


@tool("Memories mentioning something in particular.", about=P("string"))
def memory_find(about):
    if not (about or "").strip():
        return "NOTFOUND: say what to look for"
    rows = active_memory().find(about)
    return _show(rows) if rows else _soft_not_found("memory about", about, "Try other words, or memory_random.")


@tool("Keep something worth knowing next time.",
      content=P("string"), tags=P("array", optional=True, items={"type": "string"}))
def remember(content, tags=None):
    active_memory().add_memory(content, tags)
    return "kept"


@tool("Replace several memories of the SAME subject with one that says it better. "
      "Give their [numbers] and the single sentence that replaces them.",
      indexes=P("array", items={"type": "integer"}), merged=P("string"))
def consolidate(indexes, merged):
    mem = active_memory()
    try:
        ids = sorted({int(i) for i in indexes})
    except (TypeError, ValueError):
        return "ERROR: consolidate needs a list of memory numbers"
    if len(ids) < 2:
        return "ERROR: consolidating needs at least two memories"
    if not (merged or "").strip():
        return "ERROR: consolidate needs the sentence that replaces them"
    found = [mem.get(i) for i in ids]
    missing = [i for i, m in zip(ids, found) if m is None]
    if missing:
        return f"ERROR: no memories numbered {missing}"
    tags = sorted({t for m in found for t in m["tags"]} | {"consolidated"})
    mem.delete(ids)
    new = mem.add_memory(merged, tags)
    mem.archive_merge(found, new)
    return f"merged {len(ids)} memories into [{new['id']}] ({mem.count()} now)"


@tool("What is on your mind.")
def goals_list():
    goals = active_memory().open_goals()
    if not goals:
        return "nothing much on my mind"
    return "\n".join(f"[{i}] {g['content']}  ({g['passes']} passes)" for i, g in enumerate(goals))


@tool("Put something on your mind: a question, or something unfinished.", content=P("string"))
def goal_add(content):
    # A question about whether her own past is real is refused here as well as
    # in the driver's wondering paths, because this tool is a third door into
    # the same place: the model can call it mid-conversation, where neither of
    # those filters runs. Such a question can never be closed by any tool
    # here, so taking it on only fills her mind with something that will stall
    # and file another doubt on the way out.
    from babycoder.tools.dreams import unanswerable_about_self
    if unanswerable_about_self(content):
        return ("not something to take on: my own past is mine, not a case to "
                "investigate. If it is not in my memory I simply do not "
                "remember it, which settles it.")
    # No origin given: add_goal records whatever the agent's loop says she is
    # doing right now (conversation or thinking).
    active_memory().add_goal(content)
    return "on my mind now"


@tool("Let something go - answered, or no longer worth carrying.",
      index=P("integer"), outcome=P("string"))
def goal_close(index, outcome):
    mem = active_memory()
    goals = mem.open_goals()
    try:
        goal = goals[int(index)]
    except (ValueError, TypeError, IndexError):
        return _soft_not_found("goal", str(index), f"there are {len(goals)}, numbered from 0")
    mem.close_goal(goal, outcome or "let go")
    return f"let go of: {goal['content'][:50]}"


@tool("What you know about someone you have talked to, by name. Use it when they "
      "mention a person, or when you want to recall who someone is.", name=P("string"))
def about_person(name):
    mem = active_memory()
    found = mem.person_note(name)
    if found and found["note"]:
        return found["note"]
    known = ", ".join(p["name"] for p in mem.people_known() if p["note"])
    return _soft_not_found("anyone called", str(name),
                           f"People you know something about: {known}." if known
                           else "You have not got to know anyone well enough yet.")


@tool("The last few dreams you remember.")
def dreams_recent():
    mem = active_memory()
    if not mem.dreams:
        return "NOTFOUND: no dreams kept"
    return "\n\n".join(f"[{d['when'][:16]}] {d['story']}" for d in mem.dreams)
