"""
babycoder.tools.memory

AGENT MEMORY toolkit: long-term memory, per agent, as plain markdown files in
<state_dir>/memory/ that you can read, and edit, in any text editor:

  memories.md  what happened. One section per memory:
                   ## [12] 2026-09-29 21:40:05 #conversation
                   They said: it is raining
               The [number] is how the tools refer to it. Keep numbers unique
               if you add entries by hand; the next free one is in state.md.
  goals.md     what is on her mind, open and recently closed.
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
# Past this, sleep drops the oldest conversation lines (never her own
# thinking) until it is back under. Chatting adds two memories per exchange
# and merging removes at most a handful per night, so without a ceiling
# memory only ever grows.
MAX_MEMORIES = 2000
CLOSED_GOALS_KEPT = 50


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


def _write_atomic(path, text):
    """Write then rename, so a crash or Ctrl-C never leaves half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


MEMORY_HEADER = re.compile(r"^## \[(\d+)\](.*)$")
GOAL_HEADER = re.compile(r"^## (open|closed): (.*)$")
DREAM_HEADER = re.compile(r"^## (\S+ \S+)(?:\s+\(from (.*)\))?\s*$")


class MemorySystem:
    """Persistent memory: what happened, what was dreamt, what is on her mind."""

    def __init__(self, folder=None):
        self.folder = str(folder or ws().memory_path)
        os.makedirs(self.folder, exist_ok=True)
        self.paths = {name: os.path.join(self.folder, f"{name}.md")
                      for name in ("memories", "goals", "dreams", "state", "merged", "sleep_log")}
        self._seen = {}               # file -> mtime when we last read or wrote it
        self._memories = []           # dicts: id, when, content, tags; oldest first
        self.goals, self.dreams = [], []
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
                rest = header.group(2).split()
                tags = [t[1:] for t in rest if t.startswith("#")]
                when = " ".join(t for t in rest if not t.startswith("#")).replace(" ", "T")
                current = {"id": int(header.group(1)), "when": when or _now(),
                           "content": [], "tags": tags}
                memories.append(current)
            elif current is not None:
                current["content"].append(line)
        for m in memories:
            m["content"] = "\n".join(m["content"]).strip()
        self._memories = memories

    @staticmethod
    def _format_memory(m):
        tags = " ".join("#" + re.sub(r"\s+", "_", t) for t in m["tags"])
        # A line in the content that looks like a memory header would split
        # the entry in two when read back. Indent it one space.
        body = re.sub(r"^## \[", " ## [", m["content"], flags=re.MULTILINE)
        return f"## [{m['id']}] {m['when'].replace('T', ' ')} {tags}".rstrip() + f"\n{body}\n\n"

    def _memories_preamble(self):
        return ("# Memories\n\n"
                "<!-- One memory per section. The [number] is how the tools refer to it: keep it "
                "unique. Tags follow the date as #tag. You can edit or remove entries while she "
                "runs; she rereads this file when it changes. -->\n\n")

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
                           "tags": [str(t) for t in (m.get("tags") or [])]}
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

    def add_memory(self, content, tags=None):
        self._sync()
        m = {"id": self._next_id, "when": _now(), "content": content.strip(), "tags": list(tags or [])}
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

    def delete(self, ids):
        self._sync()
        ids = {int(i) for i in ids}
        self._memories = [m for m in self._memories if m["id"] not in ids]
        self._save_memories()

    def thin(self, limit=MAX_MEMORIES):
        """Drop the oldest conversation lines until at most `limit` memories
        remain. Only conversation: what she worked out herself is never
        thinned here. Returns how many went."""
        self._sync()
        excess = len(self._memories) - limit
        if excess <= 0:
            return 0
        drop = {m["id"] for m in self._memories if "conversation" in m["tags"]}
        drop = set(sorted(drop)[:excess])
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


@tool("The last few dreams you remember.")
def dreams_recent():
    mem = active_memory()
    if not mem.dreams:
        return "NOTFOUND: no dreams kept"
    return "\n\n".join(f"[{d['when'][:16]}] {d['story']}" for d in mem.dreams)
