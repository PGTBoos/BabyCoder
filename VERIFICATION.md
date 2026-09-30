# Verification report

250 checks across 7 offline suites. Run them with:

    python tests/run_all.py

No LM Studio needed. `tests/mock_lms.py` stands in for the model server and
speaks both protocols the project uses, so the agent loops are genuinely
driven rather than just imported.

## About the test count

It went **down**, from 426 to 250, and that is the point.

The old Lisa suite had 400+ behavioural assertions and was green through every
bug in her history, because it asserted what the code *did* rather than what
the design said. Several checks were written to lock in a symptom. A green
suite was how the drift got preserved.

Her tests now check two things only: that every tool she is granted exists, is
callable and returns a string, and that the design's shape is actually wired
up - one counter rather than timers, goals that can close, dreams as their own
FIFO, waking that never prunes. Everything else is judged by running her.

The coding side keeps its behavioural tests. Those assert a real contract
(sandbox holds, grants are enforced, a rejected call does not execute) and
have never been in question.

## The rewrite

`lisa_agent.py` was rewritten from scratch against `LISA_DESIGN.md`, which was
written first and agreed before any code changed. An audit of the previous
version against that design found **3 of 14 points met**:

| Design | Old code |
|---|---|
| Story frame, 25, first person | `"You are Lisa. Not an assistant playing Lisa"` |
| Context = prompt + exchange | up to 40 messages |
| 6 memory retrieval tools | `recall` only |
| Dreams: own FIFO of 10 | stored as memories with a tag |
| Goals = loose interests | three competing species |
| 10 wake / 8 dream, counted | 8 time constants, no counter |
| A cycle never spawns a competitor | a stalled goal spawned a rival entry |
| Every cycle concludes something | no "concluded" path existed at all |
| Goals capped and closable | **nothing could ever close a question** |
| Only her voice on screen | 7 machinery lines per cycle |

The three that did pass: dream order (random, story, then consolidate), sleep
being the only thing that prunes, and conversation preempting.

The user's live run showed 152 memories and **312 open questions**. That was
not a bug - there was no code path anywhere that closed a question, so the
pile could only grow, and every cycle was a lottery over garbage.

Result: 979 lines to 366, and fourteen pacing knobs
(`MAX_THREAD_ATTEMPTS`, `QUIET_AFTER_FAILURES`, `MAX_GOAL_CYCLES`,
`MUSE_INTERVAL_MINUTES`, `GOAL_REST_MINUTES`, `QUESTION_REST_MINUTES`,
`min_pull`, `failed_cycles`, `idle_announced`, `AWAY_MINUTES`,
`DREAM_INTERVAL_MINUTES`, `THINK_BREATH_SECONDS`, `IDLE_GRACE_SECONDS`,
`MAX_GRACE_SECONDS`) down to four.

## Kept from the old work

Earned, correct, and independent of the design confusion:

- `ChatInput` - character-mode typing detection, output held while you type,
  prompt drawn on the first keystroke, clean exit. Several hard bugs solved
  properly.
- `response.encoding = "utf-8"` on the SSE stream - the real cause of months
  of mojibake, which was never a console problem.
- `ensure_utf8_console` for the Windows code page.
- `run_agent(should_stop=...)` - interruptible cycles, which the design needs.
- `CHARACTER_TEMPERATURE` 0.8 against the coding default of 0.2.
- Toolsets and grants - capability as the unit of import.
- Wikipedia via the REST API (the `wikipedia` package cannot be installed on a
  current toolchain) and `ddgs` for search.
- Dream order, the `spent` guard against duplicate merges, and excluding
  dreams from the dream pool.

## Found during the rewrite

`ask_model` returned the model's content verbatim, so a model coming off
structured turns that kept emitting `{"notes":..., "final_answer":...}` had
that printed as a dream. It now unwraps the envelope, taking `final_answer`
only - `notes` is scratch rationale and is usually the stub `"thinking"`.

Also: replacing the memory section by slicing from "AGENT MEMORY" to
"TOOLSETS" silently deleted the tool registry between them. Caught by the
import failing immediately; restored from the previous archive and redone with
the correct boundary.

