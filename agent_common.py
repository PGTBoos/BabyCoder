"""
agent_common.py

The two things every coding driver needs and none of them owns: how a human
hands in the first instruction, and what the shared todo list is shaped like.

Both used to live in agent_room.py, which meant they only existed when the
room was running. That was the reason architect_agent.py and coder_agent.py
could not usefully be run on their own: the architect had no way to be told
what to plan, and nothing outside the room taught either role the todo
schema, so a board written by a standalone architect was not reliably
readable by a standalone coder. Pulling both out here is what makes
"architect alone", "coder alone" and "the room" three ways of driving the
same two roles rather than three separate programs.

Deliberately NOT part of the babycoder package: nothing in this file is a tool the
model can call. It is only how a person starts a run, and a paragraph of
prompt text. babycoder stays what it is, the sandboxed tool layer.

Run any driver with --help to see the input options.
"""

import os
import re
import sys

# The todo schema used to be spelled out here as TODO_SCHEMA_NOTE, appended to
# the coder's and architect's personas. It is gone: the shape of an item is now
# stated once, on the todo_write tool description in babycoder, and the
# discipline around it (read before write, keep what you are not changing, mark
# done rather than delete) is the planner toolset's usage note, which reaches
# any agent granted it.
#
# That was not only duplication. The two copies had drifted into disagreeing:
# the tool description said {id, task, status} while this note and agent_room
# both said {description, owner, status}. A model following the tool
# description wrote a board with no "owner" on any item, which is exactly the
# unparseable-board case decide_next had to be taught to survive.

# What a driver tells a coder to do when it is handed no task of its own.
# Phrased as a standing instruction rather than a specific job, because the
# actual work is whatever the architect already put on the board.
WORK_THE_BOARD = (
    "Check todo_read first. Find the pending todo item(s) with owner "
    '"coder" and implement the highest-priority one. When it is finished, '
    'set that item\'s status to "done" and write the full list back.'
)


def print_usage(program: str, demo_note: str = "") -> None:
    print(f"""usage:
  python {program}                     ask interactively what to work on
  python {program} "do the thing"      take the task from the command line
  python {program} --file spec.md      take the task from a file
  python {program} --todos             take no new task, work the shared todo list
  python {program} --demo              run the built-in demo task
{demo_note}
A one-line answer at the interactive prompt is often enough. For anything
longer than a line, write it in a file and use --file.""")


def read_task(argv=None, prompt: str = "Task: ", demo_task: str = None):
    """Work out what this run should do, from the command line or the person.

    Returns (task_text, mode), where mode is one of:
      "task"  - task_text is what to work on
      "demo"  - run the driver's own built-in demo (task_text is demo_task)
      "todos" - no new task, pick up whatever is already on the shared board
      "help"  - the caller should print usage and stop
      "quit"  - nothing usable was given, the caller should stop

    Returning a mode rather than just a string is what lets each driver
    decide for itself what "no task" means - the coder can go work the board,
    while the room has nothing to do without an overall task and says so.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help"):
        return None, "help"

    if argv and argv[0] == "--demo":
        if not demo_task:
            print("ERROR: this driver has no built-in demo.")
            return None, "quit"
        return demo_task, "demo"

    if argv and argv[0] == "--todos":
        return None, "todos"

    if argv and argv[0] in ("-f", "--file"):
        if len(argv) < 2:
            print("ERROR: --file needs a path.")
            return None, "quit"
        path = argv[1]
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError as exc:
            # Soft failure with the real reason, same idea as the toolkit's
            # NOTFOUND results - a missing spec file is a thing to report,
            # not a stack trace.
            print(f"ERROR: could not read {path}: {exc}")
            return None, "quit"
        if not text:
            print(f"ERROR: {path} is empty.")
            return None, "quit"
        return text, "task"

    if argv:
        # Everything else on the command line is the task itself, so an
        # unquoted sentence still works rather than being read as flags.
        return " ".join(argv).strip(), "task"

    # Nothing given, so ask. EOF (a piped or scheduled run with no input)
    # is a normal way to end up here, not an error.
    try:
        text = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None, "quit"

    if not text:
        return None, "todos"
    return text, "task"


###############################################################################
# TALKING TO A LONG-RUNNING AGENT
#
# read_task above is for a driver that takes one instruction and finishes.
# This is the other shape: an agent that keeps running and that a person can
# talk to whenever they feel like it, without the agent blocking on input
# while it waits.
#
# A background thread rather than select(): select() on stdin is POSIX-only,
# on Windows it accepts sockets and nothing else, so the obvious
# non-blocking read raises OSError there. An agent loop that swallowed that
# error would simply never see anything typed at it, on the platform where
# most people would try it first, with no message saying why. A reader
# thread works the same everywhere and gets line editing from input() for
# free.
###############################################################################


###############################################################################
# COLOUR
#
# ANSI colour codes. Windows 11 consoles (Windows Terminal and the classic
# console window) understand them once "virtual terminal processing" is
# switched on for the output handle, which enable_color() does. Anywhere they
# cannot work - output piped to a file, an old console, NO_COLOR set in the
# environment - every colour is an empty string and the text comes out plain.
###############################################################################

class Colors:
    YOU = ""      # what you type: cyan
    TOOL = ""     # tool calls and what they returned: yellow
    DREAM = ""    # sleep: dark green
    DIM = ""      # progress marks and program notices: grey
    RESET = ""
    on = False


_ANSI = {"YOU": "\033[36m", "TOOL": "\033[33m", "DREAM": "\033[32m", "DIM": "\033[90m", "RESET": "\033[0m"}


def enable_color() -> bool:
    """Switch colour on where the console supports it. Returns whether it did."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)          # standard output
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING: interpret the escape codes
            # instead of printing them as "[33m".
            if not kernel32.SetConsoleMode(handle, mode.value | 0x0004):
                return False
        except Exception:
            return False
    for name, code in _ANSI.items():
        setattr(Colors, name, code)
    Colors.on = True
    return True


def paint(text: str, color: str) -> str:
    """text in a colour, reset at the end of every line so a colour can never
    leak into the next thing printed, including your own typing."""
    if not color:
        return text
    return "\n".join(f"{color}{line}{Colors.RESET}" if line else line for line in text.split("\n"))


_CONTROL = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_]|[\x00-\x08\x0b-\x1f\x7f]")


def plain(text) -> str:
    """Strip escape codes and control characters from text we did not write
    ourselves (a web page, a model's output). A page containing its own escape
    codes could otherwise recolour, move or clear your console."""
    return _CONTROL.sub("", str(text))


class ChatInput:
    """Console input that does not block the agent's own loop, and that can
    tell the difference between "gone quiet" and "still typing".

    Reads one character at a time rather than whole lines, so a partially
    typed line is visible to the agent as `typing`. That is what lets an agent
    wait ten seconds before carrying on with its own thoughts without cutting
    someone off in the middle of a sentence: a short timeout AND a check that
    nobody is mid-word, instead of a long timeout hoping to cover both.

    Character mode needs a real terminal. Anything else - piped input, a
    scheduled run, an IDE console that does not hand over a tty - falls back
    to reading whole lines, where `typing` is always False and only the
    timeout applies. Nothing above needs to know which mode is in use.
    """

    def __init__(self, prompt: str = "You: "):
        import queue
        import threading
        self.prompt = prompt
        self._queue = queue.Queue()
        self._buffer = ""
        self._held = []          # output withheld while a line is being typed
        self._prompt_up = False  # a prompt is on screen, waiting to be typed at
        self._inline_open = False  # progress marks left the current line unfinished
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._closed = threading.Event()
        self._started = False
        self._restore = None
        self.char_mode = False

    # -- character mode ----------------------------------------------------

    def _read_char_windows(self):
        import msvcrt
        while True:
            ch = msvcrt.getwch()
            # Arrow keys, Home, End, F-keys and the like arrive as two
            # characters: a prefix (\x00 or \xe0) and a key code. Read as
            # text, the left arrow typed "àK" into the line. There is no line
            # editing here beyond backspace, so skip both and read on.
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
                continue
            return "\n" if ch in ("\r", "\n") else ch

    def _read_char_posix(self):
        while True:
            ch = sys.stdin.read(1)
            if ch != "\x1b":
                return ch
            # An arrow or function key: ESC [ ... final letter, or ESC O x.
            # Swallow the whole sequence rather than typing "[D" into the line.
            nxt = sys.stdin.read(1)
            if nxt == "[":
                while True:
                    c = sys.stdin.read(1)
                    if not c or "@" <= c <= "~":
                        break
            elif nxt == "O":
                sys.stdin.read(1)

    def _try_char_mode(self):
        """Put the terminal in character-at-a-time mode. Returns a restore
        callable, or None if this is not a terminal we can do that with."""
        if not sys.stdin.isatty():
            return None
        if os.name == "nt":
            try:
                __import__("msvcrt")   # present on Windows only
            except ImportError:
                return None
            # Windows consoles are already character-at-a-time through
            # msvcrt, so there is no terminal mode to set or restore.
            self._read_char = self._read_char_windows
            return lambda: None
        try:
            import termios
            import tty
        except ImportError:
            return None
        try:
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
        except (termios.error, ValueError, OSError):
            return None
        tty.setcbreak(fd)
        self._read_char = self._read_char_posix
        return lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def _read_loop(self):
        try:
            if self.char_mode:
                self._char_loop()
            else:
                self._line_loop()
        finally:
            if self._restore:
                self._restore()
                self._restore = None

    def _char_loop(self):
        while not self._closed.is_set():
            try:
                ch = self._read_char()
            except (OSError, ValueError, KeyboardInterrupt):
                self._closed.set()
                return
            if ch == "":                      # EOF
                self._closed.set()
                return
            if ch in ("\x03", "\x04"):        # Ctrl-C, Ctrl-D
                self._closed.set()
                return
            if ch in ("\r", "\n"):
                with self._lock:
                    line, self._buffer = self._buffer.strip(), ""
                    print()
                    self._prompt_up = False
                    # The line is finished, so anything held back while it was
                    # being typed can come out now, in the order it happened.
                    held, self._held = self._held, []
                    for text in held:
                        print(text)
                    if held:
                        print(paint(self.prompt, Colors.YOU), end="", flush=True)
                        self._prompt_up = True
                if line:
                    self._queue.put(line)
                continue
            if ch in ("\x7f", "\b"):           # backspace
                with self._lock:
                    if self._buffer:
                        self._buffer = self._buffer[:-1]
                        print("\b \b", end="", flush=True)
                continue
            with self._lock:
                if not self._buffer and self._inline_open:
                    # Progress dots were mid-line. Start the prompt on a line
                    # of its own rather than as "....You: h".
                    print()
                    self._inline_open = False
                if not self._buffer and not self._prompt_up:
                    # First character of a new line: draw the prompt first, so
                    # what you type is always labelled no matter what the agent
                    # printed before it. Relying on the agent to reprompt after
                    # every write meant any write site that forgot left your
                    # line echoing bare into the scrollback, indistinguishable
                    # from the agent's own text.
                    # Only when one is not already on screen. reprompt() draws
                    # one too, and both firing is where "You: You: " came from.
                    print(paint(self.prompt, Colors.YOU), end="", flush=True)
                    self._prompt_up = True
                self._buffer += ch
                print(f"{Colors.YOU}{ch}{Colors.RESET}", end="", flush=True)

    def _line_loop(self):
        while not self._closed.is_set():
            try:
                # No prompt of its own: the agent's loop draws it through
                # reprompt(), because only it knows when it has finished
                # printing. A prompt written from this thread lands in the
                # middle of whatever the agent was saying.
                line = input()
            except (EOFError, KeyboardInterrupt):
                # No terminal: piped input that ran out, or a scheduled run
                # with nothing attached. Not a failure, just nobody typing.
                self._closed.set()
                return
            if line.strip():
                self._queue.put(line.strip())

    # -- what the agent loop uses -----------------------------------------

    def start(self) -> "ChatInput":
        """Put the terminal in character mode, then start reading.

        The mode switch happens HERE, on the calling thread, not inside the
        reader. Doing it in the thread meant char_mode was still False for a
        moment after start() returned, so anything that asked straight away -
        a startup banner reporting whether keystroke detection is on - got the
        wrong answer and said so out loud.
        """
        if not self._started:
            self._restore = self._try_char_mode()
            self.char_mode = self._restore is not None
            self._thread.start()
            self._started = True
        return self

    def poll(self):
        """Whatever was typed since the last poll, or None. Never blocks."""
        import queue
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def write(self, text: str = "") -> None:
        """Print, unless a line is being typed - then hold it until Enter.

        This is the half of "do not talk over me" that stopping her starting
        new work does not cover. Her reply to the PREVIOUS message is already
        on its way when you begin typing the next one, and printing it lands
        in the middle of the line you are composing and destroys it. Held
        output is flushed by _char_loop the moment you press Enter.

        Taken under the same lock the keystroke echo uses, so a print and a
        keypress cannot interleave halfway through either one. In line mode
        there is no part-typed line to protect and this is a plain print.
        """
        with self._lock:
            if self._buffer:
                self._held.append(text)
                return
            self._make_room_locked()
            print(text)
            # Whatever we just printed pushed any prompt out of the way, so the
            # next keystroke has to draw a fresh one.
            self._prompt_up = False

    def _make_room_locked(self) -> None:
        """Get the cursor onto a clean line before the agent prints. Caller
        holds the lock and has checked nothing is being typed.

        An empty "You: " left on screen by reprompt() is wiped rather than
        printed after: without this every line she said while you were not
        typing came out as "You:   ~ thinking about ...", which is most of
        the "garbage" on screen. Progress dots left mid-line are ended with a
        newline for the same reason.
        """
        if self._prompt_up:
            print("\r" + " " * len(self.prompt) + "\r", end="")
            self._prompt_up = False
        if self._inline_open:
            print()
            self._inline_open = False

    def write_inline(self, mark: str = ".") -> None:
        """A progress mark, on the current line, and only when the line is
        ours to write on.

        Unlike write() there is nothing to hold for later: a dot that arrives
        after you press Enter is a dot about work that has already finished.
        Dropped silently instead - which is the whole point, since these were
        landing in the middle of what someone was typing.
        """
        with self._lock:
            if self._buffer or self._held:
                return
            if self._prompt_up:
                print("\r" + " " * len(self.prompt) + "\r", end="")
                self._prompt_up = False
            print(paint(mark, Colors.DIM), end="", flush=True)
            self._inline_open = True

    @property
    def holding(self) -> bool:
        """True when output is queued behind a line you are still typing."""
        with self._lock:
            return bool(self._held)

    @property
    def has_input(self) -> bool:
        """True when a finished line is waiting to be read. Lets a long
        running job ask "should I stop?" without consuming the line, so the
        loop that actually handles input still gets it."""
        return not self._queue.empty()

    @property
    def typing(self) -> bool:
        """True while a line is part-typed and not yet entered. An agent
        should hold off rather than talk over someone mid-sentence. Always
        False in line mode, where a partial line is invisible to us."""
        with self._lock:
            return bool(self._buffer)

    @property
    def nobody_there(self) -> bool:
        """True once stdin is exhausted or closed. An agent can use this to
        stop printing a prompt at a terminal that is not there, without
        changing anything else about how it runs."""
        return self._closed.is_set()

    def reprompt(self):
        """Redraw the prompt after the agent has printed something.

        A no-op while a line is part-typed: the prompt and everything typed so
        far are already on screen, and redrawing them printed the partial a
        second time ("you: hellouYou: hello"). Output that arrives mid-line is
        held now anyway, so there is nothing to redraw around.
        """
        if self.nobody_there:
            return
        with self._lock:
            if self._buffer or self._prompt_up:
                return
            if self._inline_open:
                print()
                self._inline_open = False
            print(paint(self.prompt, Colors.YOU), end="", flush=True)
            self._prompt_up = True

    def exit_now(self, code: int = 0):
        """End the process without waiting on the reader thread.

        The reader sits in a blocking read that cannot be cancelled. At normal
        interpreter shutdown it is still holding stdin's buffer lock, and
        Python aborts with "could not acquire lock for <_io.BufferedReader>
        ... possibly due to daemon threads" - a fatal-error dump printed at
        the user every time they typed /quit. Restoring the terminal and
        leaving immediately avoids finalisation entirely, which is the right
        trade for a CLI that has nothing left to clean up.
        """
        self.stop()
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os._exit(code)

    def stop(self):
        self._closed.set()
        # Put the terminal back even if the reader thread is parked in a
        # blocking read and will not reach its own finally.
        if self._restore:
            self._restore()
            self._restore = None


def ensure_utf8_console() -> None:
    """Make stdout/stdin carry text the model actually produced.

    A Windows console defaults to cp1252, which has no U+2019, so a model
    writing a curly apostrophe came out as "I\u00e2\u0080\u0099m still here" -
    the UTF-8 bytes shown through the wrong codec. Reconfiguring to UTF-8
    fixes it where the terminal supports it, and errors="replace" keeps a
    stubborn console from killing the run over one character.

    Called explicitly by each driver rather than run on import: quietly
    changing process-wide stream encoding as a side effect of importing a
    module is the kind of thing that is very hard to find later.
    """
    # On Windows, reconfiguring Python's streams is NOT enough on its own and
    # this is where the first attempt at this fell down. cmd.exe renders
    # output through its console code page - 850 or 1252 by default - so
    # correct UTF-8 bytes still come out as "Iam". The console itself
    # has to be switched to code page 65001 first, which is what these two
    # calls do. Harmless if it fails; the reconfigure below still helps for
    # redirected output.
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Older Python, or a stream that is not a real text wrapper
            # (a pipe under test, say). Nothing to do and nothing broken.
            pass
