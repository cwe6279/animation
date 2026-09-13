# Wiring your own agent harness to Talker

Talker's assistant characters (Clara today) hand long work to a backend agent on another
machine and carry on talking. This page is the contract that backend has to meet, the shim
that meets it for any command-line harness, and how to adapt a harness that is not headless.

## What the character does

1. The brain writes `{{task find the three cheapest 4K projectors under 300}}` inside a reply.
   The block is stripped before the voice; the words around it are spoken.
2. `talker/errands.py` queues it and, on its own thread, `POST`s it to the backend. The
   character's reply is not delayed by a millisecond; the HTTP happens elsewhere.
3. Every `--errand-poll` seconds (5) it `GET`s each open task. When one reports `done` it
   writes the summary to `faces/<face>/tasks.md` and queues an announcement.
4. The announcement is delivered the next time the room is quiet (she is not talking or
   thinking, nobody is mid-sentence, two seconds have passed): the brain gets
   `(Event: Task 3f2a "..." finished. Result: ...)` and speaks the conclusion in its own words.

Polling was chosen over callbacks on purpose: it works through any firewall, the character's
machine needs no open port, and a backend that is down is simply retried.

## The contract

Three JSON endpoints, no authentication (it is meant for a trusted LAN):

```
POST /tasks
  body     {"id": "3f2a", "task": "...", "context": "...", "from": "clara"}
  returns  {"id": "3f2a"}
```

`id` is the character's own four-hex id; use it if you can (the ledger and the log use it) or
return your own and the poller will map it. `context` is what the person had just said, for
flavour; it may be empty. `from` is the face name.

```
GET /tasks/{id}
  returns  {"id": "3f2a",
            "state": "queued" | "running" | "done" | "failed",
            "summary": "two or three spoken sentences, conclusion first",
            "result":  "the long form",
            "created": 1789253799.9, "updated": 1789253805.9}
```

Only `state`, `summary` and `result` are read. `summary` is what she will say, so write it
for the ear: no lists, no markdown, conclusion first. `result` goes into the ledger and is
never spoken; it may be long. On `failed`, put the reason in `summary`.

```
GET /tasks         -> [ ... every task, oldest first ... ]     (for people, not the poller)
GET /health        -> {"ok": true, "running": 1}
GET /capabilities  -> {"can": ["search the web and read pages", "read the calendar", ...]}   optional
```

`/capabilities` is optional but worth having: at startup the character asks for it and puts
the list in her standing instructions ("through that agent you can: ..."), so she hands
those requests off instead of saying she cannot. Without it, the `errands` field in face.json
may hold the same list by hand; the backend's answer wins when both exist. Keep the items
short and spoken-friendly; they are read by a model, not a program.

Anything else you return is ignored. Anything that raises or times out (5 s) is treated as
"backend unreachable" and retried next cycle; nothing is lost.

## The shim: `tools/agent_relay.py`

A stdlib-only HTTP server that implements the contract by running a command per task:

```bash
# on the machine with the harness
RELAY_COMMAND="claude -p" python tools/agent_relay.py             # listens on :8030
# on the character's machine, in .env
AGENT_RELAY_URL=http://agentbox:8030
```

The prompt is written to the command's **stdin** and its **stdout** is the result. That is the
whole integration surface. Per task it runs the command up to three times:

| Pass | Prompt on stdin | Output becomes |
|---|---|---|
| task | the task, plus the conversation context | `result` |
| review (only with `RELAY_ROUNDS=2`) | the task and the first answer, asked to critique and fix it | `result`, replacing the first |
| summary | the result, asked for two or three spoken sentences | `summary` |

The review pass is the "two agents talk to each other" mode: the same harness, a second time,
in the role of the colleague who checks the work. The character never sees the rounds.

Settings, as environment variables or flags: `RELAY_PORT` (8030), `RELAY_HOST` (0.0.0.0),
`RELAY_COMMAND` (`claude -p`), `RELAY_ROUNDS` (1), `RELAY_TIMEOUT` seconds per run (900),
`RELAY_WORKERS` tasks at once (2), `RELAY_STATE` the JSON file tasks persist in
(`relay_tasks.json`; a task interrupted by a restart comes back as `failed` with that reason).

Try it without any harness at all:

```bash
RELAY_COMMAND="cat" python tools/agent_relay.py --port 8031 &
curl -s -X POST localhost:8031/tasks -d '{"id":"t1","task":"say hello"}'
curl -s localhost:8031/tasks/t1
```

## Adapting a harness that is not headless

The shim needs one thing: a command that reads a prompt on stdin, does the work, prints the
answer, and exits. If your harness has a print or batch mode, `RELAY_COMMAND` is that mode
(Claude Code: `claude -p`; add `--allowedTools` or a `--permission-mode` flag as you see fit,
since nobody is there to click "allow"). If it does not, write a wrapper and point
`RELAY_COMMAND` at it. Three shapes that work:

**A harness with an HTTP or Python API.** A ten-line script: read stdin, call the API, print
the final text.

```python
#!/usr/bin/env python3
import sys, my_harness
task = sys.stdin.read()
session = my_harness.new_session(system="You are a careful research agent.")
print(session.run(task).final_text)
```

**A harness that drives an interactive terminal UI.** Start it under `pexpect` (or `expect`),
send the prompt as if typed, wait for its idle prompt, collect what it printed, quit. Brittle
but workable; prefer a real batch mode if the harness has one.

**A harness that is really a chat window.** Give it a folder it watches: the wrapper drops
`inbox/<id>.md`, waits for `outbox/<id>.md`, prints it. Whatever automation the window
supports (a hotkey, a plugin, a scheduled script) moves files from one folder to the other.

Whatever the shape, the wrapper should

- write **only the answer** to stdout; progress, logs and tool chatter go to stderr, which the
  shim keeps for its own log if the command fails;
- exit non-zero with a one-line reason on stderr when it cannot finish, so the character hears
  "the task failed: ..." instead of silence;
- run without a terminal or a display, and without needing anyone to click anything;
- finish within `RELAY_TIMEOUT` seconds, or raise the timeout.

If you would rather not go through a subprocess at all, implement the three endpoints
directly inside your harness; the whole contract is the section above and `tools/agent_relay.py`
is the reference for the JSON shapes and the state machine.

## Trying it end to end

1. Start the relay (or your own service) and `curl` a task through to `done`.
2. Put `AGENT_RELAY_URL` in the character's `.env`; the face needs `"errands": true`
   (Clara has it).
3. `python voice_loop.py --face clara --debug`. Startup prints
   `[errands] on: http://..., polled every 5s; 0 open in tasks.md`.
4. Ask for something that takes a while. You should see `[task] <id> queued`, then
   `[errands] <id> started`, and `faces/clara/tasks.md` showing `[in progress]`.
5. When it finishes: `[errands] <id> done`, then, as soon as the room is quiet, `[event] ...`
   and her spoken report. The ledger line becomes `[done]` with the summary beneath.
6. Ask "what is still open?" to see her read the ledger with `{{tool tasks}}`.

The control page (`/`) shows the backend, whether it is reachable, and the open/done/failed
counts, and has a **task** action to hand something to the backend by hand.
