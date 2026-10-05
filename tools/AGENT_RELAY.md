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
GET /tasks      -> [ ... every task, oldest first ... ]     (for people, not the poller)
GET /health     -> {"ok": true, "running": 1}
```

What the agent can do is not part of the contract: it is listed by hand under `errands` in
the face's `face.json` (a list of short spoken-friendly phrases), and goes into her standing
instructions so she hands those requests off instead of saying she cannot. Keep that list in
step with the tools your harness actually exposes on the `/tasks` path.

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

## Taking her with you

The character is a thin client: microphone, brain, voice. The agent stays at home doing the
long work. That split is the point, and it is also what breaks the moment she leaves the
house, because `http://agent.local:8080` is an mDNS name for a private address that does not
exist on hotel wifi or a phone hotspot. Her errands would queue forever.

**Put both machines on a tailnet.** It needs no router changes, exposes nothing to the
internet, works through carrier NAT, and gives one address that resolves at home *and* away,
so there is nothing to switch when a trip starts. On Fedora, on the agent box and on whatever
she travels on:

```bash
sudo dnf config-manager addrepo --from-repofile=https://pkgs.tailscale.com/stable/fedora/tailscale.repo
sudo dnf install -y tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up                      # browser login, once per machine
tailscale status                       # note the agent box's name, and whether the path is direct
```

On Raspberry Pi OS the first two lines become `curl -fsSL https://tailscale.com/install.sh | sh`;
the rest is identical. Then in her `.env`, which is gitignored:

```
AGENT_RELAY_URL=http://agentbox.<tailnet>.ts.net:8080
AGENT_RELAY_TOKEN=<the shared secret below>
```

Two things to check rather than assume: that the agent service binds `0.0.0.0` and not
`127.0.0.1`, or it will not answer on the tailnet interface; and that `tailscale status` shows
a direct connection rather than a relayed one, since a relayed path adds latency. If it is
relayed, raise `--agent-timeout` (default 10 s) or the `agent_timeout_s` tunable on the
control page.

The address can also be changed while she is running, from the control page's **Agent** tab,
and it is saved to `settings.json` for the next start. That is the escape hatch when DNS
misbehaves in a hotel.

**Then make the token mean something.** She sends `X-Relay-Token` on every request whenever
`AGENT_RELAY_TOKEN` is set; a relay that ignores it is wide open to anyone who can reach the
port. On a private tailnet that is defence in depth; the first time anything is tunnelled
publicly it is the only thing between a stranger and your calendar and your email. So:

- `POST /tasks` and the `GET /tasks*` routes compare `X-Relay-Token` against a secret from the
  relay's own environment and return **401** when it is missing or wrong.
- `GET /health` stays open, so the reachability probe and the control page's Test button keep
  working, and neither reveals anything.

**What she does when she cannot reach it.** Nothing is lost: the task waits in the outbox and
is posted the moment the agent answers again. She also says so out loud, once, rather than
appearing to ignore the request — and says again when it goes over. Idle polling costs
nothing on mobile data: a cycle with no task in flight makes no request at all.

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
