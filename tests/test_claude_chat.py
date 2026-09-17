import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx2 as httpx
import pytest
pytest.importorskip("anthropic")
from talker.brains.claude_chat import ClaudeChat


class FakeStream:
    def __init__(self, chunks): self.text_stream = iter(chunks)
    def get_final_message(self):
        class M: usage = None
        return M()
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeClient:
    """Mimics client.beta.messages.stream(...) and records the request."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []
        self.retries = None            # what with_options(max_retries=...) was given
        self.raise_on_fast = None      # an exception the fast attempt should raise
        outer = self
        class Messages:
            def stream(self, **kw):
                outer.calls.append(kw)
                if kw.get("speed") == "fast" and outer.raise_on_fast is not None:
                    raise outer.raise_on_fast
                return FakeStream(outer.chunks)
        class Beta:
            messages = Messages()
        self.beta = Beta()

    def with_options(self, **kw):
        self.retries = kw.get("max_retries")
        return self


def test_reply_streams_and_keeps_history():
    client = FakeClient(["[happy]Hi ", "there!"])
    chat = ClaudeChat(character="a robot", client=client)
    assert "".join(chat.reply("hello")) == "[happy]Hi there!"
    assert chat.messages == [{"role": "user", "content": "hello"},
                             {"role": "assistant", "content": "[happy]Hi there!"}]
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5"
    assert kw["output_config"] == {"effort": "low"}
    assert "thinking" not in kw
    assert kw["fallbacks"] == "default"
    system_text = kw["system"][0]["text"]
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Character: a robot" in system_text
    assert "[angry]" in system_text            # emotion guide included
    list(chat.reply("and again"))
    assert len(chat.messages) == 4 and client.calls[1]["messages"][0]["content"] == "hello"


def test_failed_turn_leaves_no_dangling_user_message():
    class Boom(FakeClient):
        def __init__(self):
            super().__init__([])
            def stream(**kw): raise RuntimeError("down")
            self.beta.messages.stream = stream
    chat = ClaudeChat(client=Boom())
    with pytest.raises(RuntimeError):
        list(chat.reply("hello"))
    assert chat.messages == []


def test_openai_compat_chat_streams_and_keeps_history():
    pytest.importorskip("openai")
    from talker.brains.openai_compat_chat import OpenAICompatChat
    from types import SimpleNamespace as NS

    class FakeClient:
        def __init__(self):
            self.calls = []
            outer = self
            class Completions:
                def create(self, **kw):
                    outer.calls.append(kw)
                    return iter([NS(choices=[NS(delta=NS(content="[happy]Hi "))]),
                                 NS(choices=[]),
                                 NS(choices=[NS(delta=NS(content="there!"))])])
            self.chat = NS(completions=Completions())
    client = FakeClient()
    chat = OpenAICompatChat("gpt-4o-mini", api_key=None, client=client, character="a cat")
    assert "".join(chat.reply("hello")) == "[happy]Hi there!"
    kw = client.calls[0]
    assert kw["stream"] is True and kw["messages"][0]["role"] == "system"
    assert "Character: a cat" in kw["messages"][0]["content"]
    assert chat.messages[-1] == {"role": "assistant", "content": "[happy]Hi there!"}


def test_system_prompt_starts_at_the_heading_not_the_notes():
    from talker.brains.claude_chat import load_system_prompt
    text = load_system_prompt()
    assert text.startswith("You are a character speaking out loud")
    assert "heading automatically" not in text


def test_vision_rule_only_when_the_brain_can_see():
    blind = ClaudeChat(client=FakeClient(["x"]))
    seeing = ClaudeChat(client=FakeClient(["x"]), can_see=True)
    assert "You can see" not in blind.system
    assert "You can see" in seeing.system and "You notice" in seeing.system


def test_context_notes_queue_up_and_a_failed_turn_drops_its_notice():
    from talker.brains.claude_chat import assistant_rules
    client = FakeClient(["fine."])
    chat = ClaudeChat(character="x", client=client, extra_rules=assistant_rules(memory=True, errands=True))
    assert "{{note" in chat.system and "{{task" in chat.system and "Event:" in chat.system
    chat.add_context("the door opened")
    chat.add_context("the tasks tool answered: - [done] 3f2a")
    chat.add_context("the door opened")                      # a repeat is not queued twice
    "".join(chat.reply("what's new?"))
    assert chat.messages[0] == {"role": "user", "content": "(You notice: the door opened\nthe tasks tool answered: - [done] 3f2a)"}
    assert chat.messages[1]["content"] == "what's new?"
    assert chat._pending_context == []
    # a turn that yields nothing removes both its user line and the notice pushed with it
    client.chunks = []
    chat.add_context("later")
    "".join(chat.reply("again"))
    assert [m["content"] for m in chat.messages][-1] == "fine."
    assert not any("later" in m["content"] for m in chat.messages)


def test_summarise_leaves_history_alone():
    client = FakeClient(["We planned the offsite."])
    chat = ClaudeChat(character="x", client=client)
    assert chat.summarise("sum up") == ""                    # nothing to summarise yet
    "".join(chat.reply("hello"))
    before = list(chat.messages)
    client.chunks = ["We planned the offsite."]
    assert chat.summarise("sum up") == "We planned the offsite."
    assert chat.messages == before
    assert client.calls[-1]["messages"][-1] == {"role": "user", "content": "sum up"}


def test_errand_abilities_are_told_to_her():
    from talker.brains.claude_chat import assistant_rules
    from talker.face_asset_loader import FaceManifest
    m = FaceManifest.from_dict({"errands": ["read the calendar", "search the web"]})
    assert m.errands and m.errands_can == "read the calendar, search the web"
    assert FaceManifest.from_dict({"errands": True}).errands_can == ""
    assert FaceManifest.from_dict({"errands": {"can": "run code"}}).errands_can == "run code"
    rules = assistant_rules(errands=True, can=m.errands_can)
    assert "you can: read the calendar, search the web" in rules and "Never say you cannot" in rules
    assert "Through that agent" not in assistant_rules(errands=True)


def test_fast_mode_is_a_model_suffix():
    from talker.brains.claude_chat import FAST_BETA, split_speed
    assert split_speed("claude-opus-5-fast") == ("claude-opus-5", "fast")
    assert split_speed("claude-opus-4-8-fast") == ("claude-opus-4-8", "fast")
    assert split_speed("claude-opus-5") == ("claude-opus-5", None)
    assert split_speed("claude-haiku-4-5-fast") == ("claude-haiku-4-5", None)   # not offered there
    client = FakeClient(["fast enough."])
    chat = ClaudeChat(model="claude-opus-5-fast", client=client)
    assert chat.model == "claude-opus-5" and chat.speed == "fast"
    assert "".join(chat.reply("hello")) == "fast enough."
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5" and kw["speed"] == "fast"
    assert FAST_BETA in kw["betas"] and "server-side-fallback-2026-07-01" in kw["betas"]
    assert client.retries == 0                          # no waiting through backoff on a fast turn
    # standard speed sends neither
    plain = FakeClient(["hi."])
    "".join(ClaudeChat(model="claude-opus-5", client=plain).reply("hello"))
    assert "speed" not in plain.calls[0] and FAST_BETA not in plain.calls[0]["betas"]


def test_a_rate_limited_fast_turn_still_answers_at_standard_speed():
    import anthropic
    client = FakeClient(["standard answer."])
    chat = ClaudeChat(model="claude-opus-5-fast", client=client)
    client.raise_on_fast = anthropic.RateLimitError(
        "rate limited", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)
    assert "".join(chat.reply("hello")) == "standard answer."
    assert client.calls[0].get("speed") == "fast" and "speed" not in client.calls[1]
    assert chat.speed == "fast"                         # a transient rate limit is this turn only
    assert chat.messages[-1] == {"role": "assistant", "content": "standard answer."}
    for _ in range(2):                                  # ... three in a row backs off for a while
        client.chunks = ["again."]
        "".join(chat.reply("more"))
    assert chat.speed == "fast" and chat._fast_pause_until > 0
    client.calls.clear(); client.chunks = ["quiet."]
    "".join(chat.reply("more"))
    assert "speed" not in client.calls[0]                # not asked for during the pause
    chat._fast_pause_until = 0                           # ... and asked for again after it
    client.raise_on_fast = None; client.calls.clear(); client.chunks = ["quick."]
    assert "".join(chat.reply("more")) == "quick."
    assert client.calls[0]["speed"] == "fast" and chat._fast_429 == 0

def test_no_access_to_the_preview_turns_fast_mode_off_for_the_session():
    import anthropic
    client = FakeClient(["plain answer."])
    chat = ClaudeChat(model="claude-opus-5-fast", client=client)
    client.raise_on_fast = anthropic.PermissionDeniedError(
        "not entitled", response=httpx.Response(403, request=httpx.Request("POST", "http://x")), body=None)
    assert "".join(chat.reply("hello")) == "plain answer."
    assert chat.speed is None
    client.calls.clear(); client.chunks = ["again."]
    assert "".join(chat.reply("more")) == "again."
    assert "speed" not in client.calls[0]               # not tried again this session


def test_openai_fast_mode_uses_the_service_tier():
    pytest.importorskip("openai")
    from talker.brains.openai_compat_chat import OpenAICompatChat
    assert OpenAICompatChat.split_speed("gpt-5.6-fast") == ("gpt-5.6", "fast")
    assert OpenAICompatChat.split_speed("gpt-4o-mini") == ("gpt-4o-mini", None)

    class FakeChunk:
        def __init__(self, text):
            self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})()]

    class FakeOpenAI:
        def __init__(self):
            self.calls = []
            self.fail_fast = False
            outer = self
            class Completions:
                def create(self, **kw):
                    outer.calls.append(kw)
                    if kw.get("service_tier") and outer.fail_fast:
                        raise RuntimeError("service_tier 'fast' is not available for this model")
                    return iter([FakeChunk("hello "), FakeChunk("there.")])
            class Chat:
                completions = Completions()
            self.chat = Chat()

    client = FakeOpenAI()
    chat = OpenAICompatChat("gpt-5.6-fast", api_key=None, client=client)
    assert chat.model == "gpt-5.6" and chat.speed == "fast"
    assert "".join(chat.reply("hi")) == "hello there."
    assert client.calls[0]["service_tier"] == "fast" and client.calls[0]["model"] == "gpt-5.6"
    # a model without the tier: answered at standard speed, and not retried after that
    client2 = FakeOpenAI(); client2.fail_fast = True
    chat2 = OpenAICompatChat("gpt-5.6-fast", api_key=None, client=client2)
    assert "".join(chat2.reply("hi")) == "hello there."
    assert chat2.speed is None and "service_tier" not in client2.calls[1]


def test_a_zero_fast_limit_backs_off_on_the_first_try():
    import anthropic
    client = FakeClient(["plain."])
    chat = ClaudeChat(model="claude-opus-5-fast", client=client)
    client.raise_on_fast = anthropic.RateLimitError(
        "Error code: 429 - would exceed your rate limit of 0 fast mode input tokens per minute",
        response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)
    assert "".join(chat.reply("hello")) == "plain."
    assert chat._fast_pause_until > 0                    # straight to the back-off
    client.calls.clear(); client.chunks = ["again."]
    "".join(chat.reply("more"))
    assert "speed" not in client.calls[0]
