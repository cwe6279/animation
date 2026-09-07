import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
pytest.importorskip("anthropic")
from llm_integration.claude_chat import ClaudeChat


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
        outer = self
        class Messages:
            def stream(self, **kw):
                outer.calls.append(kw)
                return FakeStream(outer.chunks)
        class Beta:
            messages = Messages()
        self.beta = Beta()


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
    from llm_integration.openai_compat_chat import OpenAICompatChat
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
    from llm_integration.claude_chat import load_system_prompt
    text = load_system_prompt()
    assert text.startswith("You are a character speaking out loud")
    assert "heading automatically" not in text


def test_vision_rule_only_when_the_brain_can_see():
    blind = ClaudeChat(client=FakeClient(["x"]))
    seeing = ClaudeChat(client=FakeClient(["x"]), can_see=True)
    assert "You can see" not in blind.system
    assert "You can see" in seeing.system and "context entries" in seeing.system
