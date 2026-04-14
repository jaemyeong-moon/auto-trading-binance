"""Tests for LLMProvider ABC and concrete implementations.

Strategy:
- All API calls replaced with unittest.mock — no real network activity.
- SDKs (anthropic, openai, google-genai) may not be installed; we inject
  fake modules via sys.modules so the import inside __init__ succeeds.
- Tests cover: chat(), chat_messages(), name property, missing-key errors,
  create_provider() auto-detection.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.llm_provider import (
    LLMProvider,
    Message,
    create_provider,
    detect_provider,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_message(role: str = "user", content: str = "hello") -> Message:
    return Message(role=role, content=content)


def _inject_fake_anthropic() -> tuple[MagicMock, MagicMock]:
    """Inject a fake 'anthropic' module into sys.modules.

    Returns (mock_client, mock_response_text_block) so callers can set return values.
    """
    mock_text_block = MagicMock()
    mock_text_block.text = "mocked anthropic response"

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    mock_anthropic_cls = MagicMock(return_value=mock_client)

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = mock_anthropic_cls  # type: ignore[attr-defined]
    sys.modules["anthropic"] = fake_module

    return mock_client, mock_text_block


def _inject_fake_openai() -> tuple[MagicMock, MagicMock]:
    """Inject a fake 'openai' module into sys.modules.

    Returns (mock_client, mock_message) so callers can set return values.
    """
    mock_message = MagicMock()
    mock_message.content = "mocked openai response"

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    mock_openai_cls = MagicMock(return_value=mock_client)

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = mock_openai_cls  # type: ignore[attr-defined]
    sys.modules["openai"] = fake_module

    return mock_client, mock_message


def _inject_fake_google_genai() -> tuple[MagicMock, MagicMock]:
    """Inject fake google / google.genai / google.genai.types modules.

    Returns (mock_client, mock_types).
    """
    mock_response = MagicMock()
    mock_response.text = "mocked gemini response"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    mock_genai_cls = MagicMock(return_value=mock_client)

    mock_types = MagicMock()
    mock_types.Content.side_effect = lambda role, parts: MagicMock(
        role=role, parts=parts
    )
    mock_types.Part.side_effect = lambda text: MagicMock(text=text)

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = mock_genai_cls  # type: ignore[attr-defined]
    fake_genai.types = mock_types  # type: ignore[attr-defined]

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai  # type: ignore[attr-defined]

    sys.modules["google"] = fake_google
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = mock_types

    return mock_client, mock_types


# ─── LLMProvider ABC contract ────────────────────────────────────────────────


class TestLLMProviderABC:
    """Verify that LLMProvider is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_methods(self):
        """A subclass missing any abstract method raises TypeError on instantiation."""

        class Incomplete(LLMProvider):  # type: ignore[abstract]
            @property
            def name(self) -> str:
                return "incomplete"

            def chat(self, prompt: str, max_tokens: int = 4096) -> str:
                return ""
            # chat_messages intentionally omitted

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ─── Message ─────────────────────────────────────────────────────────────────


class TestMessage:
    def test_role_and_content_stored(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_assistant_role(self):
        msg = Message(role="assistant", content="reply")
        assert msg.role == "assistant"


# ─── AnthropicProvider ───────────────────────────────────────────────────────


class TestAnthropicProvider:
    """Tests use a fake 'anthropic' module so the real SDK is not required."""

    def _make_provider(self) -> Any:
        mock_client, _ = _inject_fake_anthropic()
        from src.core.llm_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="test-key")
        # Direct reference to the client the provider is using
        return provider, mock_client

    def test_name_starts_with_anthropic(self):
        provider, _ = self._make_provider()
        assert provider.name.startswith("anthropic")

    def test_chat_returns_str(self):
        provider, _ = self._make_provider()
        result = provider.chat("test prompt")
        assert isinstance(result, str)
        assert result == "mocked anthropic response"

    def test_chat_passes_prompt_as_user_message(self):
        provider, mock_client = self._make_provider()
        provider.chat("my prompt")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "my prompt"

    def test_chat_messages_returns_str(self):
        provider, _ = self._make_provider()
        messages = [
            _make_message("user", "question"),
            _make_message("assistant", "answer"),
            _make_message("user", "follow-up"),
        ]
        result = provider.chat_messages(messages)
        assert isinstance(result, str)
        assert result == "mocked anthropic response"

    def test_chat_messages_maps_roles_correctly(self):
        provider, mock_client = self._make_provider()
        messages = [_make_message("user", "hi"), _make_message("assistant", "hey")]
        provider.chat_messages(messages)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"][0] == {"role": "user", "content": "hi"}
        assert call_kwargs["messages"][1] == {"role": "assistant", "content": "hey"}

    def test_max_tokens_forwarded(self):
        provider, mock_client = self._make_provider()
        provider.chat("prompt", max_tokens=512)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    def test_sdk_missing_raises_on_instantiation(self):
        """When the 'anthropic' module is absent, AnthropicProvider raises ImportError."""
        original = sys.modules.pop("anthropic", None)
        try:
            from src.core.llm_provider import AnthropicProvider
            with pytest.raises((ImportError, ModuleNotFoundError)):
                AnthropicProvider(api_key="key")
        finally:
            if original is not None:
                sys.modules["anthropic"] = original


# ─── OpenAIProvider ──────────────────────────────────────────────────────────


class TestOpenAIProvider:
    """Tests use a fake 'openai' module so the real SDK is not required."""

    def _make_provider(self) -> Any:
        mock_client, _ = _inject_fake_openai()
        from src.core.llm_provider import OpenAIProvider
        provider = OpenAIProvider(api_key="test-key")
        return provider, mock_client

    def test_name_starts_with_openai(self):
        provider, _ = self._make_provider()
        assert provider.name.startswith("openai")

    def test_chat_returns_str(self):
        provider, _ = self._make_provider()
        result = provider.chat("test prompt")
        assert isinstance(result, str)
        assert result == "mocked openai response"

    def test_chat_passes_prompt_as_user_message(self):
        provider, mock_client = self._make_provider()
        provider.chat("my prompt")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "my prompt"

    def test_chat_messages_returns_str(self):
        provider, _ = self._make_provider()
        messages = [_make_message("user", "q"), _make_message("assistant", "a")]
        result = provider.chat_messages(messages)
        assert isinstance(result, str)
        assert result == "mocked openai response"

    def test_chat_messages_maps_roles_correctly(self):
        provider, mock_client = self._make_provider()
        messages = [_make_message("user", "ping")]
        provider.chat_messages(messages)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0] == {"role": "user", "content": "ping"}

    def test_max_tokens_forwarded(self):
        provider, mock_client = self._make_provider()
        provider.chat("x", max_tokens=256)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 256

    def test_empty_content_returns_empty_string(self):
        """When OpenAI returns None content, result is an empty string."""
        provider, mock_client = self._make_provider()
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = None
        result = provider.chat("prompt")
        assert result == ""

    def test_sdk_missing_raises_on_instantiation(self):
        original = sys.modules.pop("openai", None)
        try:
            from src.core.llm_provider import OpenAIProvider
            with pytest.raises((ImportError, ModuleNotFoundError)):
                OpenAIProvider(api_key="key")
        finally:
            if original is not None:
                sys.modules["openai"] = original


# ─── GeminiProvider ──────────────────────────────────────────────────────────


class TestGeminiProvider:
    """Tests use fake google/google.genai modules so the real SDK is not required."""

    def _make_provider(self) -> Any:
        mock_client, mock_types = _inject_fake_google_genai()
        from src.core.llm_provider import GeminiProvider
        provider = GeminiProvider(api_key="test-key")
        return provider, mock_client, mock_types

    def test_name_starts_with_gemini(self):
        provider, _, __ = self._make_provider()
        assert provider.name.startswith("gemini")

    def test_chat_returns_str(self):
        provider, _, __ = self._make_provider()
        result = provider.chat("test prompt")
        assert isinstance(result, str)
        assert result == "mocked gemini response"

    def test_chat_passes_prompt_directly(self):
        provider, mock_client, _ = self._make_provider()
        provider.chat("my prompt")
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["contents"] == "my prompt"

    def test_chat_messages_returns_str(self):
        provider, _, __ = self._make_provider()
        messages = [_make_message("user", "q"), _make_message("assistant", "a")]
        result = provider.chat_messages(messages)
        assert isinstance(result, str)
        assert result == "mocked gemini response"

    def test_chat_messages_maps_assistant_to_model_role(self):
        """Gemini uses 'model' role instead of 'assistant'."""
        provider, mock_client, mock_types = self._make_provider()
        messages = [
            _make_message("user", "hi"),
            _make_message("assistant", "hello"),
        ]
        provider.chat_messages(messages)
        calls = mock_types.Content.call_args_list
        assert calls[0][1]["role"] == "user"
        assert calls[1][1]["role"] == "model"

    def test_chat_messages_user_role_preserved(self):
        provider, mock_client, mock_types = self._make_provider()
        messages = [_make_message("user", "question")]
        provider.chat_messages(messages)
        calls = mock_types.Content.call_args_list
        assert calls[0][1]["role"] == "user"

    def test_max_tokens_forwarded_as_config(self):
        provider, mock_client, _ = self._make_provider()
        provider.chat("x", max_tokens=1024)
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["config"]["max_output_tokens"] == 1024

    def test_empty_text_returns_empty_string(self):
        provider, mock_client, _ = self._make_provider()
        mock_client.models.generate_content.return_value.text = None
        result = provider.chat("prompt")
        assert result == ""

    def test_sdk_missing_raises_on_instantiation(self):
        for key in ("google", "google.genai", "google.genai.types"):
            sys.modules.pop(key, None)
        from src.core.llm_provider import GeminiProvider
        with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
            GeminiProvider(api_key="key")


# ─── Provider interface contract (parametrized) ──────────────────────────────


class TestProviderContract:
    """Each concrete provider must honour the LLMProvider ABC contract."""

    @pytest.fixture(params=["anthropic", "openai", "gemini"])
    def provider(self, request: pytest.FixtureRequest) -> LLMProvider:
        name = request.param

        if name == "anthropic":
            _inject_fake_anthropic()
            from src.core.llm_provider import AnthropicProvider
            p = AnthropicProvider(api_key="k")
            return p

        if name == "openai":
            _inject_fake_openai()
            from src.core.llm_provider import OpenAIProvider
            p = OpenAIProvider(api_key="k")
            return p

        # gemini
        _inject_fake_google_genai()
        from src.core.llm_provider import GeminiProvider
        p = GeminiProvider(api_key="k")
        return p

    def test_is_subclass_of_llmprovider(self, provider: LLMProvider):
        assert isinstance(provider, LLMProvider)

    def test_name_is_non_empty_str(self, provider: LLMProvider):
        assert isinstance(provider.name, str)
        assert len(provider.name) > 0

    def test_chat_returns_str(self, provider: LLMProvider):
        result = provider.chat("hello")
        assert isinstance(result, str)

    def test_chat_messages_returns_str(self, provider: LLMProvider):
        messages = [Message(role="user", content="hello")]
        result = provider.chat_messages(messages)
        assert isinstance(result, str)

    def test_chat_messages_multi_turn(self, provider: LLMProvider):
        messages = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
        ]
        result = provider.chat_messages(messages)
        assert isinstance(result, str)


# ─── detect_provider ─────────────────────────────────────────────────────────


class TestDetectProvider:
    def test_returns_none_when_no_keys_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert detect_provider() is None

    def test_detects_anthropic_first(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert detect_provider() == "anthropic"

    def test_detects_openai_when_anthropic_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert detect_provider() == "openai"

    def test_detects_gemini_last(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        assert detect_provider() == "gemini"

    def test_anthropic_takes_priority_over_openai(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
        monkeypatch.setenv("OPENAI_API_KEY", "o-key")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert detect_provider() == "anthropic"


# ─── create_provider ─────────────────────────────────────────────────────────


class TestCreateProvider:
    def test_raises_value_error_when_no_keys_and_no_name(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No LLM provider configured"):
            create_provider()

    def test_raises_value_error_for_unknown_provider_name(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(provider_name="unknown_llm")

    def test_creates_anthropic_provider_by_name(self):
        _inject_fake_anthropic()
        provider = create_provider(provider_name="anthropic", api_key="test-key")
        assert isinstance(provider, LLMProvider)
        assert provider.name.startswith("anthropic")

    def test_creates_openai_provider_by_name(self):
        _inject_fake_openai()
        provider = create_provider(provider_name="openai", api_key="test-key")
        assert isinstance(provider, LLMProvider)
        assert provider.name.startswith("openai")

    def test_creates_gemini_provider_by_name(self):
        _inject_fake_google_genai()
        provider = create_provider(provider_name="gemini", api_key="test-key")
        assert isinstance(provider, LLMProvider)
        assert provider.name.startswith("gemini")

    def test_auto_detects_provider_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        _inject_fake_openai()
        provider = create_provider()  # no provider_name → auto-detect
        assert provider.name.startswith("openai")

    def test_sdk_not_installed_raises_import_error(self):
        """When the SDK for the selected provider is missing, ImportError is raised."""
        # Remove any previously injected fake anthropic module
        sys.modules.pop("anthropic", None)
        with pytest.raises((ImportError, ModuleNotFoundError)):
            create_provider(provider_name="anthropic", api_key="k")
