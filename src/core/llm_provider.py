"""LLM Provider — 멀티 LLM 지원 추상화 레이어.

지원 프로바이더:
- anthropic (Claude)  : ANTHROPIC_API_KEY
- openai   (GPT)      : OPENAI_API_KEY
- gemini   (Gemini)   : GEMINI_API_KEY

.env에 설정된 키를 자동 감지하거나, ai_llm_provider 설정으로 지정.
"""

import os
from abc import ABC, abstractmethod

import structlog

logger = structlog.get_logger()


class Message:
    """대화 메시지."""
    def __init__(self, role: str, content: str) -> None:
        self.role = role      # "user" or "assistant"
        self.content = content


class LLMProvider(ABC):
    """LLM 호출 추상 인터페이스."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def chat(self, prompt: str, max_tokens: int = 4096) -> str:
        """단일 프롬프트를 보내고 텍스트 응답을 반환."""
        ...

    @abstractmethod
    def chat_messages(self, messages: list[Message],
                      max_tokens: int = 4096) -> str:
        """멀티턴 대화를 보내고 텍스트 응답을 반환."""
        ...


# ─── Anthropic (Claude) ──────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API."""

    def __init__(self, api_key: str | None = None,
                 model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def chat(self, prompt: str, max_tokens: int = 4096) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def chat_messages(self, messages: list[Message],
                      max_tokens: int = 4096) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=msgs,
        )
        return response.content[0].text


# ─── OpenAI (GPT) ────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI GPT API."""

    def __init__(self, api_key: str | None = None,
                 model: str = "gpt-4o") -> None:
        from openai import OpenAI
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self._model = model

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def chat(self, prompt: str, max_tokens: int = 4096) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def chat_messages(self, messages: list[Message],
                      max_tokens: int = 4096) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=msgs,
        )
        return response.choices[0].message.content or ""


# ─── Google Gemini ────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """Google Gemini API."""

    def __init__(self, api_key: str | None = None,
                 model: str = "gemini-2.5-flash") -> None:
        from google import genai
        from google.genai import types
        self._client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        )
        self._model = model
        self._types = types

    @property
    def name(self) -> str:
        return f"gemini/{self._model}"

    def chat(self, prompt: str, max_tokens: int = 4096) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={"max_output_tokens": max_tokens},
        )
        return response.text or ""

    def chat_messages(self, messages: list[Message],
                      max_tokens: int = 4096) -> str:
        # Gemini: Content 리스트로 변환
        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            contents.append(
                self._types.Content(
                    role=role,
                    parts=[self._types.Part(text=m.content)],
                )
            )
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config={"max_output_tokens": max_tokens},
        )
        return response.text or ""


# ─── 프로바이더 자동 감지 & 팩토리 ───────────────────────

# 우선순위: 설정 > 환경변수 자동 감지
_PROVIDER_MAP = {
    "anthropic": ("ANTHROPIC_API_KEY", AnthropicProvider),
    "openai": ("OPENAI_API_KEY", OpenAIProvider),
    "gemini": ("GEMINI_API_KEY", GeminiProvider),
}

# 자동 감지 순서 (키가 있는 첫 번째 프로바이더 사용)
_AUTO_DETECT_ORDER = ["anthropic", "openai", "gemini"]


def detect_provider() -> str | None:
    """환경변수에서 사용 가능한 LLM 프로바이더를 자동 감지.

    Returns:
        프로바이더 이름 ("anthropic", "openai", "gemini") 또는 None
    """
    for provider_name in _AUTO_DETECT_ORDER:
        env_key, _ = _PROVIDER_MAP[provider_name]
        if os.environ.get(env_key):
            return provider_name
    return None


def create_provider(provider_name: str | None = None,
                    api_key: str | None = None) -> LLMProvider:
    """LLM 프로바이더 인스턴스 생성.

    Args:
        provider_name: "anthropic", "openai", "gemini". None이면 자동 감지.
        api_key: API 키. None이면 환경변수에서 읽음.

    Raises:
        ValueError: 프로바이더를 찾을 수 없을 때
        ImportError: 해당 SDK가 설치되지 않았을 때
    """
    if provider_name is None:
        provider_name = detect_provider()

    if provider_name is None:
        raise ValueError(
            "No LLM provider configured. Set one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY"
        )

    if provider_name not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(_PROVIDER_MAP.keys())}"
        )

    env_key, provider_cls = _PROVIDER_MAP[provider_name]

    try:
        provider = provider_cls(api_key=api_key)
    except ImportError:
        pkg_map = {
            "anthropic": "anthropic",
            "openai": "openai",
            "gemini": "google-genai",
        }
        pkg = pkg_map.get(provider_name, provider_name)
        raise ImportError(
            f"{provider_name} SDK not installed. Run: pip install {pkg}"
        )

    logger.info("llm.provider_created", provider=provider.name)
    return provider
