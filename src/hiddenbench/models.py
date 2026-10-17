from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

from dotenv import load_dotenv


Message = dict[str, str]


class ModelClient(Protocol):
    """Minimal interface needed by HiddenBench."""

    model_name: str

    def chat(self, messages: list[Message]) -> str:
        """Return a free-form assistant message."""

    def chat_json(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON object matching the requested schema as closely as possible."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from model text."""

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unbalanced JSON object in model response")


PROVIDER_DEFAULTS = {
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": None},
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "together": {"api_key_env": "TOGETHER_API_KEY", "base_url": "https://api.together.xyz/v1"},
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
    "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
}


class OpenAICompatibleClient:
    """Adapter for OpenAI and OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        model: str,
        *,
        provider: str = "openai",
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> None:
        load_dotenv()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI dependency with `pip install hiddenbench[openai]`.") from exc

        defaults = PROVIDER_DEFAULTS.get(provider, {})
        env_name = api_key_env or defaults.get("api_key_env")
        resolved_key = api_key or (os.getenv(env_name) if env_name else None)
        if not resolved_key:
            raise ValueError(f"Missing API key. Set {env_name or 'an API key'} or pass api_key.")

        resolved_base_url = base_url if base_url is not None else defaults.get("base_url")
        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if timeout is not None:
            kwargs["timeout"] = timeout

        self.client = OpenAI(**kwargs)
        self.model_name = model
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        # Keep request/response details for reproducible diagnostics. API keys are
        # intentionally never included in these records.
        self.json_call_traces: list[dict[str, Any]] = []

    def _completion(self, messages: list[Message], response_format: dict[str, Any] | None = None) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover - exercised in live runs.
                last_error = exc
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error

    def chat(self, messages: list[Message]) -> str:
        return self._completion(messages)

    def chat_json(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        json_instruction = {
            "role": "user",
            "content": "Return only one valid JSON object. Do not include markdown or commentary.",
        }
        request_messages = messages + [json_instruction]
        # OpenAI JSON mode guarantees syntax only. Strict Structured Outputs also
        # enforces the caller's schema (including the task's allowed vote options).
        # Keep JSON mode for compatible providers whose schema support varies.
        if self.provider == "openai":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "hiddenbench_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        trace: dict[str, Any] = {
            "model": self.model_name,
            "provider": self.provider,
            "messages": request_messages,
            "response_format": response_format,
        }
        try:
            raw_response = self._completion(request_messages, response_format=response_format)
            trace["raw_response"] = raw_response
            parsed = extract_json_object(raw_response)
            trace["parsed_response"] = parsed
            return parsed
        except Exception as exc:
            trace["error_type"] = type(exc).__name__
            trace["error"] = str(exc)
            raise
        finally:
            self.json_call_traces.append(trace)


def build_model_client(
    *,
    provider: str,
    model: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_retries: int = 3,
) -> ModelClient:
    if provider == "openai-compatible":
        if not base_url:
            raise ValueError("--base-url is required for provider=openai-compatible")
        return OpenAICompatibleClient(
            model,
            provider=provider,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_retries=max_retries,
        )
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported provider {provider!r}. Use one of {sorted(PROVIDER_DEFAULTS)} or openai-compatible.")
    return OpenAICompatibleClient(
        model,
        provider=provider,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        temperature=temperature,
        max_retries=max_retries,
    )
