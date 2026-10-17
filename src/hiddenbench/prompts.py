from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSet:
    system_prompt: str
    first_user_prompt: str
    user_prompt: str
    first_vote_prompt: str
    vote_prompt: str
    generation_prompt: str | None = None
    prompt_version: str = "custom"
    prompt_file: str | None = None
    coordinator_system_prompt: str | None = None
    coordinator_runtime_prompt: str | None = None
    coordinator_user_prompt: str | None = None
    checkpoint_vote_prompt: str | None = None


PROMPT_KEYS = (
    "system_prompt",
    "first_user_prompt",
    "user_prompt",
    "first_vote_prompt",
    "vote_prompt",
    "generation_prompt",
)
V3_PROMPT_KEYS = ("coordinator_system_prompt", "coordinator_runtime_prompt", "coordinator_user_prompt", "checkpoint_vote_prompt")


def default_prompt_file(version: str = "v1") -> Path:
    """Return the single JSON file containing all bundled prompt templates."""
    if version not in {"v1", "v2", "v3", "v4", "v4.1"}:
        raise ValueError("prompt version must be v1, v2, v3, v4 or v4.1")
    filename = "prompts.json" if version == "v1" else "prompts_" + version.replace(".", "_") + ".json"
    return Path(__file__).resolve().parent / filename


def load_prompts(path: str | Path | None = None, *, version: str = "v1") -> PromptSet:
    prompt_file = Path(path) if path is not None else default_prompt_file(version)
    data = json.loads(prompt_file.read_text(encoding="utf-8"))
    unknown = set(data) - set(PROMPT_KEYS + V3_PROMPT_KEYS)
    missing = set(PROMPT_KEYS) - set(data)
    if unknown or missing:
        raise ValueError(
            f"Invalid prompt file {prompt_file}: missing keys={sorted(missing)}, "
            f"unknown keys={sorted(unknown)}"
        )
    prompts: dict[str, str | None] = {}
    for key in data:
        value = data[key]
        if key == "generation_prompt" and value is None:
            prompts[key] = None
            continue
        if isinstance(value, str):
            # Keep support for a single string per prompt for custom prompt files.
            prompts[key] = value
        elif isinstance(value, list) and all(isinstance(line, str) for line in value):
            prompts[key] = "\n\n".join(value)
        else:
            raise ValueError(f"Prompt {key!r} in {prompt_file} must be a string, string array, or null")
    resolved = prompt_file.resolve()
    selected_version = next(
        (v for v in ("v1", "v2", "v3", "v4", "v4.1") if resolved == default_prompt_file(v)), "custom"
    )
    if selected_version in {"v3", "v4", "v4.1"} and any(not prompts.get(key) for key in V3_PROMPT_KEYS):
        raise ValueError(f"{selected_version.upper()} requires coordinator and checkpoint prompt templates")
    return PromptSet(**prompts, prompt_version=selected_version, prompt_file=str(resolved))


def render_prompt(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"%{key}%", value)
    return rendered
