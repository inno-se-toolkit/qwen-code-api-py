"""Message transformations: system prompt injection and cache_control tagging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_config import log

_cached_system_prompt: str | None = None


def _load_system_prompt() -> str:
    global _cached_system_prompt
    if _cached_system_prompt is not None:
        return _cached_system_prompt

    path = Path.cwd() / "sys-prompt.txt"
    try:
        _cached_system_prompt = path.read_text().strip()
    except FileNotFoundError:
        _cached_system_prompt = ""
    return _cached_system_prompt


def _add_cache_control(message: dict[str, Any]) -> dict[str, Any]:
    """Add cache_control to the last content item of a message."""
    content = message.get("content")

    if isinstance(content, str):
        return {
            **message,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }

    if isinstance(content, list) and content:
        new_parts: list[Any] = [*content]
        last: Any = new_parts[-1]
        if isinstance(last, dict):
            new_parts[-1] = {**last, "cache_control": {"type": "ephemeral"}}
        return {**message, "content": new_parts}

    return message


def transform_messages(
    messages: list[dict[str, Any]],
    model: str,
    *,
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Inject system prompt and add cache_control matching the real client.

    Streaming: cache_control on system message + last message.
    Non-streaming: cache_control on system message only.
    """
    transformed = list(messages)

    prompt = _load_system_prompt()

    # Inject system prompt
    sys_idx = next(
        (i for i, m in enumerate(transformed) if m.get("role") == "system"), None
    )

    if prompt:
        if sys_idx is not None:
            existing = transformed[sys_idx]
            if isinstance(existing.get("content"), list):
                old_text = "".join(
                    p.get("text", "")  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType,reportUnknownVariableType]
                    for p in existing["content"]
                    if isinstance(p, dict)
                )
            else:
                old_text = str(existing.get("content", ""))
            merged = f"{prompt}\n\n---\n\n{old_text}"
            transformed[sys_idx] = {**existing, "content": merged}
        else:
            transformed.insert(0, {"role": "system", "content": prompt})
            sys_idx = 0
        log.debug("System prompt injected (%d chars)", len(prompt))

    # Apply cache_control to system message (always) and last message (streaming only)
    if sys_idx is not None:
        transformed[sys_idx] = _add_cache_control(transformed[sys_idx])

    if streaming and transformed:
        last_idx = len(transformed) - 1
        if last_idx != sys_idx:
            transformed[last_idx] = _add_cache_control(transformed[last_idx])

    return transformed
