# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert smolagents message and tool shapes into util-genai GenAI types.

smolagents passes ``generate(messages=...)`` a list of ``ChatMessage`` objects
or plain dicts, and returns a ``ChatMessage``. This module maps those, and the
``tools_to_call_from`` tool objects, onto the types in
``opentelemetry.util.genai.types``. util-genai then serializes them into
``gen_ai.input.messages``, ``gen_ai.output.messages``, and
``gen_ai.tool.definitions``.
"""

from __future__ import annotations

import base64
import binascii
import logging
from enum import Enum
from typing import Any

from opentelemetry.util.genai.types import (
    Blob,
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolDefinition,
    Uri,
)

_logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_MIME_TYPE = "image/png"
_DATA_URL_PREFIX = "data:"

# smolagents-internal roles -> semconv ``gen_ai`` message roles. smolagents
# applies the same mapping (``models.tool_role_conversions``) inside
# ``generate``, after the wrapper has already read ``messages``, so the wrapper
# has to apply it too. Without this map the input messages would carry roles
# the spec doesn't define. A model configured with ``custom_role_conversions``
# overrides the mapping and can send different roles. Roles that are already
# spec values pass through unchanged.
_ROLE_MAP: dict[str, str] = {
    "tool-call": "assistant",
    "tool-response": "user",
}


def _unwrap_role(role: Any) -> str | None:
    if role is None:
        return None
    if isinstance(role, Enum):
        role = role.value
    name = str(role)
    return _ROLE_MAP.get(name, name)


def _decode_base64_image(image: str) -> tuple[bytes, str] | None:
    """Decode a base64 payload or data URL into ``(bytes, mime_type)``."""
    mime_type = _DEFAULT_IMAGE_MIME_TYPE
    if image.startswith(_DATA_URL_PREFIX):
        header, _, image = image.partition(",")
        media_type = header[len(_DATA_URL_PREFIX) :].split(";")[0]
        if media_type:
            mime_type = media_type
    try:
        return base64.b64decode(image, validate=True), mime_type
    except (binascii.Error, ValueError):
        _logger.debug("Failed to decode a base64 image", exc_info=True)
        return None


def _encode_image_base64(image: Any) -> str | None:
    try:
        from smolagents.utils import (  # pylint: disable=import-outside-toplevel
            encode_image_base64,
        )
    except ImportError:
        _logger.debug("smolagents.utils.encode_image_base64 is unavailable")
        return None
    try:
        encoded = encode_image_base64(image)
    except Exception:  # pylint: disable=broad-except
        _logger.debug(
            "Failed to encode image of type %s, dropping it from telemetry",
            type(image).__name__,
            exc_info=True,
        )
        return None
    return encoded if isinstance(encoded, str) else None


def _image_blob(image: Any) -> Blob | None:
    """Build a ``Blob`` part from a base64 string, data URL, or PIL image."""
    if isinstance(image, str):
        decoded = _decode_base64_image(image)
    else:
        encoded = _encode_image_base64(image)
        decoded = (
            _decode_base64_image(encoded) if encoded is not None else None
        )
    if decoded is None:
        return None
    content, mime_type = decoded
    return Blob(mime_type=mime_type, modality="image", content=content)


def _image_part_from_element(element: dict[str, Any]) -> Uri | Blob | None:
    content_type = element.get("type")
    if content_type == "image_url":
        image_url = element.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else None
        if isinstance(url, str) and url:
            return Uri(mime_type=None, modality="image", uri=url)
        return None
    if content_type == "image":
        image = element.get("image")
        if image is not None:
            return _image_blob(image)
    return None


def _parts_from_content(content: Any) -> list[MessagePart]:
    parts: list[MessagePart] = []
    if isinstance(content, str):
        parts.append(Text(content=content))
        return parts
    if isinstance(content, list):
        for element in content:
            if not isinstance(element, dict):
                _logger.debug(
                    "Unknown message content dropped from telemetry: %s",
                    type(element).__name__,
                )
                continue
            if element.get("type") == "text" and (text := element.get("text")):
                parts.append(Text(content=text))
                continue
            if image_part := _image_part_from_element(element):
                parts.append(image_part)
            else:
                _logger.debug(
                    "Unknown message part dropped from telemetry: %s",
                    element.get("type"),
                )
    return parts


def _get_role_and_content(message: Any) -> tuple[Any, Any]:
    if isinstance(message, dict):
        return message.get("role"), message.get("content")
    return getattr(message, "role", None), getattr(message, "content", None)


def to_input_messages(messages: Any) -> list[InputMessage]:
    """Map smolagents ``generate`` input messages to ``InputMessage`` objects."""
    result: list[InputMessage] = []
    if not isinstance(messages, list):
        return result
    for message in messages:
        raw_role, content = _get_role_and_content(message)
        role = _unwrap_role(raw_role)
        if not role:
            continue
        result.append(
            InputMessage(role=role, parts=_parts_from_content(content))
        )
    return result


def to_output_message(output_message: Any) -> OutputMessage:
    """Map a smolagents ``ChatMessage`` response to an ``OutputMessage``.

    The in-process runtimes return the generated text and nothing else: no tool
    calls (the agent parses those out of the text afterwards), no reasoning
    content, and no finish reason. ``OutputMessage`` requires
    ``finish_reason``, and util-genai drops an empty value when it emits
    ``gen_ai.response.finish_reasons``. Defaulting it to ``"stop"`` instead
    would make a generation cut short by ``max_new_tokens`` look like a natural
    stop.
    """
    role = _unwrap_role(getattr(output_message, "role", None)) or "assistant"
    parts = _parts_from_content(getattr(output_message, "content", None))
    return OutputMessage(role=role, parts=parts, finish_reason="")


def _tool_parameters(tool: Any) -> dict[str, Any] | None:
    """Return the JSON Schema ``parameters`` object for a smolagents tool.

    A tool's ``inputs`` map is not a JSON Schema on its own: smolagents wraps it
    in an object schema, derives ``required`` from ``nullable``, and rewrites its
    non-JSON-Schema ``"any"`` type. ``get_tool_json_schema`` builds exactly the
    schema the provider receives.
    """
    try:
        from smolagents.models import (  # pylint: disable=import-outside-toplevel
            get_tool_json_schema,
        )
    except ImportError:
        _logger.debug("smolagents.models.get_tool_json_schema is unavailable")
        return None
    try:
        schema = get_tool_json_schema(tool)
        parameters = schema["function"]["parameters"]
    except Exception:  # pylint: disable=broad-except
        _logger.debug(
            "Failed to build a JSON Schema for tool %s",
            getattr(tool, "name", None),
            exc_info=True,
        )
        return None
    return parameters if isinstance(parameters, dict) else None


def to_tool_definitions(tools: Any) -> list[ToolDefinition] | None:
    """Map smolagents tool objects to function tool definitions."""
    if not isinstance(tools, list) or not tools:
        return None
    definitions: list[ToolDefinition] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not name:
            continue
        definitions.append(
            FunctionToolDefinition(
                name=name,
                description=getattr(tool, "description", None),
                parameters=_tool_parameters(tool),
            )
        )
    return definitions or None
