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
from collections.abc import Mapping
from enum import Enum
from typing import Any

from opentelemetry.util.genai.types import (
    Blob,
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCallRequest,
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

# Amazon Bedrock reports the stop reason as ``stopReason`` using Anthropic's
# vocabulary. Normalize it the way the anthropic instrumentation does
# (``anthropic/utils.py``); anything unmapped passes through.
_STOP_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
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
        from smolagents.utils import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
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


def _raw_value(raw: Any, key: str) -> Any:
    """Read ``key`` off a provider response that may be an object or a dict.

    ``ChatMessage.raw`` is whatever the provider handed back: an OpenAI-shaped
    object for the API-backed models, and a dict for ``AmazonBedrockModel``
    (the boto3 ``converse`` response) and the local runtimes.
    """
    if isinstance(raw, Mapping):
        return raw.get(key)
    return getattr(raw, key, None)


def _first_choice(output_message: Any) -> Any:
    choices = _raw_value(getattr(output_message, "raw", None), "choices")
    if isinstance(choices, list) and choices:
        return choices[0]
    return None


def _reasoning_from_raw(output_message: Any) -> str | None:
    message = _raw_value(_first_choice(output_message), "message")
    reasoning = _raw_value(message, "reasoning_content")
    return reasoning if isinstance(reasoning, str) and reasoning else None


def _tool_call_requests(output_message: Any) -> list[ToolCallRequest]:
    # ChatMessage.__post_init__ coerces every entry into a
    # ChatMessageToolCall, so the id/function/name/arguments are all present.
    tool_calls = getattr(output_message, "tool_calls", None) or []
    return [
        ToolCallRequest(
            name=tool_call.function.name,
            id=tool_call.id,
            arguments=tool_call.function.arguments,
        )
        for tool_call in tool_calls
    ]


def finish_reason(output_message: Any) -> str | None:
    """Why the provider stopped generating, or ``None`` if it didn't say.

    The local runtimes (``TransformersModel``, ``VLLMModel``, ``MLXModel``) put
    ``{"out": ..., "completion_kwargs": ...}`` on ``raw`` and report no finish
    reason at all. Defaulting those to ``"stop"`` would make a generation cut
    short by ``max_new_tokens`` look like a natural stop. A response carrying
    tool calls is the one case where the reason follows without guessing.
    """
    reason = _raw_value(_first_choice(output_message), "finish_reason")
    if isinstance(reason, str) and reason:
        return reason
    raw = getattr(output_message, "raw", None)
    stop_reason = _raw_value(raw, "stopReason")
    if isinstance(stop_reason, str) and stop_reason:
        return _STOP_REASON_MAP.get(stop_reason, stop_reason)
    if getattr(output_message, "tool_calls", None):
        return "tool_calls"
    return None


def to_output_message(output_message: Any) -> OutputMessage:
    """Map a smolagents ``ChatMessage`` response to an ``OutputMessage``."""
    role = _unwrap_role(getattr(output_message, "role", None)) or "assistant"
    parts = _parts_from_content(getattr(output_message, "content", None))
    if reasoning := _reasoning_from_raw(output_message):
        parts.append(Reasoning(content=reasoning))
    tool_call_requests = _tool_call_requests(output_message)
    parts.extend(tool_call_requests)
    # OutputMessage requires the field; util-genai drops an empty value when it
    # emits gen_ai.response.finish_reasons.
    reason = finish_reason(output_message)
    return OutputMessage(role=role, parts=parts, finish_reason=reason or "")


def response_id(output_message: Any) -> str | None:
    """Extract ``gen_ai.response.id`` from the provider response on ``.raw``."""
    value = _raw_value(getattr(output_message, "raw", None), "id")
    return value if isinstance(value, str) and value else None


def response_model_name(output_message: Any) -> str | None:
    """Extract ``gen_ai.response.model`` from the provider response on ``.raw``."""
    value = _raw_value(getattr(output_message, "raw", None), "model")
    return value if isinstance(value, str) and value else None


def _tool_parameters(tool: Any) -> dict[str, Any] | None:
    """Return the JSON Schema ``parameters`` object for a smolagents tool.

    A tool's ``inputs`` map is not a JSON Schema on its own: smolagents wraps it
    in an object schema, derives ``required`` from ``nullable``, and rewrites its
    non-JSON-Schema ``"any"`` type. ``get_tool_json_schema`` builds exactly the
    schema the provider receives.
    """
    try:
        from smolagents.models import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
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
