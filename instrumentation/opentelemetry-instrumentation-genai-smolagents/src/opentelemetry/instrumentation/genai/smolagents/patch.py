# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""wrapt wrapper factories for smolagents instrumentation.

Each factory takes the shared :class:`TelemetryHandler` and returns a wrapper
suitable for :func:`wrapt.wrap_function_wrapper`:

- :func:`model_generate` wraps each defining ``Model.generate`` -> ``chat`` span.
- :func:`model_generate_stream` wraps each defining ``Model.generate_stream``
  -> ``chat`` span, held open until the stream is drained.

Original library exceptions are always re-raised unmodified; telemetry is
finalized via ``invocation.stop()`` / ``invocation.fail(exc)``.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.stream import SyncStreamWrapper
from opentelemetry.util.genai.types import (
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
)

from ._messages import (
    response_id,
    response_model_name,
    to_input_messages,
    to_output_message,
    to_tool_definitions,
)
from .provider import resolve_provider, resolve_server_address_port

_logger = logging.getLogger(__name__)

_Wrapper = Callable[
    [Callable[..., Any], Any, tuple[Any, ...], dict[str, Any]], Any
]


def _bind_arguments(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Bind call args to the wrapped callable's signature, applying defaults.

    smolagents passes the interesting arguments positionally
    (``model.generate(input_messages)``), so binding is what makes them
    readable by name. On a binding failure the keyword arguments are returned
    on their own, without the positional ones and without the defaults.
    """
    try:
        bound = signature(wrapped).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        _logger.debug(
            "Failed to bind arguments of %s; falling back to keyword arguments",
            getattr(wrapped, "__qualname__", wrapped),
            exc_info=True,
        )
        return dict(kwargs)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _remove_parameter_sentinel() -> Any:
    """Return smolagents' ``REMOVE_PARAMETER`` sentinel, or ``None`` if absent."""
    try:
        from smolagents.models import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
            REMOVE_PARAMETER,
        )
    except ImportError:
        _logger.debug("smolagents.models.REMOVE_PARAMETER is unavailable")
        return None
    return REMOVE_PARAMETER


# Model classes that never forward ``stop_sequences``, whatever
# ``supports_stop_parameter`` answers. ``AmazonBedrockModel`` overrides
# ``_prepare_completion_kwargs`` and calls the base with a hardcoded
# ``stop_sequences=None`` (``models.py``), so its ``converse`` request carries
# no stop sequences at all. ``_forwards_stop_sequences`` matches these names
# along the MRO, so subclasses are covered too.
_MODELS_DROPPING_STOP_SEQUENCES = frozenset({"AmazonBedrockModel"})


def _forwards_stop_sequences(instance: Any) -> bool:
    if any(
        cls.__name__ in _MODELS_DROPPING_STOP_SEQUENCES
        for cls in type(instance).__mro__
    ):
        return False
    return bool(getattr(instance, "supports_stop_parameter", False))


def _merged_request_kwargs(
    instance: Any, bound: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild the request keyword arguments smolagents will send.

    ``Model._prepare_completion_kwargs`` seeds ``stop`` from the
    ``stop_sequences`` argument and ``response_format`` from its own argument,
    applies the per-call ``**kwargs`` on top and the model-level ``self.kwargs``
    last, so a key set at two levels reaches the provider with the model-level
    value and a model-level ``REMOVE_PARAMETER`` drops it from the request
    entirely. Following that order here is what keeps a removed key off the span
    as well.
    """
    merged: dict[str, Any] = {}
    stop_sequences = bound.get("stop_sequences")
    if stop_sequences is not None and _forwards_stop_sequences(instance):
        merged["stop"] = stop_sequences
    response_format = bound.get("response_format")
    if response_format is not None:
        merged["response_format"] = response_format
    call_kwargs = bound.get("kwargs")
    if isinstance(call_kwargs, dict):
        merged.update(call_kwargs)
    model_kwargs = getattr(instance, "kwargs", None)
    if not isinstance(model_kwargs, dict):
        return merged
    remove = _remove_parameter_sentinel()
    for name, value in model_kwargs.items():
        if remove is not None and value is remove:
            merged.pop(name, None)
        else:
            merged[name] = value
    return merged


def _stop_sequences(merged: dict[str, Any]) -> list[str] | None:
    """Return the stop sequences the request carries, if any."""
    stop = merged.get("stop", merged.get("stop_sequences"))
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [str(item) for item in stop]
    return None


# smolagents' ``response_format`` type -> ``gen_ai.output.type`` value, for the
# types whose provider spelling differs from the semconv one. The openai
# instrumentation maps the same two.
_OUTPUT_TYPE_MAP: dict[str, str] = {
    "json_object": GenAI.GenAiOutputTypeValues.JSON.value,
    "json_schema": GenAI.GenAiOutputTypeValues.JSON.value,
}

_OUTPUT_TYPE_VALUES = frozenset(
    value.value for value in GenAI.GenAiOutputTypeValues
)


def _output_type(merged: dict[str, Any]) -> str | None:
    """Map the request's ``response_format`` to ``gen_ai.output.type``.

    smolagents forwards ``response_format`` to the provider unchanged, so its
    ``type`` is whatever the provider accepts. Only the values the semconv
    defines are recorded; a provider-specific one is dropped rather than put on
    an enum attribute.
    """
    response_format = merged.get("response_format")
    if not isinstance(response_format, Mapping):
        return None
    format_type = response_format.get("type")
    if not isinstance(format_type, str):
        return None
    output_type = _OUTPUT_TYPE_MAP.get(format_type, format_type)
    if output_type not in _OUTPUT_TYPE_VALUES:
        _logger.debug(
            "No gen_ai.output.type value for response_format type %r",
            format_type,
        )
        return None
    return output_type


def _apply_request_parameters(
    invocation: InferenceInvocation, instance: Any, bound: dict[str, Any]
) -> None:
    """Copy the request parameters smolagents will send onto the span."""
    merged = _merged_request_kwargs(instance, bound)

    invocation.temperature = _coerce_float(merged.get("temperature"))
    invocation.top_p = _coerce_float(merged.get("top_p"))
    invocation.top_k = _coerce_float(merged.get("top_k"))
    invocation.frequency_penalty = _coerce_float(
        merged.get("frequency_penalty")
    )
    invocation.presence_penalty = _coerce_float(merged.get("presence_penalty"))
    # TransformersModel takes the generation limit as max_new_tokens (which its
    # constructor defaults to 4096) and treats max_tokens as an alias for it.
    invocation.max_tokens = _coerce_int(
        merged.get("max_tokens", merged.get("max_new_tokens"))
    )
    invocation.seed = _coerce_int(merged.get("seed"))
    invocation.stop_sequences = _stop_sequences(merged)
    invocation.output_type = _output_type(merged)


def _apply_token_usage(
    invocation: InferenceInvocation, output_message: Any
) -> None:
    # ChatMessage.token_usage is the only source: the per-model
    # last_input_token_count / last_output_token_count counters were removed
    # before the oldest supported smolagents. It is None for the local runtimes
    # (TransformersModel, VLLMModel, MLXModel), which report no usage.
    token_usage = getattr(output_message, "token_usage", None)
    if token_usage is None:
        return
    invocation.input_tokens = token_usage.input_tokens
    invocation.output_tokens = token_usage.output_tokens


def _start_inference(
    handler: TelemetryHandler,
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> InferenceInvocation:
    """Start the ``chat`` span and record the request.

    ``generate`` and ``generate_stream`` take the same parameters.
    """
    provider = resolve_provider(instance)
    server_address, server_port = resolve_server_address_port(instance)
    invocation = handler.inference(
        provider,
        request_model=getattr(instance, "model_id", None),
        server_address=server_address,
        server_port=server_port,
    )
    bound = _bind_arguments(wrapped, args, kwargs)
    _apply_request_parameters(invocation, instance, bound)
    invocation.tool_definitions = to_tool_definitions(
        bound.get("tools_to_call_from")
    )
    if handler.should_capture_content():
        invocation.input_messages = to_input_messages(bound.get("messages"))
    return invocation


def model_generate(handler: TelemetryHandler) -> _Wrapper:
    """Wrap a defining ``Model.generate`` to emit a ``chat`` span."""

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_inference(handler, wrapped, instance, args, kwargs)
        with invocation:
            output_message = wrapped(*args, **kwargs)
            _apply_token_usage(invocation, output_message)
            invocation.response_model_name = response_model_name(
                output_message
            )
            invocation.response_id = response_id(output_message)
            output = to_output_message(output_message)
            # to_output_message leaves finish_reason empty when the provider
            # reported none, and an empty value is dropped rather than guessed
            # at.
            if output.finish_reason:
                invocation.finish_reasons = [output.finish_reason]
            if handler.should_capture_content():
                invocation.output_messages = [output]
            return output_message

    return wrapper


@dataclass
class _StreamedToolCall:
    """A tool call assembled from stream deltas."""

    id: str | None = None
    name: str = ""
    arguments: str = ""


class _ModelStreamWrapper(SyncStreamWrapper[Any]):
    """Keep the ``chat`` span open until the delta stream is drained.

    Passing the invocation to ``super().__init__()`` turns on
    ``gen_ai.request.stream`` and the per-chunk timing metrics.
    """

    def __init__(
        self,
        stream: Generator[Any, Any, Any],
        invocation: InferenceInvocation,
        handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_inference = invocation
        self._self_handler = handler
        self._self_content: list[str] = []
        self._self_tool_calls: dict[int, _StreamedToolCall] = {}
        self._self_input_tokens = 0
        self._self_output_tokens = 0
        self._self_saw_token_usage = False

    def _accumulate_tool_call(self, delta: Any) -> None:
        index = getattr(delta, "index", None)
        if not isinstance(index, int):
            # agglomerate_stream_deltas raises here; telemetry must not.
            _logger.debug("Dropping a tool call delta that carries no index")
            return
        tool_call = self._self_tool_calls.setdefault(
            index, _StreamedToolCall()
        )
        if delta.id:
            tool_call.id = delta.id
        function = getattr(delta, "function", None)
        if function is None:
            return
        if function.name:
            tool_call.name = function.name
        if function.arguments:
            tool_call.arguments += function.arguments

    def _process_chunk(self, chunk: Any) -> None:
        content = getattr(chunk, "content", None)
        if content:
            self._self_content.append(content)
        token_usage = getattr(chunk, "token_usage", None)
        if token_usage is not None:
            # Summed like agglomerate_stream_deltas, so the span agrees with the
            # totals the agent's monitor reports.
            self._self_saw_token_usage = True
            self._self_input_tokens += token_usage.input_tokens
            self._self_output_tokens += token_usage.output_tokens
        for delta in getattr(chunk, "tool_calls", None) or []:
            self._accumulate_tool_call(delta)

    def _output_message(self) -> OutputMessage | None:
        parts: list[MessagePart] = []
        content = "".join(self._self_content)
        if content:
            parts.append(Text(content=content))
        parts.extend(
            ToolCallRequest(
                name=tool_call.name,
                id=tool_call.id,
                arguments=tool_call.arguments or None,
            )
            for tool_call in self._self_tool_calls.values()
        )
        if not parts:
            # Closed before it was drained, so there is no response to report.
            return None
        # Deltas carry no finish reason, so tool calls are the only evidence.
        # Defaulting to "stop" would hide a generation cut short by a token
        # limit.
        finish_reason = "tool_calls" if self._self_tool_calls else ""
        return OutputMessage(
            role="assistant", parts=parts, finish_reason=finish_reason
        )

    def _finalize(self, error: BaseException | None = None) -> None:
        invocation = self._self_inference
        if self._self_saw_token_usage:
            invocation.input_tokens = self._self_input_tokens
            invocation.output_tokens = self._self_output_tokens
        output = self._output_message()
        if output is not None:
            if output.finish_reason:
                invocation.finish_reasons = [output.finish_reason]
            if self._self_handler.should_capture_content():
                invocation.output_messages = [output]
        if error is not None:
            invocation.fail(error)
        else:
            invocation.stop()

    def _on_stream_end(self) -> None:
        self._finalize()

    def _on_stream_error(self, error: BaseException) -> None:
        # Records what was streamed before the failure.
        self._finalize(error)


def model_generate_stream(handler: TelemetryHandler) -> _Wrapper:
    """Wrap a defining ``Model.generate_stream`` to emit a ``chat`` span.

    ``stream_outputs=True`` routes an agent's model calls here. The span stays
    open until the caller drains the deltas.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_inference(handler, wrapped, instance, args, kwargs)

        try:
            stream = wrapped(*args, **kwargs)
        except Exception as error:  # pylint: disable=broad-except
            invocation.fail(error)
            raise

        return _ModelStreamWrapper(stream, invocation, handler)

    return wrapper
