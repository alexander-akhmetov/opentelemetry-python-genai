# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers, tools, and model stubs for smolagents instrumentation tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from smolagents import OpenAIModel, Tool

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)


class GetWeatherTool(Tool):
    name = "get_weather"
    description = "Get the weather for a given city"
    inputs = {
        "location": {
            "type": "string",
            "description": "The city to get the weather for",
        }
    }
    output_type = "string"

    def forward(self, location: str) -> str:
        return "sunny"


def openai_model() -> OpenAIModel:
    return OpenAIModel(
        model_id="gpt-4o",
        api_key="test_openai_api_key",
        api_base="https://api.openai.com/v1",
    )


def stub_openai_client(
    content: str, finish_reason: str = "stop", error: Exception | None = None
) -> Any:
    """An object shaped like the bits of ``openai.OpenAI`` that ``generate`` uses.

    Building a real ``ChatCompletion`` keeps the response shape honest (the
    wrapper reads ``raw.model``, ``raw.id``, and ``raw.choices[0].finish_reason``)
    without needing a cassette for a deployment we can't record against.
    """
    from openai.types.chat import (  # noqa: PLC0415
        ChatCompletion,
        ChatCompletionMessage,
    )
    from openai.types.chat.chat_completion import Choice  # noqa: PLC0415
    from openai.types.completion_usage import CompletionUsage  # noqa: PLC0415

    completion = ChatCompletion(
        id="chatcmpl-stub",
        model="gpt-4o-2024-08-06",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,
                message=ChatCompletionMessage(
                    role="assistant", content=content
                ),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=3, completion_tokens=1, total_tokens=4
        ),
    )

    def create(**_: Any) -> ChatCompletion:
        if error is not None:
            raise error
        return completion

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def stub_streaming_openai_client(
    chunks: list[Any], error: Exception | None = None
) -> Any:
    """An ``openai.OpenAI`` stand-in whose ``create`` returns a chunk stream.

    ``OpenAIModel.generate_stream`` reads ``event.usage`` and
    ``event.choices[0].delta``, so the chunks are real ``ChatCompletionChunk``
    objects. ``error`` is raised after the chunks are yielded, which is how a
    provider failure part-way through a stream reaches the caller.
    """

    def create(**_: Any) -> Any:
        def stream() -> Any:
            yield from chunks
            if error is not None:
                raise error

        return stream()

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def text_chunk(content: str) -> Any:
    from openai.types.chat import ChatCompletionChunk  # noqa: PLC0415
    from openai.types.chat.chat_completion_chunk import (  # noqa: PLC0415
        Choice,
        ChoiceDelta,
    )

    return ChatCompletionChunk(
        id="chatcmpl-stub",
        model="gpt-4o-2024-08-06",
        object="chat.completion.chunk",
        created=0,
        choices=[Choice(index=0, delta=ChoiceDelta(content=content))],
    )


def tool_call_chunk(
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    from openai.types.chat import ChatCompletionChunk  # noqa: PLC0415
    from openai.types.chat.chat_completion_chunk import (  # noqa: PLC0415
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    return ChatCompletionChunk(
        id="chatcmpl-stub",
        model="gpt-4o-2024-08-06",
        object="chat.completion.chunk",
        created=0,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=index,
                            id=call_id,
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name=name, arguments=arguments
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def usage_chunk(prompt_tokens: int, completion_tokens: int) -> Any:
    from openai.types.chat import ChatCompletionChunk  # noqa: PLC0415
    from openai.types.completion_usage import CompletionUsage  # noqa: PLC0415

    return ChatCompletionChunk(
        id="chatcmpl-stub",
        model="gpt-4o-2024-08-06",
        object="chat.completion.chunk",
        created=0,
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def spans_by_operation(
    spans: list[ReadableSpan], operation: str
) -> list[ReadableSpan]:
    return [
        span
        for span in spans
        if (span.attributes or {}).get(GenAI.GEN_AI_OPERATION_NAME)
        == operation
    ]


def attr(span: ReadableSpan, name: str) -> Any:
    return (span.attributes or {}).get(name)


def parse_messages(span: ReadableSpan, name: str) -> list[dict[str, Any]]:
    raw = attr(span, name)
    return json.loads(raw) if isinstance(raw, str) else []


def part_types(messages: list[dict[str, Any]]) -> list[str]:
    return [part["type"] for message in messages for part in message["parts"]]


def metrics_by_name(metric_reader: Any) -> dict[str, Any]:
    data = metric_reader.get_metrics_data()
    if data is None:
        return {}
    return {
        metric.name: metric
        for resource_metric in data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }


def data_point_attributes(metric: Any) -> list[dict[str, Any]]:
    return [dict(point.attributes) for point in metric.data.data_points]
