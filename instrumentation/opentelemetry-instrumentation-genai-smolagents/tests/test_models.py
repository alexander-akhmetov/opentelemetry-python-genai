# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Model (``chat``) instrumentation tests: VCR-backed runs of the real
smolagents model classes, provider/endpoint resolution, request-parameter
mapping, and message conversion.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Generator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from smolagents import LiteLLMModel, OpenAIModel
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
)

from opentelemetry.instrumentation.genai.smolagents._messages import (
    response_id,
    response_model_name,
    to_input_messages,
    to_output_message,
    to_tool_definitions,
)
from opentelemetry.instrumentation.genai.smolagents.provider import (
    resolve_provider,
    resolve_server_address_port,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.types import (
    Blob,
    Reasoning,
    Text,
    ToolCallRequest,
    Uri,
)

from .test_utils import (
    GetWeatherTool,
    attr,
    data_point_attributes,
    metrics_by_name,
    openai_model,
    parse_messages,
    part_types,
    spans_by_operation,
    stub_openai_client,
    stub_streaming_openai_client,
    text_chunk,
    tool_call_chunk,
    usage_chunk,
)

IMAGE_URL = (
    "https://fastly.picsum.photos/id/237/200/300.jpg"
    "?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
)


def test_openai_model_basic(
    instrument_with_content, span_exporter, metric_reader, vcr
) -> None:
    model = openai_model()
    text = "Who won the World Cup in 2018? Answer in one word with no punctuation."
    with vcr.use_cassette("openai_model_basic.yaml"):
        output = model.generate(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": text}]}
            ]
        )
    assert output.content == "France"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.name == "chat gpt-4o"
    assert span.status.status_code == StatusCode.UNSET
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "openai"
    assert attr(span, GenAI.GEN_AI_REQUEST_MODEL) == "gpt-4o"
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 25
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == 2
    assert (
        attr(span, GenAI.GEN_AI_RESPONSE_ID)
        == "chatcmpl-Ax6UoZOGLTVmdQxp0ToJi1tv1FUkb"
    )
    assert attr(span, GenAI.GEN_AI_RESPONSE_MODEL) == "gpt-4o-2024-08-06"
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)
    assert attr(span, server_attributes.SERVER_ADDRESS) == "api.openai.com"
    assert attr(span, server_attributes.SERVER_PORT) is None

    inputs = parse_messages(span, GenAI.GEN_AI_INPUT_MESSAGES)
    assert inputs[0]["role"] == "user"
    assert inputs[0]["parts"][0] == {"type": "text", "content": text}

    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["role"] == "assistant"
    assert outputs[0]["parts"][0] == {"type": "text", "content": "France"}

    metrics = metrics_by_name(metric_reader)
    duration = metrics[gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION]
    assert duration.unit == "s"
    assert data_point_attributes(duration) == [
        {
            GenAI.GEN_AI_OPERATION_NAME: "chat",
            GenAI.GEN_AI_PROVIDER_NAME: "openai",
            GenAI.GEN_AI_REQUEST_MODEL: "gpt-4o",
            GenAI.GEN_AI_RESPONSE_MODEL: "gpt-4o-2024-08-06",
            server_attributes.SERVER_ADDRESS: "api.openai.com",
        }
    ]
    token_usage = metrics[gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE]
    assert {
        point.attributes[GenAI.GEN_AI_TOKEN_TYPE]: point.sum
        for point in token_usage.data.data_points
    } == {"input": 25, "output": 2}


def test_openai_model_no_content(
    instrument_no_content, span_exporter, vcr
) -> None:
    model = openai_model()
    with vcr.use_cassette("openai_model_basic.yaml"):
        model.generate(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
            ]
        )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "openai"
    assert isinstance(attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS), int)
    assert attr(span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None


def test_openai_model_image_url(
    instrument_with_content, span_exporter, vcr
) -> None:
    model = openai_model()
    with vcr.use_cassette("openai_model_image_url.yaml"):
        model.generate(
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {"type": "text", "text": "What breed is this dog?"},
                        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    ],
                )
            ]
        )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    inputs = parse_messages(span, GenAI.GEN_AI_INPUT_MESSAGES)
    parts = inputs[0]["parts"]
    assert parts[0] == {"type": "text", "content": "What breed is this dog?"}
    assert parts[1]["type"] == "uri"
    assert parts[1]["modality"] == "image"
    assert parts[1]["uri"] == IMAGE_URL


def test_openai_model_with_tools(
    instrument_with_content, span_exporter, vcr
) -> None:
    model = openai_model()
    with vcr.use_cassette("openai_model_tool.yaml"):
        output = model.generate(
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": "What is the weather in Paris?",
                        }
                    ],
                )
            ],
            tools_to_call_from=[GetWeatherTool()],
        )
    assert output.tool_calls[0].function.name == "get_weather"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert json.loads(attr(span, GenAI.GEN_AI_TOOL_DEFINITIONS)) == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city to get the weather for",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    tool_call_parts = [
        part for part in outputs[0]["parts"] if part["type"] == "tool_call"
    ]
    assert tool_call_parts[0]["name"] == "get_weather"
    # smolagents hands the provider's raw argument payload through unparsed.
    assert tool_call_parts[0]["arguments"] == '{"location":"Paris"}'
    assert tool_call_parts[0]["id"] == "call_EUuviydGIG5Jau3DLw4v4cue"
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("tool_calls",)


def _litellm_supports_reasoning() -> bool:
    """litellm surfaces Anthropic ``reasoning_content`` only from ~1.63 onward.

    The oldest supported litellm (smolagents' 1.60.2 floor) doesn't parse the
    Anthropic thinking blocks into ``reasoning_content``, so the reasoning part
    can't be mapped there. Gate the reasoning-specific assertion on the version.
    """
    from importlib.metadata import version  # noqa: PLC0415

    parts = version("litellm").split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (1, 63)
    except (IndexError, ValueError):
        return True


def test_litellm_reasoning(
    instrument_with_content,
    span_exporter,
    litellm_local_cost_map,
    patch_tiktoken_encoding,
    vcr,
) -> None:
    model = LiteLLMModel(
        model_id="anthropic/claude-3-7-sonnet-20250219",
        api_key="test_anthropic_api_key",
        thinking={"type": "enabled", "budget_tokens": 4000},
    )
    with vcr.use_cassette("litellm_reasoning.yaml"):
        model.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Who won the World Cup in 2018? Answer in one "
                                "word with no punctuation."
                            ),
                        }
                    ],
                }
            ]
        )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "anthropic"
    assert (
        attr(span, GenAI.GEN_AI_REQUEST_MODEL)
        == "anthropic/claude-3-7-sonnet-20250219"
    )
    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    if _litellm_supports_reasoning():
        assert "reasoning" in part_types(outputs)


def test_model_generate_reraises_and_records_error(
    instrument_with_content, span_exporter
) -> None:
    from smolagents.models import Model  # noqa: PLC0415

    model = Model(model_id="broken-model")
    with pytest.raises(NotImplementedError):
        model.generate(messages=[{"role": "user", "content": "hi"}])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "NotImplementedError"


def test_inherited_azure_generate_emits_one_chat_span(
    instrument_with_content, span_exporter
) -> None:
    # AzureOpenAIModel inherits generate from OpenAIModel, so only the defining
    # class is patched. Exercise the inherited method end to end to prove the
    # single patch still produces exactly one span with the Azure provider.
    from smolagents import AzureOpenAIModel  # noqa: PLC0415

    model = AzureOpenAIModel(
        model_id="gpt-4o-deployment",
        azure_endpoint="https://example-resource.openai.azure.com",
        api_key="test_azure_api_key",
        api_version="2024-10-21",
    )
    model.client = stub_openai_client("Bonjour")

    output = model.generate(messages=[{"role": "user", "content": "Hi"}])
    assert output.content == "Bonjour"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "azure.ai.openai"
    assert attr(span, GenAI.GEN_AI_REQUEST_MODEL) == "gpt-4o-deployment"
    assert (
        attr(span, server_attributes.SERVER_ADDRESS)
        == "example-resource.openai.azure.com"
    )
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 3
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == 1


def _fake_model(class_name: str, **attrs: Any) -> Any:
    instance = type(class_name, (), {})()
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


@pytest.mark.parametrize(
    "class_name, attrs, expected",
    [
        ("OpenAIModel", {}, "openai"),
        ("AzureOpenAIModel", {}, "azure.ai.openai"),
        ("AmazonBedrockModel", {}, "aws.bedrock"),
        # No GenAI registry value exists for these runtimes, so the product
        # name is used rather than the class name.
        ("InferenceClientModel", {}, "huggingface"),
        ("TransformersModel", {}, "huggingface"),
        ("VLLMModel", {}, "vllm"),
        ("MLXModel", {}, "mlx"),
        # LiteLLM vendor prefixes: remapped where the slug differs from the
        # semconv value, passed through otherwise.
        ("LiteLLMModel", {"model_id": "anthropic/claude-3"}, "anthropic"),
        ("LiteLLMModel", {"model_id": "mistral/large"}, "mistral_ai"),
        ("LiteLLMModel", {"model_id": "xai/grok"}, "x_ai"),
        ("LiteLLMModel", {"model_id": "gemini/gemini-2.0"}, "gcp.gemini"),
        (
            "LiteLLMModel",
            {"model_id": "vertex_ai/gemini-2.0"},
            "gcp.vertex_ai",
        ),
        ("LiteLLMModel", {"model_id": "watsonx/granite"}, "ibm.watsonx.ai"),
        (
            "LiteLLMModel",
            {"model_id": "azure_ai/phi-4"},
            "azure.ai.inference",
        ),
        ("LiteLLMModel", {"model_id": "ollama/llama3"}, "ollama"),
        # LiteLLMRouterModel takes a model-group name, not a provider/model
        # slug, so there is nothing to resolve.
        ("LiteLLMRouterModel", {"model_id": "model-group-1"}, "unknown"),
        # gen_ai.provider.name is also a metric attribute. An unmapped model
        # must not fall back to the deployment-specific host or a class name.
        (
            "CustomModel",
            {"api_base": "https://llm.example.com/v1"},
            "unknown",
        ),
        ("CustomModel", {}, "unknown"),
    ],
)
def test_resolve_provider(
    class_name: str, attrs: dict[str, Any], expected: str
) -> None:
    assert resolve_provider(_fake_model(class_name, **attrs)) == expected


@pytest.mark.parametrize(
    "model_class, expected",
    [
        ("OpenAIModel", "openai"),
        ("AzureOpenAIModel", "azure.ai.openai"),
        ("AmazonBedrockModel", "aws.bedrock"),
        ("InferenceClientModel", "huggingface"),
        ("TransformersModel", "huggingface"),
        ("VLLMModel", "vllm"),
        ("MLXModel", "mlx"),
        ("LiteLLMModel", "unknown"),
        ("LiteLLMRouterModel", "unknown"),
    ],
)
def test_resolve_provider_covers_every_real_model_class(
    model_class: str, expected: str
) -> None:
    # The mapping is keyed by class name, so pin it against the real classes
    # rather than only against synthetic stand-ins.
    import smolagents  # noqa: PLC0415

    instance = object.__new__(getattr(smolagents, model_class))
    assert resolve_provider(instance) == expected


@pytest.mark.parametrize(
    "attrs, expected",
    [
        ({"api_base": "https://api.openai.com/v1"}, ("api.openai.com", None)),
        # The default HTTPS port is omitted per the semconv server.port guidance.
        (
            {"api_base": "https://api.openai.com:443/v1"},
            ("api.openai.com", None),
        ),
        ({"api_base": "http://localhost:11434/v1"}, ("localhost", 11434)),
        (
            {
                "client_kwargs": {
                    "azure_endpoint": "https://x.openai.azure.com"
                }
            },
            ("x.openai.azure.com", None),
        ),
        ({}, (None, None)),
    ],
)
def test_resolve_server_address_port(
    attrs: dict[str, Any], expected: tuple[str | None, int | None]
) -> None:
    assert resolve_server_address_port(_fake_model("M", **attrs)) == expected


def test_server_address_falls_back_to_the_sdk_client() -> None:
    # The common configuration: no api_base, so the URL is only known to the
    # client the model built for itself.
    model = OpenAIModel(model_id="gpt-4o", api_key="test_openai_api_key")
    assert resolve_server_address_port(model) == ("api.openai.com", None)


def test_request_parameters_recorded(
    instrument_with_content, span_exporter
) -> None:
    from smolagents.models import Model  # noqa: PLC0415

    model = Model(
        model_id="broken-model",
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        frequency_penalty=0.25,
        presence_penalty=1,
        max_tokens=256,
        seed=7,
    )
    with pytest.raises(NotImplementedError):
        model.generate(
            messages=[{"role": "user", "content": "hi"}],
            stop_sequences=["<end>"],
        )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_TEMPERATURE) == 0.5
    assert attr(span, GenAI.GEN_AI_REQUEST_TOP_P) == 0.9
    # top_k is a float attribute in the spec even though callers pass an int.
    assert attr(span, GenAI.GEN_AI_REQUEST_TOP_K) == 40.0
    assert isinstance(attr(span, GenAI.GEN_AI_REQUEST_TOP_K), float)
    assert attr(span, GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY) == 0.25
    assert attr(span, GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY) == 1.0
    assert attr(span, GenAI.GEN_AI_REQUEST_MAX_TOKENS) == 256
    assert isinstance(attr(span, GenAI.GEN_AI_REQUEST_MAX_TOKENS), int)
    assert attr(span, GenAI.GEN_AI_REQUEST_SEED) == 7
    assert attr(span, GenAI.GEN_AI_REQUEST_STOP_SEQUENCES) == ("<end>",)


def test_model_kwargs_win_over_call_kwargs(
    instrument_with_content, span_exporter
) -> None:
    # _prepare_completion_kwargs applies the call kwargs first and the model
    # kwargs on top, and drops any key whose model-level value is the
    # REMOVE_PARAMETER sentinel.
    from smolagents.models import (  # noqa: PLC0415
        REMOVE_PARAMETER,
        Model,
    )

    model = Model(
        model_id="broken-model",
        temperature=0.1,
        max_tokens=REMOVE_PARAMETER,
    )
    with pytest.raises(NotImplementedError):
        model.generate(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.9,
            max_tokens=512,
            top_p=0.5,
        )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_TEMPERATURE) == 0.1
    assert attr(span, GenAI.GEN_AI_REQUEST_MAX_TOKENS) is None
    assert attr(span, GenAI.GEN_AI_REQUEST_TOP_P) == 0.5


@pytest.mark.parametrize(
    "model_id, model_kwargs, expected",
    [
        # gpt-5 doesn't accept `stop`, so smolagents truncates the generated
        # text locally instead of sending the sequences.
        ("gpt-5", {}, None),
        ("gpt-4o", {}, ("<end>",)),
        # An explicit `stop` overrides the stop_sequences argument.
        ("gpt-4o", {"stop": ["STOP"]}, ("STOP",)),
        # The model-level sentinel pops the `stop` that
        # _prepare_completion_kwargs seeded from stop_sequences, leaving the
        # request with none.
        ("gpt-4o", {"stop": "REMOVE"}, None),
    ],
)
def test_stop_sequences_follow_what_is_sent(
    instrument_with_content,
    span_exporter,
    model_id: str,
    model_kwargs: dict[str, Any],
    expected: tuple[str, ...] | None,
) -> None:
    from smolagents.models import (  # noqa: PLC0415
        REMOVE_PARAMETER,
        Model,
    )

    model_kwargs = {
        key: REMOVE_PARAMETER if value == "REMOVE" else value
        for key, value in model_kwargs.items()
    }
    model = Model(model_id=model_id, **model_kwargs)
    with pytest.raises(NotImplementedError):
        model.generate(
            messages=[{"role": "user", "content": "hi"}],
            stop_sequences=["<end>"],
        )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_STOP_SEQUENCES) == expected


def _bedrock_model(response: dict[str, Any]) -> Any:
    from smolagents import AmazonBedrockModel  # noqa: PLC0415

    # A caller-supplied client keeps boto3 out of the test.
    return AmazonBedrockModel(
        model_id="us.amazon.nova-pro-v1:0",
        client=SimpleNamespace(converse=lambda **_: response),
    )


BEDROCK_RESPONSE: dict[str, Any] = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "done"}],
            "tool_calls": None,
        }
    },
    "usage": {"inputTokens": 3, "outputTokens": 2},
    "stopReason": "end_turn",
}


def test_bedrock_stop_sequences_are_not_recorded(
    instrument_with_content, span_exporter
) -> None:
    # supports_stop_parameter says yes, but the prepared request carries no
    # stop sequences, so the span must not claim any either.
    model = _bedrock_model(BEDROCK_RESPONSE)
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    request = model._prepare_completion_kwargs(  # noqa: SLF001
        messages=messages, stop_sequences=["<end>"]
    )
    assert model.supports_stop_parameter is True
    assert "stop" not in request

    model.generate(messages=messages, stop_sequences=["<end>"])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_STOP_SEQUENCES) is None
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "aws.bedrock"
    # Bedrock's "end_turn" is normalized to the semconv "stop".
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)


@pytest.mark.parametrize(
    "model_kwargs, call_kwargs, expected",
    [
        ({"max_tokens": 256}, {}, 256),
        ({}, {"max_tokens": 256}, 256),
        # max_new_tokens is the TransformersModel spelling of the same limit,
        # and max_tokens wins when a caller sets both.
        ({"max_new_tokens": 4096}, {}, 4096),
        ({"max_new_tokens": 4096, "max_tokens": 256}, {}, 256),
        ({}, {}, None),
    ],
)
def test_max_tokens_covers_both_spellings(
    instrument_with_content,
    span_exporter,
    model_kwargs: dict[str, Any],
    call_kwargs: dict[str, Any],
    expected: int | None,
) -> None:
    from smolagents.models import Model  # noqa: PLC0415

    model = Model(model_id="broken-model", **model_kwargs)
    with pytest.raises(NotImplementedError):
        model.generate(
            messages=[{"role": "user", "content": "hi"}], **call_kwargs
        )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_MAX_TOKENS) == expected


@pytest.mark.parametrize(
    "response_format, model_kwargs, expected",
    [
        ({"type": "json_object"}, {}, "json"),
        ({"type": "json_schema", "json_schema": {}}, {}, "json"),
        ({"type": "text"}, {}, "text"),
        (None, {}, None),
        # smolagents forwards response_format unchanged, so its type is
        # whatever the provider accepts. An unknown one is dropped rather than
        # recorded on an enum attribute.
        ({"type": "xml"}, {}, None),
        ({}, {}, None),
        # The model-level kwargs win over the argument, and the sentinel drops
        # the key from the request, the same as for every other parameter.
        (
            {"type": "json_object"},
            {"response_format": {"type": "text"}},
            "text",
        ),
        ({"type": "json_object"}, {"response_format": "REMOVE"}, None),
        (None, {"response_format": {"type": "json_object"}}, "json"),
    ],
)
def test_output_type_follows_the_response_format(
    instrument_with_content,
    span_exporter,
    response_format: dict[str, Any] | None,
    model_kwargs: dict[str, Any],
    expected: str | None,
) -> None:
    from smolagents.models import (  # noqa: PLC0415
        REMOVE_PARAMETER,
        Model,
    )

    model_kwargs = {
        key: REMOVE_PARAMETER if value == "REMOVE" else value
        for key, value in model_kwargs.items()
    }
    model = Model(model_id="broken-model", **model_kwargs)
    with pytest.raises(NotImplementedError):
        model.generate(
            messages=[{"role": "user", "content": "hi"}],
            response_format=response_format,
        )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_OUTPUT_TYPE) == expected


def test_user_subclass_of_a_patched_model_keeps_its_provider(
    instrument_with_content, span_exporter, metric_reader
) -> None:
    # A subclass inherits the patched generate, so it is instrumented; resolving
    # the provider by exact class name would report "unknown" on both the span
    # and the metrics.
    class TenantOpenAIModel(OpenAIModel):
        pass

    model = TenantOpenAIModel(
        model_id="gpt-4o",
        api_key="test_openai_api_key",
        api_base="http://localhost:11434/v1",
    )
    model.client = stub_openai_client("Bonjour")
    model.generate(messages=[{"role": "user", "content": "Hi"}])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "openai"
    # A non-default port is part of the endpoint, unlike the HTTPS default.
    assert attr(span, server_attributes.SERVER_ADDRESS) == "localhost"
    assert attr(span, server_attributes.SERVER_PORT) == 11434
    duration = metrics_by_name(metric_reader)[
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ]
    assert data_point_attributes(duration)[0][GenAI.GEN_AI_PROVIDER_NAME] == (
        "openai"
    )


def test_provider_error_is_recorded_and_reraised(
    instrument_with_content, span_exporter
) -> None:
    model = openai_model()
    model.client = stub_openai_client(
        "", error=ConnectionError("connection reset")
    )

    with pytest.raises(ConnectionError, match="connection reset"):
        model.generate(messages=[{"role": "user", "content": "Hi"}])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "ConnectionError"
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None


def _drain_stream(model: OpenAIModel, **kwargs: Any) -> list[Any]:
    return list(
        model.generate_stream(
            messages=[{"role": "user", "content": "Hi"}], **kwargs
        )
    )


def test_generate_stream_is_lazy_and_records_the_drained_response(
    instrument_with_content, span_exporter
) -> None:
    model = openai_model()
    model.client = stub_streaming_openai_client(
        [text_chunk("Bon"), text_chunk("jour"), usage_chunk(3, 2)]
    )

    stream = model.generate_stream(
        messages=[{"role": "user", "content": "Hi"}]
    )
    # A streamed response isn't finished until the caller drains it.
    assert spans_by_operation(span_exporter.get_finished_spans(), "chat") == []

    deltas = list(stream)
    assert "".join(delta.content or "" for delta in deltas) == "Bonjour"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "openai"
    assert attr(span, GenAI.GEN_AI_REQUEST_MODEL) == "gpt-4o"
    assert attr(span, GenAI.GEN_AI_REQUEST_STREAM) is True
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 3
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == 2
    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"] == [{"type": "text", "content": "Bonjour"}]
    # Deltas carry no finish reason, and "stop" would hide a truncation.
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None


def test_generate_stream_stays_a_generator(
    instrument_with_content, span_exporter
) -> None:
    # Instrumentation observes; it must not change what generate_stream returns.
    model = openai_model()
    model.client = stub_streaming_openai_client([text_chunk("Bonjour")])

    stream = model.generate_stream(
        messages=[{"role": "user", "content": "Hi"}]
    )
    assert isinstance(stream, Generator)
    assert inspect.isgenerator(stream)
    list(stream)


def test_generate_stream_accumulates_tool_calls(
    instrument_with_content, span_exporter
) -> None:
    model = openai_model()
    model.client = stub_streaming_openai_client(
        [
            tool_call_chunk(0, call_id="call_1", name="get_weather"),
            tool_call_chunk(0, arguments='{"location":'),
            tool_call_chunk(0, arguments='"Paris"}'),
        ]
    )

    _drain_stream(model, tools_to_call_from=[GetWeatherTool()])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    (part,) = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)[0]["parts"]
    assert part == {
        "type": "tool_call",
        "id": "call_1",
        "name": "get_weather",
        "arguments": '{"location":"Paris"}',
    }
    # Tool calls are the only evidence of why a streamed generation stopped.
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("tool_calls",)


def test_generate_stream_error_mid_iteration_is_recorded_and_reraised(
    instrument_with_content, span_exporter
) -> None:
    model = openai_model()
    model.client = stub_streaming_openai_client(
        [text_chunk("Bon")], error=ConnectionError("stream died")
    )

    with pytest.raises(ConnectionError, match="stream died"):
        _drain_stream(model)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "ConnectionError"
    # What was streamed before the failure is still recorded.
    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"] == [{"type": "text", "content": "Bon"}]


def test_generate_stream_close_before_drain_finalizes_once(
    instrument_with_content, span_exporter
) -> None:
    model = openai_model()
    model.client = stub_streaming_openai_client([text_chunk("Bonjour")])

    stream = model.generate_stream(
        messages=[{"role": "user", "content": "Hi"}]
    )
    stream.close()
    stream.close()  # idempotent

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.UNSET
    assert attr(span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None


def test_generate_stream_records_chunk_metrics(
    instrument_with_content, span_exporter, metric_reader
) -> None:
    model = openai_model()
    model.client = stub_streaming_openai_client(
        [text_chunk("Bon"), text_chunk("jour"), usage_chunk(3, 2)]
    )

    _drain_stream(model)

    metrics = metrics_by_name(metric_reader)
    assert gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION in metrics
    assert gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE in metrics
    assert (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK in metrics
    )
    assert (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK in metrics
    )


def test_generate_stream_no_content(
    instrument_no_content, span_exporter
) -> None:
    model = openai_model()
    model.client = stub_streaming_openai_client(
        [text_chunk("Bonjour"), usage_chunk(3, 2)]
    )

    _drain_stream(model)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == 2


def test_event_only_content_capture(
    instrument_event_only, span_exporter, log_exporter
) -> None:
    model = openai_model()
    model.client = stub_openai_client("Bonjour")
    model.generate(messages=[{"role": "user", "content": "Hi"}])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None
    # Metadata still goes on the span.
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 3

    (log,) = log_exporter.get_finished_logs()
    record = log.log_record
    assert record.event_name == "gen_ai.client.inference.operation.details"
    # Event attributes carry the messages as structured values, not JSON text.
    attributes = record.attributes or {}
    inputs = attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    outputs = attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert inputs[0]["parts"][0]["content"] == "Hi"
    assert outputs[0]["parts"][0]["content"] == "Bonjour"
    assert outputs[0]["finish_reason"] == "stop"


def test_to_tool_definitions_uses_json_schema() -> None:
    (definition,) = to_tool_definitions([GetWeatherTool()])
    assert definition.name == "get_weather"
    assert definition.description == "Get the weather for a given city"
    # smolagents' raw ``inputs`` map is not a JSON Schema on its own.
    assert definition.parameters == {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city to get the weather for",
            }
        },
        "required": ["location"],
    }


def test_to_input_messages_maps_smolagents_only_roles() -> None:
    # smolagents converts these roles itself inside generate(), after the
    # wrapper has already read the messages.
    messages = to_input_messages(
        [
            {"role": MessageRole.TOOL_CALL, "content": "call"},
            {"role": MessageRole.TOOL_RESPONSE, "content": "response"},
            {"role": MessageRole.SYSTEM, "content": "sys"},
        ]
    )
    assert [message.role for message in messages] == [
        "assistant",
        "user",
        "system",
    ]


def test_to_input_messages_dict_and_chatmessage() -> None:
    messages = to_input_messages(
        [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
    )
    assert messages[0].role == "user"
    assert messages[0].parts == [Text(content="Hello")]
    assert messages[1].role == "assistant"
    assert messages[1].parts == [Text(content="Hi there")]


def test_to_input_messages_image_and_base64() -> None:
    messages = to_input_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {"type": "image", "image": "aVZCT1J3MEtHZ28="},
                ],
            }
        ]
    )
    parts = messages[0].parts
    assert parts[0] == Uri(mime_type=None, modality="image", uri=IMAGE_URL)
    assert isinstance(parts[1], Blob)
    assert parts[1].modality == "image"
    assert parts[1].mime_type == "image/png"
    assert isinstance(parts[1].content, bytes)


def test_to_input_messages_data_url_keeps_media_type() -> None:
    messages = to_input_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "data:image/jpeg;base64,aVZCT1J3MEtHZ28=",
                    }
                ],
            }
        ]
    )
    (part,) = messages[0].parts
    assert isinstance(part, Blob)
    assert part.mime_type == "image/jpeg"


@pytest.mark.parametrize(
    "image",
    [
        "not base64!!!!",
        # A path is not base64, but a non-validating decode silently turns this
        # one into 15 bytes of garbage after dropping the invalid characters.
        "/tmp/photos/my-cat.png",
    ],
)
def test_to_input_messages_drops_malformed_base64_image(image: str) -> None:
    messages = to_input_messages(
        [{"role": "user", "content": [{"type": "image", "image": image}]}]
    )
    assert messages[0].parts == []


def test_to_output_message_text_reasoning_and_tool_calls() -> None:
    class _Msg:
        role = MessageRole.ASSISTANT
        content = "The answer"
        tool_calls = [
            ChatMessageToolCall(
                id="call_1",
                type="function",
                function=ChatMessageToolCallFunction(
                    name="get_weather", arguments='{"location": "Paris"}'
                ),
            )
        ]

        class raw:  # noqa: N801
            class _Choice:
                class message:  # noqa: N801
                    reasoning_content = "thinking about it"

            choices = [_Choice()]

    output = to_output_message(_Msg())
    assert output.role == "assistant"
    assert Text(content="The answer") in output.parts
    assert Reasoning(content="thinking about it") in output.parts
    tool_calls = [p for p in output.parts if isinstance(p, ToolCallRequest)]
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].id == "call_1"
    assert output.finish_reason == "tool_calls"


def test_to_output_message_from_a_dict_raw_response() -> None:
    # AmazonBedrockModel, TransformersModel, VLLMModel and MLXModel put a plain
    # dict on ChatMessage.raw rather than an SDK object.
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="done",
        raw={"stopReason": "max_tokens"},
    )
    output = to_output_message(message)
    assert output.parts == [Text(content="done")]
    # "max_tokens" is Bedrock's stopReason spelling for a length cutoff.
    assert output.finish_reason == "length"
    assert response_id(message) is None
    assert response_model_name(message) is None


def test_to_output_message_unwraps_the_role_enum() -> None:
    output = to_output_message(
        ChatMessage(role=MessageRole.ASSISTANT, content="done")
    )
    assert output.role == "assistant"


def test_to_output_message_maps_image_content() -> None:
    # A response whose content is a list carries the same element shapes as a
    # request, so an image in it maps the same way.
    output = to_output_message(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=[
                {"type": "text", "text": "Here it is"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {"type": "image", "image": "aVZCT1J3MEtHZ28="},
            ],
        )
    )
    assert output.parts[0] == Text(content="Here it is")
    assert output.parts[1] == Uri(
        mime_type=None, modality="image", uri=IMAGE_URL
    )
    blob = output.parts[2]
    assert isinstance(blob, Blob)
    assert blob.mime_type == "image/png"
    assert blob.modality == "image"


LOCAL_RUNTIME_RAW = {
    "out": "done",
    "completion_kwargs": {"max_new_tokens": 4096},
}


@pytest.mark.parametrize(
    "raw, tool_calls, expected",
    [
        # API-backed models: the provider's own value, whatever it is.
        (
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]),
            [],
            "stop",
        ),
        (
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")]),
            [],
            "length",
        ),
        # Bedrock's stopReason values map onto the semconv vocabulary.
        ({"stopReason": "max_tokens"}, [], "length"),
        ({"stopReason": "tool_use"}, [], "tool_calls"),
        # An unmapped value passes through rather than being guessed at.
        ({"stopReason": "guardrail_intervened"}, [], "guardrail_intervened"),
        # The local runtimes report no reason, so none is recorded and
        # util-genai omits the empty value.
        (LOCAL_RUNTIME_RAW, [], ""),
        (None, [], ""),
        # Tool calls in the response give the reason without guessing.
        (
            LOCAL_RUNTIME_RAW,
            [
                ChatMessageToolCall(
                    id="call_1",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="get_weather", arguments="{}"
                    ),
                )
            ],
            "tool_calls",
        ),
    ],
)
def test_finish_reason_follows_the_provider_response(
    raw: Any, tool_calls: list[ChatMessageToolCall], expected: str
) -> None:
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="done",
        tool_calls=tool_calls or None,
        raw=raw,
    )
    assert to_output_message(message).finish_reason == expected


def test_model_reporting_no_finish_reason_omits_the_attribute(
    instrument_with_content, span_exporter
) -> None:
    # InferenceClientModel is a patched class whose provider response can come
    # back without a finish reason; the span must then carry none.
    from smolagents import InferenceClientModel  # noqa: PLC0415

    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token="hf_test"
    )
    model.client = SimpleNamespace(
        chat_completion=lambda **_: SimpleNamespace(
            id="hf-1",
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant", content="ok", tool_calls=None
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )
    )

    model.generate(messages=[{"role": "user", "content": "hi"}])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "huggingface"
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None
    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["finish_reason"] == ""


# The local runtimes flatten a message's content as text, so the content has to
# be a list of parts rather than a bare string.
LOCAL_RUNTIME_MESSAGES: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Where is the Louvre?"}],
    }
]


def _mlx_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    """An ``MLXModel`` whose ``mlx_lm`` pieces are stubbed.

    ``MLXModel.generate`` imports nothing itself; it drives ``stream_generate``
    over ``self.model`` and ``self.tokenizer``, which ``__init__`` loads from
    ``mlx_lm``. Bypassing ``__init__`` is therefore enough to run the real
    ``generate``, and the runtime doesn't have to be installed. ``monkeypatch``
    is unused here; it keeps the factory signature uniform with the vllm one,
    which does have modules to stub.
    """
    from smolagents.models import MLXModel  # noqa: PLC0415

    model = object.__new__(MLXModel)
    model.model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    model.kwargs = {}
    model.flatten_messages_as_text = True
    model.apply_chat_template_kwargs = {}
    model.model = object()
    model.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, tools=None, **_: [1, 2, 3]
    )
    model.stream_generate = lambda *_, **__: iter(
        [SimpleNamespace(text="In "), SimpleNamespace(text="Paris")]
    )
    return model


def _vllm_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A ``VLLMModel`` with ``vllm`` itself stubbed.

    ``VLLMModel.generate`` imports ``SamplingParams`` and
    ``StructuredOutputsParams`` from ``vllm`` when it runs, and neither test env
    installs vllm, so both modules are faked for the duration of the test.
    Everything the wrapper reads still comes from the real ``generate``.
    """
    from smolagents.models import VLLMModel  # noqa: PLC0415

    def fake_params(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    vllm = ModuleType("vllm")
    sampling_params = ModuleType("vllm.sampling_params")
    setattr(vllm, "SamplingParams", fake_params)
    setattr(sampling_params, "StructuredOutputsParams", fake_params)
    setattr(vllm, "sampling_params", sampling_params)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    completion = SimpleNamespace(
        prompt_token_ids=[1, 2, 3, 4],
        outputs=[SimpleNamespace(text="In Paris", token_ids=[5, 6])],
    )
    model = object.__new__(VLLMModel)
    model.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model.kwargs = {}
    model.flatten_messages_as_text = True
    model._is_vlm = False
    model.apply_chat_template_kwargs = {}
    model.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **_: "prompt"
    )
    model.model = SimpleNamespace(generate=lambda *_, **__: [completion])
    return model


@pytest.mark.parametrize(
    "model_factory, provider, request_model, input_tokens, output_tokens",
    [
        (
            _mlx_model,
            "mlx",
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            3,
            2,
        ),
        (_vllm_model, "vllm", "Qwen/Qwen2.5-0.5B-Instruct", 4, 2),
    ],
    ids=["mlx", "vllm"],
)
def test_local_runtime_response_is_recorded(
    instrument_with_content,
    span_exporter,
    monkeypatch: pytest.MonkeyPatch,
    model_factory: Any,
    provider: str,
    request_model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    output = model_factory(monkeypatch).generate(
        messages=LOCAL_RUNTIME_MESSAGES
    )
    assert output.content == "In Paris"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == provider
    assert attr(span, GenAI.GEN_AI_REQUEST_MODEL) == request_model
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == input_tokens
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == output_tokens
    # A local runtime returns no provider response envelope and listens on no
    # socket, so it reports no finish reason, id or response model, and there is
    # no endpoint to derive server.address from.
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None
    assert attr(span, GenAI.GEN_AI_RESPONSE_ID) is None
    assert attr(span, GenAI.GEN_AI_RESPONSE_MODEL) is None
    assert attr(span, server_attributes.SERVER_ADDRESS) is None
    assert attr(span, server_attributes.SERVER_PORT) is None
    assert span.status.status_code == StatusCode.UNSET

    inputs = parse_messages(span, GenAI.GEN_AI_INPUT_MESSAGES)
    assert inputs[0]["parts"] == [
        {"type": "text", "content": "Where is the Louvre?"}
    ]
    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["role"] == "assistant"
    assert outputs[0]["parts"] == [{"type": "text", "content": "In Paris"}]
    assert outputs[0]["finish_reason"] == ""
