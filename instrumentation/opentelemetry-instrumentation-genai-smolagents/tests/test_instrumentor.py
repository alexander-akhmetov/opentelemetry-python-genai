# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle, entry-point, completion-hook, and restoration tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import smolagents
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
    _model_classes_defining,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.genai.completion_hook import CompletionHook

from .test_utils import openai_model, stub_openai_client


class RecordingHook(CompletionHook):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def on_completion(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _generate_a_chat_span() -> None:
    model = openai_model()
    model.client = stub_openai_client("Bonjour")
    model.generate(messages=[{"role": "user", "content": "Hi"}])


def test_entrypoint_loads_instrumentor() -> None:
    (entrypoint,) = entry_points(
        group="opentelemetry_instrumentor", name="smolagents"
    )
    instrumentor = entrypoint.load()()
    assert isinstance(instrumentor, SmolagentsInstrumentor)


def test_instrumentation_dependencies() -> None:
    dependencies = SmolagentsInstrumentor().instrumentation_dependencies()
    assert "smolagents >= 1.24.0" in dependencies


def test_instrument_uninstrument_restores_originals(
    tracer_provider, logger_provider, meter_provider
) -> None:
    original_generate = smolagents.OpenAIModel.generate
    original_base_generate = smolagents.Model.generate
    original_generate_stream = smolagents.OpenAIModel.generate_stream

    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    assert smolagents.OpenAIModel.generate is not original_generate
    assert smolagents.Model.generate is not original_base_generate
    assert (
        smolagents.OpenAIModel.generate_stream is not original_generate_stream
    )

    instrumentor.uninstrument()

    assert smolagents.OpenAIModel.generate is original_generate
    assert smolagents.Model.generate is original_base_generate
    assert smolagents.OpenAIModel.generate_stream is original_generate_stream


def test_uninstrument_through_a_new_constructor_call(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # BaseInstrumentor is a per-class singleton, so the documented
    # SmolagentsInstrumentor().instrument() / SmolagentsInstrumentor()
    # .uninstrument() form must restore everything even though the second
    # constructor call re-runs __init__ on the live instance.
    original_generate = smolagents.OpenAIModel.generate

    SmolagentsInstrumentor().instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    SmolagentsInstrumentor().uninstrument()

    assert smolagents.OpenAIModel.generate is original_generate


@pytest.mark.parametrize("method", ["generate", "generate_stream"])
def test_model_classes_defining(method: str) -> None:
    classes = _model_classes_defining(smolagents, method)

    # Each class object appears once, whatever names it is exported under.
    assert len(classes) == len(set(classes))
    # Every entry owns the method, so no class is wrapped for one it only
    # inherits.
    for model_cls in classes:
        assert method in model_cls.__dict__

    # The API-backed model classes define both; the classes that inherit them
    # are reached through the base they inherit from.
    assert {
        smolagents.OpenAIModel,
        smolagents.LiteLLMModel,
        smolagents.InferenceClientModel,
    } <= set(classes)
    assert smolagents.AzureOpenAIModel not in classes
    assert smolagents.LiteLLMRouterModel not in classes


def test_only_generate_covers_the_base_class_and_bedrock() -> None:
    # The base Model and AmazonBedrockModel define generate but no
    # generate_stream, so streaming is patched on fewer classes.
    generate = set(_model_classes_defining(smolagents, "generate"))
    generate_stream = set(
        _model_classes_defining(smolagents, "generate_stream")
    )

    assert {smolagents.Model, smolagents.AmazonBedrockModel} <= generate
    assert generate_stream.isdisjoint(
        {smolagents.Model, smolagents.AmazonBedrockModel}
    )


def test_repeated_instrument_uninstrument(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # BaseInstrumentor returns a per-class singleton, so the wrapped-class
    # bookkeeping has to survive being filled and drained more than once.
    original_generate = smolagents.OpenAIModel.generate

    instrumentor = SmolagentsInstrumentor()
    for _ in range(2):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        assert smolagents.OpenAIModel.generate is not original_generate
        instrumentor.uninstrument()
        assert smolagents.OpenAIModel.generate is original_generate


def test_uninstrument_without_instrument() -> None:
    # BaseInstrumentor.uninstrument() short-circuits, but _uninstrument() must
    # also be a no-op on unpatched attributes: the rollback in _instrument()
    # calls it after a partial patch.
    original_generate = smolagents.OpenAIModel.generate

    SmolagentsInstrumentor().uninstrument()
    SmolagentsInstrumentor()._uninstrument()

    assert smolagents.OpenAIModel.generate is original_generate


def test_instrument_with_no_providers() -> None:
    # Without providers the handler falls back to the globals; instrumenting
    # must not require a caller to pass them.
    original_generate = smolagents.OpenAIModel.generate

    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument()
    try:
        assert smolagents.OpenAIModel.generate is not original_generate
    finally:
        instrumentor.uninstrument()

    assert smolagents.OpenAIModel.generate is original_generate


def test_failed_instrument_rolls_back_partial_patches(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # A failure part-way through must leave no class patched, because
    # uninstrument() cannot clean up after a failed _instrument().
    model_classes = _model_classes_defining(smolagents, "generate")
    assert len(model_classes) > 1, (
        "the rollback needs more than one class to patch"
    )
    originals = {
        model_cls: model_cls.__dict__["generate"]
        for model_cls in model_classes
    }
    stream_classes = _model_classes_defining(smolagents, "generate_stream")
    stream_originals = {
        model_cls: model_cls.__dict__["generate_stream"]
        for model_cls in stream_classes
    }

    real_wrap = wrap_function_wrapper
    calls = 0

    def fail_on_the_second_class(target: Any, name: str, wrapper: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("boom")
        real_wrap(target, name, wrapper)

    with patch(
        "opentelemetry.instrumentation.genai.smolagents.wrap_function_wrapper",
        fail_on_the_second_class,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            SmolagentsInstrumentor().instrument(
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
            )

    assert calls == 2, "the first class was expected to be patched"
    for model_cls, original in originals.items():
        assert model_cls.__dict__["generate"] is original
    for model_cls, original in stream_originals.items():
        assert model_cls.__dict__["generate_stream"] is original


def test_inherited_generate_wrapped_only_on_defining_classes(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # AzureOpenAIModel and LiteLLMRouterModel inherit generate; they must not be
    # wrapped separately or they would emit duplicate chat spans.
    assert "generate" not in smolagents.AzureOpenAIModel.__dict__
    assert "generate" not in smolagents.LiteLLMRouterModel.__dict__

    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        # wrapt returns a fresh bound wrapper per attribute access, so the
        # wrapper objects differ; the underlying wrapped function is shared,
        # proving AzureOpenAIModel inherits the single wrapped generate.
        assert (
            smolagents.AzureOpenAIModel.generate.__wrapped__
            is smolagents.OpenAIModel.generate.__wrapped__
        )
    finally:
        instrumentor.uninstrument()


def test_explicit_completion_hook_takes_precedence(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    explicit_hook = RecordingHook()
    with patch(
        "opentelemetry.instrumentation.genai.smolagents.load_completion_hook"
    ) as load_hook:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
            completion_hook=explicit_hook,
        ):
            _generate_a_chat_span()

    load_hook.assert_not_called()
    assert explicit_hook.calls, "explicit completion hook was not invoked"


def test_env_completion_hook_used_when_no_explicit_hook(
    tracer_provider, logger_provider, meter_provider
) -> None:
    env_hook = RecordingHook()
    with patch(
        "opentelemetry.instrumentation.genai.smolagents.load_completion_hook",
        return_value=env_hook,
    ) as load_hook:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            _generate_a_chat_span()

    load_hook.assert_called_once()
    assert env_hook.calls, "env-resolved completion hook was not invoked"
