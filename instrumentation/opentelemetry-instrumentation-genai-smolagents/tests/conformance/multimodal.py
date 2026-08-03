# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios for non-text message parts: an image ``uri`` on a chat
input, and a ``reasoning`` part on a chat output."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from smolagents import LiteLLMModel, OpenAIModel
from smolagents.models import ChatMessage, MessageRole

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

from ._helpers import attr, chat_spans, part_fields

_IMAGE_URL = (
    "https://fastly.picsum.photos/id/237/200/300.jpg"
    "?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
)


class MultimodalScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("openai_model_image_url.yaml"):
                model = OpenAIModel(
                    model_id="gpt-4o",
                    api_key="test_openai_api_key",
                    api_base="https://api.openai.com/v1",
                )
                model.generate(
                    messages=[
                        ChatMessage(
                            role=MessageRole.USER,
                            content=[
                                {
                                    "type": "text",
                                    "text": "What breed is this dog?",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _IMAGE_URL},
                                },
                            ],
                        )
                    ]
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        input_parts = {
            fields
            for span in chat_spans(report)
            for fields in part_fields(attr(span, "gen_ai.input.messages"))
        }
        assert ("uri", "image") in input_parts, (
            f"expected an image uri input part, saw {input_parts}"
        )


class ReasoningScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = (
        # LiteLLM routes to the provider host internally and a LiteLLMModel
        # built without an explicit api_base exposes no endpoint URL, so there
        # is nothing to derive server.address from on the chat span.
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
        ),
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        env = {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("tiktoken.get_encoding") as get_encoding,
        ):
            get_encoding.return_value = mock.MagicMock(
                encode=lambda *_: [1, 2, 3]
            )
            with instrument(
                SmolagentsInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette("litellm_reasoning.yaml"):
                    model = LiteLLMModel(
                        model_id="anthropic/claude-3-7-sonnet-20250219",
                        api_key="test_anthropic_api_key",
                        thinking={"type": "enabled", "budget_tokens": 4000},
                    )
                    model.generate(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "Who won the World Cup in 2018? "
                                            "Answer in one word with no "
                                            "punctuation."
                                        ),
                                    }
                                ],
                            }
                        ]
                    )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        output_parts = {
            part_type
            for span in chat_spans(report)
            for part_type, _ in part_fields(
                attr(span, "gen_ai.output.messages")
            )
        }
        assert "reasoning" in output_parts, (
            f"expected a reasoning output part, saw {output_parts}"
        )
