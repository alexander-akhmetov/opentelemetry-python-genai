# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios for the ``chat`` operation (plain and tool-calling)."""

from __future__ import annotations

from typing import Any

from smolagents.models import ChatMessage, MessageRole

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from ..test_utils import GetWeatherTool, openai_model  # noqa: TID252
from ._helpers import attr, chat_spans, part_fields


class ChatScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

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
            with vcr.use_cassette("openai_model_basic.yaml"):
                openai_model().generate(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Who won the World Cup in 2018? Answer "
                                        "in one word with no punctuation."
                                    ),
                                }
                            ],
                        }
                    ]
                )


class ToolCallingScenario(Scenario):
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
            with vcr.use_cassette("openai_model_tool.yaml"):
                openai_model().generate(
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

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        output_part_types = {
            part_type
            for span in chat_spans(report)
            for part_type, _ in part_fields(
                attr(span, "gen_ai.output.messages")
            )
        }
        assert "tool_call" in output_part_types, (
            f"expected a tool_call output part, saw {output_part_types}"
        )
