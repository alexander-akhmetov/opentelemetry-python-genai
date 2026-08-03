# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for smolagents instrumentation tests."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.test_util_genai.vcr import (
    scrub_response_headers_overwrite,
)

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("cookie", "test_cookie"),
            ("authorization", "Bearer test_openai_api_key"),
            ("x-api-key", "test_anthropic_api_key"),
            ("openai-organization", "test_openai_org_id"),
            ("openai-project", "test_openai_project_id"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "openai-organization": "test_openai_org_id",
                "openai-project": "test_openai_project_id",
                "Set-Cookie": "test_set_cookie",
            }
        ),
    }


@pytest.fixture
def litellm_local_cost_map():
    """Use LiteLLM's bundled model-cost map so it doesn't fetch prices over the
    network during cassette playback."""
    previous = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = previous


@pytest.fixture
def patch_tiktoken_encoding():
    """Patch ``tiktoken.get_encoding`` so LiteLLM doesn't download an encoding."""
    with patch("tiktoken.get_encoding") as mock_get_encoding:
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding
        yield


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    with instrument(
        SmolagentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    with instrument(
        SmolagentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_event_only(tracer_provider, logger_provider, meter_provider):
    with instrument(
        SmolagentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="EVENT_ONLY",
        emit_event=True,
    ) as instrumentor:
        yield instrumentor
