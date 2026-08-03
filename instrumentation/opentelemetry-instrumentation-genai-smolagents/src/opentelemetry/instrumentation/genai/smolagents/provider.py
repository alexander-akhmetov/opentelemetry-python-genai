# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a smolagents model instance to a ``gen_ai.provider.name`` value.

Resolution order:

1. For the LiteLLM model classes, the ``model_id`` vendor prefix
   (``anthropic/claude-...`` -> ``anthropic``).
2. The model class name (e.g. ``OpenAIModel`` -> ``openai``), looked up along the
   class hierarchy so a user subclass resolves to the provider of the base class
   whose ``generate`` it inherits.
3. ``unknown``.

``gen_ai.provider.name`` is a metric attribute as well as a span attribute, so
every value has to stay low cardinality: deployment-specific detail is reported
as ``server.address`` instead. ``TelemetryHandler.inference`` requires
``provider`` as a string, so this always returns a value rather than ``None``.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

_logger = logging.getLogger(__name__)

_PROVIDER = GenAI.GenAiProviderNameValues

_UNKNOWN_PROVIDER = "unknown"

# Model class name -> provider value. The GenAI registry has no value for the
# Hugging Face, vLLM, and MLX runtimes, so those use the product name; a class
# name would look like a provider without being one.
_CLASS_NAME_TO_PROVIDER: dict[str, str] = {
    "OpenAIModel": _PROVIDER.OPENAI.value,
    "AzureOpenAIModel": _PROVIDER.AZURE_AI_OPENAI.value,
    "AmazonBedrockModel": _PROVIDER.AWS_BEDROCK.value,
    "InferenceClientModel": "huggingface",
    "TransformersModel": "huggingface",
    "VLLMModel": "vllm",
    "MLXModel": "mlx",
}

# LiteLLM model_id prefix -> semconv provider value, for the prefixes whose
# LiteLLM vendor slug differs from the semconv value. Every other prefix is
# passed through as-is (``ollama/llama3`` -> ``ollama``). LiteLLM's slugs are a
# closed vocabulary, which keeps the cardinality bounded.
_LITELLM_PREFIX_TO_PROVIDER: dict[str, str] = {
    "azure": _PROVIDER.AZURE_AI_OPENAI.value,
    "azure_ai": _PROVIDER.AZURE_AI_INFERENCE.value,
    "bedrock": _PROVIDER.AWS_BEDROCK.value,
    "gemini": _PROVIDER.GCP_GEMINI.value,
    "mistral": _PROVIDER.MISTRAL_AI.value,
    "vertex_ai": _PROVIDER.GCP_VERTEX_AI.value,
    "watsonx": _PROVIDER.IBM_WATSONX_AI.value,
    "xai": _PROVIDER.X_AI.value,
}

_LITELLM_CLASS_NAMES = frozenset({"LiteLLMModel", "LiteLLMRouterModel"})


def _class_names(instance: Any) -> list[str]:
    """The instance's class names, most derived first.

    Only the classes that define ``generate`` are patched, so an instrumented
    model can be a user subclass of one of them. Matching the exact class name
    alone would report ``unknown`` for every such subclass.
    """
    return [cls.__name__ for cls in type(instance).__mro__]


def _provider_from_litellm(instance: Any) -> str | None:
    model_id = getattr(instance, "model_id", None)
    if not isinstance(model_id, str) or "/" not in model_id:
        return None
    prefix = model_id.split("/", 1)[0].lower()
    return _LITELLM_PREFIX_TO_PROVIDER.get(prefix, prefix)


def _endpoint(instance: Any) -> str | None:
    """The endpoint URL the model calls, or ``None`` if it exposes none.

    Three places can hold it, checked in this order:

    1. ``api_base`` on the instance (``LiteLLMModel``).
    2. ``base_url`` or ``azure_endpoint`` in the SDK client kwargs
       (``OpenAIModel``, ``AzureOpenAIModel``, ``InferenceClientModel``).
    3. ``base_url`` on the client the model constructed (e.g. openai's httpx
       URL), which holds the effective URL when the caller left it at the
       provider default.
    """
    api_base = getattr(instance, "api_base", None)
    if api_base:
        return str(api_base)
    client_kwargs = getattr(instance, "client_kwargs", None)
    if isinstance(client_kwargs, dict):
        for key in ("base_url", "azure_endpoint"):
            value = client_kwargs.get(key)
            if value:
                return str(value)
    client_base_url = getattr(
        getattr(instance, "client", None), "base_url", None
    )
    if client_base_url:
        return str(client_base_url)
    return None


def resolve_server_address_port(
    instance: Any,
) -> tuple[str | None, int | None]:
    """Return ``(server.address, server.port)`` from the model's endpoint URL.

    Models that don't expose an ``api_base`` / ``base_url`` / ``azure_endpoint``
    (e.g. ``LiteLLMModel`` resolving the host internally, local runtimes)
    yield ``(None, None)`` and the caller omits the attributes.
    """
    endpoint = _endpoint(instance)
    if endpoint is None:
        return None, None
    parsed = urlparse(endpoint)
    port = parsed.port
    if port == 443:
        port = None
    return parsed.hostname or None, port


def resolve_provider(instance: Any) -> str:
    """Return the ``gen_ai.provider.name`` value for a smolagents model instance."""
    class_names = _class_names(instance)

    if not _LITELLM_CLASS_NAMES.isdisjoint(class_names):
        provider = _provider_from_litellm(instance)
        if provider is not None:
            return provider

    for class_name in class_names:
        provider = _CLASS_NAME_TO_PROVIDER.get(class_name)
        if provider is not None:
            return provider

    _logger.debug(
        "No known gen_ai.provider.name for model class %s", class_names[0]
    )
    return _UNKNOWN_PROVIDER
