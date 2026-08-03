# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry smolagents Instrumentation
========================================

Instrumentation for `smolagents <https://github.com/huggingface/smolagents>`_.

Model calls are recorded as ``chat`` spans. Agent runs and tool calls are not
instrumented yet.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.smolagents import (
        SmolagentsInstrumentor,
    )
    from smolagents import InferenceClientModel

    SmolagentsInstrumentor().instrument()

    model = InferenceClientModel()
    model.generate([{"role": "user", "content": "How many seconds are in a week?"}])

Configuration
-------------

Message content capture can be configured by setting the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``. Supported values are
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, and ``SPAN_AND_EVENT``.

Captured content can be forwarded to external storage with a completion hook.
Set ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` (with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``), or pass one programmatically
via ``instrument(completion_hook=...)`` which takes precedence over the
environment variable.

API
---
"""

from __future__ import annotations

from collections.abc import Collection
from types import ModuleType
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .package import _instruments
from .patch import model_generate, model_generate_stream

__all__ = ["SmolagentsInstrumentor"]


def _model_classes_defining(smolagents: ModuleType, method: str) -> list[type]:
    """The exported model classes whose ``method`` gets wrapped.

    Only classes that define ``method`` in their own ``__dict__`` are patched,
    so a class that inherits it (``AzureOpenAIModel``, ``LiteLLMRouterModel``)
    isn't wrapped a second time and can't produce duplicate ``chat`` spans.
    A user-defined subclass that overrides the method shadows the patched base
    method and emits no ``chat`` span; that limitation is documented in
    ``README.rst``.

    Deduplicated by class object, because smolagents exports some classes under
    two names (``OpenAIServerModel`` is ``OpenAIModel``) and wrapping the same
    class twice would double every ``chat`` span.
    """
    from smolagents.models import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        Model,
    )

    classes: dict[type, None] = {}
    for obj in vars(smolagents).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, Model)
            and method in obj.__dict__
        ):
            classes.setdefault(obj, None)
    return list(classes)


class SmolagentsInstrumentor(BaseInstrumentor):
    """An instrumentor for smolagents."""

    # ``BaseInstrumentor.__new__`` returns a per-class singleton, but Python
    # still runs ``__init__`` on every construction. Initializing this state in
    # ``__init__`` would let the documented ``SmolagentsInstrumentor()
    # .uninstrument()`` form wipe the live instance's bookkeeping and leave
    # smolagents permanently patched, so these are class-level defaults that
    # only ``_instrument`` / ``_uninstrument`` rebind.
    _wrapped_generate_classes: list[type] = []
    _wrapped_generate_stream_classes: list[type] = []

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable smolagents instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
                - completion_hook: CompletionHook instance
        """
        import smolagents  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )

        wrapped_generate_classes: list[type] = []
        self._wrapped_generate_classes = wrapped_generate_classes
        wrapped_generate_stream_classes: list[type] = []
        self._wrapped_generate_stream_classes = wrapped_generate_stream_classes
        try:
            for model_cls in _model_classes_defining(smolagents, "generate"):
                wrap_function_wrapper(
                    model_cls,
                    "generate",
                    model_generate(handler),
                )
                wrapped_generate_classes.append(model_cls)

            for model_cls in _model_classes_defining(
                smolagents, "generate_stream"
            ):
                wrap_function_wrapper(
                    model_cls,
                    "generate_stream",
                    model_generate_stream(handler),
                )
                wrapped_generate_stream_classes.append(model_cls)
        except Exception:
            # BaseInstrumentor.instrument() doesn't mark the instrumentor as
            # instrumented when _instrument raises, so uninstrument() would
            # refuse to run and leave the patches applied with no way to undo.
            self._uninstrument()
            raise

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable smolagents instrumentation and restore patched originals."""
        for model_cls in self._wrapped_generate_classes:
            unwrap(model_cls, "generate")
        self._wrapped_generate_classes = []

        for model_cls in self._wrapped_generate_stream_classes:
            unwrap(model_cls, "generate_stream")
        self._wrapped_generate_stream_classes = []
