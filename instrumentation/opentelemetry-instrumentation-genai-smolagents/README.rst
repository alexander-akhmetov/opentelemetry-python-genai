OpenTelemetry smolagents Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-smolagents.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-smolagents/

This library provides OpenTelemetry instrumentation for `smolagents
<https://github.com/huggingface/smolagents>`_.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-smolagents

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.smolagents import (
        SmolagentsInstrumentor,
    )

    # Instrument smolagents
    SmolagentsInstrumentor().instrument()

Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message
content, set the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of ``NO_CONTENT``,
``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `smolagents Documentation <https://huggingface.co/docs/smolagents>`_
* `smolagents GitHub Repository <https://github.com/huggingface/smolagents>`_
