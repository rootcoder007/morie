Language models and the curated tables
======================================

Language models: bring your own
-------------------------------

The project hosts no language model, and no prompt goes anywhere you did not set up. The
assistant (``morie ask``, ``morie chat``, ``morie percy``, :func:`morie.llm.ask`) uses the model
you bring, in the same order rmoriebricklayer uses:

1. **Your own server**: any OpenAI-compatible endpoint, such as LM Studio
   (``http://localhost:1234/v1``), vLLM, llama.cpp's ``llama-server`` or a company gateway:
   ``morie config set own.url URL`` (and ``own.model``, ``own.key``).
2. **A provider's key**: OpenAI, Anthropic, Google Gemini, OpenRouter, Groq, Together AI,
   Fireworks AI, DeepSeek, Mistral, xAI, Cerebras, Perplexity, Hugging Face, OpenCode Zen,
   Cohere and NVIDIA NIM (``morie.providers.PRESETS``), each from its usual variable
   (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GEMINI_API_KEY``, ...) or saved with
   ``morie config set anthropic.key`` (asked for, never echoed or printed). The model is
   ``MORIE_<PROVIDER>_MODEL`` or ``<provider>.model``; ``morie models --route PROVIDER`` lists
   what your key can use. Anthropic's Messages API and Gemini's ``generateContent`` are spoken
   natively; the rest are OpenAI-compatible.
3. **Ollama**: a server on this machine or another (``ollama serve``, the Ollama app, or its
   system service), with a model pulled (``ollama pull NAME``); ``ollama.url`` points at another
   machine, and ``ollama.url https://ollama.com`` with ``ollama.key`` reaches Ollama's own cloud.
4. Nothing set up: a keyword answer from the package's own help, marked as such.

``morie config setup`` walks through the choices; ``morie doctor`` says which route ``ask`` takes
and why the others are not used; ``morie config set route anthropic`` (or ``--route`` on one
call) pins one, and a pinned route never falls back to another. Settings live in
``$XDG_CONFIG_HOME/morie/llm.json`` (owner-only), the file rmoriebricklayer's ``rmbl config``
reads too, and an environment variable that is set wins over a saved value.

.. code-block:: bash

   export ANTHROPIC_API_KEY=...                  # or: morie config set anthropic.key
   morie doctor                                  # ask uses: Anthropic, model claude-opus-5-5
   morie models                                  # the models the route offers (the one in use marked)
   morie ask "which module fits a treatment-control design?"
   morie config set own.url http://localhost:1234/v1   # LM Studio first, from now on
   morie ask --route groq "..."                  # one question on another route

A provider with a key but no model set (and no default) is not asked: ``ask`` says how to name
one. The hosted tier that used to answer at ``llm.rmorie.com`` is retired; ``route = hosted`` is
an error that says so.

The curated tables
------------------

The curated tables at https://data.rmorie.com open without sign-in and without a key; none is
read or sent (a key an earlier version stored stays in its file, unread):

.. code-block:: bash

   morie list-datasets
   morie pull chicago_crime/incidents --out incidents.csv

The service's address comes from the signed services document (``morie.services``), verified
with a pinned ML-DSA-44 key before use. Every table is checked against the SHA-256 the manifest
lists before it is used. A request carries the package User-Agent the service asks of
downloads. ICPSR studies (and the NCVS 38604 extract) are not served (ICPSR's terms do not
allow redistribution); asking for one says where to get it.
