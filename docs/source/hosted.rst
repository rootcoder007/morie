Hosted LLM tier
===============

``https://llm.rmorie.com`` is an authenticated, rate-limited inference
endpoint run by the MORIE project for the ``morie`` (Python) and
``rmorie`` (R) packages. It exists so that ``morie ask`` and
``morie_llm_ask()`` work on a machine with no local model and no API key
of your own, and it is the last resort: a local Ollama is tried first, then
every key of your own (an attached endpoint, Gemini, OpenAI; see
`Your own model endpoint`_ below), and the hosted tier only when none of
those answers. Keys are issued on request at https://rmorie.com/access.
The endpoints, modes and model list come from a signed services document
(``https://rmorie.com/.well-known/morie-services.json``, ML-DSA-44, public
key pinned in the package) that the package verifies before use, so they
can change without a release; ``MORIE_SERVICES_URL`` names a mirror, with
the same signature check.

Getting a key
-------------

One key per person, shared by both languages. Request it at
https://rmorie.com/access and store it:

.. code-block:: bash

   morie login --token                           # paste the key you were issued

The GitHub and emailed-code sign-ins remain for accounts that have them:

.. code-block:: bash

   morie login                                   # GitHub device flow
   morie login --email you@example.com           # 6-digit code by email
   morie login --email you@example.com --to-email   # key sent to your inbox

.. code-block:: r

   rmorie::morie_llm_login(token = "<key>")               # the key you were issued
   rmorie::morie_llm_login()                              # GitHub
   rmorie::morie_llm_login(email = "you@example.com")     # emailed code
   rmorie::morie_llm_login(email = "you@example.com", to_email = TRUE)

The R package also ships the same verbs as a shell command:
``rmorie::install_cli()`` links ``rmorie`` onto your PATH, after which
``rmorie login``, ``rmorie login --email …``, ``rmorie login --token``,
``rmorie logout``, ``rmorie doctor``, ``rmorie models`` and ``rmorie ask …``
work like their ``morie`` counterparts; ``rmoriebricklayer`` has the same
verbs once ``rmoriebricklayer::install_cli()`` has run.

Which models you can ask
------------------------

.. code-block:: bash

   morie models                                  # the hosted tier's list for your key (default marked *), then local Ollama
   morie doctor                                  # the same list on the hosted line
   morie ask --model gpt-oss:120b-cloud "..."    # one call with a named model

The same verbs exist as ``rmorie models`` / ``rmorie ask --model NAME`` and
``rmoriebricklayer models`` / ``rmoriebricklayer ask --model NAME``; in R,
``rmorie::morie_llm_hosted_models()`` and
``rmoriebricklayer::bricklayer_llm_models()`` return the list with the
default as an attribute. ``MORIE_HOSTED_MODEL`` changes the default.

The site at https://llm.rmorie.com describes the tier and points to the
request form at https://rmorie.com/access; it offers no sign-in of its own.

Where the key lives
-------------------

``$XDG_CONFIG_HOME/morie/credentials.json`` (``~/.config/morie/`` when
the variable is unset), written owner-only (mode 0600 on POSIX). Both
packages read and write the same file, so signing in from R signs you
in for Python and vice versa. ``morie logout`` / ``morie_llm_logout()``
remove it.

Environment overrides:

``MORIE_HOSTED_KEY``
   Use this key instead of the stored one (CI, containers).
``MORIE_HOSTED_BASE_URL``
   Another gateway (the default comes from the services document), or
   ``off`` to disable the tier entirely (``""`` also
   disables it on POSIX; Windows drops an empty variable, hence ``off``).
``MORIE_HOSTED_MODEL``
   Model to request (default: the services document's default model). When the gateway no
   longer lists the requested model, the packages use the first model it
   does list instead of failing, since cloud models get retired upstream.
``MORIE_HOSTED_AUTH_URL``
   The sign-in service (default from the services document).

Which models
------------

``morie models`` (``morie_llm_models()`` in R) is the list to trust: the
gateway serves the ollama.com cloud models (``minimax-m3:cloud``,
``gpt-oss:120b-cloud`` and the rest) and additional AI models,
whose ids end in ``:cf``: kimi-k2.6:cf, kimi-k2.7-code:cf, deepseek-v4-pro:cf, deepseek-v4-flash:cf, glm-5.2:cf, glm-5.3:cf, glm-5.3-flash:cf, gpt-oss-120b:cf, gpt-oss-20b:cf, llama-4-scout:cf, qwen3.8-27b:cf, nemotron-3-120b:cf and gemma-4-26b:cf.
Pick one per call with ``morie ask --model gpt-oss-120b:cf "..."``. When an
ollama.com model is rate limited or down, the gateway answers the same
request from one of the additional models, so a busy hour does not turn
into an error.

The same key opens data.rmorie.com
----------------------------------

The curated datasets at https://data.rmorie.com (:doc:`learn/datasets`) are
gated by this key too (issued on request at https://rmorie.com/access, under
https://rmorie.com/data-license): ``morie pull chicago_crime/incidents``,
``rmorie::morie_load_hosted_dataset()``, or any HTTP client with
``Authorization: Bearer <key>``.

Your own model endpoint
-----------------------

The hosted tier is one route; any OpenAI-compatible endpoint can be attached
instead or as well, and the assistant verbs (``ask``, ``percy``, ``agent``,
``chat``) use it when no local Ollama answers, before the hosted tier is
tried. OpenAI, Anthropic's compatibility endpoint
(``https://api.anthropic.com/v1``), OpenRouter, Mistral, Groq, a local LM
Studio / vLLM / llama.cpp server: anything that serves
``POST BASE_URL/chat/completions``.

.. code-block:: bash

   morie provider set --base-url https://api.openai.com/v1 --key sk-... --model gpt-4o-mini
   morie provider show                 # endpoint, model, masked key
   morie models                        # "Your endpoint (...)" is listed first
   morie ask "which module fits a treatment-control design?"
   morie provider unset

.. code-block:: r

   rmorie::morie_llm_provider_set("https://api.openai.com/v1", "sk-...", model = "gpt-4o-mini")
   rmorie::morie_llm_provider_show()
   rmorie::morie_llm_provider_unset()
   # or, from the shell: rmorie provider set --base-url URL --key KEY [--model NAME]

The setting is stored in the same credentials file as the hosted key, so
both languages see it. The environment variables ``LLM_API_BASE_URL``,
``LLM_API_KEY`` and ``MORIE_API_MODEL`` take precedence when set (CI,
containers). The full order the packages try, as ``morie doctor`` /
``rmorie doctor`` report it: local Ollama, your own keys (your endpoint,
``GEMINI_API_KEY``, ``OPENAI_API_KEY``), the hosted tier as the last resort,
then a local keyword fallback that says it is one.

Emailed keys and pasted tokens
------------------------------

``--to-email`` (``to_email = TRUE``) is for the case where the machine
you are signing in from is not the machine that will use the key: the
gateway mails the key to the address that just proved it owns the
inbox, stores nothing locally, and you paste it later with
``morie login --token`` (``morie_llm_login(token = )``), which probes the
gateway once and tells you whether it was accepted.

Limits and models
-----------------

Per key: 10 requests a minute, 30,000 tokens a minute, 100 requests a
day. Signing in again replaces your previous key. Only cloud-hosted open
models are exposed (``GET /v1/models`` lists the current set); local GPU
models on the host are never reachable. Any OpenAI-compatible client can
use the key against ``https://llm.rmorie.com/v1``.

What is logged
--------------

Per-key request and token counters, and the edge access log (IP, path,
status). No prompts, no responses, no GitHub tokens. The GitHub sign-in
reads only your public login to name the key; an email account is a hash
of the address. Keys can be revoked at any time, and abuse (automated
scraping, illegal content) gets the key revoked without notice.
