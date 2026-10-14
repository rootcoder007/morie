# SPDX-License-Identifier: AGPL-3.0-or-later
"""``morie help [TOPIC]``: short guides, so nobody has to read the source to get going."""

from __future__ import annotations

TOPICS: dict[str, tuple[str, str]] = {
    "getting-started": (
        "first steps: install check, data, analyses, asking questions",
        """\
Getting started with morie

  morie doctor                  what is installed and reachable from here, and the route `ask` takes
  morie list-datasets           the bundled datasets and the curated tables (keys for --dataset)
  morie list-modules            the analysis modules and the files each writes
  morie run-module NAME         run one module; morie pipeline --all -y runs them all
  morie cheatsheet              a one-page command reference
  morie explain FILE            what an output file holds

Asking questions (needs a language model; see `morie help llm`):

  morie config setup            bring a model: a provider's key, your own server, or Ollama
  morie ask "how do I run an IPW analysis?"
  morie models                  the models the route offers

The curated tables at data.rmorie.com: `morie login` (GitHub; --email for a code).

Settings: `morie help config`. Every command has its own --help (morie ask --help).""",
    ),
    "llm": (
        "how `morie ask` finds a language model, and how to choose one",
        """\
How `morie ask` reaches a model

Bring your own; nothing is hosted for you and no prompt goes anywhere you did not set up.
Three kinds of route:
  own        your own OpenAI-compatible server: own.url (+ own.key, own.model), e.g.
             LM Studio (http://localhost:1234/v1), vLLM, llama.cpp, a company gateway.
  PROVIDER   a provider whose key is set: openai, anthropic, gemini, openrouter, groq,
             together, fireworks, deepseek, mistral, xai, cerebras, perplexity, huggingface,
             opencode, cohere, nvidia. The key from its usual variable (ANTHROPIC_API_KEY,
             OPENAI_API_KEY, ...) or `morie config set anthropic.key` (asked for, saved privately).
  ollama     an Ollama server (`ollama serve`, the Ollama app or its service), on this machine,
             another one, or Ollama's cloud (ollama.url https://ollama.com with ollama.key).
             It counts only when it has a model: one pulled, or ollama.model set.

With route = auto (the default) `ask` takes the first that is set up, in the order own,
each provider with a key, ollama; when a request fails it moves on to the next. Choose one:

  morie config set route anthropic          always that one (unset route: back to auto)
  morie ask --route groq "..."              one question only
  MORIE_LLM_ROUTE=ollama morie ask ...      one shell only

  morie doctor            which route `ask` takes now, and why the others are not used
  morie models            the models the route offers (--route ROUTE for another)
  morie ask --model NAME  one question with another model

All settings: `morie help config`.""",
    ),
    "config": (
        "the language-model settings (`morie config`)",
        """\
Language-model settings

  morie config                    every setting, its value and where it comes from
  morie config help               what each setting means, and its environment variable
  morie config setup              walk through all of them (Enter keeps a value)
  morie config get KEY
  morie config set KEY VALUE      keys (own.key, ollama.key, anthropic.key, ...) prompt when VALUE is left out
  morie config unset KEY
  morie config path               where they are saved

Keys: route (auto | own | ollama | a provider), own.url, own.key, own.model, ollama.url,
ollama.model, ollama.key, PROVIDER.key and PROVIDER.model for each provider, and r.mismatch
(warn | quiet | strict): what the R-backed commands do when the installed R package is
another version than this morie. warn, the default, uses it and says so once;
`morie config set r.mismatch strict` refuses instead.

They are saved in $XDG_CONFIG_HOME/morie/llm.json (default ~/.config/morie/llm.json,
private), shared with rmoriebricklayer's `rmbl config`. A key is never printed: it shows only
as "set". An environment variable that is set (MORIE_LLM_ROUTE, ANTHROPIC_API_KEY,
OLLAMA_HOST, OLLAMA_MODEL, MORIE_ANTHROPIC_MODEL, ...) wins over a saved value.

From Python:
  from morie.llm import config, config_get, config_unset
  config()                                   # the table
  config(route="anthropic", anthropic_model="claude-opus-5-5")
  config_get("route"); config_unset("route")""",
    ),
}


def topics_text() -> str:
    """The topic list (``morie help`` with no topic)."""
    lines = ["morie help TOPIC:"]
    lines += [f"  {name:<16} {desc}" for name, (desc, _) in TOPICS.items()]
    lines.append("  commands         every command (same as morie --help)")
    return "\n".join(lines)


def show(topic: str | None) -> str | None:
    """The text of a topic, the topic list for None, or None for an unknown topic."""
    if not topic:
        return topics_text()
    t = topic.strip().lower().replace("_", "-")
    t = {"start": "getting-started", "quickstart": "getting-started", "models": "llm", "ask": "llm"}.get(t, t)
    entry = TOPICS.get(t)
    return entry[1] if entry else None
