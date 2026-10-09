# SPDX-License-Identifier: AGPL-3.0-or-later
"""``morie help [TOPIC]``: short guides, so nobody has to read the source to get going."""

from __future__ import annotations

TOPICS: dict[str, tuple[str, str]] = {
    "getting-started": (
        "first steps: install check, data, analyses, asking questions",
        """\
Getting started with morie

  morie doctor                  what is installed and reachable from here, and the route `ask` takes
  morie list-datasets           the bundled and hosted datasets (keys for --dataset)
  morie list-modules            the analysis modules and the files each writes
  morie run-module NAME         run one module; morie pipeline --all -y runs them all
  morie cheatsheet              a one-page command reference
  morie explain FILE            what an output file holds

Asking questions (needs a language model; see `morie help llm`):

  morie login                   sign in to the hosted MORIE tier (GitHub; --email for a code)
  morie ask "how do I run an IPW analysis?"
  morie models                  the models you can ask

Settings: `morie help config`. Every command has its own --help (morie ask --help).""",
    ),
    "llm": (
        "how `morie ask` finds a language model, and how to choose one",
        """\
How `morie ask` reaches a model

Three routes:
  ollama   a local or LAN Ollama server (private, no key). It counts only when it has a
           model: one pulled (`ollama pull gemma4:e2b`) or ollama.model set. A server with
           nothing pulled is skipped.
  own      your own OpenAI-compatible server or key: own.url (+ own.key, own.model),
           `morie provider set`, GEMINI_API_KEY or OPENAI_API_KEY.
  hosted   the hosted MORIE tier, after `morie login` (or `morie login --token KEY`).

With route = auto (the default) `ask` takes the first that is set up, in the order
ollama, own, hosted; when a request fails it moves on to the next. Choose one:

  morie config set route hosted          always the hosted tier (unset route: back to auto)
  morie ask --route hosted "..."         one question only
  MORIE_LLM_ROUTE=hosted morie ask ...   one shell only

  morie doctor            which route `ask` takes now, and why the others are not used
  morie models            the hosted models for your key, then the local Ollama models
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
  morie config set KEY VALUE      keys (own.key, ollama.key, hosted.key) prompt when VALUE is left out
  morie config unset KEY
  morie config path               where they are saved

Keys: route (auto | own | ollama | hosted), own.url, own.key, own.model, ollama.url,
ollama.model, ollama.key, hosted.url, hosted.model, hosted.key.

They are saved in $XDG_CONFIG_HOME/morie/llm.json (default ~/.config/morie/llm.json,
private), shared with rmoriebricklayer's `rmbl config`. hosted.key is checked with the
gateway and kept in credentials.json, as `morie login` does. An environment variable that
is set (MORIE_LLM_ROUTE, OLLAMA_HOST, OLLAMA_MODEL, MORIE_HOSTED_MODEL, ...) wins over a
saved value.

From Python:
  from morie.llm import config, config_get, config_unset
  config()                                   # the table
  config(route="hosted", hosted_model="gpt-oss-120b:cf")
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
