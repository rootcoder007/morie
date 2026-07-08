# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.agent — call the rmorie terminal agent (from rmorie-cli).

morie already ships a native LLM layer (``morie.ask`` / ``morie.llm``). This
is the *ecosystem-uniform* entry point that shells out to the optional
``rmorie`` CLI agent — the same agent ``rmorie::agent`` /
``rmoriedata::ask`` / ``rmoriebricklayer::agent_bundle`` call — so every MORIE
package exposes one consistent agent. Optional dependency: needs the
``rmorie`` binary on PATH and, for cloud backends, ``RMORIE_AGENT_API_KEY`` in
the environment (the key is never passed as an argument).
"""
from __future__ import annotations

import shutil
import subprocess


def agent(
    task: str,
    *,
    model: str | None = None,
    backend: str = "auto",
    dry_run: bool = False,
) -> str:
    """Run the rmorie CLI agent on ``task`` and return its output.

    Parameters
    ----------
    task:
        The prompt/task for the agent.
    model:
        Model id, e.g. ``"claude-sonnet-5"``, ``"gpt-4o"``,
        ``"gemini-2.5-flash"``, or an Ollama tag such as ``"llama3.2"``.
        Defaults to the CLI's default model.
    backend:
        ``"auto"`` (route on the model id), ``"anthropic"``, ``"openai"``,
        ``"gemini"``, or ``"ollama"``.
    dry_run:
        If ``True`` the agent only plans (no network call).

    Returns
    -------
    str
        The agent's combined stdout/stderr, or a clear message if the
        ``rmorie`` binary is not installed.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    binary = shutil.which("rmorie")
    if not binary:
        return (
            "rmorie CLI not found on PATH. Install rmorie-cli to use "
            "morie.agent()."
        )
    args = [binary, "agent", "--backend", backend]
    if model:
        args += ["-m", model]
    if dry_run:
        args += ["--dry-run"]
    args.append(task)
    proc = subprocess.run(args, capture_output=True, text=True)
    return (proc.stdout + proc.stderr).strip()
