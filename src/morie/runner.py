"""CLI entrypoints for MORIE package workflows and assistant mode.

Provides the ``morie`` command with subcommands for running analysis modules,
querying the LLM agent, and checking environment health.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Emissions tracking -- use vendored morie.emissions, fall back to codecarbon
# ---------------------------------------------------------------------------
from morie.emissions import EmissionsTracker as _EmissionsTracker

from . import __version__
from .modules import DEFAULT_CPADS_CSV, list_modules, run_module
from .perseus import ask_percy

_CODECARBON_AVAILABLE = True  # the vendored tracker is always present


def execute_pipeline(
    modules: list[str] | None = None,
    *,
    cpads_csv: str = DEFAULT_CPADS_CSV,
    dataset_key: str | None = None,
    output_dir: str | None = None,
    silent: bool = False,
    track_carbon: bool = True,
) -> int:
    """
    Run the specified epidemiologic analysis modules.

    :param modules: A list of module names to execute, defaults to None.
    :type modules: list[str], optional
    :param silent: If True, skips the safety confirmation prompt, defaults to False.
    :type silent: bool, optional
    :param track_carbon: If True, track CO₂ emissions with the vendored morie.emissions tracker.
    :type track_carbon: bool, optional
    """
    selected = modules or [item["name"] for item in list_modules()]

    if not silent:
        print("Selected modules:", ", ".join(selected))
        confirm = input(f"This will run the MORIE module pipeline for: {', '.join(selected)}. Continue? [y/N]: ")
        if confirm.lower() != "y":
            print("Pipeline aborted.")
            return 1
    else:
        print("Selected modules:", ", ".join(selected))

    # Start pipeline-level emissions tracker
    tracker = None
    if track_carbon and _CODECARBON_AVAILABLE:
        import os

        emissions_dir = os.path.join(output_dir, "emissions") if output_dir else "data/manifest/outputs/emissions"
        os.makedirs(emissions_dir, exist_ok=True)
        try:
            tracker = _EmissionsTracker(
                project_name="morie-pipeline",
                output_dir=emissions_dir,
                log_level="error",
            )
            tracker.start()
        except Exception:  # pragma: no cover
            tracker = None

    results = {}
    total = len(selected)

    # rich progress (rich is a core dependency; enlighten is gone).
    _progress = None
    _task = None
    try:
        from rich.progress import Progress

        _progress = Progress()
        _progress.start()
        _task = _progress.add_task("Pipeline", total=total)
    except Exception:  # pragma: no cover - non-tty edge cases
        _progress = None

    failed: dict[str, str] = {}
    for idx, module_name in enumerate(selected, start=1):
        if _progress:
            _progress.update(_task, description=f"Running: {module_name}")
        else:
            print(f"[{idx}/{total}] Running: {module_name}", flush=True)
        try:
            results[module_name] = run_module(
                module_name,
                cpads_csv=cpads_csv,
                dataset_key=dataset_key,
                output_dir=output_dir,
            )
            status = "Done"
        except Exception as exc:
            failed[module_name] = str(exc)
            print(f"  ERROR in {module_name}: {exc}")
            status = "FAILED"
        if _progress:
            _progress.advance(_task)
        else:
            print(f"[{idx}/{total}] {status}: {module_name}", flush=True)

    if _progress:
        _progress.stop()

    if tracker is not None:
        try:
            emissions = tracker.stop()
            if emissions is not None:
                print(f"Pipeline CO₂ emissions: {emissions:.3g} kg CO₂eq")
                cap = getattr(tracker, "capsule", None)
                if cap:
                    print(f"Capsule: {cap['manifest']}{' (signed)' if cap.get('signed') else ' (unsigned)'}")
        except Exception:  # pragma: no cover
            pass

    n_ok = len(results)
    if failed:
        print(f"Pipeline completed {n_ok}/{total} ({len(failed)} failed: {', '.join(failed)}).")
        if results:
            print("Succeeded modules:", ", ".join(results.keys()))
        return 1
    print(f"Pipeline completed successfully ({n_ok}/{total}).")
    print("Completed modules:", ", ".join(results.keys()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="MORIE package runner")
    parser.add_argument("--version", action="version", version=f"morie {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    pipeline = subparsers.add_parser("pipeline", help="Run the MORIE module pipeline")
    pipeline.add_argument("--all", action="store_true", help="Run the current MORIE module surface")
    pipeline.add_argument("--modules", nargs="+", help="Override the default module list")
    pipeline.add_argument("--cpads-csv", default=DEFAULT_CPADS_CSV, help="Path to the CPADS CSV")
    pipeline.add_argument("--dataset", default=None, help="Dataset key (e.g. ocp21, hibp). Overrides --cpads-csv")
    pipeline.add_argument("--output-dir", help="Optional directory for CSV outputs")
    pipeline.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    pipeline.add_argument(
        "--no-carbon",
        action="store_true",
        default=False,
        help="Disable CodeCarbon emissions tracking for this run",
    )

    parity = subparsers.add_parser(
        "parity-review",
        help=argparse.SUPPRESS,
        description="Internal compatibility audit for maintainers.",
    )
    parity.add_argument("--epiml-root", required=True, help="Path to the internal reference checkout")
    parity.add_argument("--output", help="Optional CSV path for the audit matrix")

    subparsers.add_parser("list-modules", help="List implemented MORIE CPADS modules")

    run_cmd = subparsers.add_parser("run-module", help="Run one MORIE module")
    run_cmd.add_argument("module", help="Module name to run")
    run_cmd.add_argument("--cpads-csv", default=DEFAULT_CPADS_CSV, help="Path to the CPADS CSV")
    run_cmd.add_argument("--dataset", default=None, help="Dataset key (e.g. ocp21, hibp). Overrides --cpads-csv")
    run_cmd.add_argument("--output-dir", help="Optional directory for CSV outputs")

    run_all = subparsers.add_parser("run-modules", help="Run multiple MORIE modules")
    run_all.add_argument("--modules", nargs="+", help="Module names to run; defaults to all implemented modules")
    run_all.add_argument("--cpads-csv", default=DEFAULT_CPADS_CSV, help="Path to the CPADS CSV")
    run_all.add_argument("--dataset", default=None, help="Dataset key (e.g. ocp21, hibp). Overrides --cpads-csv")
    run_all.add_argument("--output-dir", help="Optional directory for CSV outputs")

    agent_cmd = subparsers.add_parser("agent", help="Ask Perseus, the MORIE agent")
    agent_cmd.add_argument("question", help="Question to answer")
    agent_cmd.add_argument("--context", help="Optional context string")
    agent_cmd.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming (default: stream to terminal)",
    )

    ask_cmd = subparsers.add_parser(
        "ask",
        help="Ask the MORIE agent (streaming by default)",
    )
    ask_cmd.add_argument("question", help="Question to answer")
    ask_cmd.add_argument("--context", help="Optional context string")
    ask_cmd.add_argument("--model", default=None, help="Override the LLM model name")
    ask_cmd.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming and print the full response at once",
    )

    # ── chat ─────────────────────────────────────────────────────────────
    chat_cmd = subparsers.add_parser(
        "chat",
        help="Launch interactive chat REPL (streaming LLM, slash commands)",
    )
    chat_cmd.add_argument(
        "--agent",
        default=None,
        help="Load a specific agent persona (e.g., morie-architect)",
    )

    percy_talk = subparsers.add_parser(
        "percy",
        aliases=["perseus"],
        help="Talk to Perseus -- MORIE's expert AI (auto-detects best connection)",
    )
    percy_talk.add_argument("question", nargs="?", default=None, help="Question (omit for interactive mode)")
    percy_talk.add_argument("--model", default=None, help="Override model name")
    percy_talk.add_argument("--remote", action="store_true", help="Use Perseus cloud (MORIE-hosted, no local setup)")
    percy_talk.add_argument("--local", action="store_true", help="Force local Ollama only (no network)")
    percy_talk.add_argument("--pi", default=None, help="Connect to Pi (e.g. --pi host or MORIE_PI_HOST)")
    percy_talk.add_argument("--cloud", default=None, help="Custom Perseus relay URL (e.g. https://your-relay:8421)")
    percy_talk.add_argument("--no-stream", action="store_true", help="Disable streaming output")

    relay_cmd = subparsers.add_parser(
        "serve",
        help="Start Perseus relay server (expose Perseus over the network)",
    )
    relay_cmd.add_argument("--port", type=int, default=8421, help="Port to listen on (default: 8421)")
    relay_cmd.add_argument("--token", default=None, help="Require auth token for API access")
    relay_cmd.add_argument("--bind", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")

    login_cmd = subparsers.add_parser(
        "login",
        help="Sign in to the hosted MORIE LLM tier (llm.rmorie.com) with GitHub; stores a per-user key",
    )
    login_cmd.add_argument("--no-browser", action="store_true", help="Print the sign-in URL instead of opening it")
    login_cmd.add_argument(
        "--email", default=None, metavar="ADDRESS", help="Sign in with a code emailed to this address instead of GitHub"
    )
    login_cmd.add_argument(
        "--to-email", action="store_true", help="With --email: have the key emailed to you instead of stored here"
    )
    login_cmd.add_argument(
        "--token",
        nargs="?",
        const="",
        default=None,
        metavar="KEY",
        help="Store a key you already have (from the website or your email); prompts when KEY is omitted",
    )
    subparsers.add_parser("logout", help="Forget the hosted LLM key")
    inter = subparsers.add_parser(
        "interactive",
        help="Add, inspect or remove the interactive layer (polyglot REPL, agent, TUI, exec) that the published package leaves out",
    )
    inter_sub = inter.add_subparsers(dest="interactive_cmd")
    inter_install = inter_sub.add_parser(
        "install",
        help="Fetch the five modules for this version from the tagged source on GitHub, verify them against the bundled manifest, and enable them for this user",
    )
    inter_install.add_argument(
        "--ref", help="git ref to fetch (default: the tag of the installed version, or main for a dev build)"
    )
    inter_install.add_argument(
        "--from", dest="source", metavar="DIR", help="a local src/morie directory instead of GitHub (offline)"
    )
    inter_install.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the SHA-256 check (needed for a --ref that is not this version's tag)",
    )
    inter_sub.add_parser("status", help="Show where the layer lives and whether it is active for this version")
    inter_sub.add_parser("remove", help="Delete the per-user copy of the layer")
    subparsers.add_parser("models", help="List the models you can ask: the hosted tier for your key, then local Ollama")

    doctor_cmd = subparsers.add_parser(
        "doctor",
        help="Run MORIE environment diagnostics",
    )
    doctor_cmd.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to remediate failed checks (install missing deps, etc.)",
    )

    # ── update ───────────────────────────────────────────────────────────
    em_p = subparsers.add_parser(
        "emissions", help="Measure this machine's compute emissions (CodeCarbon method) and seal a capsule"
    )
    em_p.add_argument("--seconds", type=float, default=3.0, help="How long to load the CPU (default 3)")
    em_p.add_argument("--output-dir", default="emissions", help="Where emissions.csv and the capsule go")
    em_p.add_argument("--country", default="", help="ISO-3 country code for the grid's carbon intensity")
    em_p.add_argument("--no-capsule", action="store_true", help="Skip the manifest and signature")
    update_cmd = subparsers.add_parser(
        "update",
        help="Check PyPI for a newer morie release and optionally install it",
    )
    update_cmd.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Install the update without prompting",
    )

    # ── profile-dataset ──────────────────────────────────────────────────
    profile_cmd = subparsers.add_parser(
        "profile-dataset",
        help="Profile a dataset: infer variable types, roles, and suggest analyses",
    )
    profile_cmd.add_argument("path", nargs="?", default=None, help="Dataset file (CSV, TSV, Excel, Parquet, JSON)")
    profile_cmd.add_argument("--csv", default=None, help="Same as the positional path (kept for scripts)")
    profile_cmd.add_argument("--treatment", default=None, help="Hint: column to use as treatment")
    profile_cmd.add_argument("--outcome", default=None, help="Hint: column to use as outcome")
    profile_cmd.add_argument("--weights", default=None, help="Hint: column to use as survey weight")
    profile_cmd.add_argument("--suggest", action="store_true", help="Also print a suggested analysis plan")

    # ── sample ───────────────────────────────────────────────────────────
    sample_cmd = subparsers.add_parser(
        "sample",
        help="Draw a sample from a dataset (SRS, stratified, cluster, PPS)",
    )
    sample_cmd.add_argument("path", nargs="?", default=None, help="Input dataset file")
    sample_cmd.add_argument("--csv", default=None, help="Same as the positional path (kept for scripts)")
    sample_cmd.add_argument(
        "--method",
        choices=["srs", "stratified", "cluster", "pps"],
        default="srs",
        help="Sampling method (default: srs)",
    )
    sample_cmd.add_argument("--n", type=int, required=True, help="Sample size (or total for proportional stratified)")
    sample_cmd.add_argument("--strata-col", default=None, help="Column for stratified sampling")
    sample_cmd.add_argument("--cluster-col", default=None, help="Column for cluster sampling")
    sample_cmd.add_argument("--size-col", default=None, help="Column for PPS sampling")
    sample_cmd.add_argument("--proportional", action="store_true", help="Use proportional allocation (stratified only)")
    sample_cmd.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    sample_cmd.add_argument("--output", default=None, help="Output CSV path for the sample")

    # ── inspect ──────────────────────────────────────────────────────────
    inspect_cmd = subparsers.add_parser(
        "inspect",
        help="Browse and summarise output CSVs (schema, rows, stats)",
    )
    inspect_cmd.add_argument("path", help="Path to a CSV file or output directory")
    inspect_cmd.add_argument(
        "--module",
        default=None,
        help="Scope inspection to expected outputs of this module",
    )

    # ── verify ───────────────────────────────────────────────────────────
    verify_cmd = subparsers.add_parser(
        "verify",
        help="Validate statistical outputs for correctness",
    )
    verify_cmd.add_argument("path", help="Path to a CSV file or output directory")
    verify_cmd.add_argument(
        "--module",
        default=None,
        help="Scope verification to expected outputs of this module",
    )

    # ── tui ──────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "tui",
        help="Launch the full-screen terminal IDE (requires textual)",
    )

    # ── selftest ─────────────────────────────────────────────────────────
    subparsers.add_parser(
        "selftest",
        help="Run a quick smoke test of all MORIE subsystems",
    )

    # ── percysuits ──────────────────────────────────────────────────────
    percy_cmd = subparsers.add_parser(
        "percysuits",
        help="Pull all Perseus LLM models (auto-detects Ollama)",
    )
    percy_cmd.add_argument(
        "--host",
        default=None,
        help="Ollama host URL (default: OLLAMA_HOST or localhost)",
    )
    percy_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Show model status without pulling",
    )
    percy_cmd.add_argument(
        "--ssh",
        default=None,
        help="Pull via SSH (e.g. --ssh user@host)",
    )

    # ── exec ────────────────────────────────────────────────────────────
    exec_cmd = subparsers.add_parser(
        "exec",
        help="Execute code inline, from stdin, or create-and-open (co) a file",
    )
    exec_cmd.add_argument("code", nargs="?", default=None, help="Code string, or 'co' for create-and-open mode")
    exec_cmd.add_argument("filename", nargs="?", default=None, help="Filename for 'co' mode (e.g. test.py)")
    exec_cmd.add_argument("--lang", choices=["python", "r"], default="python", help="Language (default: python)")
    exec_cmd.add_argument("--file", default=None, dest="exec_file", help="Read code from file instead")
    exec_cmd.add_argument("--dir", default=None, dest="co_dir", help="Directory for co files (default: ./cofs)")

    # ── edit ────────────────────────────────────────────────────────────
    edit_cmd = subparsers.add_parser(
        "edit",
        help="Open a file in the built-in scientific editor (Textual)",
    )
    edit_cmd.add_argument("file", help="File to edit (creates if not found)")
    edit_cmd.add_argument("--run", action="store_true", help="Run the file after saving (ctrl+r)")
    edit_cmd.add_argument("--lang", default=None, help="Language hint (python/r/shell)")

    # ── repl ────────────────────────────────────────────────────────────
    repl_cmd = subparsers.add_parser(
        "repl",
        help="Launch headless polyglot REPL (36 languages)",
    )
    repl_cmd.add_argument("--lang", default="python", help="Default language (default: python)")
    repl_cmd.add_argument(
        "--polyglot",
        action="store_true",
        default=True,
        help="Enable cross-language variable bridging (default: on)",
    )
    repl_cmd.add_argument("--no-polyglot", action="store_true", help="Disable cross-language variable bridging")
    repl_cmd.add_argument("--no-detect", action="store_true", help="Disable auto-detection (use --lang only)")

    # ── convert-checkpoint ───────────────────────────────────────────────
    convert_cmd = subparsers.add_parser(
        "convert-checkpoint",
        help="Convert a .pt training checkpoint to GGUF (optionally TurboQuant-compressed)",
    )
    convert_cmd.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    convert_cmd.add_argument("--output", required=True, help="Output .gguf file path")
    convert_cmd.add_argument("--tokenizer-dir", default=None, help="Autoresearch tokenizer directory")
    convert_cmd.add_argument(
        "--turbo-bits",
        type=int,
        default=0,
        choices=[0, 2, 3, 4],
        help="TurboQuant compression bits (0 = no compression, default: 0)",
    )

    # ── crypto ──────────────────────────────────────────────────────────
    crypto_cmd = subparsers.add_parser(
        "crypto",
        help="Post-quantum cryptography utilities (ML-KEM-768 + ChaCha20)",
    )
    crypto_sub = crypto_cmd.add_subparsers(dest="crypto_command")

    crypto_keygen = crypto_sub.add_parser("keygen", help="Generate an ML-KEM-768 key pair")
    crypto_keygen.add_argument("--name", required=True, help="Key pair identifier")
    crypto_keygen.add_argument("--output", default=None, help="Output directory for key files (default: keystore)")

    crypto_encrypt = crypto_sub.add_parser("encrypt", help="Hybrid-encrypt a file")
    crypto_encrypt.add_argument("file", help="File to encrypt")
    crypto_encrypt.add_argument("--to", required=True, dest="recipient", help="Recipient key name or public key file")

    crypto_decrypt = crypto_sub.add_parser("decrypt", help="Decrypt a hybrid-encrypted file")
    crypto_decrypt.add_argument("file", help="File to decrypt")
    crypto_decrypt.add_argument("--key", required=True, help="Key name in keystore")

    # ── list-datasets ────────────────────────────────────────────────────
    subparsers.add_parser(
        "list-datasets",
        help="List all built-in datasets and their cache status",
    )

    # ── download-bootstrap ───────────────────────────────────────────────
    bootstrap_cmd = subparsers.add_parser(
        "download-bootstrap",
        help="Download large bootstrap weight files from CKAN API",
    )
    bootstrap_cmd.add_argument(
        "--survey",
        choices=["csads_2021", "csads_2023", "csus_2019", "csus_2023", "all"],
        default="all",
        help="Which bootstrap file to download (default: all)",
    )
    bootstrap_cmd.add_argument(
        "--limit",
        type=int,
        default=32000,
        help="Max records to fetch from CKAN (default: 32000)",
    )

    # ── verify-pollution (W6 pollution-health CLI) ─────────────────────
    try:
        from morie.verify_pollution import register_subparser as _vp_register

        _vp_register(subparsers)
    except ImportError:
        pass

    # ── verify-earth-engine (EE auth smoke check) ──────────────────────
    try:
        from morie.verify_earth_engine import register_subparser as _vee_register

        _vee_register(subparsers)
    except ImportError:
        pass

    # ── provider: attach your own OpenAI-compatible endpoint ───────────
    prov = subparsers.add_parser(
        "provider",
        help="Attach your own model endpoint (OpenAI-compatible: OpenAI, Anthropic, "
        "OpenRouter, LM Studio, vLLM, ...): set | show | unset",
    )
    prov_sub = prov.add_subparsers(dest="provider_cmd", required=True)
    prov_set = prov_sub.add_parser("set", help="store the endpoint, key and optional model")
    prov_set.add_argument(
        "--base-url", required=True, help="e.g. https://api.openai.com/v1 or https://api.anthropic.com/v1"
    )
    prov_set.add_argument("--key", required=True, help="the API key for that endpoint")
    prov_set.add_argument("--model", default=None, help="model name to ask by default (optional)")
    prov_sub.add_parser("show", help="what is attached")
    prov_sub.add_parser("unset", help="detach it")

    # ── bricklayer: offer to install the rest of the morie family ──────
    try:
        from morie.bricklayer import register_subparser as _bl_register

        _bl_register(subparsers)
    except ImportError:
        pass

    # ── ingest: pull open-data feeds (CKAN, TPS ArcGIS, SIU PDFs, A2AJ) ─
    ingest_cmd = subparsers.add_parser(
        "ingest",
        help="Pull open-data feeds (CKAN portals, TPS ArcGIS layers, SIU PDFs, A2AJ legal data)",
    )
    ingest_cmd.add_argument(
        "portal",
        choices=["ckan", "tps", "siu", "a2aj"],
        help="Which portal adapter to invoke",
    )
    ingest_cmd.add_argument(
        "portal_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the adapter (use `--help` to see them)",
    )

    # ── tutorial: interactive walkthrough for fresh users ─────────────
    subparsers.add_parser(
        "tutorial",
        help="Interactive first-time-user walkthrough (no Python knowledge required)",
    )

    # ── cheatsheet: one-page command reference ────────────────────────
    subparsers.add_parser(
        "cheatsheet",
        help="Print a one-page reference of the most-used morie commands",
    )

    # ── explain: human-readable description of an output CSV ──────────
    explain_cmd = subparsers.add_parser(
        "explain",
        help="Describe what a module-output CSV contains and how to read it",
    )
    explain_cmd.add_argument("filename", help="Output CSV filename (e.g. power_summary.csv)")

    # ── generate-template: first-paper methods scaffold ───────────────
    template_cmd = subparsers.add_parser(
        "generate-template",
        help="Write a methods+results scaffold for your first paper",
    )
    template_cmd.add_argument(
        "--module", default="power-design", help="Module name to scaffold methods for (default: power-design)"
    )
    template_cmd.add_argument(
        "--out", type=Path, default=Path("first-paper.md"), help="Output markdown path (default: first-paper.md)"
    )

    # ── pull: one-line CLI shortcuts to named morie.datasets loaders ───
    # This is the non-coder entry point.  Users never have to write
    # `python -c "import morie.datasets ..."` — they say
    #   morie pull tps-major --year 2024 --out file.csv
    # and a DataFrame lands on disk.
    pull_cmd = subparsers.add_parser(
        "pull",
        help="One-line dataset shortcut (CLI for morie.datasets); writes CSV to disk",
    )
    pull_cmd.add_argument(
        "dataset",
        nargs="?",
        help="A catalog key (morie list-datasets, e.g. ocp21) or a shortcut: tps-major, tps-shootings, "
        "tps-homicide, tps-layers, tps-major-toy, cpads, otis-a01-toy, siu-toy, siu-index",
    )
    pull_cmd.add_argument("--all", action="store_true", help="Download every catalog dataset into --out (a directory)")
    pull_cmd.add_argument("--year", type=int, help="Filter to a single year (TPS only)")
    pull_cmd.add_argument("--max", type=int, dest="max_features", help="Cap rows fetched (TPS only)")
    pull_cmd.add_argument("--out", type=Path, help="Output CSV path (stdout if omitted)")

    return parser


def _load_dotenv_if_present() -> None:
    """Auto-load a nearby .env file into os.environ so subcommands pick up
    GOOGLE_CLOUD_PROJECT, MORIE_EE_KEY_PATH, etc. without requiring the
    user to `source` before calling `morie`. Zero-dep parser -- only simple
    KEY=VALUE lines, ignores comments and blanks, won't overwrite
    already-exported variables (shell wins).

    Tests that need a deterministic "no credentials" baseline set
    MORIE_SKIP_DOTENV=1 so this loader becomes a no-op -- otherwise
    a .env with valid keys would defeat the purge."""
    import os

    if os.environ.get("MORIE_SKIP_DOTENV") == "1":
        return
    from pathlib import Path

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[5] / ".env",  # dev/sphinx/project/.env
    ]
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            for raw in resolved.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


def _friendly_error(exc: BaseException) -> str | None:
    """Translate the most common cryptic morie errors into actionable hints.

    Returns the user-facing message, or None to let the raw traceback show.
    Centralised here so fresh users never see a bare Python stack on the
    most-hit failure modes.
    """

    msg = str(exc)
    cls = type(exc).__name__

    # PEP 668 — externally-managed-environment
    if "externally-managed-environment" in msg:
        return (
            "Your system Python forbids `pip install` outside a virtualenv "
            "(this is PEP 668, enforced on modern Debian / Ubuntu / Pi). "
            "morie isn't broken; your system pip is locked.\n\n"
            "Fix: install morie into a managed venv using the one-liner:\n"
            "    curl -fsSL https://rootcoder007.github.io/morie/install.sh | bash\n"
        )

    # Missing CPADS data (pre-v0.5.0 — we ship synthetic now, but a stale
    # install might still hit this)
    if isinstance(exc, FileNotFoundError) and "cpads" in msg.lower():
        return (
            "Could not find a CPADS CSV at the expected project path.\n\n"
            "If you have the real Statistics Canada PUMF, pass it:\n"
            "    morie run-module power-design --cpads-csv /path/to/cpads.csv\n\n"
            "Otherwise upgrade to morie v0.5.0+, which ships a synthetic\n"
            "fallback frame so first-run analyses work without a download:\n"
            "    pip install -U morie\n"
        )

    # Missing optional dep (xgboost / lxml / textual / openai)
    if isinstance(exc, ImportError) and "No module named" in msg:
        missing = msg.split("'")[1] if "'" in msg else "<unknown>"
        return (
            f"morie tried to use {missing!r} but it isn't installed.\n\n"
            f"This is usually an optional dependency.  Try:\n"
            f"    pip install {missing}\n\n"
            f"Or reinstall morie with all the extras:\n"
            f"    pip install 'morie[interactive,ml]'\n"
        )

    # Module name typo
    if "MODULE_SPECS" in msg or ("module" in msg.lower() and "unknown" in msg.lower()):
        return (
            "That module name didn't match any registered module.\n\n"
            "List the available ones with:\n"
            "    morie list-modules\n"
        )

    # Connection / TLS / network failures
    if cls in ("ConnectError", "ReadTimeout", "ConnectTimeout") or "Connection reset" in msg:
        return (
            f"Network call failed: {msg}\n\n"
            "Common causes:\n"
            "  - The portal is rate-limiting or blocking automated clients\n"
            "    (some Cloudflare-protected sites do this).\n"
            "  - You're offline or behind a captive Wi-Fi portal.\n"
            "  - The endpoint URL changed (governments reorganise sites).\n\n"
            "If you can curl the URL from the same machine, the problem is\n"
            "in our client; please file an issue at\n"
            "https://github.com/rootcoder007/morie/issues.\n"
        )

    return None


def _drain_stream(stream) -> int:
    """Write a streamed answer to stdout and return how many chunks came.

    The count is the only evidence that a backend answered: a selected
    backend that prints an error and stops yields nothing, and the
    payload's ``mode`` was fixed before any bytes arrived.
    """
    n = 0
    for chunk in stream:
        sys.stdout.write(chunk)
        sys.stdout.flush()
        n += 1
    sys.stdout.write("\n")
    return n


def _llm_exit_code(payload, chunks=None) -> int:
    """0 when an LLM answered; 1 when the static local fallback did or
    the selected backend produced no output at all.

    `morie agent`/`morie ask` returned 0 unconditionally, so a scripted
    call could not tell an answer from "no backend reachable".
    """
    if isinstance(payload, dict) and payload.get("mode") == "local_fallback":
        sys.stdout.flush()
        print("\nno LLM backend was reachable; this is the local fallback text", file=sys.stderr)
        return 1
    if isinstance(payload, dict) and payload.get("failed"):
        print("the LLM backend request failed", file=sys.stderr)
        return 1
    if chunks is not None and chunks == 0:
        print("the LLM backend produced no output", file=sys.stderr)
        return 1
    if (
        chunks is None
        and isinstance(payload, dict)
        and "output_stream" not in payload
        and not str(payload.get("output_text") or "").strip()
    ):
        print("the LLM backend produced no output", file=sys.stderr)
        return 1
    return 0


def _local_models_line() -> str:
    """One line on the local Ollama server for `morie models`."""
    try:
        from .loc import LocalOllama

        client = LocalOllama()
        if not client.is_running():
            return f"Local Ollama: not reachable at {client.base_url}"
        names = [m.name for m in client.list_models()]
        return f"Local Ollama ({client.base_url}): " + (", ".join(names) if names else "running, no models pulled")
    except Exception:  # noqa: BLE001 - a missing or broken local server is not an error here
        return "Local Ollama: not reachable"


def _add_interactive_layer(what: str) -> bool:
    """A verb needs a module the published package leaves out: say how to add it, and on a terminal offer to do it now.

    Returns True when the layer was just installed and joined to the package path, so
    the caller can import again; False when the person should run the install first.
    Never reads stdin unless both ends are a terminal (CI and pipes get the message only).
    """
    import os

    from . import _interactive

    print(f"{what} is not bundled in this install. Run `morie interactive install` to add it for this user.")
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not os.environ.get("MORIE_NO_PROMPT", "").strip()
    if not interactive:
        return False
    try:
        answer = input("Add it now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if answer not in ("y", "yes"):
        return False
    if _interactive.install() != 0:
        return False
    import morie

    _interactive.activate(morie.__path__, morie.__version__)
    return True


def main() -> int:
    """
    Entry point for the MORIE command line interface.
    """
    try:
        return _main_impl()
    except KeyboardInterrupt:
        print("\ninterrupted (Ctrl-C)", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except PermissionError as exc:
        print(f"morie: cannot write {exc.filename or 'here'}: permission denied", file=sys.stderr)
        return 1
    except BaseException as exc:  # noqa: BLE001
        hint = _friendly_error(exc)
        if hint is not None:
            print(hint, file=sys.stderr)
            # Show class+message at the bottom for debuggers, but spare the traceback
            print(f"  ({type(exc).__name__}: {exc})", file=sys.stderr)
            return 1
        # Unknown errors — re-raise so the traceback shows.
        raise


def _main_impl() -> int:
    _load_dotenv_if_present()

    if len(sys.argv) > 1 and sys.argv[1].startswith("?"):
        question = sys.argv[1][1:].strip()
        if not question and len(sys.argv) > 2:
            question = " ".join(sys.argv[2:])
        elif not question:
            print("Usage: morie ?<question>")
            return 1
        payload = ask_percy(question, stream=sys.stdout.isatty())
        if "output_stream" in payload:
            return _llm_exit_code(payload, _drain_stream(payload["output_stream"]))
        print(payload.get("output_text", ""))
        return _llm_exit_code(payload)

    parser = build_parser()
    args = parser.parse_args()

    # Auto-detect: no subcommand + interactive TTY -> launch TUI or chat REPL.
    if args.command is None:
        if sys.stdout.isatty() and sys.stdin.isatty():
            try:
                from .tui import _TEXTUAL_AVAILABLE, launch_tui

                if _TEXTUAL_AVAILABLE:
                    return launch_tui()
            except ImportError:
                pass
            # Textual not available -- try chat REPL.
            from .chat import run_chat_repl

            return run_chat_repl()
        parser.print_help()
        return 0

    if args.command == "pipeline":
        if not args.all and not args.modules:
            parser.print_help()
            return 0
        # Use rich progress display when running in an interactive terminal.
        if sys.stdout.isatty():
            from .progress import execute_pipeline_with_progress

            return execute_pipeline_with_progress(
                modules=args.modules,
                cpads_csv=args.cpads_csv,
                dataset_key=getattr(args, "dataset", None),
                output_dir=args.output_dir,
                silent=args.yes,
                track_carbon=not args.no_carbon,
            )
        return execute_pipeline(
            modules=args.modules,
            cpads_csv=args.cpads_csv,
            dataset_key=getattr(args, "dataset", None),
            output_dir=args.output_dir,
            silent=args.yes,
            track_carbon=not args.no_carbon,
        )

    if args.command == "parity-review":
        from ._parity import build_parity_matrix, summarize_parity_matrix, write_parity_matrix

        root = Path(args.epiml_root)
        if not root.is_dir():
            print(f"parity-review: {root} is not a directory", file=sys.stderr)
            return 2
        try:
            matrix = build_parity_matrix(args.epiml_root)
        except Exception as exc:  # noqa: BLE001 - an empty or foreign tree, reported in words
            print(
                f"parity-review: {root} does not look like the reference checkout ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            return 2
        if len(matrix) == 0:
            print(f"parity-review: no reference modules found under {root}", file=sys.stderr)
            return 2
        summary = summarize_parity_matrix(matrix)
        if args.output:
            write_parity_matrix(matrix, args.output)
            print(f"Wrote audit matrix to {args.output}")
        print(
            "Audit summary:",
            f"total={summary.total_rows}",
            f"present={summary.already_present}",
            f"scaffolded={summary.scaffolded_incomplete}",
            f"missing={summary.missing}",
            f"deferred={summary.deferred}",
        )
        return 0

    if args.command == "list-modules":
        for spec in list_modules():
            print(spec["name"])
            print(f"  {spec['description']}")
            print(f"  outputs: {', '.join(spec['output_files'])}")
        return 0

    if args.command == "run-module":
        out_dir = Path(args.output_dir) if args.output_dir else Path("morie-output") / args.module
        outputs = run_module(
            args.module,
            cpads_csv=args.cpads_csv,
            dataset_key=getattr(args, "dataset", None),
            output_dir=out_dir,
        )
        print(f"Completed module: {args.module}")
        print("Generated tables:", ", ".join(outputs.keys()))
        n_files = len(list(out_dir.iterdir())) if out_dir.is_dir() else 0
        print(f"Written to {out_dir} ({n_files} files)")
        return 0

    if args.command == "run-modules":
        selected = args.modules or [item["name"] for item in list_modules()]
        ds_key = getattr(args, "dataset", None)
        results = {
            module_name: run_module(
                module_name,
                cpads_csv=args.cpads_csv,
                dataset_key=ds_key,
                output_dir=args.output_dir,
            )
            for module_name in selected
        }
        print("Completed modules:", ", ".join(results.keys()))
        return 0

    if args.command in ("percy", "perseus"):
        return _handle_percy(args)

    if args.command == "serve":
        from .perseus_relay import serve

        try:
            serve(port=args.port, token=args.token, bind=args.bind)
        except OSError as exc:  # port taken, bind address refused
            print(f"serve: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command in ("agent", "assistant"):
        use_stream = not getattr(args, "no_stream", False)
        try:
            from .agent import create_agent

            agent = create_agent()
            answered = True
            if use_stream:
                answered = _drain_stream(agent.chat_stream(args.question)) > 0
            else:
                resp = agent.chat(args.question)
                answered = bool(str(resp.text or "").strip()) and not getattr(resp, "failed", False)
                if answered:  # a failed backend's text ("LLM request failed ...") is not an answer
                    print(resp.text)
                    if resp.tool_calls_made:
                        print(f"\n[{len(resp.tool_calls_made)} tool calls in {resp.iterations} iterations]")
            agent.close()
            if not answered:
                print("the local agent produced no output; answering through the provider chain", file=sys.stderr)
                payload = ask_percy(args.question, context=getattr(args, "context", None), stream=use_stream)
                if use_stream:
                    return _llm_exit_code(payload, _drain_stream(payload["output_stream"]))
                print(payload["output_text"])
                return _llm_exit_code(payload)
        except Exception:
            # the agent needs an LLM backend; say why it stepped aside
            # rather than hiding a real bug in create_agent()
            print("no local Ollama for the tool-calling agent; answering through the provider chain", file=sys.stderr)
            payload = ask_percy(
                args.question,
                context=getattr(args, "context", None),
                stream=use_stream,
            )
            if use_stream:
                return _llm_exit_code(payload, _drain_stream(payload["output_stream"]))
            print(payload["output_text"])
            return _llm_exit_code(payload)
        return 0

    if args.command == "ask":
        use_stream = not getattr(args, "no_stream", False)
        payload = ask_percy(
            args.question,
            context=args.context,
            model=getattr(args, "model", None),
            stream=use_stream,
        )
        if use_stream:
            return _llm_exit_code(payload, _drain_stream(payload["output_stream"]))
        print(payload["output_text"])
        return _llm_exit_code(payload)

    if args.command == "chat":
        from .chat import run_chat_repl

        return run_chat_repl(agent=args.agent)

    if args.command == "tui":
        try:
            from .tui import launch_tui
        except ImportError:
            if not _add_interactive_layer('The TUI (also: pip install "morie[interactive]")'):
                return 1
            try:
                from .tui import launch_tui
            except ImportError:
                print('The TUI also needs the interactive extra: pip install "morie[interactive]"')
                return 1
        return launch_tui()

    if args.command == "selftest":
        from .selftest import run_selftest

        return run_selftest()

    if args.command == "percysuits":
        return _handle_percysuits(args)

    if args.command == "exec":
        return _handle_exec(args)

    if args.command == "edit":
        return _handle_edit(args)

    if args.command == "repl":
        return _handle_repl(args)

    if args.command == "list-datasets":
        from .data import list_datasets

        datasets = list_datasets()
        kw = max(20, max((len(d["key"]) for d in datasets), default=20))
        print(f"{'Key':<{kw}} {'Type':<12} {'Rows':>10}  {'Route'}")
        print("-" * (kw + 76))
        for d in datasets:
            known = d["cached"] or (d["type"] == "hosted" and d.get("rows"))  # hosted rows come from the manifest
            status = f"{d['rows']:,}" if known else "not cached"
            print(f"{d['key']:<{kw}} {d['type']:<12} {status:>10}  {d['route']}")
        n_own = sum(d["route"].startswith("own file") for d in datasets)
        print("-" * (kw + 76))
        print(
            f"{len(datasets)} keys: {len(datasets) - n_own} download from their portal or rmoriedata on first use; "
            f"{n_own} are files you place under $MORIE_DATA_DIR/datasets/ (restricted or your own data)."
        )
        n_hub = sum(d["type"] == "hosted" for d in datasets)
        if n_hub:
            print(
                f"{n_hub} curated tables at data.rmorie.com (db/table keys), opened by your MORIE key: morie pull KEY"
            )
        else:
            from .datahub import hosted_key

            if hosted_key():
                print("data.rmorie.com was not reachable just now; its tables appear once a fetch succeeds.")
            else:
                from .datahub import hosted_key

            if hosted_key():
                print("data.rmorie.com was not reachable just now; its tables appear once a fetch succeeds.")
            else:
                print("Curated tables at data.rmorie.com appear here after `morie login` (they need the MORIE key).")
        return 0

    if args.command == "verify-pollution":
        from .verify_pollution import handle_verify_pollution

        return handle_verify_pollution(args)

    if args.command == "verify-earth-engine":
        from .verify_earth_engine import handle_verify_earth_engine

        return handle_verify_earth_engine(args)

    if args.command in ("bricklayer", "r-install"):
        from .bricklayer import run as _bl_run

        return _bl_run(args)

    if args.command == "tutorial":
        from .tutorial import run as _run_tutorial

        return _run_tutorial()

    if args.command == "cheatsheet":
        from .explain import print_cheatsheet

        print_cheatsheet()
        return 0

    if args.command == "explain":
        from .explain import describe

        print(describe(args.filename))
        return 0

    if args.command == "generate-template":
        # Copy the bundled first-paper template (shipped alongside the
        # source repo at templates/first-paper.md, mirrored into the
        # wheel as morie/data/first-paper.md).
        from importlib.resources import as_file, files

        try:
            with as_file(files("morie").joinpath("data/first-paper.md")) as src:
                content = src.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Fallback: try the project-tree path (dev install)
            src = Path(__file__).resolve().parents[2] / "templates" / "first-paper.md"
            content = src.read_text(encoding="utf-8")
        # Light placeholder substitution
        content = content.replace("[MODULE_NAME]", args.module)
        args.out = Path(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
        print(f"wrote {args.out}  ({len(content):,} chars)", file=sys.stderr)
        return 0

    if args.command == "pull":
        # One-line dataset shortcuts — the non-coder entry point.
        # Resolves a name like "tps-major" into the matching morie.datasets
        # function, or any catalog key into load_dataset(), and writes the
        # DataFrame to disk (or stdout). Downloads are cached, so a module
        # run after `morie pull ocp21` uses the real PUMF.
        import morie.datasets as md

        from .data import DATASET_CATALOG, _fuzzy_match_key, load_dataset
        from .datahub import is_hosted_key

        if args.all:
            out_dir = Path(args.out or "datasets")
            out_dir.mkdir(parents=True, exist_ok=True)
            failed = 0
            for key in sorted(DATASET_CATALOG):
                try:
                    frame = load_dataset(key)
                except MemoryError:
                    failed += 1
                    print(
                        f"  {key:<12} FAILED: out of memory (this table needs more RAM than this machine has)",
                        file=sys.stderr,
                    )
                    continue
                except Exception as exc:  # noqa: BLE001 - report and continue
                    failed += 1
                    print(f"  {key:<12} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                dest = out_dir / f"{key}.csv"
                frame.to_csv(dest, index=False)
                print(f"  {key:<12} {len(frame):,} rows -> {dest}")
            return 1 if failed else 0
        if not args.dataset:
            print(
                "usage: morie pull KEY [--out FILE.csv] | morie pull --all [--out DIR]   (keys: morie list-datasets)",
                file=sys.stderr,
            )
            return 2

        try:
            if args.dataset == "tps-major":
                df = md.tps_major_crime(year=args.year, max_features=args.max_features)
            elif args.dataset == "tps-major-toy":
                df = md.tps_major_crime(year=args.year, max_features=args.max_features, offline=True)
            elif args.dataset == "tps-shootings":
                df = md.tps_shootings(year=args.year, max_features=args.max_features)
            elif args.dataset == "tps-homicide":
                df = md.tps_homicide(year=args.year, max_features=args.max_features)
            elif args.dataset == "tps-layers":
                df = md.tps_layers()
            elif args.dataset == "cpads":
                df = md.cpads()
            elif args.dataset == "otis-a01-toy":
                df = md.otis_a01()
            elif args.dataset == "siu-toy":
                # Single-row "DataFrame" carrying the synthetic report text
                # for parity with the other pull verbs.
                from morie.fn import _frame_core as _pd

                df = _pd.DataFrame([{"report_id": "24-OFD-001", "text": md.siu_report_text(offline=True)}])
            elif (
                args.dataset in DATASET_CATALOG
                or _fuzzy_match_key(args.dataset) is not None
                or is_hosted_key(args.dataset)
            ):
                df = load_dataset(args.dataset)
            elif args.dataset == "siu-index":
                df = md.siu_director_reports()
            else:
                import difflib

                near = difflib.get_close_matches(args.dataset, list(DATASET_CATALOG), n=5, cutoff=0.5)
                near += [k for k in DATASET_CATALOG if k.startswith(args.dataset[:3]) and k not in near][:5]
                hint = f"; did you mean: {', '.join(near)}" if near else " (morie list-datasets shows every key)"
                print(f"unknown dataset {args.dataset!r}{hint}", file=sys.stderr)
                return 2
        except Exception as exc:  # noqa: BLE001 — surface origin-API errors plainly
            print(f"pull failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if len(df) == 0:
            print(
                f"pull {args.dataset}: the source returned no rows (the portal may have changed; nothing written)",
                file=sys.stderr,
            )
            return 1
        if args.out:
            args.out = Path(args.out)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, index=False)
            print(f"wrote {args.out}  ({len(df):,} rows, {len(df.columns)} cols)", file=sys.stderr)
        else:
            sys.stdout.write(df.to_csv(index=False))
        return 0

    if args.command == "ingest":
        # Delegate to the chosen portal sub-module's cli() handler.
        # Each sub-module owns its own argparse for the portal-specific flags.
        portal = args.portal
        portal_args = list(args.portal_args or [])
        # argparse.REMAINDER swallows a leading "--" separator; strip it
        if portal_args and portal_args[0] == "--":
            portal_args = portal_args[1:]
        if portal == "ckan":
            from .ingest.ckan import cli as _cli
        elif portal == "tps":
            from .ingest.tps import cli as _cli
        elif portal == "siu":
            from .ingest.siu import cli as _cli
        elif portal == "a2aj":
            from .ingest.a2aj import cli as _cli
        else:
            print(f"unknown portal {portal!r}; valid: ckan, tps, siu, a2aj")
            return 2
        return _cli(portal_args)

    if args.command == "download-bootstrap":
        from .data import DATASET_CATALOG, fetch_ckan_to_cache

        boot = {k: e for k, e in DATASET_CATALOG.items() if e.get("type") == "bootstrap"}
        if args.survey == "all":
            targets = sorted(boot)
        else:
            survey, _, year = args.survey.partition("_")
            targets = sorted(k for k, e in boot.items() if e.get("survey") == survey and year in str(e.get("year", "")))
            if not targets:
                print(f"  Unknown survey {args.survey!r}; bootstrap keys: {', '.join(sorted(boot))}")
                return 1
        for key in targets:
            entry = DATASET_CATALOG[key]
            rid = entry.get("ckan_resource_id", "")
            if not rid:
                print(f"  {key}: no CKAN resource ID -- download the CSV manually to {entry['local_path']}")
                continue
            print(f"  Downloading {key} ({entry['name']}) from CKAN (limit={args.limit})...")
            try:
                df = fetch_ckan_to_cache(key, limit=args.limit)
                if args.limit and len(df) > args.limit:
                    df = df.head(args.limit)
                print(
                    f"    OK: {len(df):,} rows cached"
                    + (f" (first {args.limit:,}; --limit)" if args.limit and len(df) == args.limit else "")
                )
            except Exception as e:
                print(f"    ERROR: {e}")
        return 0

    if args.command == "crypto":
        import getpass
        from pathlib import Path as _CryptoPath

        def _crypto_password() -> str | None:
            try:
                return getpass.getpass("Keystore password: ")
            except (EOFError, OSError):
                print(
                    "a keystore password is needed: run this in a terminal, or use file keys "
                    "(keygen --output DIR; encrypt --to DIR/NAME.moriepk; decrypt --key DIR/NAME.moriesk)",
                    file=sys.stderr,
                )
                return None

        if args.crypto_command == "keygen":
            from .crypto.hybrid import keygen as _hybrid_keygen
            from .crypto.keystore import create_keystore as _ks_create
            from .crypto.keystore import store_keypair as _ks_store

            if args.output:
                pk, sk = _hybrid_keygen()
                out_dir = _CryptoPath(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)
                pk_path = out_dir / f"{args.name}.moriepk"
                sk_path = out_dir / f"{args.name}.moriesk"
                pk_path.write_bytes(pk)
                sk_path.write_bytes(sk)
                with contextlib.suppress(OSError):
                    sk_path.chmod(0o600)  # a secret key is owner-only
                print(f"Public key:  {pk_path}")
                print(f"Secret key:  {sk_path}")
            else:
                pw = _crypto_password()
                if pw is None:
                    return 1
                ks_path = str(_CryptoPath("~/.morie/keys/keystore.json").expanduser())
                if not _CryptoPath(ks_path).exists():
                    _ks_create(pw, path=ks_path)
                    print(f"Created keystore at {ks_path}")
                pk, sk = _hybrid_keygen()
                _ks_store(args.name, pk, sk, pw, path=ks_path)
                print(f"Key pair '{args.name}' stored in keystore")
            return 0

        if args.crypto_command == "encrypt":
            from .crypto.hybrid import hybrid_encrypt as _h_enc

            src = _CryptoPath(args.file)
            if not src.exists():
                print(f"File not found: {src}")
                return 1
            pk_path = _CryptoPath(args.recipient)
            if pk_path.exists():
                pk = pk_path.read_bytes()
            else:
                from .crypto.keystore import load_keypair as _ks_load

                pw = _crypto_password()
                if pw is None:
                    return 1
                try:
                    pk, _ = _ks_load(args.recipient, pw)
                except ValueError:
                    print("wrong keystore password, or the keystore is corrupted", file=sys.stderr)
                    return 1
            ct = _h_enc(src.read_bytes(), pk)
            out = src.with_suffix(src.suffix + ".morieenc")
            out.write_bytes(ct)
            print(f"Encrypted: {out}")
            return 0

        if args.crypto_command == "decrypt":
            from .crypto.hybrid import hybrid_decrypt as _h_dec
            from .crypto.keystore import load_keypair as _ks_load2

            src = _CryptoPath(args.file)
            if not src.exists():
                print(f"File not found: {src}")
                return 1
            key_file = _CryptoPath(args.key)
            if key_file.is_file():  # a secret key written by `keygen --output DIR`
                sk = key_file.read_bytes()
            else:
                pw = _crypto_password()
                if pw is None:
                    return 1
                try:
                    _, sk = _ks_load2(args.key, pw)
                except ValueError:
                    print("wrong keystore password, or the keystore is corrupted", file=sys.stderr)
                    return 1
            try:
                plaintext = _h_dec(src.read_bytes(), sk)
            except (ValueError, KeyError) as exc:
                print(f"decrypt failed: {exc} (wrong key, or the file is not a morie ciphertext)", file=sys.stderr)
                return 1
            out_name = str(src).replace(".morieenc", "")
            if out_name == str(src):
                out_name = str(src) + ".dec"
            _CryptoPath(out_name).write_bytes(plaintext)
            print(f"Decrypted: {out_name}")
            return 0

        crypto_cmd_parser = build_parser()._subparsers._group_actions[0].choices.get("crypto")
        if crypto_cmd_parser:
            crypto_cmd_parser.print_help()
        return 2  # usage error, as argparse reports for the others

    if args.command == "convert-checkpoint":
        from .pt2gguf import convert

        if not Path(args.checkpoint).is_file():
            print(f"convert-checkpoint: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 2
        try:
            convert(args.checkpoint, args.output, args.tokenizer_dir, args.turbo_bits)
        except ImportError as exc:  # torch / gguf are optional extras
            print(f"convert-checkpoint: {exc}", file=sys.stderr)
            return 1
        except RuntimeError as exc:  # the MORIE_TRUST_CHECKPOINT gate, a malformed file
            print(f"convert-checkpoint: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "login":
        from .hosted import device_login, email_login, store_token

        try:
            if getattr(args, "token", None) is not None:
                token = args.token
                if not token:
                    import getpass

                    token = getpass.getpass("Paste your MORIE key: ")
                store_token(token)
            elif getattr(args, "email", None):
                email_login(args.email, to_email=getattr(args, "to_email", False))
            else:
                device_login(open_browser=not getattr(args, "no_browser", False))
        except Exception as exc:  # network or user abandoned the flow
            print(f"login failed: {exc}")
            return 1
        return 0

    if args.command == "interactive":
        from . import _interactive

        sub = getattr(args, "interactive_cmd", None) or "status"
        if sub == "install":
            return _interactive.install(ref=args.ref, source=args.source, verify=not args.no_verify)
        if sub == "remove":
            return _interactive.remove()
        return _interactive.status()

    if args.command == "logout":
        from .hosted import logout

        logout()
        return 0

    if args.command == "models":
        from .hosted import models_lines

        for line in models_lines():
            print(line)
        from .hosted import provider_line

        own = provider_line()
        if own:
            print(own)
        print(_local_models_line())
        print("Pick one per call with `morie ask --model NAME ...`, or set MORIE_HOSTED_MODEL / MORIE_OLLAMA_MODEL.")
        from . import hosted as _hosted

        if getattr(_hosted, "_hosted_failure", None) == "rejected":
            print("the gateway rejected the stored key; run `morie login` again", file=sys.stderr)
            return 1
        return 0

    if args.command == "provider":
        from .hosted import provider_set, provider_show, provider_unset

        if args.provider_cmd == "set":
            try:
                provider_set(args.base_url, args.key, args.model)
            except ValueError as exc:
                print(f"morie provider: {exc}", file=sys.stderr)
                return 2
        elif args.provider_cmd == "show":
            provider_show()
        else:
            provider_unset()
        return 0

    if args.command == "doctor":
        from .doctor import run_doctor

        return run_doctor(fix=getattr(args, "fix", False))

    if args.command == "emissions":
        from .emissions import run_check, summary_text

        data = run_check(args.seconds, args.output_dir, capsule=not args.no_capsule, country_iso_code=args.country)
        print(summary_text(data, getattr(data, "capsule", None)))
        return 0

    if args.command == "update":
        from ._update_check import run_update

        return run_update(yes=getattr(args, "yes", False))

    if args.command in ("profile-dataset", "sample"):
        args.csv = args.csv or getattr(args, "path", None)
        if not args.csv:
            print(f"usage: morie {args.command} PATH  (or --csv PATH)", file=sys.stderr)
            return 2

    if args.command == "profile-dataset":
        from .dataset import load_dataset, profile_dataset, suggest_analysis_plan

        try:
            df = load_dataset(args.csv)
        except FileNotFoundError:
            print(f"file not found: {args.csv}", file=sys.stderr)
            return 1
        except UnicodeDecodeError:
            print(f"{args.csv} is not UTF-8 text; re-save it as UTF-8 (iconv -f latin1 -t utf-8)", file=sys.stderr)
            return 1
        except (IndexError, ValueError) as exc:
            print(f"{args.csv} has no data rows ({exc})", file=sys.stderr)
            return 1
        if len(df.columns) != len(set(df.columns)):
            print("note: duplicate column names in the header; only the first of each is kept", file=sys.stderr)
        profile = profile_dataset(
            df,
            hint_treatment=args.treatment,
            hint_outcome=args.outcome,
            hint_weights=args.weights,
        )
        print(profile.summary_table())
        if profile.suggested_treatment:
            print(f"\nSuggested treatment: {profile.suggested_treatment}")
        if profile.suggested_outcome:
            print(f"Suggested outcome:   {profile.suggested_outcome}")
        if profile.suggested_weights:
            print(f"Suggested weights:   {profile.suggested_weights}")
        if args.suggest:
            plan = suggest_analysis_plan(profile)
            print("\n--- Suggested Analysis Plan ---")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. [{step['analysis']}] {step['rationale']}")
        return 0

    if args.command == "sample":
        from .dataset import load_dataset
        from .sampling import (
            cluster_sample,
            pps_sample,
            simple_random_sample,
            stratified_sample,
        )

        df = load_dataset(args.csv)
        method = args.method

        if method == "srs":
            sample = simple_random_sample(df, args.n, seed=args.seed)
        elif method == "stratified":
            if not args.strata_col:
                print("Error: --strata-col is required for stratified sampling")
                return 1
            sample = stratified_sample(
                df,
                args.strata_col,
                args.n,
                proportional=args.proportional,
                seed=args.seed,
            )
        elif method == "cluster":
            if not args.cluster_col:
                print("Error: --cluster-col is required for cluster sampling")
                return 1
            sample = cluster_sample(df, args.cluster_col, args.n, seed=args.seed)
        elif method == "pps":
            if not args.size_col:
                print("Error: --size-col is required for PPS sampling")
                return 1
            sample = pps_sample(df, args.size_col, args.n, seed=args.seed)
        else:
            print(f"Unknown method: {method}")
            return 1

        print(f"Sampled {len(sample)} rows using {method}")
        if args.output:
            sample.to_csv(args.output, index=False)
            print(f"Wrote sample to {args.output}")
        else:
            print(sample.head(20).to_string())
        return 0

    if args.command == "inspect":
        from pathlib import Path as _Path

        from .inspector import (
            inspect_directory,
            inspect_output,
            render_inspection,
        )

        target = _Path(args.path)
        if target.is_file():
            render_inspection(inspect_output(target))
        elif target.is_dir():
            results = inspect_directory(target, module_name=args.module)
            if not results:
                print(f"No CSV files found in {target}")
                return 1
            for result in results:
                render_inspection(result)
                print()
        else:
            print(f"Path not found: {target}")
            return 1
        return 0

    if args.command == "verify":
        from pathlib import Path as _Path

        from .inspector import (
            render_verification,
            verify_directory,
            verify_statistical_output,
        )

        target = _Path(args.path)
        if target.is_file():
            report = verify_statistical_output(target)
            render_verification(report)
            return 0 if report.passed else 1
        elif target.is_dir():
            reports = verify_directory(target, module_name=args.module)
            if not reports:
                print(f"No CSV files found in {target}")
                return 1
            all_passed = True
            for report in reports:
                render_verification(report)
                print()
                if not report.passed:
                    all_passed = False
            return 0 if all_passed else 1
        else:
            print(f"Path not found: {target}")
            return 1

    parser.print_help()
    return 0


def _handle_exec(args: argparse.Namespace) -> int:
    # TRUST BOUNDARY: 'morie exec' runs code the LOCAL USER supplies on
    # their own machine -- same trust model as `python -c` / `Rscript -e`.
    # It never executes remote or network-supplied code. Set
    # MORIE_NO_EXEC=1 (CI, shared hosts) to disable it entirely.
    import os
    import subprocess
    import tempfile

    # Absent from the published wheel by design: without the guard module
    # there is no exec surface to authorise, so refuse rather than raise
    # ModuleNotFoundError at the user.
    try:
        from morie._exec_guard import ExecGuardError, ensure_exec_allowed
    except ModuleNotFoundError:
        if not _add_interactive_layer("'morie exec'"):
            return 1
        from morie._exec_guard import ExecGuardError, ensure_exec_allowed

    try:
        ensure_exec_allowed("'morie exec'")
    except ExecGuardError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.code == "co":
        return _handle_exec_co(args)

    code = args.code
    if args.exec_file:
        with open(args.exec_file) as f:
            code = f.read()
    elif code is None:
        if sys.stdin.isatty():
            print("Enter code (ctrl+d to execute):")
        code = sys.stdin.read()
    if not code or not code.strip():
        print("No code provided.")
        return 1

    if args.lang == "r":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            return subprocess.call(["Rscript", tmp])
        finally:
            os.unlink(tmp)

    ns = {"__name__": "__morie_exec__"}
    try:
        from morie.fn import _array_core as np
        from morie.fn import _frame_core as pd

        ns["np"] = np
        ns["pd"] = pd
    except ImportError:
        pass
    try:
        from morie import fn
        from morie.fn._registry import REGISTRY

        ns["REGISTRY"] = REGISTRY
        ns["fn"] = fn
    except ImportError:
        pass
    try:
        exec(compile(code, "<morie-exec>", "exec"), ns)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _handle_exec_co(args: argparse.Namespace) -> int:
    from pathlib import Path

    filename = args.filename
    if not filename:
        print("Usage: morie exec co <filename>")
        print("  e.g. morie exec co test.py")
        return 1

    co_dir = Path(args.co_dir) if args.co_dir else Path.cwd() / "cofs"
    co_dir.mkdir(parents=True, exist_ok=True)
    filepath = co_dir / filename

    ext = filepath.suffix.lower()
    lang_map = {
        ".py": "python",
        ".r": "r",
        ".sh": "shell",
        ".js": "javascript",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".ml": "ocaml",
        ".lua": "lua",
        ".ts": "typescript",
        ".tex": "latex",
    }
    lang = args.lang if args.lang != "python" else lang_map.get(ext, "python")

    if not filepath.exists():
        filepath.write_text("")
        print(f"Created {filepath}")
    else:
        print(f"Opening {filepath}")

    try:
        from .editor import launch_editor

        return launch_editor(str(filepath), run_on_save=True, lang_hint=lang)
    except ImportError:
        import subprocess

        return subprocess.call(["nano", str(filepath)])


def _handle_edit(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .editor import edit_file

    filepath = Path(args.file).resolve()
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text("")

    rc = edit_file(str(filepath), lang_hint=args.lang)
    if rc != 0 or not args.run:
        return rc
    lang = args.lang or ("r" if filepath.suffix.lower() == ".r" else "python")
    if lang not in ("python", "r"):
        print(f"--run supports python and r files, not {lang}", file=sys.stderr)
        return 2
    return _handle_exec(argparse.Namespace(code=None, filename=None, lang=lang, exec_file=str(filepath), co_dir=None))


def _handle_repl(args: argparse.Namespace) -> int:
    try:
        from .polyglot import run_headless_repl
    except ImportError:
        if not _add_interactive_layer("The polyglot multi-language REPL"):
            return 1
        from .polyglot import run_headless_repl

    polyglot = not getattr(args, "no_polyglot", False)
    auto_detect = not getattr(args, "no_detect", False)
    return run_headless_repl(
        polyglot=polyglot,
        auto_detect=auto_detect,
        lang=args.lang,
    )


PERCY_MODELS = [
    ("functiongemma:270m", "0.3 GB", "ToolCall", "FunctionGemma 270M -- fast tool calling"),
    ("functiongemma", "0.3 GB", "ToolCall", "FunctionGemma (default tag)"),
    ("gemma4:e2b", "7.2 GB", "LLM", "Perseus base model (Gemma 4)"),
    ("gemma4:e2b-it-q4_K_M", "7.2 GB", "LLM", "Gemma 4 instruction-tuned Q4"),
    ("gemma4:e2b-it-q8_0", "8.1 GB", "LLM", "Gemma 4 instruction-tuned Q8"),
    ("gemma4:e4b-it-q4_K_M", "9.6 GB", "LLM", "Gemma 4 large Q4"),
    ("gemma4:e4b-it-q8_0", "11 GB", "LLM", "Gemma 4 large Q8"),
    ("gemma3:4b", "3.3 GB", "LLM", "Gemma 3 lightweight"),
    ("gemma3n:e2b", "5.6 GB", "LLM", "Gemma 3 Nano small"),
    ("gemma3n:e4b", "7.5 GB", "LLM", "Gemma 3 Nano large"),
    ("mistral-nemo", "7.1 GB", "LLM", "Mistral Nemo 12B"),
    ("qwen3.5:0.8b", "1.0 GB", "LLM", "Qwen 3.5 tiny"),
    ("qwen3.5:2b", "2.7 GB", "LLM", "Qwen 3.5 small"),
    ("qwen3.5:4b", "3.4 GB", "LLM", "Qwen 3.5 medium"),
    ("qwen3.5:9b", "6.6 GB", "LLM", "Qwen 3.5 large"),
    ("nemotron-3-nano:4b", "2.8 GB", "LLM", "NVIDIA Nemotron Nano Q4"),
    ("nemotron-3-nano:4b-q8_0", "4.2 GB", "LLM", "NVIDIA Nemotron Nano Q8"),
    ("nemotron-3-nano:4b-bf16", "8.0 GB", "LLM", "NVIDIA Nemotron Nano BF16"),
    ("nemotron-cascade-2", "14 GB", "LLM", "NVIDIA Nemotron Cascade 24B"),
    ("deepseek-r1:1.5b", "1.1 GB", "Reasoning", "DeepSeek R1 tiny"),
    ("deepseek-r1:8b", "5.2 GB", "Reasoning", "DeepSeek R1 medium"),
    ("deepseek-r1:14b", "9.0 GB", "Reasoning", "DeepSeek R1 large"),
    ("magistral:24b", "14 GB", "Reasoning", "Mistral Magistral 24B"),
    ("lfm2", "14 GB", "LLM", "Liquid LFM2"),
    ("lfm2:24b-q4_K_M", "14 GB", "LLM", "Liquid LFM2 24B Q4"),
    ("lfm2.5-thinking", "0.7 GB", "Reasoning", "Liquid LFM2.5 thinking"),
    ("qwen3-vl:8b", "6.1 GB", "Vision", "Qwen 3 Vision 8B"),
    ("glm-ocr", "2.2 GB", "Vision", "GLM OCR model"),
    ("gpt-oss:20b", "12 GB", "LLM", "Open-source GPT 20B"),
    ("translategemma", "3.3 GB", "Translation", "Google TranslateGemma"),
    ("phi4-mini", "2.5 GB", "LLM", "Microsoft Phi-4 Mini"),
    ("phi4-mini:3.8b-q8_0", "4.1 GB", "LLM", "Microsoft Phi-4 Mini Q8"),
    ("smollm2:1.7b", "1.0 GB", "LLM", "HuggingFace SmolLM2 tiny"),
    ("llama3.2:3b", "2.0 GB", "LLM", "Meta Llama 3.2 3B"),
    ("granite3.3:8b", "4.9 GB", "LLM", "IBM Granite 3.3 8B"),
    ("granite3.3:2b", "1.6 GB", "LLM", "IBM Granite 3.3 2B"),
    ("kimi-k2.5:cloud", "0 GB", "Cloud", "Moonshot Kimi K2.5 (cloud)"),
    ("minimax-m2.7:cloud", "0 GB", "Cloud", "MiniMax M2.7 (cloud)"),
    ("nomic-embed-text", "0.3 GB", "Embedding", "Nomic text embeddings"),
    ("snowflake-arctic-embed2", "0.4 GB", "Embedding", "Snowflake embeddings"),
]


def _handle_percy(args: argparse.Namespace) -> int:
    import os
    import sys

    try:
        from .agent import create_agent
    except ImportError:
        if not _add_interactive_layer("The agent"):
            return 1
        from .agent import create_agent

    use_stream = not getattr(args, "no_stream", False)
    model = getattr(args, "model", None)
    base_url = None

    cloud_url = getattr(args, "cloud", None) or os.environ.get("PERSEUS_CLOUD_URL")
    cloud_token = os.environ.get("PERSEUS_CLOUD_TOKEN")

    if getattr(args, "local", False):
        print("Perseus [local mode] -- using local Ollama only...")
    elif cloud_url:
        print(f"Perseus [cloud] -- connecting to {cloud_url}...")
    elif getattr(args, "remote", False):
        cloud_url = os.environ.get("PERSEUS_CLOUD_URL")
        if cloud_url:
            print(f"Perseus [cloud] -- connecting to {cloud_url}...")
        else:
            pi = os.environ.get("MORIE_PI_HOST")
            if pi:
                host_part = pi.split("@")[-1] if "@" in pi else pi
                cloud_url = f"http://{host_part}:8421"
                print(f"Perseus [cloud via Pi] -- connecting to {cloud_url}...")
            else:
                print("Perseus [cloud] -- no relay configured.", file=sys.stderr)
                print("  Set PERSEUS_CLOUD_URL or MORIE_PI_HOST, or run `morie serve` on your Pi.", file=sys.stderr)
                return 1
    elif getattr(args, "pi", None):
        pi_host = args.pi
        base_url = f"http://{pi_host}:11434" if "://" not in pi_host else pi_host
        print(f"Perseus [Pi] -- connecting to {base_url}...")
    elif os.environ.get("MORIE_PI_HOST"):
        pi = os.environ["MORIE_PI_HOST"]
        host_part = pi.split("@")[-1] if "@" in pi else pi
        base_url = f"http://{host_part}:11434"

    try:
        agent = create_agent(model=model, base_url=base_url, cloud_url=cloud_url, cloud_token=cloud_token)
        model_name = getattr(agent, "_model", "unknown")
        if hasattr(agent, "is_running") and not agent.is_running():
            raise RuntimeError("no Ollama server answers at " + str(base_url or "the default address"))
    except Exception as exc:
        # Perseus's tool-calling agent needs a local Ollama; without one, answer
        # through the provider chain (hosted tier, your own endpoint, ...)
        print(f"Perseus: no local Ollama ({exc}); answering through the provider chain instead.", file=sys.stderr)
        question = getattr(args, "question", None)
        if question is None:
            print('Give the question on the command line: morie percy "..."  (or run `morie chat`)', file=sys.stderr)
            return 1
        payload = ask_percy(question, stream=use_stream, model=model)
        if use_stream:
            return _llm_exit_code(payload, _drain_stream(payload["output_stream"]))
        print(payload["output_text"])
        return _llm_exit_code(payload)
    print(f"Perseus [{model_name}] ready.\n")

    def _answer(q: str) -> int:
        try:
            _percy_answer(agent, q, use_stream, sys.stdout)
            return 0
        except Exception as exc:  # the Ollama backend died mid-answer: provider chain instead
            print(
                f"Perseus: the local agent backend failed ({exc}); answering through the provider chain.",
                file=sys.stderr,
            )
            payload = ask_percy(q, stream=use_stream, model=model)
            if use_stream:
                return _llm_exit_code(payload, _drain_stream(payload["output_stream"]))
            print(payload["output_text"])
            return _llm_exit_code(payload)

    question = getattr(args, "question", None)
    rc = 0
    if question is None:
        print("Type your question (or 'quit' to exit):\n")
        while True:
            try:
                q = input("percy> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            rc = _answer(q)
            print()
    else:
        rc = _answer(question)

    if hasattr(agent, "close"):
        agent.close()
    return rc


def _percy_answer(agent, question, use_stream, out):
    if use_stream and hasattr(agent, "chat_stream"):
        for chunk in agent.chat_stream(question):
            out.write(chunk)
            out.flush()
        out.write("\n")
    else:
        resp = agent.chat(question)
        out.write(resp.text + "\n")
        if resp.tool_calls_made:
            out.write(f"\n[{len(resp.tool_calls_made)} tool calls in {resp.iterations} iterations]\n")


def _handle_percysuits(args: argparse.Namespace) -> int:
    import os

    ssh_target = getattr(args, "ssh", None)
    host = args.host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    dry_run = args.dry_run

    if ssh_target:
        print("Perseus Model Suite")
        print(f"Target: {ssh_target} (SSH)")
        print()
        installed, installed_sizes = _percysuits_get_installed_ssh(ssh_target)
    else:
        print("Perseus Model Suite")
        print(f"Target: {host}")
        print()
        installed, installed_sizes = _percysuits_get_installed(host)
    if installed is None:
        return 1

    to_pull = []
    have = []
    for name, size, category, desc in PERCY_MODELS:
        if name in installed:
            have.append((name, size, category, desc))
        else:
            to_pull.append((name, size, category, desc))

    print(f"{'Model':<30} {'Size':>7}  {'Type':<12} Status")
    print("-" * 72)
    for name, size, category, desc in PERCY_MODELS:
        tag = "INSTALLED" if name in installed else "MISSING"
        sym = "+" if tag == "INSTALLED" else "-"
        print(f"  {sym} {name:<28} {size:>7}  {category:<12} {desc}")
    print()
    print(f"Installed: {len(have)}/{len(PERCY_MODELS)} | Missing: {len(to_pull)}")

    if installed_sizes:
        total = sum(installed_sizes.values()) / (1024**3)
        print(f"Total installed: {total:.1f} GB")
    print()

    if not to_pull:
        print("All Perseus models installed!")
        return 0

    if dry_run:
        print("Would pull:")
        for name, size, _cat, desc in to_pull:
            print(f"  {name} ({size}) -- {desc}")
        return 0

    if ssh_target:
        return _percysuits_pull_ssh(ssh_target, to_pull)
    return _percysuits_pull(host, to_pull)


def _percysuits_get_installed_ssh(ssh_target: str):
    import json as _json
    import subprocess

    try:
        result = subprocess.run(
            ["ssh", ssh_target, "curl -s http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        models_data = _json.loads(result.stdout).get("models", [])
        installed = set()
        sizes = {}
        for m in models_data:
            name = m.get("name", "")
            installed.add(name)
            if ":" in name:
                installed.add(name.split(":")[0])
            sz = m.get("size", 0)
            if sz:
                sizes[name] = sz
        return installed, sizes
    except Exception as exc:
        print(f"ERROR: cannot reach Ollama via SSH {ssh_target}: {exc}")
        return None, None


def _percysuits_pull_ssh(ssh_target: str, to_pull: list) -> int:
    import subprocess

    pulled = 0
    failed = []
    for i, (name, size, _cat, _desc) in enumerate(to_pull, 1):
        print(f"[{i}/{len(to_pull)}] {name} ({size}) ...")
        try:
            proc = subprocess.Popen(
                ["ssh", ssh_target, f"ollama pull {name}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    print(f"  {stripped}")
            proc.wait(timeout=600)
            if proc.returncode == 0:
                pulled += 1
                print(f"  Done: {name}")
            else:
                failed.append(name)
                print(f"  FAILED: {name} (exit {proc.returncode})")
        except Exception as exc:
            failed.append(name)
            print(f"  FAILED: {name} ({exc})")

    print()
    print(f"Pulled: {pulled} | Failed: {len(failed)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    return 0 if not failed else 1


def _percysuits_get_installed(host: str):
    try:
        import httpx

        resp = httpx.get(f"{host}/api/tags", timeout=15)
        resp.raise_for_status()
        models_data = resp.json().get("models", [])
        installed = set()
        sizes = {}
        for m in models_data:
            name = m.get("name", "")
            installed.add(name)
            if ":" in name:
                installed.add(name.split(":")[0])
            sz = m.get("size", 0)
            if sz:
                sizes[name] = sz
        return installed, sizes
    except Exception as exc:
        print(f"ERROR: cannot reach Ollama at {host}: {exc}")
        print()
        print("Make sure Ollama is running:")
        print("  ollama serve")
        print()
        print("Or set OLLAMA_HOST if Ollama is on another machine:")
        print("  export OLLAMA_HOST=http://your-server.local:11434")
        return None, None


def _percysuits_pull(host: str, to_pull: list) -> int:
    import json as _json

    import httpx

    pulled = 0
    failed = []
    for i, (name, size, _cat, _desc) in enumerate(to_pull, 1):
        print(f"[{i}/{len(to_pull)}] {name} ({size}) ...")
        try:
            with httpx.stream(
                "POST",
                f"{host}/api/pull",
                json={"name": name, "stream": True},
                timeout=httpx.Timeout(
                    connect=15,
                    read=600,
                    write=15,
                    pool=15,
                ),
            ) as resp:
                resp.raise_for_status()
                last_status = ""
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                    except ValueError:
                        continue
                    status = data.get("status", "")
                    if status != last_status:
                        print(f"  {status}")
                        last_status = status
                    if "error" in data:
                        raise RuntimeError(data["error"])
            pulled += 1
            print(f"  Done: {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAILED: {name} ({exc})")

    print()
    print(f"Pulled: {pulled} | Failed: {len(failed)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
