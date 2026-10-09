"""MORIE environment self-diagnostics.

``morie doctor`` checks that all required components are installed and
reachable before the user runs analysis modules.  Each check prints a
pass/fail row; the overall exit code is 0 only if all required checks pass.

Uses :mod:`rich` for formatted terminal output when available, with a plain
text fallback for minimal environments.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Individual check functions -- each returns (bool, str) = (passed, message)
# ---------------------------------------------------------------------------


def _check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    label = f"{v.major}.{v.minor}.{v.micro}"
    return ok, label if ok else f"{label} (need >= 3.10)"


def _check_import(package: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(package)
    except ImportError:
        return False, "not installed"
    version = getattr(mod, "__version__", None)
    if not version:  # rich, textual and others keep their version in the package metadata only
        try:
            from importlib.metadata import version as _dist_version

            version = _dist_version(package)
        except Exception:
            version = "installed"
    return True, str(version)


def _check_interactive_layer() -> tuple[bool, str]:
    """The repl/exec/agent/edit/tui verbs need five modules the wheel leaves out."""
    try:
        from . import _interactive as inter

        pkg_dir = Path(__file__).resolve().parent
        if all((pkg_dir / n).is_file() for n in inter.FILES):
            return True, "bundled (source checkout)"
        d = inter.data_dir()
        have = inter.present(d)
        iv = inter.installed_version(d)
        if len(have) == len(inter.FILES) and iv == inter.package_version():
            return True, f"installed for morie {iv} in {d}"
        if have:
            return False, f"installed for morie {iv} (this is {inter.package_version()}): morie interactive install"
        return False, "not installed (morie repl/exec/agent/edit/tui): morie interactive install"
    except Exception as exc:
        return False, f"error: {exc}"


def _check_r() -> tuple[bool, str]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False, "Rscript not found (optional)"
    try:
        out = subprocess.check_output(["Rscript", "--version"], stderr=subprocess.STDOUT, timeout=5).decode().strip()
        # "R scripting front-end version 4.4.1 (2024-06-14)"
        version = out.split("version")[-1].strip().split()[0] if "version" in out else out
        return True, version
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False, "Rscript found but failed to run"


def _check_ollama() -> tuple[bool, str]:
    """The Ollama server ollama.url names (OLLAMA_HOST, else localhost): its models, or why it is not used."""
    from . import llm

    base = llm._ollama_base_url()
    if not base:
        return False, "switched off (ollama.url = off)"
    tags = llm._ollama_tags()
    if tags is None:
        return False, f"not reachable at {base} (optional)"
    if not tags:
        if llm._cfg.value("ollama.model"):
            return True, f"running at {base}, no models listed; ollama.model = {llm._ollama_model()}"
        return False, f"running at {base} but no model pulled, so ask skips it (`ollama pull gemma4:e2b`)"
    labels = [t["name"] for t in tags[:3]]
    more = f" (+{len(tags) - 3} more)" if len(tags) > 3 else ""
    return True, f"{', '.join(labels)}{more}; default {llm._ollama_model()}"


def _check_route() -> tuple[bool, str]:
    """The route `morie ask` takes now (route = auto|own|ollama|hosted, `morie config`)."""
    from .llm import route_summary

    try:
        line = route_summary()
    except Exception as exc:  # noqa: BLE001 - a bad saved route, an unreadable services document
        return False, f"error: {exc}"
    return (not line.startswith("ask has no route")), f"{line}; change: morie config set route auto|own|ollama|hosted"


def _check_hosted() -> tuple[bool, str]:
    """The hosted tier: logged in and answering, logged in but unreachable, or not logged in."""
    from .hosted import status

    try:
        s = status()
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"error: {exc}"
    if not s["base_url"]:
        return False, "disabled (MORIE_HOSTED_BASE_URL or hosted.url is off, or the services document switched it off)"
    if not s["logged_in"]:
        return False, "not logged in -- run `morie login` (GitHub) or `morie login --email you@example.com`"
    who = f" as {s['user']}" if s.get("user") else ""
    if not s["reachable"]:
        from .hosted import hosted_problem_line

        return False, f"{hosted_problem_line()}{who}"
    from .hosted import hosted_model_available, hosted_models

    listed = hosted_models() or []
    tail = f"; models: {', '.join(listed)} (default {hosted_model_available()})" if listed else ""
    return True, f"logged in{who}, gateway answering{tail}"


def _check_gemini() -> tuple[bool, str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return True, f"key set, model={model}"
    return False, "GEMINI_API_KEY not set (optional)"


def _check_openai_compat() -> tuple[bool, str]:
    from .llm import _api_base_url, _api_model

    base = _api_base_url()
    if base:
        return True, f"{base} (model {_api_model()})"
    return False, "not set (optional): morie config set own.url URL, or morie provider set"


def _check_openai() -> tuple[bool, str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return (True, "key set") if key else (False, "OPENAI_API_KEY not set (optional)")


def _check_datasets() -> tuple[bool, str]:
    """Check built-in MORIE datasets database."""
    try:
        from .data import list_datasets, morie_db

        db_path = morie_db()
        if not db_path.exists():
            return (
                False,
                "morie.db not found -- it is built on first use: run `morie pull KEY` (keys: morie list-datasets)",
            )
        size_mb = db_path.stat().st_size // (1024 * 1024)
        ds = list_datasets()
        catalog = [d for d in ds if d["type"] != "hosted"]
        hosted = [d for d in ds if d["type"] == "hosted"]
        cached = [d for d in catalog if d["cached"]]
        extra = f", {len(hosted)} curated tables at data.rmorie.com" if hosted else ""
        return True, f"{len(cached)} of {len(catalog)} catalog keys cached ({size_mb}MB store){extra}"
    except Exception as e:
        return False, f"error: {e}"


def _check_docker() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if docker is None:
        return False, "docker not found (optional)"
    try:
        out = subprocess.check_output(["docker", "--version"], stderr=subprocess.STDOUT, timeout=5).decode().strip()
        return True, out
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False, "docker found but not running (optional)"


def _check_morie_version() -> tuple[bool, str]:
    """Check whether a newer morie release is available on PyPI.

    Uses the daily-cached result from the import-time update check, so
    this never makes a network call itself.
    """
    try:
        import morie

        from ._update_check import _parse_version, _read_cache

        installed = getattr(morie, "__version__", "0.0.0+unknown")
        latest = _read_cache().get("latest")
        if not latest:
            return True, f"{installed} (latest not yet checked)"
        if _parse_version(latest) > _parse_version(installed):
            return False, f"{installed} -- {latest} available; run `morie update`"
        return True, f"{installed} (up to date)"
    except Exception as exc:  # noqa: BLE001
        return True, f"version check unavailable ({exc})"


# ---------------------------------------------------------------------------
# Master check list
# ---------------------------------------------------------------------------

_REQUIRED_IMPORTS = [
    "httpx",
    "rich",
]

# morie's core is native: these only speed up or extend a few paths and are reported, not required
_OPTIONAL_IMPORTS: list[str] = ["pandas", "numpy", "scipy", "sklearn", "statsmodels", "textual"]


# The trust knobs as morie._exec_guard defines them (that module ships with the interactive layer):
# environment variables, read the same way, so a plain install reports its posture too.
_KNOB_DETAILS = (
    ("MORIE_NO_EXEC", "when set: ALL dynamic execution (REPL/exec/shell) is disabled"),
    (
        "MORIE_ALLOW_REMOTE_INSTALL",
        "when set: the shell launcher (bin/morie, git checkouts) may run the downloaded Ollama install.sh",
    ),
    (
        "MORIE_TRUST_CHECKPOINT",
        "when set: convert-checkpoint / pt2gguf deserialize a .pt (tensors and plain containers only)",
    ),
    ("MORIE_ALLOW_RC", "when set: the shell launcher (bin/morie, git checkouts) sources the ESML_RC shell config"),
    (
        "MORIE_ALLOW_CRON",
        "when set: the shell launcher's `cron add/remove` (bin/morie, git checkouts) may edit your crontab",
    ),
)


def _knob_status_without_layer() -> list[dict[str, Any]]:
    out = []
    for name, detail in _KNOB_DETAILS:
        v = os.environ.get(name, "").strip()
        on = v not in ("", "0") if name == "MORIE_NO_EXEC" else v.lower() in ("1", "true", "yes", "on")
        out.append({"name": name, "enabled": on, "detail": detail})
    return out


def run_checks() -> dict[str, Any]:
    """Run all diagnostics and return a structured results dict."""
    results: dict[str, Any] = {"checks": [], "all_required_passed": True}

    def _add(label: str, passed: bool, detail: str, required: bool = True) -> None:
        results["checks"].append({"label": label, "passed": passed, "detail": detail, "required": required})
        if required and not passed:
            results["all_required_passed"] = False

    # Python
    ok, detail = _check_python_version()
    _add("Python version", ok, detail, required=True)

    # morie itself -- newer release available?
    ok, detail = _check_morie_version()
    _add("morie version", ok, detail, required=False)

    # Required Python packages
    for pkg in _REQUIRED_IMPORTS:
        ok, detail = _check_import(pkg)
        _add(f"import {pkg}", ok, detail, required=True)

    # Optional Python packages
    for pkg in _OPTIONAL_IMPORTS:
        ok, detail = _check_import(pkg)
        if not ok:
            detail = (
                'not installed (morie tui needs it): pip install "morie[interactive]"'
                if pkg == "textual"
                else "not installed (optional: morie's native cores need none of these)"
            )
        _add(f"import {pkg}", ok, detail, required=False)

    ok, detail = _check_interactive_layer()
    _add("Interactive layer", ok, detail, required=False)

    # R
    ok, detail = _check_r()
    _add("R (Rscript)", ok, detail, required=False)

    # LLM providers
    ok, detail = _check_ollama()
    _add("Ollama (local)", ok, detail, required=False)

    ok, detail = _check_hosted()
    _add("Hosted LLM (llm.rmorie.com)", ok, detail, required=False)

    ok, detail = _check_gemini()
    _add("Gemini API key", ok, detail, required=False)

    ok, detail = _check_openai_compat()
    _add("OpenAI-compat API", ok, detail, required=False)

    ok, detail = _check_openai()
    _add("OpenAI API key", ok, detail, required=False)

    ok, detail = _check_route()
    _add("LLM route (morie ask)", ok, detail, required=False)

    # Data
    ok, detail = _check_datasets()
    _add("Built-in datasets", ok, detail, required=False)

    # Infrastructure
    ok, detail = _check_docker()
    _add("Docker", ok, detail, required=False)

    # Trust knobs -- report the active security posture. Each row is
    # informational (never a required failure): the SAFE default is the
    # risky path disabled, so an enabled knob is shown but not marked FAIL.
    try:
        from morie._exec_guard import knob_status
    except ImportError:
        knob_status = _knob_status_without_layer

    if knob_status is not None:
        for knob in knob_status():
            state = "ENABLED" if knob["enabled"] else "default (off)"
            _add(f"trust: {knob['name']}", True, f"{state} -- {knob['detail']}", required=False)

    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_plain(results: dict[str, Any]) -> None:
    from .i18n import t

    print(t("doctor.heading"))
    print("=" * 50)
    for check in results["checks"]:
        status = "OK " if check["passed"] else ("FAIL" if check["required"] else "WARN")
        print(f"  [{status}] {check['label']:<30} {check['detail']}")
    print()
    if results["all_required_passed"]:
        print("All required checks passed.")
    else:
        print("Some required checks failed. See FAIL rows above.")


def _render_rich(results: dict[str, Any]) -> None:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title=__import__("morie.i18n", fromlist=["t"]).t("doctor.heading"),
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Status", width=6, justify="center")
    table.add_column("Check", style="bold", min_width=28)
    table.add_column("Detail")

    for check in results["checks"]:
        if check["passed"]:
            status = "[green]  OK [/green]"
        elif check["required"]:
            status = "[red] FAIL[/red]"
        else:
            status = "[yellow] WARN[/yellow]"
        table.add_row(status, check["label"], check["detail"])

    console.print(table)

    if results["all_required_passed"]:
        console.print("[green]All required checks passed.[/green]")
    else:
        console.print("[red]Some required checks failed. See FAIL rows above.[/red]")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


# import-name -> pip-install-name, where the two differ
_PIP_NAME = {"sklearn": "scikit-learn"}


def _render(results: dict[str, Any]) -> None:
    """Render the results table, picking rich or plain output.

    Respects ``NO_COLOR`` (https://no-color.org) and falls back to the
    plain renderer when stdout isn't a TTY, so screen readers and
    pipelines get a clean linear stream instead of box-drawing.
    """
    no_color = bool(os.environ.get("NO_COLOR"))
    not_tty = not sys.stdout.isatty()
    if no_color or not_tty:
        _render_plain(results)
    else:
        try:
            _render_rich(results)
        except ImportError:
            _render_plain(results)


def _installer() -> list[str] | None:
    """The command that installs into this interpreter: pip when it has pip, uv when uv made it."""
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install"]
    import shutil

    uv = shutil.which("uv")
    return [uv, "pip", "install", "--python", sys.executable] if uv else None


def _heal(results: dict[str, Any]) -> bool:
    """Attempt to remediate failed checks (``morie doctor --fix``).

    Auto-fixable: missing Python packages (``pip install``) and the
    morie cache directory.  Everything else gets an actionable hint
    rather than a silent failure.  Returns True if anything was fixed.
    """
    print()
    print("Healing -- attempting to fix failed checks ...")

    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "morie"
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"  [ok]   cache directory ready: {cache_dir}")
    except OSError as exc:
        print(f"  [fail] could not create {cache_dir}: {exc}")

    fixed_any = False
    installer = _installer()
    for check in results["checks"]:
        if check["passed"]:
            continue
        label = check["label"]
        if label.startswith("import "):
            pkg = label[len("import ") :]
            pip_name = _PIP_NAME.get(pkg, pkg)
            if installer is None:
                # a uv-made venv has no pip: say what to run instead of printing "No module named pip"
                print(f"  [hint] {pip_name}: this environment has no pip; run `uv pip install {pip_name}`")
                continue
            print(f"  ...    installing {pip_name} ({' '.join(installer[-3:])}) ...")
            rc = subprocess.run([*installer, pip_name]).returncode
            print(f"  [{'ok' if rc == 0 else 'fail'}]   install {pip_name}")
            fixed_any = fixed_any or rc == 0
        elif label == "morie version":
            print("  [hint] run `morie update` to upgrade morie itself.")
        elif label == "Built-in datasets":
            print("  [hint] the built-in DB is downloaded, not shipped: run `morie download-bootstrap`.")
        elif label == "R (Rscript)":
            print("  [hint] install R from https://www.r-project.org/ (optional -- only the R bridge needs it).")
        else:
            print(f"  [hint] {label}: optional component -- configure it only if you need that feature.")

    if fixed_any:
        importlib.invalidate_caches()
    return fixed_any


def run_doctor(fix: bool = False) -> int:
    """Run diagnostics and print a summary table.

    Parameters
    ----------
    fix : bool
        When True (``morie doctor --fix``), attempt to remediate failed
        checks -- install missing Python packages, create the cache
        directory -- then re-run and re-render the diagnostics.

    Returns
    -------
    int
        ``0`` if all required checks pass, ``1`` otherwise.
    """
    results = run_checks()
    _render(results)

    if fix:
        healed = _heal(results)
        if healed:
            print()
            print("Re-running diagnostics after fixes ...")
        results = run_checks()
        _render(results)

    return 0 if results["all_required_passed"] else 1
