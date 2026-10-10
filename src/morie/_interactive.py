# SPDX-License-Identifier: AGPL-3.0-or-later
"""The interactive layer (polyglot REPL, agent, TUI, ``morie exec``, program launcher) for installed copies.

Six modules stay out of the published wheel and sdist on purpose: five execute
code that a person or a model types, and ``_launch`` starts every other program
morie uses (R, Docker, the editor, the C compiler, pip, ssh, gcloud, quarto).
Package scanners flag both surfaces (see ``wheel.exclude`` in pyproject.toml). ``morie interactive install`` fetches
those files for the installed version from the tagged source on GitHub, checks
each one against the SHA-256 manifest that ships inside the wheel, and places
them in a per-user directory that ``import morie`` adds to the package path.
Nothing under site-packages changes; ``morie interactive remove`` deletes the
directory again. A source checkout has the files already and never needs this.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

FILES: tuple[str, ...] = ("polyglot.py", "agent.py", "tui.py", "_exec_guard.py", "repl_init.py", "_launch.py")
RAW_URL = "https://raw.githubusercontent.com/rootcoder007/morie/{ref}/src/morie/{name}"
_MANIFEST = Path(__file__).with_name("_interactive_manifest.json")
_RELEASE = re.compile(r"^\d+\.\d+\.\d+$")


def data_dir() -> Path:
    """Where the layer lives for this user (``MORIE_INTERACTIVE_DIR`` overrides)."""
    env = os.environ.get("MORIE_INTERACTIVE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / "morie" / "interactive"


def manifest() -> dict[str, str]:
    """``{file name: sha256}`` for the five files, as built into this wheel (empty when missing)."""
    try:
        files = json.loads(_MANIFEST.read_text(encoding="utf-8"))["files"]
        return {str(k): str(v) for k, v in files.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def sha256_of(path: Path) -> str:
    """SHA-256 of the file with CRLF line endings read as LF.

    A Windows checkout with autocrlf turns every .py into CRLF, and the manifest
    describes the LF bytes GitHub serves; the hash must not depend on that.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def package_version() -> str:
    from morie import __version__

    return __version__


def default_ref(version: str) -> str:
    """The tag of a release version; ``main`` for anything else (dev builds, unknown)."""
    return f"v{version}" if _RELEASE.match(version or "") else "main"


def installed_version(d: Path | None = None) -> str | None:
    d = d or data_dir()
    try:
        v = (d / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return v or None


def present(d: Path | None = None) -> list[str]:
    d = d or data_dir()
    return [n for n in FILES if (d / n).is_file()]


def activate(package_path: list, version: str) -> bool:
    """Called by ``import morie``: join the per-user directory to the package path when it matches this version.

    One ``stat`` when nothing is installed. A directory installed for another
    version is left out (``morie interactive status`` says so) rather than mixing
    two releases' files.
    """
    d = data_dir()
    if installed_version(d) != version:
        return False
    if changed_files(d):
        return False  # edited since a verified install: never load it (``status`` names the files)
    p = str(d)
    if p not in package_path:
        package_path.append(p)
    return True


def changed_files(d: Path | None = None) -> list[str]:
    """Files of a verified install that no longer match the bundled manifest.

    A layer installed with verification is re-checked at every ``import morie``
    (five files, about 400 KB of SHA-256), so a file edited after install is not
    loaded. A layer installed with ``--no-verify`` is the user's explicit choice
    and is not checked.
    """
    d = d or data_dir()
    if not (d / "VERIFIED").is_file():
        return []
    expected = manifest()
    if not expected:
        return []
    return [n for n in FILES if not (d / n).is_file() or sha256_of(d / n) != expected.get(n)]


class LayerMissingError(RuntimeError):
    """A feature needs the interactive layer, and it is not installed for this user."""


def _offer_install(what: str) -> bool:
    """Say how to add the layer; on a terminal offer to add it now. True when it was just installed."""
    changed = changed_files()
    if changed:
        print(
            f"{what} is not loaded: {', '.join(changed)} changed since the verified install "
            "(`morie interactive status` shows it; `morie interactive install` restores it)."
        )
        return False
    print(f"{what} is not bundled in this install. Run `morie interactive install` to add it for this user.")
    interactive = (
        sys.stdin is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
        and not os.environ.get("MORIE_NO_PROMPT", "").strip()
    )
    if not interactive:
        return False
    try:
        answer = input("Add it now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if answer not in ("y", "yes") or install() != 0:
        return False
    import morie

    activate(morie.__path__, morie.__version__)
    return True


def launcher(what: str = "This command", *, offer: bool = True):
    """The module that starts other programs (``_launch``, part of the interactive layer).

    The published package never starts a program itself: R, Docker, the editor, the C compiler,
    pip, ssh and the rest all go through ``_launch``, which ``morie interactive install`` adds.
    ``what`` names the feature in the message shown when the layer is missing. Raises
    :class:`LayerMissingError` when it is missing and not installed on the spot. ``offer=False``
    raises at once, without printing or prompting: for optional probes that have a fallback.
    """
    try:
        from morie import _launch

        return _launch
    except ImportError:
        pass
    if offer and _offer_install(what):
        try:
            from morie import _launch

            return _launch
        except ImportError:
            pass
    raise LayerMissingError(
        f"{what} starts another program, which needs the interactive layer: run `morie interactive install`"
    )


def _download(url: str, dest: Path, label: str) -> None:
    from morie._progress import download_url

    download_url(url, dest, label, timeout=60)


def _textual_available() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def install(
    ref: str | None = None, source: str | None = None, verify: bool = True, out=print, force: bool = False
) -> int:
    """Fetch, verify and enable the five files. Returns a process exit code."""
    from urllib.error import HTTPError, URLError

    version = package_version()
    ref = ref or default_ref(version)
    expected = manifest()
    d = data_dir()
    if (
        not force
        and not source
        and expected
        and installed_version(d) == version
        and len(present(d)) == len(FILES)
        and all(sha256_of(d / n) == expected.get(n) for n in FILES)
    ):
        out(
            f"The interactive layer for morie {version} is already installed in {d} (verified). Pass --force to fetch it again."
        )
        return 0
    if verify and not expected:
        out("This build carries no manifest for the interactive layer, so the files cannot be verified.")
        out("Pass --no-verify to install them on trust, or run morie from a source checkout.")
        return 1
    with tempfile.TemporaryDirectory(prefix="morie-interactive-") as tmp:
        tmpd = Path(tmp)
        for name in FILES:
            if source:
                src = Path(source).expanduser() / name
                if not src.is_file():
                    out(f"{src} is missing; --from needs a src/morie directory with all five files.")
                    return 1
                shutil.copyfile(src, tmpd / name)
                continue
            url = RAW_URL.format(ref=ref, name=name)
            try:
                _download(url, tmpd / name, name)
            except HTTPError as exc:
                if exc.code == 404:
                    if ref == default_ref(version) and ref != "main":
                        out(
                            f"morie {version} has no release tag {ref} on GitHub yet (a pre-release build), so its "
                            "interactive layer cannot be fetched by version. Install it from the branch it was "
                            "built from: morie interactive install --ref <branch> --no-verify"
                        )
                    else:
                        out(
                            f"{url} does not exist: the ref {ref!r} has no such file. Pass --ref (a tag, branch or commit)."
                        )
                else:
                    out(f"download of {name} failed: HTTP {exc.code}")
                return 1
            except (URLError, OSError, TimeoutError) as exc:
                out(f"download of {name} failed: {exc}")
                return 1
        if verify:
            bad = [n for n in FILES if sha256_of(tmpd / n) != expected.get(n)]
            if bad:
                out(f"{', '.join(bad)}: contents differ from the manifest built into morie {version}.")
                out(
                    f"Fetch the matching release with --ref {default_ref(version)}, or pass --no-verify to keep these files anyway."
                )
                return 1
        d.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(d / "__pycache__", ignore_errors=True)
        for name in FILES:
            shutil.copyfile(tmpd / name, d / name)
        (d / "VERSION").write_text(version + "\n", encoding="utf-8")
        marker = d / "VERIFIED"
        if verify:
            marker.write_text("sha256 against the bundled manifest\n", encoding="utf-8")
        elif marker.exists():
            marker.unlink()
    where = "a local directory" if source else ref
    checked = "verified against the bundled manifest" if verify else "not verified"
    out(f"Installed the interactive layer for morie {version} from {where} into {d} ({checked}).")
    if _textual_available():
        out("morie repl, morie exec, morie agent and morie tui work now.")
    else:
        out(
            "morie repl, exec and agent work now, as do the features that start other programs "
            "(R and the R bridge, Docker, morie edit, doctor --fix, notebooks); "
            'morie tui also needs: pip install "morie[interactive]"'
        )
    out("Remove it again with: morie interactive remove")
    return 0


def remove(out=print) -> int:
    d = data_dir()
    if not d.exists():
        out(f"Nothing to remove: {d} does not exist.")
        return 0
    from ._safe_rm import refuse_reason, rmtree_owned

    # the installer writes VERSION: without it the directory (MORIE_INTERACTIVE_DIR can point anywhere) is not ours
    why = None if (d / "VERSION").is_file() else "it holds no interactive layer (no VERSION file)"
    why = why or refuse_reason(d)
    if why:
        out(f"Not removing {d}: {why}.")
        return 1
    rmtree_owned(d, owned=True)
    out(f"Removed {d}.")
    return 0


def status(out=print) -> int:
    version = package_version()
    d = data_dir()
    have = present(d)
    iv = installed_version(d)
    pkg_dir = Path(__file__).resolve().parent
    bundled = [n for n in FILES if (pkg_dir / n).is_file()]
    out(f"interactive layer directory: {d}")
    if len(bundled) == len(FILES):
        out("state: active (this is a source checkout; all five files are part of the package)")
        return 0
    if not have:
        out("state: not installed")
        out("Add it with: morie interactive install")
        return 0
    if len(have) < len(FILES) or iv is None:
        out(f"state: incomplete ({len(have)} of {len(FILES)} files); run: morie interactive install")
        return 0
    if iv != version:
        out(f"state: installed for morie {iv}, this is morie {version}; run: morie interactive install")
        return 0
    bad = changed_files(d)
    if bad:
        out(f"state: NOT loaded: {', '.join(bad)} changed since the verified install; run: morie interactive install")
        return 1
    checked = (
        "verified against the bundled manifest"
        if (d / "VERIFIED").is_file()
        else "NOT verified (installed with --no-verify)"
    )
    out(f"state: active for morie {version}, {checked} (files: {', '.join(FILES)})")
    return 0
