# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie._datapaths — portable data-directory resolution.

The dataset loaders in ``morie.arsau_datasets`` / ``morie.otis_datasets``
must NEVER hardcode absolute paths from the maintainer's workstation
(e.g. ``/Volumes/VSR/...``). Other people install morie from PyPI /
r-universe / CRAN. Their files live wherever they put them.

This module provides one function — :func:`resolve_data_dir` — which
walks a documented cascade and returns the directory containing the
requested domain's data files. If nothing is found, it raises
``FileNotFoundError`` with a clear message telling the user what to do.

The cascade (highest priority first):

1. Explicit ``data_dir=`` argument from the caller
2. Domain-specific env var ``MORIE_<DOMAIN>_DIR`` (e.g.
   ``MORIE_ARSAU_DIR``, ``MORIE_OTIS_DIR``)
3. Generic ``MORIE_DATA_DIR`` env var + ``/<domain>`` subdirectory
4. Persistent user cache (only if it already exists — never created
   here; created only by ``morie_<domain>_download(...)`` which is the
   user's explicit opt-in):
     - prefer ``platformdirs.user_data_dir("morie")`` if installed
     - else ``~/.local/share/morie`` on Linux,
       ``~/Library/Application Support/morie`` on macOS, or
       ``%APPDATA%/morie`` on Windows
5. Bundled fixture under the morie installation
   (``inst/extdata/<domain>/`` on the R side; ``morie/data/<domain>/``
   on the Python side). These are TINY toy fixtures for tests + tutorials.
6. Raise ``FileNotFoundError`` with a remediation paragraph.

This matches the CRAN-Policy compliant pattern documented in
``r-package/morie/R/database.R`` for ``morie_cache_dir`` /
``morie_cache_clear``: nothing is written outside of an explicit user
opt-in, and tests/tutorials work against bundled toy fixtures.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Iterable


def _env(name: str) -> str | None:
    """Return the environment variable, or ``None`` if unset or empty."""
    v = os.environ.get(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def _user_data_dir() -> Path:
    """Platform-appropriate persistent user-data directory for morie.

    Tries ``platformdirs`` first (more accurate); falls back to a
    hand-rolled per-platform default. Either way, this function does
    NOT create the directory — it only returns the path.
    """
    try:
        import platformdirs

        return Path(platformdirs.user_data_dir("morie", appauthor=False))
    except ImportError:
        system = platform.system()
        home = Path.home()
        if system == "Darwin":
            return home / "Library" / "Application Support" / "morie"
        if system == "Windows":
            base = _env("APPDATA")
            return Path(base) / "morie" if base else home / "AppData" / "Roaming" / "morie"
        # Linux / BSD / other unix
        xdg = _env("XDG_DATA_HOME")
        return Path(xdg) / "morie" if xdg else home / ".local" / "share" / "morie"


def _bundled_fixture(domain: str) -> Path:
    """The bundled tiny fixture path for ``domain``.

    Lives at ``<morie-pkg>/data/<domain>/``. These fixtures ship
    inside the wheel + sdist and are versioned alongside the code.
    They are SMALL — 5-row toy samples per record type — so unit tests
    + tutorials run with zero network, zero downloads, zero
    environment configuration. The real, full datasets are out of
    scope for shipping inside the package (CRAN 5-MB cap + the actual
    Ontario datasets are >>5 MB).
    """
    return Path(__file__).resolve().parent / "data" / domain


def resolve_data_dir(
    domain: str,
    *,
    data_dir: str | os.PathLike | None = None,
    extra_env_vars: Iterable[str] | None = None,
    allow_bundled_fixture: bool = True,
    require_exists: bool = True,
) -> Path:
    """Resolve the directory containing the requested ``domain``'s data files.

    Parameters
    ----------
    domain : str
        Lower-case short name (e.g. ``"arsau"``, ``"otis"``).
    data_dir : str | PathLike | None, default None
        Explicit user override; highest priority.
    extra_env_vars : iterable of str, optional
        Additional environment variables to check (after the standard
        ``MORIE_<DOMAIN>_DIR`` and ``MORIE_DATA_DIR``). Useful for
        legacy variable names.
    allow_bundled_fixture : bool, default True
        If True, fall back to the bundled tiny fixture under
        ``morie/data/<domain>/`` when no user-supplied location works.
        Set False in production code that needs the real dataset, not
        a 5-row toy sample.
    require_exists : bool, default True
        If True, raise ``FileNotFoundError`` when no candidate
        directory exists. If False, return the first candidate path
        regardless of existence (useful for download targets).

    Returns
    -------
    Path
        Absolute path to the resolved directory.

    Raises
    ------
    FileNotFoundError
        If ``require_exists=True`` and no candidate path exists.

    Examples
    --------
    >>> resolve_data_dir('arsau', data_dir='/srv/morie-data/ARSAU')
    PosixPath('/srv/morie-data/ARSAU')

    >>> import os
    >>> os.environ['MORIE_ARSAU_DIR'] = '/srv/x'   # doctest: +SKIP
    >>> resolve_data_dir('arsau')                  # doctest: +SKIP
    PosixPath('/srv/x')
    """
    domain_lc = domain.strip().lower()
    domain_uc = domain_lc.upper()

    candidates: list[tuple[str, Path]] = []

    # 1. Explicit argument
    if data_dir is not None:
        candidates.append(("data_dir argument", Path(data_dir).expanduser().resolve()))

    # 2. Domain-specific env var
    e = _env(f"MORIE_{domain_uc}_DIR")
    if e:
        candidates.append((f"MORIE_{domain_uc}_DIR env var", Path(e).expanduser().resolve()))

    # 2b. Caller-specified extra env vars
    if extra_env_vars:
        for var in extra_env_vars:
            e2 = _env(var)
            if e2:
                candidates.append((f"{var} env var", Path(e2).expanduser().resolve()))

    # 3. Generic MORIE_DATA_DIR + /<domain>
    e3 = _env("MORIE_DATA_DIR")
    if e3:
        candidates.append((
            "MORIE_DATA_DIR + /" + domain_lc,
            (Path(e3).expanduser() / domain_lc).resolve(),
        ))
        # Also try with original-case domain (some users prefer it):
        if domain_lc != domain:
            candidates.append((
                "MORIE_DATA_DIR + /" + domain,
                (Path(e3).expanduser() / domain).resolve(),
            ))

    # 4. Persistent user data dir (only if it already exists)
    user_dir = _user_data_dir() / domain_lc
    candidates.append(("user-data dir (opt-in)", user_dir))

    # 5. Bundled tiny fixture
    if allow_bundled_fixture:
        candidates.append(("bundled fixture", _bundled_fixture(domain_lc)))

    if not require_exists:
        # Return the highest-priority candidate path regardless of
        # existence — caller will use this for download targets etc.
        return candidates[0][1]

    for label, path in candidates:
        if path.exists() and path.is_dir():
            return path

    # Nothing worked — give the user a remediation paragraph.
    tried = "\n".join(f"  - {label}: {path}" for label, path in candidates)
    raise FileNotFoundError(
        f"morie: could not find data directory for domain {domain_lc!r}.\n"
        f"Tried (in order):\n{tried}\n\n"
        f"To fix, do one of:\n"
        f"  - pass data_dir='/path/to/{domain_lc}' to the loader\n"
        f"  - set MORIE_{domain_uc}_DIR=/path/to/{domain_lc} in your environment\n"
        f"  - set MORIE_DATA_DIR=/path/to/morie-data (with a {domain_lc}/ subdir)\n"
        f"  - call morie.{domain_lc}_download(target_dir=...) to fetch into a "
        f"local cache\n"
    )


__all__ = ["resolve_data_dir"]
