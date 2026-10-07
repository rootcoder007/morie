# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recursive deletes that cannot take a user's work with them.

A directory is removed only when the caller proves it owns it; the root, the home directory, the
working directory and its ancestors, and a git checkout are refused whatever the caller says.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = ["refuse_reason", "rmtree_owned"]


def refuse_reason(path: str | Path) -> str | None:
    """Why ``path`` must never be removed recursively, or None."""
    p = Path(path).expanduser().resolve()
    if p == Path(p.anchor):
        return "it is the filesystem root"
    if p == Path.home().resolve():
        return "it is the home directory"
    cwd = Path.cwd().resolve()
    if p == cwd or p in cwd.parents:
        return "it is the working directory or one of its parents"
    if (p / ".git").exists():
        return "it is a git checkout"
    return None


def rmtree_owned(path: str | Path, *, owned: bool) -> None:
    """Remove ``path`` and everything in it, only if ``owned`` and not a protected directory."""
    if not owned:
        raise PermissionError(f"refusing to remove {path}: it was not created by morie")
    why = refuse_reason(path)
    if why:
        raise PermissionError(f"refusing to remove {path}: {why}")
    shutil.rmtree(path)
