"""Minimal file editor entry point (native, no TUI framework).

The previous implementation embedded a Textual TUI. Textual was the
last interactive-extra dependency; the native replacement opens the
user's own editor ($VISUAL / $EDITOR, falling back to vi) on the file,
which is what a terminal user expects anyway, and never imports a UI
framework.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def edit_file(path: str, lang_hint: str | None = None) -> int:
    """Open ``path`` in the user's editor; returns the exit code."""
    del lang_hint
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") \
        or ("vi" if shutil.which("vi") else None)
    if editor is None:
        print("no editor found: set $EDITOR", file=sys.stderr)
        return 1
    return subprocess.call([editor, path])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: morie-edit <file>", file=sys.stderr)
        return 2
    return edit_file(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
