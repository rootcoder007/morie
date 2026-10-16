"""Minimal file editor entry point (native, no TUI framework).

The previous implementation embedded a Textual TUI. Textual was the
last interactive-extra dependency; the native replacement opens the
user's own editor ($VISUAL / $EDITOR, falling back to vi) on the file,
which is what a terminal user expects anyway, and never imports a UI
framework.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys

_TERMINAL_EDITORS = frozenset({"vi", "vim", "nvim", "nano", "pico", "emacs", "ed", "micro", "joe", "ne", "helix", "hx"})


def edit_file(path: str, lang_hint: str | None = None) -> int:
    """Open ``path`` in the user's editor; returns the exit code."""
    del lang_hint
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("vi" if shutil.which("vi") else None)
    if editor is None:
        print("no editor found: set $EDITOR", file=sys.stderr)
        return 1
    # $EDITOR may carry arguments ("code --wait"): split it ourselves, never through a shell
    parts = [p.strip('"') for p in shlex.split(editor, posix=False)] if os.name == "nt" else shlex.split(editor)
    if not parts:
        print("no editor found: set $EDITOR", file=sys.stderr)
        return 1
    binary = os.path.basename(parts[0])
    if binary in _TERMINAL_EDITORS and not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            f"{binary} is a terminal editor and this is not a terminal; run `morie edit {path}` from a shell, "
            "or set $EDITOR to a graphical editor",
            file=sys.stderr,
        )
        return 1
    if shutil.which(parts[0]) is None and not os.path.isfile(parts[0]):
        print(f"editor {parts[0]!r} not found on PATH (set $EDITOR to one that is)", file=sys.stderr)
        return 1
    from ._interactive import LayerMissingError, launcher

    try:
        sp = launcher("Opening an editor")
    except LayerMissingError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        return sp.call([*parts, path])
    except OSError as exc:
        print(f"could not start {parts[0]!r}: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``morie-edit <file>``: open the file in the built-in editor.

    Returns:
        2 with a usage line when no file is given, else the editor's exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: morie-edit <file>", file=sys.stderr)
        return 2
    return edit_file(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
