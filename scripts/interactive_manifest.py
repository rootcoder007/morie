#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write (or --check) src/morie/_interactive_manifest.json: SHA-256 of the five source-tree-only modules.

The wheel ships the manifest; `morie interactive install` verifies the files it fetches
against it. tests/test_interactive.py fails when the manifest drifts from the tree.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "morie"
OUT = PKG / "_interactive_manifest.json"
FILES = ("polyglot.py", "agent.py", "tui.py", "_exec_guard.py", "repl_init.py")


def build() -> str:
    # CRLF read as LF: a Windows checkout must produce the same hashes (see morie._interactive.sha256_of)
    files = {n: hashlib.sha256((PKG / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in FILES}
    return json.dumps({"files": files}, indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = build()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is stale; run scripts/interactive_manifest.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
