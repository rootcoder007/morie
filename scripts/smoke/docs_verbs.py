# SPDX-License-Identifier: AGPL-3.0-or-later
"""Docs drift check: every `morie <verb> ...` command in the docs must name a verb and
flags the parser actually has. Catches a renamed verb or a dropped flag that the
pages still advertise.

    python scripts/smoke/docs_verbs.py docs/source README.md WHATS_NEW.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CMD = re.compile(r"(?m)^[ \t]*(?:\$[ \t]*)?morie[ \t]+([a-z][a-z-]*)((?:[ \t]+[^\s\\]+)*)")


def parser_table() -> dict[str, set[str]]:
    from morie.runner import build_parser

    table: dict[str, set[str]] = {}
    for a in build_parser()._actions:
        if isinstance(a, argparse._SubParsersAction):
            for verb, sub in a.choices.items():
                flags = set()
                for act in sub._actions:
                    flags.update(act.option_strings)
                    if isinstance(act, argparse._SubParsersAction):
                        for s2 in act.choices.values():
                            for act2 in s2._actions:
                                flags.update(act2.option_strings)
                table[verb] = flags
    return table


def main(paths: list[str]) -> int:
    table = parser_table()
    problems = []
    files: list[Path] = []
    for p in map(Path, paths):
        files += [p] if p.is_file() else [f for f in p.rglob("*") if f.suffix in (".rst", ".md")]
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in CMD.finditer(text):
            verb, rest = m.group(1), m.group(2)
            if verb in ("-h", "--help"):
                continue
            if verb not in table:
                if verb in ("does", "is", "can", "will", "has", "runs", "uses", "ships", "and", "or"):
                    continue  # prose, not a command
                problems.append(f"{f}: unknown verb `morie {verb}`")
                continue
            for flag in re.findall(r"(?<![\w-])(--?[a-z][\w-]*)", rest):
                if flag in ("--help", "-h"):
                    continue
                if flag not in table[verb] and not any(flag in v for v in table.values() if verb == "ingest"):
                    problems.append(f"{f}: `morie {verb}` has no flag {flag}")
    for p in sorted(set(problems)):
        print(p)
    print(f"docs_verbs: {len(files)} files, {len(set(problems))} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["docs/source", "README.md", "WHATS_NEW.md"]))
