#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bundle morie.fn's ~36k per-callable .py and ~36k describe_*.md files into
two archives so the wheel ships a few hundred files instead of ~73,000.

Outputs (into --out, default = --src):
  _fnsrc.zip             DEFLATE zip of the per-callable <short>.py modules.
                         morie.fn.__init__ appends it to __path__ so zipimport
                         resolves ``morie.fn.<short>`` from inside it.
  describe_docs.json.xz  lzma-compressed {short: markdown} map, read by
                         morie.fn.describe(). The ~36k near-identical markdown
                         files compress ~5x better as one stream than as
                         individually-deflated wheel members.

Run from the repo root or via CMake at build time. Idempotent. The loaders
fall back to loose files, so generating these is purely a packaging step.
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
from pathlib import Path

_PREFIX = "describe_"
_SUFFIX = ".md"

# fn/*.py files that must stay LOOSE in the wheel: infra, not per-callable
# implementations. This is the exact complement of the wheel.exclude negation
# in pyproject.toml (every fn/*.py is zipped + excluded from the wheel EXCEPT
# these), so any loose file dropped from the wheel is recoverable via zipimport
# -- including orphan .py not referenced by _lazy_map. Underscore-prefixed
# modules (_registry, _richresult, ...) are infra and skipped by the
# startswith("_") test. ``describe.py`` is a lazy module but reads package data
# via __file__, so it must stay loose too.
_INFRA_PY = frozenset({"describe.py", "__init__.py"})


def build_fnsrc_xz(fn_dir: Path, out: Path) -> int:
    """Solid-lzma {short: source} for every per-callable <short>.py (all of
    fn/*.py except the infra files). Solid compression exploits the heavy
    cross-file redundancy in the generated modules (~76 MB source -> ~3 MB, vs
    ~28 MB as an individually-deflated zip). The loader decompresses this once
    into an on-disk cache zip. Returns the number of modules written."""
    sources: dict[str, str] = {}
    for py in sorted(fn_dir.glob("*.py")):
        if py.name.startswith("_") or py.name in _INFRA_PY:
            continue
        sources[py.stem] = py.read_text(encoding="utf-8", errors="replace")
    blob = json.dumps(sources, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with lzma.open(out, "wb", preset=9 | lzma.PRESET_EXTREME) as fh:
        fh.write(blob)
    return len(sources)


def build_describe_docs(fn_dir: Path, out: Path) -> int:
    """lzma-compress {short: markdown} for every describe_<short>.md."""
    docs: dict[str, str] = {}
    for md in fn_dir.glob(f"{_PREFIX}*{_SUFFIX}"):
        short = md.name[len(_PREFIX) : -len(_SUFFIX)]
        docs[short] = md.read_text(encoding="utf-8")
    blob = json.dumps(docs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with lzma.open(out, "wb", preset=9 | lzma.PRESET_EXTREME) as fh:
        fh.write(blob)
    return len(docs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="path to src/morie/fn")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: --src)")
    args = ap.parse_args(argv)

    fn_dir: Path = args.src
    out_dir: Path = args.out or fn_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (fn_dir / "_lazy_map.json").exists():
        ap.error(f"{fn_dir}/_lazy_map.json not found -- is --src the morie.fn dir?")

    fnsrc_out = out_dir / "_fnsrc.json.xz"
    md_out = out_dir / "describe_docs.json.xz"
    nz = build_fnsrc_xz(fn_dir, fnsrc_out)
    nd = build_describe_docs(fn_dir, md_out)

    print(f"_fnsrc.json.xz        : {nz} modules -> {fnsrc_out.stat().st_size / 1e6:.1f} MB")
    print(f"describe_docs.json.xz : {nd} docs       -> {md_out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
