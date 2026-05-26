# morie.fn -- function file (rootcoder007/morie)
"""Inspect a single output file (schema, row counts, stats)."""

from morie.inspector import inspect_output as _fn

inspct = _fn
inspect_output = _fn


def cheatsheet() -> str:
    return "inspct() -> Inspect a single output file (schema, row counts, stats)."
