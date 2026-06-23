# morie.fn -- function file (rootcoder007/morie)
"""Design effect (DEFF) from survey weights."""

from morie.sampling import design_effect as _fn

deff = _fn
design_effect = _fn


def cheatsheet() -> str:
    return "deff() -> Design effect (DEFF) from survey weights."
