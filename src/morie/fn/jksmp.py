# morie.fn -- function file (rootcoder007/morie)
"""Delete-1 jackknife variance estimate."""

from morie.sampling import jackknife_estimate as _fn

jksmp = _fn
jackknife_estimate = _fn


def cheatsheet() -> str:
    return 'jksmp() -> Delete-1 jackknife variance estimate.'
