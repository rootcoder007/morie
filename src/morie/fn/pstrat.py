# morie.fn -- function file (rootcoder007/morie)
"""Post-stratification weights to match population distribution."""

from morie.survey import poststratification_weights as _fn

pstrat = _fn
poststratification_weights = _fn


def cheatsheet() -> str:
    return "pstrat() -> Post-stratification weights to match population distribution"
