# morie.fn -- function file (rootcoder007/morie)
"""Gamma function."""

from ._sci_core import gamma as _g


def gammfn(x):
    """Gamma function Γ(x) = (x-1)!."""
    return float(_g(x)) if isinstance(x, (int, float)) else _g(x)
