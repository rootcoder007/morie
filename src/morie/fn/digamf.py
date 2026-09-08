# morie.fn -- function file (rootcoder007/morie)
"""Digamma function."""

from ._sci_core import digamma


def digamf(x):
    """Digamma function ψ(x) = d/dx ln Γ(x)."""
    return float(digamma(x)) if isinstance(x, (int, float)) else digamma(x)
