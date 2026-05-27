# morie.fn -- function file (rootcoder007/morie)
"""Digamma function."""

from scipy.special import digamma
def digamf(x):
    """Digamma function ψ(x) = d/dx ln Γ(x)."""
    return float(digamma(x)) if isinstance(x, (int, float)) else digamma(x)
