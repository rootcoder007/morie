# morie.fn -- function file (rootcoder007/morie)
"""Beta function."""

from ._sci_core import beta as _b


def betafn(a, b):
    """Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)."""
    return float(_b(a, b))
