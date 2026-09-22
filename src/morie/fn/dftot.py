# morie.fn -- function file (rootcoder007/morie)
"""Total degrees of freedom."""


def dftot(n: int) -> int:
    """Total degrees of freedom: n − 1."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    return n - 1


def cheatsheet() -> str:
    return "dftot(n) -> Total degrees of freedom: n − 1."
