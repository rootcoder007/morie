# morie.fn -- function file (rootcoder007/morie)
"""Double centering matrix B"""

# double_center was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .hmmds import double_center  # noqa: E402,F401

doub = double_center


def cheatsheet() -> str:
    return "double_center({}) -> Double centering matrix B"
