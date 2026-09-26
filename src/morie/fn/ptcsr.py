# morie.fn -- function file (rootcoder007/morie)
"""Complete Spatial Randomness test"""

# csr_test was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .mrkcsr import csr_test  # noqa: E402,F401

csr_ = csr_test


def cheatsheet() -> str:
    return "csr_test({}) -> Complete Spatial Randomness test"
