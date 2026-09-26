"""ICC(1,1) one-way random effects."""

__all__ = ["icc_one_way"]


# icc_one_way was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .icc1 import icc_one_way  # noqa: E402,F401


def cheatsheet():
    return "icc31c: ICC(1,1) one-way random effects"
