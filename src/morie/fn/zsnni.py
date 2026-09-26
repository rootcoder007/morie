"""Natural neighbor interpolation"""

# natural_neighbor was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .sintf import natural_neighbor  # noqa: E402,F401

natu = natural_neighbor


def cheatsheet() -> str:
    return "natural_neighbor({}) -> Natural neighbor interpolation"
