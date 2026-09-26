"""Anisotropy ratio estimation"""

# anisotropy_ratio was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .sganr import anisotropy_ratio  # noqa: E402,F401

anis = anisotropy_ratio


def cheatsheet() -> str:
    return "anisotropy_ratio({}) -> Anisotropy ratio estimation"
