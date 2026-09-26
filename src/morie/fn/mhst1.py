"""Mantel-Haenszel pooled OR."""

__all__ = ["mantel_haenszel_or"]


# mantel_haenszel_or was a placeholder that shadowed the real implementation of the same
# name; it now is that implementation.
from .mhors import mantel_haenszel_or  # noqa: E402,F401


def cheatsheet():
    return "mhst1: Mantel-Haenszel pooled OR"
