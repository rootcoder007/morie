"""Tajima's D -- re-export of the canonical implementation."""

# This module used to carry its own body: a verbatim one-sample
# Kolmogorov-Smirnov test against a fitted normal, pasted in by the
# generator and unrelated to tajimas_d. A correct implementation already
# lived in `tajd`, and `_lazy_map.json` already resolved
# `tajimas_d` there, so the two would silently disagree depending on
# whether a caller reached the function by module path or by name.
# Rather than write a second, divergent copy of the method, this module
# is now a re-export of the canonical one.

from .tajd import tajimas_d

__all__ = ["tajimas_d"]


def cheatsheet():
    return "taji_d: alias of morie.fn.tajd.tajimas_d -- Tajima's D"
