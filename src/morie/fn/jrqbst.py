"""Jarque-Bera normality test -- re-export of the canonical implementation."""

# This module used to carry its own body: a verbatim one-sample
# Kolmogorov-Smirnov test against a fitted normal, pasted in by the
# generator and unrelated to jarque_bera. A correct implementation already
# lived in `jarber`, and `_lazy_map.json` already resolved
# `jarque_bera` there, so the two would silently disagree depending on
# whether a caller reached the function by module path or by name.
# Rather than write a second, divergent copy of the method, this module
# is now a re-export of the canonical one.

from .jarber import jarque_bera

__all__ = ["jarque_bera"]


def cheatsheet():
    return "jrqbst: alias of morie.fn.jarber.jarque_bera -- Jarque-Bera normality test"
