"""Johansen cointegration test -- re-export of the canonical implementation."""

# This module used to carry its own body: a verbatim one-sample
# Kolmogorov-Smirnov test against a fitted normal, pasted in by the
# generator and unrelated to johansen_test. A correct implementation already
# lived in `johcg`, and `_lazy_map.json` already resolved
# `johansen_test` there, so the two would silently disagree depending on
# whether a caller reached the function by module path or by name.
# Rather than write a second, divergent copy of the method, this module
# is now a re-export of the canonical one.

from .johcg import johansen_test

__all__ = ["johansen_test"]


def cheatsheet():
    return "johnsen: alias of morie.fn.johcg.johansen_test -- Johansen cointegration test"
