"""Lilliefors normality test -- re-export of the canonical implementation."""

# This module used to carry its own body: a verbatim one-sample
# Kolmogorov-Smirnov test against a fitted normal, pasted in by the
# generator and unrelated to lilliefors_test. A correct implementation already
# lived in `lilf`, and `_lazy_map.json` already resolved
# `lilliefors_test` there, so the two would silently disagree depending on
# whether a caller reached the function by module path or by name.
# Rather than write a second, divergent copy of the method, this module
# is now a re-export of the canonical one.

from .lilf import lilliefors_test

__all__ = ["lilliefors_test"]


def cheatsheet():
    return "lbrst: alias of morie.fn.lilf.lilliefors_test -- Lilliefors normality test"
