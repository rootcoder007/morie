"""Sobel test for an indirect effect -- re-export of the canonical implementation."""

# This module used to carry its own body: a verbatim one-sample
# Kolmogorov-Smirnov test against a fitted normal, pasted in by the
# generator and unrelated to sobel_test. A correct implementation already
# lived in `sobel`, and `_lazy_map.json` already resolved
# `sobel_test` there, so the two would silently disagree depending on
# whether a caller reached the function by module path or by name.
# Rather than write a second, divergent copy of the method, this module
# is now a re-export of the canonical one.

from .sobel import sobel_test

__all__ = ["sobel_test"]


def cheatsheet():
    return "sobtst: alias of morie.fn.sobel.sobel_test -- Sobel test for an indirect effect"
