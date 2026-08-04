"""KPSS stationarity test -- re-export of the canonical implementation."""

# This module used to carry its own body: a verbatim one-sample
# Kolmogorov-Smirnov test against a fitted normal, pasted in by the
# generator and unrelated to kpss_test. A correct implementation already
# lived in `kpss`, and `_lazy_map.json` already resolved
# `kpss_test` there, so the two would silently disagree depending on
# whether a caller reached the function by module path or by name.
# Rather than write a second, divergent copy of the method, this module
# is now a re-export of the canonical one.

from .kpss import kpss_test

__all__ = ["kpss_test"]


def cheatsheet():
    return "kpsstst: alias of morie.fn.kpss.kpss_test -- KPSS stationarity test"
