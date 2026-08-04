"""IRT test information function -- re-export of the canonical implementation."""

# This module used to carry its own body: a verbatim one-sample
# Kolmogorov-Smirnov test against a fitted normal, pasted in by the
# generator and unrelated to test_information. A correct implementation already
# lived in `tinfo`, and `_lazy_map.json` already resolved
# `test_information` there, so the two would silently disagree depending on
# whether a caller reached the function by module path or by name.
# Rather than write a second, divergent copy of the method, this module
# is now a re-export of the canonical one.

from .tinfo import test_information

__all__ = ["test_information"]


def cheatsheet():
    return "tstinf: alias of morie.fn.tinfo.test_information -- IRT test information function"
