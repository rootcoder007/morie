# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.14: Sentence Mover's Distance."""

from ._richresult import RichResult
from .km123 import kamath_ch8_moverscore_distance

__all__ = ["kamath_ch8_smd"]


def kamath_ch8_smd(x, y, E=None):
    r"""SMD(x^n, y^n) = || E(x_1^{l_x}) - E(y_1^{l_y}) ||.

    When n exceeds the sentence length each side collapses to a single
    n-gram and MoverScore degenerates to one Euclidean distance --
    exactly Eq 8.11 -- so this delegates to ``morie.fn.km123``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.14, printed
    p. 326.

    Examples
    --------
    >>> out = kamath_ch8_smd([0.0, 0.0], [3.0, 4.0])
    >>> out["estimate"]
    5.0
    """
    d = kamath_ch8_moverscore_distance(x, y, E=E)
    return RichResult(payload={
        "estimate": d["estimate"], "difference": d["difference"],
        "n": d["n"],
        "method": "Sentence Mover's Distance (Kamath Eq 8.14; the "
                  "Eq 8.11 core in km123)"})


def cheatsheet():
    return "km126: sentence-level MoverScore = one L2 distance"


# compact alias per ledger/NAMING.md
kamathch8smd = kamath_ch8_smd
