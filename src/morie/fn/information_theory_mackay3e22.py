# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayes factor: free-bias coin against a fixed-bias coin.

MacKay (2003) eq. (3.12), (3.20), (3.22), pp. 52-53
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["bcoinbf", "information_theory_mackay_chapter_3_equation_22"]

_METHOD = "Bayes factor: free-bias coin against a fixed-bias coin"


def bcoinbf(fa, fb, p0=0.16666666666666666):
    """Bayes factor: free-bias coin against a fixed-bias coin.

    (3.12), (3.20), (3.22) pp.52-53 -- H1 (free p_a) vs H0 (p_a = p0).

    Parameters
    ----------
    fa : as documented for the shelf core
        See ``morie.fn._itila.bcoinbf``.
    fb : as documented for the shelf core
        See ``morie.fn._itila.bcoinbf``.
    p0 : as documented for the shelf core
        See ``morie.fn._itila.bcoinbf``.

    Returns
    -------
    result : RichResult
        Payload keys: ratio, evidence1, evidence0.

    References
    ----------
    MacKay (2003) eq. (3.12), (3.20), (3.22), pp. 52-53
    """
    res = _core.bcoinbf(fa=fa, fb=fb, p0=p0)
    return RichResult(
        title=_METHOD,
        summary_lines=[("ratio", res["ratio"]), ("evidence1", res["evidence1"]), ("evidence0", res["evidence0"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_3_equation_22 = bcoinbf


def cheatsheet():
    return "bcoinbf: Bayes factor: free-bias coin against a fixed-bias coin -- MacKay (2003) eq. (3.12), (3.20), (3.22), pp. 52-53"
