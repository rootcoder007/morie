# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Posterior odds as a running product of likelihood ratios.

MacKay (2003) eq. (3.31), p. 63
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["lrprod", "information_theory_mackay_chapter_3_equation_31"]

_METHOD = "Posterior odds as a running product of likelihood ratios"


def lrprod(num, den):
    """Posterior odds as a running product of likelihood ratios.

    (3.31) p.63 -- posterior odds as a running product of ratios.

    Parameters
    ----------
    num : as documented for the shelf core
        See ``morie.fn._itila.lrprod``.
    den : as documented for the shelf core
        See ``morie.fn._itila.lrprod``.

    Returns
    -------
    result : RichResult
        Payload keys: ratio, p1, logratio.

    References
    ----------
    MacKay (2003) eq. (3.31), p. 63
    """
    res = _core.lrprod(num=num, den=den)
    return RichResult(
        title=_METHOD,
        summary_lines=[("ratio", res["ratio"]), ("p1", res["p1"]), ("logratio", res["logratio"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_3_equation_31 = lrprod


def cheatsheet():
    return "lrprod: Posterior odds as a running product of likelihood ratios -- MacKay (2003) eq. (3.31), p. 63"
