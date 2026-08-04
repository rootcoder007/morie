# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian sum identity behind the central binomial approximation.

MacKay (2003) eq. (1.41), p. 17
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["binsumga", "information_theory_mackay_chapter_1_equation_41"]

_METHOD = "Gaussian sum identity behind the central binomial approximation"


def binsumga(n):
    """Gaussian sum identity behind the central binomial approximation.

    (1.41) p.17 -- the Gaussian sum that proves (1.40).

    MacKay writes ``sigma = sqrt(N/4)`` in the running text but then
    uses ``sqrt(2 pi sigma)``; only the reading ``sigma = N/4`` (the
    VARIANCE) makes (1.41) agree with (1.40).  This routine takes the
    variance reading and reports both quantities so the discrepancy is
    visible rather than silently resolved.

    Parameters
    ----------
    n : as documented for the shelf core
        See ``morie.fn._itila.binsumga``.

    Returns
    -------
    result : RichResult
        Payload keys: total, gsum, var.

    References
    ----------
    MacKay (2003) eq. (1.41), p. 17
    """
    res = _core.binsumga(n=n)
    return RichResult(
        title=_METHOD,
        summary_lines=[("total", res["total"]), ("gsum", res["gsum"]), ("var", res["var"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_1_equation_41 = binsumga


def cheatsheet():
    return "binsumga: Gaussian sum identity behind the central binomial approximation -- MacKay (2003) eq. (1.41), p. 17"
