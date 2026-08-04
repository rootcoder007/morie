# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian approximation to the central binomial coefficient.

MacKay (2003) eq. (1.40), p. 17
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["cbcapx", "information_theory_mackay_chapter_1_equation_40"]

_METHOD = "Gaussian approximation to the central binomial coefficient"


def cbcapx(n):
    """Gaussian approximation to the central binomial coefficient.

    (1.40) p.17 -- Gaussian approximation to the central binomial.

    Parameters
    ----------
    n : as documented for the shelf core
        See ``morie.fn._itila.cbcapx``.

    Returns
    -------
    result : RichResult
        Payload keys: approx, exact, relerr.

    References
    ----------
    MacKay (2003) eq. (1.40), p. 17
    """
    res = _core.cbcapx(n=n)
    return RichResult(
        title=_METHOD,
        summary_lines=[("approx", res["approx"]), ("exact", res["exact"]), ("relerr", res["relerr"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_1_equation_40 = cbcapx


def cheatsheet():
    return "cbcapx: Gaussian approximation to the central binomial coefficient -- MacKay (2003) eq. (1.40), p. 17"
