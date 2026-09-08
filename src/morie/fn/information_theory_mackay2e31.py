# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Posterior predictive probability for the next draw.

MacKay (2003) eq. (2.29)-(2.31), p. 29
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["urnpred", "information_theory_mackay_chapter_2_equation_31"]

_METHOD = "Posterior predictive probability for the next draw"


def urnpred(nb, ntot, nurns=10):
    """Posterior predictive probability for the next draw.

    (2.29)-(2.31) p.29 -- predictive P(next ball black).

    Parameters
    ----------
    nb : as documented for the shelf core
        See ``morie.fn._itila.urnpred``.
    ntot : as documented for the shelf core
        See ``morie.fn._itila.urnpred``.
    nurns : as documented for the shelf core
        See ``morie.fn._itila.urnpred``.

    Returns
    -------
    result : RichResult
        Payload keys: p, pnot, pmap.

    References
    ----------
    MacKay (2003) eq. (2.29)-(2.31), p. 29
    """
    res = _core.urnpred(nb=nb, ntot=ntot, nurns=nurns)
    return RichResult(
        title=_METHOD,
        summary_lines=[("p", res["p"]), ("pnot", res["pnot"]), ("pmap", res["pmap"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_2_equation_31 = urnpred


def cheatsheet():
    return "urnpred: Posterior predictive probability for the next draw -- MacKay (2003) eq. (2.29)-(2.31), p. 29"
