# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Marginal likelihood of a straight-line regression model.

MacKay (2003) eq. (28.22) and Exercise 28.2, p. 352
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["linevid", "information_theory_mackay_chapter_28_equation_22"]

_METHOD = "Marginal likelihood of a straight-line regression model"


def linevid(x, t, sigma=1.0, slope=True, priorsd=1.0):
    """Marginal likelihood of a straight-line regression model.

    (28.22) p.352, Exercise 28.2 -- evidence for a straight-line model.

    ``slope=False`` is H1 (horizontal line, w1 = 0); ``slope=True`` is
    H2 (w1 free with a Normal(0, priorsd^2) prior).  Closed form:
    t ~ Normal(0, priorsd^2 X X' + sigma^2 I).

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._itila.linevid``.
    t : as documented for the shelf core
        See ``morie.fn._itila.linevid``.
    sigma : as documented for the shelf core
        See ``morie.fn._itila.linevid``.
    slope : as documented for the shelf core
        See ``morie.fn._itila.linevid``.
    priorsd : as documented for the shelf core
        See ``morie.fn._itila.linevid``.

    Returns
    -------
    result : RichResult
        Payload keys: logevidence, quadform, logdet.

    References
    ----------
    MacKay (2003) eq. (28.22) and Exercise 28.2, p. 352
    """
    res = _core.linevid(x=x, t=t, sigma=sigma, slope=slope, priorsd=priorsd)
    return RichResult(
        title=_METHOD,
        summary_lines=[("logevidence", res["logevidence"]), ("quadform", res["quadform"]), ("logdet", res["logdet"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_28_equation_22 = linevid


def cheatsheet():
    return "linevid: Marginal likelihood of a straight-line regression model -- MacKay (2003) eq. (28.22) and Exercise 28.2, p. 352"
