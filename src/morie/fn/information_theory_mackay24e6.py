# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian log likelihood in sufficient statistics.

MacKay (2003) eq. (24.5)-(24.6), p. 319
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["gllsuff", "information_theory_mackay_chapter_24_equation_6"]

_METHOD = "Gaussian log likelihood in sufficient statistics"


def gllsuff(xbar, s, n, mu, sigma):
    """Gaussian log likelihood in sufficient statistics.

    (24.5)-(24.6) p.319 -- Gaussian log likelihood via (xbar, S).

    Parameters
    ----------
    xbar : as documented for the shelf core
        See ``morie.fn._itila.gllsuff``.
    s : as documented for the shelf core
        See ``morie.fn._itila.gllsuff``.
    n : as documented for the shelf core
        See ``morie.fn._itila.gllsuff``.
    mu : as documented for the shelf core
        See ``morie.fn._itila.gllsuff``.
    sigma : as documented for the shelf core
        See ``morie.fn._itila.gllsuff``.

    Returns
    -------
    result : RichResult
        Payload keys: loglik, sigman.

    References
    ----------
    MacKay (2003) eq. (24.5)-(24.6), p. 319
    """
    res = _core.gllsuff(xbar=xbar, s=s, n=n, mu=mu, sigma=sigma)
    return RichResult(
        title=_METHOD,
        summary_lines=[("loglik", res["loglik"]), ("sigman", res["sigman"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_24_equation_6 = gllsuff


def cheatsheet():
    return "gllsuff: Gaussian log likelihood in sufficient statistics -- MacKay (2003) eq. (24.5)-(24.6), p. 319"
