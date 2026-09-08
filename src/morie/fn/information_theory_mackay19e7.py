# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dynamic-equilibrium fitness-variance factor.

MacKay (2003) eq. (19.7), p. 271
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["sexbeta", "information_theory_mackay_chapter_19_equation_7"]

_METHOD = "Dynamic-equilibrium fitness-variance factor"


def sexbeta(gamma):
    """Dynamic-equilibrium fitness-variance factor.

    (19.7) p.271 -- dynamic-equilibrium variance factor 1/(1 - gamma).

    Parameters
    ----------
    gamma : as documented for the shelf core
        See ``morie.fn._itila.sexbeta``.

    Returns
    -------
    result : RichResult
        Payload keys: onepbeta, beta.

    References
    ----------
    MacKay (2003) eq. (19.7), p. 271
    """
    res = _core.sexbeta(gamma=gamma)
    return RichResult(
        title=_METHOD,
        summary_lines=[("onepbeta", res["onepbeta"]), ("beta", res["beta"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_19_equation_7 = sexbeta


def cheatsheet():
    return "sexbeta: Dynamic-equilibrium fitness-variance factor -- MacKay (2003) eq. (19.7), p. 271"
