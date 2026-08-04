# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quadratic (Gaussian) approximation to a posterior.

MacKay (2003) eq. (28.5), p. 344
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["postgapx", "information_theory_mackay_chapter_28_equation_5"]

_METHOD = "Quadratic (Gaussian) approximation to a posterior"


def postgapx(dw, a):
    """Quadratic (Gaussian) approximation to a posterior.

    (28.5) p.344 -- quadratic (Gaussian) approximation to the posterior.

    Parameters
    ----------
    dw : as documented for the shelf core
        See ``morie.fn._itila.postgapx``.
    a : as documented for the shelf core
        See ``morie.fn._itila.postgapx``.

    Returns
    -------
    result : RichResult
        Payload keys: quadform, logratio, ratio.

    References
    ----------
    MacKay (2003) eq. (28.5), p. 344
    """
    res = _core.postgapx(dw=dw, a=a)
    return RichResult(
        title=_METHOD,
        summary_lines=[("quadform", res["quadform"]), ("logratio", res["logratio"]), ("ratio", res["ratio"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_28_equation_5 = postgapx


def cheatsheet():
    return "postgapx: Quadratic (Gaussian) approximation to a posterior -- MacKay (2003) eq. (28.5), p. 344"
