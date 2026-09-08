# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Uniform prior density on the bent-coin bias.

MacKay (2003) eq. (3.9), p. 51
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["bcoinpri", "information_theory_mackay_chapter_3_equation_9"]

_METHOD = "Uniform prior density on the bent-coin bias"


def bcoinpri(pa):
    """Uniform prior density on the bent-coin bias.

    (3.9) p.51 -- the uniform prior density P(p_a | H1) = 1.

    Parameters
    ----------
    pa : as documented for the shelf core
        See ``morie.fn._itila.bcoinpri``.

    Returns
    -------
    result : RichResult
        Payload keys: density, logdensity.

    References
    ----------
    MacKay (2003) eq. (3.9), p. 51
    """
    res = _core.bcoinpri(pa=pa)
    return RichResult(
        title=_METHOD,
        summary_lines=[("density", res["density"]), ("logdensity", res["logdensity"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_3_equation_9 = bcoinpri


def cheatsheet():
    return "bcoinpri: Uniform prior density on the bent-coin bias -- MacKay (2003) eq. (3.9), p. 51"
