# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bent-coin likelihood.

MacKay (2003) eq. (3.8), p. 51
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["bcoinlik", "information_theory_mackay_chapter_3_equation_8"]

_METHOD = "Bent-coin likelihood"


def bcoinlik(pa, fa, fb):
    """Bent-coin likelihood.

    (3.8) p.51 -- bent-coin likelihood P(s | p_a, F, H1).

    Parameters
    ----------
    pa : as documented for the shelf core
        See ``morie.fn._itila.bcoinlik``.
    fa : as documented for the shelf core
        See ``morie.fn._itila.bcoinlik``.
    fb : as documented for the shelf core
        See ``morie.fn._itila.bcoinlik``.

    Returns
    -------
    result : RichResult
        Payload keys: likelihood, loglik.

    References
    ----------
    MacKay (2003) eq. (3.8), p. 51
    """
    res = _core.bcoinlik(pa=pa, fa=fa, fb=fb)
    return RichResult(
        title=_METHOD,
        summary_lines=[("likelihood", res["likelihood"]), ("loglik", res["loglik"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_3_equation_8 = bcoinlik


def cheatsheet():
    return "bcoinlik: Bent-coin likelihood -- MacKay (2003) eq. (3.8), p. 51"
