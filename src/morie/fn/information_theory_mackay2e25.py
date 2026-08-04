# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Posterior over a discrete urn hypothesis.

MacKay (2003) eq. (2.25)-(2.26), p. 28
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["urnpost", "information_theory_mackay_chapter_2_equation_25"]

_METHOD = "Posterior over a discrete urn hypothesis"


def urnpost(nb, ntot, nurns=10):
    """Posterior over a discrete urn hypothesis.

    (2.25)-(2.26) p.28 -- posterior over the urn index u.

    Parameters
    ----------
    nb : as documented for the shelf core
        See ``morie.fn._itila.urnpost``.
    ntot : as documented for the shelf core
        See ``morie.fn._itila.urnpost``.
    nurns : as documented for the shelf core
        See ``morie.fn._itila.urnpost``.

    Returns
    -------
    result : RichResult
        Payload keys: evidence, map, prior.

    References
    ----------
    MacKay (2003) eq. (2.25)-(2.26), p. 28
    """
    res = _core.urnpost(nb=nb, ntot=ntot, nurns=nurns)
    return RichResult(
        title=_METHOD,
        summary_lines=[("evidence", res["evidence"]), ("map", res["map"]), ("prior", res["prior"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_2_equation_25 = urnpost


def cheatsheet():
    return "urnpost: Posterior over a discrete urn hypothesis -- MacKay (2003) eq. (2.25)-(2.26), p. 28"
