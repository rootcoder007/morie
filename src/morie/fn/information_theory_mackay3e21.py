# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Posterior odds from a likelihood ratio and prior odds.

MacKay (2003) eq. (3.21), p. 53
"""

from . import _itila as _core

from ._richresult import RichResult

__all__ = ["postodds", "information_theory_mackay_chapter_3_equation_21"]

_METHOD = "Posterior odds from a likelihood ratio and prior odds"


def postodds(lik1, lik0, prior1=0.5, prior0=0.5):
    """Posterior odds from a likelihood ratio and prior odds.

    (3.21) p.53 -- posterior odds = likelihood ratio x prior odds.

    Parameters
    ----------
    lik1 : as documented for the shelf core
        See ``morie.fn._itila.postodds``.
    lik0 : as documented for the shelf core
        See ``morie.fn._itila.postodds``.
    prior1 : as documented for the shelf core
        See ``morie.fn._itila.postodds``.
    prior0 : as documented for the shelf core
        See ``morie.fn._itila.postodds``.

    Returns
    -------
    result : RichResult
        Payload keys: odds, logodds, p1.

    References
    ----------
    MacKay (2003) eq. (3.21), p. 53
    """
    res = _core.postodds(lik1=lik1, lik0=lik0, prior1=prior1, prior0=prior0)
    return RichResult(
        title=_METHOD,
        summary_lines=[("odds", res["odds"]), ("logodds", res["logodds"]), ("p1", res["p1"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
information_theory_mackay_chapter_3_equation_21 = postodds


def cheatsheet():
    return "postodds: Posterior odds from a likelihood ratio and prior odds -- MacKay (2003) eq. (3.21), p. 53"
