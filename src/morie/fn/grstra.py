# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stratified test split by proportional allocation.

Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 2 pp. 60-61
"""

from . import _geron as _core

from ._richresult import RichResult

__all__ = ["stratsplt", "geron_stratified_split"]

_METHOD = "Stratified test split by proportional allocation"


def stratsplt(strata, testratio=0.2):
    """Stratified test split by proportional allocation.

    Stratified test split by proportional allocation, pp. 60-61.

    The book's point is the GUARANTEE, not the shuffle: "the right
    number of instances are sampled from each stratum to guarantee that
    the test set is representative".  Allocation here is exactly that
    -- ``round(n_s * testratio)`` from each stratum -- with the members
    taken in their original order so the result is reproducible without
    a seed.  ``maxdev`` reports the largest gap between a stratum's
    share of the test set and its share of the population, which is the
    quantity the book checks on p. 61.

    Parameters
    ----------
    strata : as documented for the shelf core
        See ``morie.fn._geron.stratsplt``.
    testratio : as documented for the shelf core
        See ``morie.fn._geron.stratsplt``.

    Returns
    -------
    result : RichResult
        Payload keys: ntest, ntrain, maxdev, nstrata.

    References
    ----------
    Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 2 pp. 60-61
    """
    res = _core.stratsplt(strata=strata, testratio=testratio)
    return RichResult(
        title=_METHOD,
        summary_lines=[("ntest", res["ntest"]), ("ntrain", res["ntrain"]), ("maxdev", res["maxdev"]), ("nstrata", res["nstrata"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
geron_stratified_split = stratsplt


def cheatsheet():
    return "stratsplt: Stratified test split by proportional allocation"
