# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimum-cost detection-to-track assignment.

Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 12 p. 476 (objective stated in words; no formula printed)
"""

from . import _geron as _core

from ._richresult import RichResult

__all__ = ["trkassign", "geron_object_tracking"]

_METHOD = "Minimum-cost detection-to-track assignment"


def trkassign(posdist, appdist=None, weight=0.5, maxn=8):
    """Minimum-cost detection-to-track assignment.

    Minimum-cost detection-to-track assignment, p. 476.

    p. 476 describes DeepSORT in words and prints no formula, but it
    does state the objective the assignment step solves: it "finds the
    combination of mappings that minimizes the distance between the
    detections and the predicted positions of tracked objects, while
    also minimizing the appearance discrepancy".  That objective is
    what is implemented -- the combined cost
    ``(1 - weight) * position + weight * appearance`` minimized over
    all one-to-one mappings.  The Kalman prediction step and the
    appearance network are the CALLER's; this routine takes their
    output as the two cost matrices.

    ponytail: exact optimum by exhaustive search over permutations,
    capped at ``maxn`` tracks.  The Hungarian algorithm the book names
    gives the same answer in O(n^3) -- swap it in if n ever exceeds a
    handful.

    Parameters
    ----------
    posdist : as documented for the shelf core
        See ``morie.fn._geron.trkassign``.
    appdist : as documented for the shelf core
        See ``morie.fn._geron.trkassign``.
    weight : as documented for the shelf core
        See ``morie.fn._geron.trkassign``.
    maxn : as documented for the shelf core
        See ``morie.fn._geron.trkassign``.

    Returns
    -------
    result : RichResult
        Payload keys: cost, nmatched, meancost.

    References
    ----------
    Geron, A. (2026). Hands-On Machine Learning with Scikit-Learn and PyTorch. O'Reilly, ch. 12 p. 476 (objective stated in words; no formula printed)
    """
    res = _core.trkassign(posdist=posdist, appdist=appdist, weight=weight, maxn=maxn)
    return RichResult(
        title=_METHOD,
        summary_lines=[("cost", res["cost"]), ("nmatched", res["nmatched"]), ("meancost", res["meancost"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
geron_object_tracking = trkassign


def cheatsheet():
    return "trkassign: Minimum-cost detection-to-track assignment"
