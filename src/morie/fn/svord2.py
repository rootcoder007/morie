"""Ordered Logit Spatial Model."""

from . import _array_core as np
from ._containers import SpatialResult


def svord2(voter, candidates, *, beta=1.0):
    """Spatial-voting choice probabilities under a conditional logit.

    With Euclidean distance d_j from the voter's ideal point to candidate
    j, utility u_j = -beta d_j and McFadden's conditional logit gives

        P_j = exp(u_j) / sum_k exp(u_k).

    Despite the module title this is not an ordered-logit model: there
    are no cut points and the candidates are unordered alternatives. The
    ``statistic`` is the probability of choosing the nearest candidate.

    Returns
    -------
    SpatialResult
        ``statistic`` = P(nearest); ``extra["probabilities"]`` = P_j in
        candidate order.
    """
    voter = np.asarray(voter, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    dists = np.linalg.norm(candidates - voter, axis=1)
    ranks = dists.argsort().argsort()
    utils = -beta * dists
    exp_u = np.exp(utils - utils.max())
    probs = exp_u / exp_u.sum()
    stat = float(probs[ranks == 0][0]) if len(probs[ranks == 0]) > 0 else float(probs[0])
    _extra = {"probabilities": probs.tolist()}

    return SpatialResult(
        name="Ordered Logit Spatial Model",
        statistic=float(stat),
        extra=_extra,
    )


svord2 = svord2  # alias


def cheatsheet() -> str:
    return "svord2({}) -> Ordered Logit Spatial Model."
