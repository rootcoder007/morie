# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decision boundary for logistic regression: theta^T x = 0."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_decision_boundary"]


def geron_decision_boundary(theta, X_grid, fit_intercept=True):
    """
    Decision boundary for logistic regression: theta^T x = 0.

    Formula: {x : theta^T x = 0}

    The boundary is a hyperplane, so membership is decided by the *signed
    distance* ``theta^T x / ||w||`` (with ``w`` the non-intercept part),
    not by the raw score: two models with the same boundary but different
    scales give different scores and identical distances. Points within
    ``tol`` of zero distance are flagged as on the boundary rather than
    silently assigned to a side.

    For a 2-D input the explicit line ``x2 = -(b + w1 x1)/w2`` is returned
    as ``line`` (slope, intercept) when ``w2 != 0``, and the vertical
    boundary ``x1 = -b/w1`` otherwise.

    Parameters
    ----------
    theta : array-like
        Parameters; ``theta[0]`` is the bias when ``fit_intercept``.
    X_grid : array-like, shape (m, n)
        Points to classify (without a bias column).
    fit_intercept : bool, default True

    Returns
    -------
    result : RichResult
        Keys: scores, signed_distance, labels, probabilities, on_boundary,
        normal, margin, line, estimate, n, method.

    Examples
    --------
    The boundary ``x1 + x2 = 1`` splits the unit square's corners:

    >>> r = geron_decision_boundary([-1.0, 1.0, 1.0],
    ...                             [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    >>> r["labels"]
    [0, 1, 0]
    >>> [round(v, 9) for v in r["signed_distance"]]
    [-0.707106781, 0.707106781, 0.0]
    >>> r["on_boundary"]
    [False, False, True]
    >>> [round(v, 6) for v in r["line"]]
    [-1.0, 1.0]

    Probabilities at the boundary are exactly one half:

    >>> round(r["probabilities"][2], 12)
    0.5

    References
    ----------
    Géron Ch 4
    """
    th = np.atleast_1d(np.asarray(theta, dtype=float))
    G = np.atleast_2d(np.asarray(X_grid, dtype=float))
    if th.size == 0 or G.size == 0:
        raise ValueError("geron_decision_boundary: theta and X_grid must be non-empty")
    if not np.all(np.isfinite(th)) or not np.all(np.isfinite(G)):
        raise ValueError("geron_decision_boundary: theta and X_grid must be finite")

    if fit_intercept:
        if th.size != G.shape[1] + 1:
            raise ValueError(
                f"geron_decision_boundary: theta has {th.size} entries but X_grid has {G.shape[1]} columns "
                "(expected columns + 1 with fit_intercept=True)"
            )
        b, w = float(th[0]), th[1:]
    else:
        if th.size != G.shape[1]:
            raise ValueError(f"geron_decision_boundary: theta has {th.size} entries but X_grid has {G.shape[1]} columns")
        b, w = 0.0, th

    nw = float(np.linalg.norm(w))
    if nw == 0:
        raise ValueError("geron_decision_boundary: the weight vector is zero, so theta^T x = 0 defines no hyperplane")

    scores = G @ w + b
    dist = scores / nw
    tol = 1e-12
    on = np.abs(dist) <= tol
    labels = (scores > 0).astype(int)
    probs = 1.0 / (1.0 + np.exp(-scores))

    line = None
    if w.size == 2:
        if w[1] != 0:
            line = [float(-w[0] / w[1]), float(-b / w[1])]
        else:
            line = [float("inf"), float(-b / w[0])]

    return RichResult(
        title="Decision boundary",
        summary_lines=[("Dimensions", int(w.size)), ("On boundary", int(on.sum()))],
        interpretation="Signed distance is scale-free; the raw score is not, which is why it is reported separately.",
        payload={
            "scores": scores.tolist(),
            "signed_distance": dist.tolist(),
            "labels": labels.tolist(),
            "probabilities": probs.tolist(),
            "on_boundary": on.tolist(),
            "normal": (w / nw).tolist(),
            "bias": b,
            "margin": float(np.min(np.abs(dist))),
            "line": line,
            "estimate": float(np.min(np.abs(dist))),
            "n": int(G.shape[0]),
            "method": "hyperplane theta^T x = 0 with signed distances",
        },
    )


def cheatsheet():
    return "hmdbd: Decision boundary for logistic regression: theta^T x = 0"
