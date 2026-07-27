# morie.fn -- function file (rootcoder007/morie)
"""Plot-ready coordinates for ideal-point maps."""

import numpy as np

from ._richresult import RichResult

__all__ = ["plot_spatial"]


def plot_spatial(ideal_points, party_labels=None, stimuli=None, stimuli_labels=None):
    r"""Assemble everything a spatial map needs, without plotting.

    Returns the coordinates (padded to 2-D), per-party centroids, the
    axis limits with a 10% margin, and optional stimulus annotations
    -- the data layer of Armstrong's ideal-point figures, kept
    separate from any plotting backend so the R and Python sides can
    render identically.

    Parameters
    ----------
    ideal_points : array-like, shape (n,) or (n, k)
        Legislator coordinates (k <= 2 used).
    party_labels : array-like, optional
        Group label per legislator.
    stimuli : array-like, optional
        Stimulus coordinates to annotate.
    stimuli_labels : sequence, optional
        Names for the stimuli.

    Returns
    -------
    RichResult
        keys: ``coords`` (n, 2), ``parties`` (labels or None),
        ``centroids`` (dict label -> (2,)), ``xlim``, ``ylim``,
        ``stimuli`` ((m, 2) or None), ``stimuli_labels``, ``n``,
        ``method``.

    References
    ----------
    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Sec. 5.3.5.1 (plotting
    legislator estimates), p. 144.
    """
    X = np.atleast_2d(np.asarray(ideal_points, dtype=float))
    if X.shape[0] == 1 and np.ndim(ideal_points) == 1:
        X = X.T
    n = X.shape[0]
    if X.shape[1] == 1:
        X = np.column_stack([X[:, 0], np.zeros(n)])
    coords = X[:, :2]

    parties = None
    centroids = {}
    if party_labels is not None:
        parties = np.asarray(party_labels).ravel()
        if parties.size != n:
            raise ValueError("party_labels must have one entry per legislator.")
        for p in dict.fromkeys(parties.tolist()):
            centroids[p] = coords[parties == p].mean(axis=0)

    S = None
    if stimuli is not None:
        S = np.atleast_2d(np.asarray(stimuli, dtype=float))
        if S.shape[1] == 1:
            S = np.column_stack([S[:, 0], np.zeros(S.shape[0])])
        S = S[:, :2]
        if stimuli_labels is not None and len(stimuli_labels) != S.shape[0]:
            raise ValueError("stimuli_labels must match the number of stimuli.")

    allpts = coords if S is None else np.vstack([coords, S])
    span = np.maximum(allpts.max(axis=0) - allpts.min(axis=0), 1e-9)
    lo = allpts.min(axis=0) - 0.1 * span
    hi = allpts.max(axis=0) + 0.1 * span

    return RichResult(
        payload={
            "coords": coords,
            "parties": parties,
            "centroids": centroids,
            "xlim": (float(lo[0]), float(hi[0])),
            "ylim": (float(lo[1]), float(hi[1])),
            "stimuli": S,
            "stimuli_labels": list(stimuli_labels) if stimuli_labels is not None else None,
            "n": int(n),
            "method": "Spatial-map data layer (coords, centroids, limits, annotations)",
        }
    )


def cheatsheet():
    return "plpol: coords padded to 2-D, party centroids, 10%-margin limits"
