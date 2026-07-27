# morie.fn -- function file (rootcoder007/morie)
"""Bridge observations for cross-period ideal-point comparison."""

import numpy as np

from ._richresult import RichResult
from .procs import procrustes_rotation

__all__ = ["bridge_observations"]


def bridge_observations(ideal_points_periods, bridge_ids):
    r"""Anchor two periods' ideal points on their shared members.

    Ideal points estimated separately per period are only identified
    up to rotation/reflection (and here, translation and scale), so
    cross-period comparison needs *bridges*: actors present in both.
    The second period is mapped onto the first by the similarity
    transform (centre, scale, then Schoenemann rotation) fitted on the
    bridge actors only, and the residual bridge misfit is the honesty
    check -- large misfit means the common-space assumption is doing
    real work.

    Parameters
    ----------
    ideal_points_periods : sequence of two (dict or (ids, coords))
        Per period: a mapping id -> coordinate vector, or a tuple of
        (ids, (m, k) array).
    bridge_ids : sequence
        Actor ids present in both periods.

    Returns
    -------
    RichResult
        keys: ``aligned`` (dict id -> coords, period 2 mapped into
        period 1's space), ``bridge_residual`` (RMS misfit over the
        bridges after alignment), ``scale``, ``rotation``,
        ``n_bridges``, ``method``.

    References
    ----------
    Bailey, M. A. (2007). Comparable preference estimates across time
    and institutions for the court, congress, and presidency. *AJPS*,
    51(3), 433-448. (bridging as the identification device)

    Schoenemann, P. H. (1966). *Psychometrika*, 31(1), 1-10. (the
    orthogonal alignment step)
    """

    def as_map(p):
        if isinstance(p, dict):
            return {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in p.items()}
        ids, coords = p
        coords = np.atleast_2d(np.asarray(coords, dtype=float))
        if coords.shape[0] != len(ids):
            raise ValueError("ids and coordinates disagree in length.")
        return {i: coords[j] for j, i in enumerate(ids)}

    if len(ideal_points_periods) != 2:
        raise ValueError("exactly two periods are supported.")
    p1, p2 = (as_map(p) for p in ideal_points_periods)
    bridges = [b for b in bridge_ids]
    missing = [b for b in bridges if b not in p1 or b not in p2]
    if missing:
        raise ValueError(f"bridge ids absent from a period: {missing}.")
    k = len(next(iter(p1.values())))
    if len(bridges) < k + 1:
        raise ValueError(f"need at least {k + 1} bridges for a {k}-dimensional alignment.")

    A = np.vstack([p1[b] for b in bridges])
    B = np.vstack([p2[b] for b in bridges])
    muA, muB = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - muA, B - muB
    sB = np.linalg.norm(Bc)
    scale = float(np.linalg.norm(Ac) / sB) if sB > 0 else 1.0
    rot = procrustes_rotation(Ac, Bc * scale)
    T = rot["rotation"]

    aligned = {i: (np.atleast_1d(v) - muB) * scale @ T + muA for i, v in p2.items()}
    resid = np.sqrt(np.mean([((aligned[b] - p1[b]) ** 2).sum() for b in bridges]))

    return RichResult(
        payload={
            "aligned": aligned,
            "bridge_residual": float(resid),
            "scale": scale,
            "rotation": T,
            "n_bridges": len(bridges),
            "method": "Bridge alignment: centre + scale + Procrustes on shared actors",
        }
    )


def cheatsheet():
    return "brdgo: fit similarity transform on bridge actors, map period 2 into period 1"
