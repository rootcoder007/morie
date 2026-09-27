"""Bliss points from preference ratings (external unfolding, PREFMAP ideal-point model) and
kernel-smoothed utility surfaces.

Carroll, J. D. (1972). Individual differences and multidimensional scaling. In Multidimensional
Scaling: Theory and Applications in the Behavioral Sciences, vol. 1, 105-155 (PREFMAP).
Nadaraya, E. A. (1964). On estimating regression. Theory of Probability and its Applications 9,
141-142. Watson, G. S. (1964). Smooth regression analysis. Sankhya A 26, 359-372.
"""

import math

from ._qpcore import solve
from ._richresult import RichResult

__all__ = ["bliss_points"]


def bliss_points(ratings, stimuli, grid=None, bandwidth=None):
    r"""Ideal points of raters from their ratings of stimuli at known positions.

    The ideal-point model r_ij = a_i - b_i ||z_j - x_i||^2 + e_ij is linear in
    (1, z_j, ||z_j||^2): r_ij = c0 + c . z_j + c2 ||z_j||^2 with c2 = -b_i and c = 2 b_i x_i, so
    ordinary least squares per rater gives x_i = -c / (2 c2) (Carroll 1972, PREFMAP phase III).
    Raters with c2 >= 0 have an anti-ideal point (a disliked location) and are flagged.
    With ``grid``, each rater's utility surface is the Nadaraya-Watson smooth of the ratings,
    u_i(g) = sum_j K(||g - z_j|| / h) r_ij / sum_j K(...), Gaussian K, bandwidth h (default the
    median inter-stimulus distance).

    Parameters
    ----------
    ratings : raters x stimuli matrix
    stimuli : list of stimulus positions
    grid : list of points, optional
    bandwidth : float, optional

    Returns
    -------
    RichResult
        Keys: ideal (per rater, None when anti-ideal), salience b_i, anti_ideal (bool), r2,
        surface (raters x grid, with ``grid``).

    References
    ----------
    Carroll, J. D. (1972). In Multidimensional Scaling, vol. 1, 105-155.
    Nadaraya, E. A. (1964). Theory of Probability and its Applications 9, 141-142.

    Examples
    --------
    >>> z = [[0.0], [1.0], [2.0], [3.0]]
    >>> r = bliss_points([[10 - (v[0] - 1.2) ** 2 for v in z]], z)
    >>> round(r["ideal"][0][0], 12)
    1.2
    """
    Z = (
        [[float(t)] for t in stimuli]
        if isinstance(stimuli[0], (int, float))
        else [[float(t) for t in r] for r in stimuli]
    )
    d = len(Z[0])
    D = [[1.0] + z + [sum(t * t for t in z)] for z in Z]
    p = d + 2
    XtX = [[sum(r[a] * r[b] for r in D) for b in range(p)] for a in range(p)]
    ideal, sal, anti, r2 = [], [], [], []
    for row in ratings:
        r = [float(v) for v in row]
        Xty = [sum(D[j][a] * r[j] for j in range(len(r))) for a in range(p)]
        c = solve(XtX, Xty)
        c2 = c[-1]
        anti.append(c2 >= 0)
        sal.append(-c2)
        ideal.append(None if c2 == 0 else [-c[1 + q] / (2 * c2) for q in range(d)])
        fit = [sum(D[j][a] * c[a] for a in range(p)) for j in range(len(r))]
        m = sum(r) / len(r)
        sst = sum((v - m) ** 2 for v in r)
        r2.append(1 - sum((v - f) ** 2 for v, f in zip(r, fit)) / sst if sst > 0 else 1.0)
    out = {"ideal": ideal, "salience": sal, "anti_ideal": anti, "r2": r2}
    if grid is not None:
        G = [[float(t)] for t in grid] if isinstance(grid[0], (int, float)) else [[float(t) for t in g] for g in grid]
        if bandwidth is None:
            ds = sorted(math.dist(Z[a], Z[b]) for a in range(len(Z)) for b in range(a + 1, len(Z)))
            h = ds[len(ds) // 2]
        else:
            h = float(bandwidth)
        surf = []
        for row in ratings:
            s = []
            for g in G:
                w = [math.exp(-0.5 * (math.dist(g, z) / h) ** 2) for z in Z]
                s.append(sum(wi * float(v) for wi, v in zip(w, row)) / sum(w))
            surf.append(s)
        out["surface"] = surf
        out["bandwidth"] = h
    return RichResult(title="Bliss points", summary_lines=[("raters", len(ideal))], payload=out)


def cheatsheet():
    return "svbliss: PREFMAP ideal points from ratings and kernel-smoothed utility surfaces"
