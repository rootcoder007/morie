"""Monte Carlo spatial significance test with simulation envelopes."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._rng import random_uniform


def _inverse_distance_moran(z, coords):
    """Moran's I with inverse-distance weights w_ij = 1/d_ij, i != j."""
    n = len(z)
    m = 0.0
    for v in z:
        m += v
    m /= n
    d = [v - m for v in z]
    s0 = 0.0
    num = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            if dist <= 0.0:
                raise ValueError("coincident locations: inverse-distance weight is infinite")
            w = 1.0 / dist
            s0 += w
            num += w * d[i] * d[j]
    den = 0.0
    for v in d:
        den += v * v
    return n / s0 * num / den


def _q7(xs, p):
    h = (len(xs) - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def monte_carlo_spatial_test(Z, coords, stat_fn=None, n_sim=999, seed=42):
    """Monte Carlo (random relabelling) test for spatial pattern.

    Under the null of no spatial structure every assignment of the observed
    values to the locations is equally likely, so the statistic is
    recomputed on ``n_sim`` random permutations of ``Z`` over the fixed
    ``coords`` and the observed value is ranked among them (Besag and
    Diggle 1977; Hope 1968). The default statistic is Moran's I with
    inverse-distance weights,
    ``I = (n / S0) sum_ij w_ij (z_i - zbar)(z_j - zbar) / sum_i (z_i - zbar)^2``,
    ``w_ij = 1 / d_ij``. A statistic that ignores the locations -- the
    sample variance this function used to default to -- is invariant to the
    permutations, so its test can never reject.

    Permutations are Fisher-Yates shuffles driven by the package's Philox
    stream (``seed``), identical in the R arm.

    .. epigraph:: There is no royal road to geometry. -- Euclid

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    stat_fn : callable, optional
        Statistic ``(Z, coords) -> float``. Defaults to inverse-distance
        Moran's I.
    n_sim : int
        Number of permutations.
    seed : int
        Philox seed for the permutations.

    Returns
    -------
    DescriptiveResult
        ``value`` = observed statistic; ``extra``: ``observed``,
        ``envelope_lo`` / ``envelope_hi`` (2.5 and 97.5 percent points of
        the permutation distribution), ``p_value`` (upper tail,
        ``(1 + #{T_b >= T}) / (n_sim + 1)``), ``p_two_sided``, ``n_sim``,
        ``significant`` (observed outside the envelope), ``sim_mean``.

    References
    ----------
    Besag, J. and Diggle, P. J. (1977). Simple Monte Carlo tests for spatial
    pattern. Applied Statistics 26, 327-333.
    Hope, A. C. A. (1968). A simplified Monte Carlo significance test
    procedure. JRSS B 30, 582-598.
    Cliff, A. D. and Ord, J. K. (1981). Spatial Processes. Pion.

    Examples
    --------
    >>> xy = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    >>> r = monte_carlo_spatial_test([1.0, 1.2, 0.9, 2.0, 2.4, 2.2], xy, n_sim=19)
    >>> round(r.value, 12), r.extra["p_value"]
    (-0.151622612225, 0.3)
    """
    z = [float(v) for v in (Z.tolist() if hasattr(Z, "tolist") else Z)]
    C = coords.tolist() if hasattr(coords, "tolist") else [list(r) for r in coords]
    C = [[float(a) for a in r] for r in C]
    n = len(z)
    if len(C) != n:
        raise ValueError("Z and coords must have the same number of rows")
    if n < 3:
        raise ValueError("need at least 3 locations")
    n_sim = int(n_sim)
    if n_sim < 1:
        raise ValueError("n_sim must be at least 1")
    stat = _inverse_distance_moran if stat_fn is None else stat_fn
    observed = float(stat(z, C))
    u = random_uniform(n_sim * (n - 1), seed=seed, stream=0)
    sims = []
    pos = 0
    for _ in range(n_sim):
        perm = list(z)
        for i in range(n - 1, 0, -1):
            j = int(math.floor(float(u[pos]) * (i + 1)))
            pos += 1
            perm[i], perm[j] = perm[j], perm[i]
        sims.append(float(stat(perm, C)))
    ge = sum(1 for t in sims if t >= observed)
    le = sum(1 for t in sims if t <= observed)
    p_up = (1.0 + ge) / (n_sim + 1.0)
    p_two = min(1.0, 2.0 * min(p_up, (1.0 + le) / (n_sim + 1.0)))
    ss = sorted(sims)
    lo = _q7(ss, 0.025)
    hi = _q7(ss, 0.975)
    mean_sim = 0.0
    for t in sims:
        mean_sim += t
    mean_sim /= n_sim
    return DescriptiveResult(
        name="monte_carlo_spatial_test",
        value=observed,
        extra={
            "observed": observed,
            "envelope_lo": lo,
            "envelope_hi": hi,
            "p_value": p_up,
            "p_two_sided": p_two,
            "n_sim": n_sim,
            "sim_mean": mean_sim,
            "significant": observed < lo or observed > hi,
            "statistic": "inverse-distance Moran's I" if stat_fn is None else "user",
        },
    )


sgmci = monte_carlo_spatial_test


def cheatsheet() -> str:
    return "monte_carlo_spatial_test({}) -> permutation test (default Moran's I, inverse distance) with envelopes"
