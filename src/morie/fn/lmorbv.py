# morie.fn -- function file (rootcoder007/morie)
"""Local bivariate Moran's I with a conditional permutation test (Anselin et al. 2002)."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats
from ._containers import SpatialResult
from ._rng import random_uniform

__all__ = ["local_moran_bivariate"]


def _scale(v):
    n = len(v)
    m = sum(v) / n
    sd = (sum((a - m) ** 2 for a in v) / (n - 1)) ** 0.5
    return [(a - m) / sd for a in v]


def local_moran_bivariate(x, y, W, nsim: int = 199, seed: int = 1, scale: bool = True) -> SpatialResult:
    r"""Local bivariate Moran statistic of each unit.

    .. math::

        I^{B}_i = x_i \sum_j w_{ij} y_j

    with ``x`` and ``y`` standardised when ``scale`` (as
    ``spdep::localmoran_bv``).  Inference is by conditional permutation:
    for unit ``i`` the ``|N(i)|`` neighbour values of ``y`` are redrawn
    ``nsim`` times with replacement from the other ``n - 1`` values of
    ``y``; the draws come from the Philox stream ``i`` of ``seed`` (index
    ``floor(u (n - 1))``), so both morie arms give the same draws, while
    ``spdep`` uses R's generator.  Reported per unit: the permutation mean
    and variance, the deviate ``(I_i - mean) / sqrt(var)``, its two-sided
    normal p-value and the folded pseudo p-value
    ``(min(#{sim >= I_i}, nsim - #{sim >= I_i}) + 1) / (nsim + 1)``.
    Quadrants compare ``x`` with its mean and the (unscaled) spatial lag of
    ``y`` with the lag's mean (values at the mean count as ``Low``).

    :param x: Values (n,) at the focal unit.
    :param y: Values (n,) whose spatial lag is taken.
    :param W: Spatial weights (n, n); every unit needs a neighbour.
    :param nsim: Permutations per unit.
    :param seed: Philox seed.
    :param scale: Standardise ``x`` and ``y`` first.
    :return: :class:`SpatialResult` with ``local_values`` the ``I^B_i``
        and ``extra`` keys ``expected``, ``variance``, ``z``, ``p_normal``,
        ``p_folded``, ``quadrant``.

    References
    ----------
    Anselin, L., Syabri, I. and Smirnov, O. (2002). Visualizing
    multivariate spatial correlation with dynamically linked windows.
    In *New Tools for Spatial Data Analysis: Proceedings of the
    Specialist Meeting*. CSISS, Santa Barbara.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> r = local_moran_bivariate([1.0, 2.0, 4.0, 8.0], [2.0, 1.0, 5.0, 7.0], W, nsim=9)
    >>> [round(v, 6) for v in r.local_values]
    [0.887109, 0.102641, 0.014663, 0.623176]
    """
    xs = [float(v) for v in np.asarray(x, dtype=float).tolist()]
    ys = [float(v) for v in np.asarray(y, dtype=float).tolist()]
    n = len(xs)
    Wl = np.asarray(W, dtype=float).tolist()
    if len(ys) != n or len(Wl) != n or any(len(r) != n for r in Wl):
        raise ValueError("x, y and W must share n")
    nb = [[j for j in range(n) if j != i and Wl[i][j] != 0.0] for i in range(n)]
    if any(len(a) == 0 for a in nb):
        raise ValueError("every unit needs at least one neighbour")
    if int(nsim) < 1:
        raise ValueError("nsim must be at least 1")
    ly0 = [sum(Wl[i][j] * ys[j] for j in nb[i]) for i in range(n)]
    mx, ml = sum(xs) / n, sum(ly0) / n
    quad = [("High" if xs[i] > mx else "Low") + "-" + ("High" if ly0[i] > ml else "Low") for i in range(n)]
    if scale:
        xs, ys = _scale(xs), _scale(ys)
    obs = [xs[i] * sum(Wl[i][j] * ys[j] for j in nb[i]) for i in range(n)]
    ns = int(nsim)
    ex, va, zz, pn, pf = [], [], [], [], []
    for i in range(n):
        yi = ys[:i] + ys[i + 1 :]
        wi = [Wl[i][j] for j in nb[i]]
        c = len(wi)
        u = [float(v) for v in random_uniform(ns * c, seed=seed, stream=i)]
        sims = []
        for s in range(ns):
            acc = 0.0
            for t in range(c):
                acc += yi[min(int(u[s * c + t] * (n - 1)), n - 2)] * wi[t]
            sims.append(xs[i] * acc)
        m = sum(sims) / ns
        v = sum((a - m) ** 2 for a in sims) / (ns - 1) if ns > 1 else float("nan")
        z = (obs[i] - m) / v**0.5 if v > 0.0 else float("nan")
        ge = sum(1 for a in sims if a >= obs[i])
        ex.append(m)
        va.append(v)
        zz.append(z)
        pn.append(float(2.0 * stats.norm.sf(abs(z))) if z == z else float("nan"))
        pf.append((min(ge, ns - ge) + 1) / (ns + 1))
    return SpatialResult(
        name="local_moran_bivariate",
        statistic=sum(obs) / n,
        local_values=obs,
        extra={
            "expected": ex,
            "variance": va,
            "z": zz,
            "p_normal": pn,
            "p_folded": pf,
            "quadrant": quad,
            "nsim": ns,
        },
    )


def cheatsheet() -> str:
    return (
        "local_moran_bivariate(x, y, W) -> local bivariate Moran with conditional permutation (spdep::localmoran_bv)."
    )
