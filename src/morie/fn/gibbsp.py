# morie.fn -- function file (rootcoder007/morie)
"""Maximum pseudolikelihood for Gibbs point processes (Geyer, soft-core, Diggle-Gratton, Strauss)."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import solve, ssum


def _stat(interaction, u, P, skip, par, near):
    """Canonical interaction statistic V(u | x) with the data point ``skip`` removed."""
    d = [math.dist(u, p) if j != skip else math.inf for j, p in enumerate(P)]
    if interaction == "strauss":
        return float(sum(1 for t in d if t <= par["r"]))
    if interaction == "softcore":
        v = 0.0
        for t in d:
            if t < math.inf:
                pr = -((t / par["sigma0"]) ** (-2.0 / par["kappa"]))
                if pr < -25.0:
                    return -math.inf
                v += pr
        return v
    if interaction == "diggle_gratton":
        s = 0.0
        for t in d:
            if t <= par["delta"]:
                return -math.inf
            if t <= par["rho"]:
                s += math.log((t - par["delta"]) / (par["rho"] - par["delta"]))
        return s
    if interaction == "geyer":
        r, sat = par["r"], par["sat"]
        nb = [j for j, t in enumerate(d) if t <= r]
        v = min(sat, len(nb))
        for j in nb:
            tj = near[j] - (1 if skip is not None and math.dist(P[j], P[skip]) <= r else 0)
            v += min(sat, tj + 1) - min(sat, tj)
        return float(v)
    raise ValueError("interaction must be 'strauss', 'geyer', 'softcore' or 'diggle_gratton'")


def gibbs_pseudolikelihood(
    points,
    window,
    interaction,
    *,
    r=None,
    sat=None,
    kappa=None,
    sigma0=None,
    delta=None,
    rho=None,
    nx: int = 12,
    ny: int = 12,
    max_iter: int = 100,
) -> DescriptiveResult:
    """Besag maximum pseudolikelihood for a pairwise or saturated Gibbs process.

    The conditional intensity is ``log lambda(u | x) = log beta + theta V(u | x)``
    with the canonical statistic ``V``:

    - ``strauss`` (radius ``r``): the number of points within ``r``;
      ``gamma = exp(theta)``;
    - ``geyer`` (radius ``r``, saturation ``sat``): the change in
      ``S(x) = sum_i min(sat, t_r(x_i, x))`` when ``u`` is added;
      ``gamma = exp(theta)``;
    - ``softcore`` (index ``kappa``, scale ``sigma0``, by default the
      smallest nearest-neighbour distance): ``-sum_j (d_j/sigma0)^(-2/kappa)``,
      with zero conditional intensity once a single term falls below
      ``-25``; ``sigma = theta^(kappa/2) sigma0`` for ``theta > 0``;
    - ``diggle_gratton`` (hard core ``delta``, range ``rho``):
      ``sum_j log((d_j - delta) / (rho - delta))`` over pairs closer than
      ``rho``, ``-inf`` inside the hard core; ``theta`` is Diggle and
      Gratton's ``kappa``.

    The pseudolikelihood is approximated by the Berman-Turner quadrature
    (data points plus the ``nx`` by ``ny`` grid of tile centres, counting
    weights; data points are evaluated with themselves removed; quadrature
    points of zero conditional intensity drop out) and maximised by
    Newton's method on the standardised statistic. These equal ``spatstat.model::ppm(Q ~ 1, Strauss(r) /
    Geyer(r, sat) / Softcore(kappa) / DiggleGratton(delta, rho), correction
    = "none")`` with ``Q = quadscheme(X, D, method = "grid", ntile = c(nx,
    ny))``, ``D`` the grid of tile centres. Standard errors come from the
    inverse Hessian of the log pseudolikelihood (they treat the fit as
    Poisson and understate the uncertainty of a Gibbs model).

    :param points: (n, 2) event locations inside ``window``.
    :param window: Rectangle ``(xmin, xmax, ymin, ymax)``.
    :param interaction: ``"strauss"``, ``"geyer"``, ``"softcore"`` or
        ``"diggle_gratton"``.
    :param r: Interaction radius (Strauss, Geyer).
    :param sat: Saturation (Geyer).
    :param kappa: Soft-core index in (0, 1).
    :param sigma0: Soft-core scale (default the smallest nearest-neighbour
        distance, as spatstat's self-start).
    :param delta: Hard-core distance (Diggle-Gratton).
    :param rho: Interaction range (Diggle-Gratton).
    :param nx: Dummy grid columns.
    :param ny: Dummy grid rows.
    :param max_iter: Newton iterations.
    :return: DescriptiveResult; ``value`` is ``[log beta, theta]``;
        ``extra`` has ``se``, ``beta``, ``gamma`` (Strauss, Geyer),
        ``sigma`` and ``sigma0`` (soft-core), ``loglik`` and ``n_quadrature``.

    References
    ----------
    Besag, J. (1977). Some methods of statistical analysis for spatial
    data. Bulletin of the International Statistical Institute 47, 77-92.

    Baddeley, A. and Turner, R. (2000). Practical maximum pseudolikelihood
    for spatial point patterns. Australian and New Zealand Journal of
    Statistics 42, 283-322.

    Geyer, C. J. (1999). Likelihood inference for spatial point processes.
    In Stochastic Geometry: Likelihood and Computation, 79-140.

    Ogata, Y. and Tanemura, M. (1984). Likelihood analysis of spatial point
    patterns. Journal of the Royal Statistical Society B 46, 496-518.

    Diggle, P. J. and Gratton, R. J. (1984). Monte Carlo methods of
    inference for implicit statistical models. Journal of the Royal
    Statistical Society B 46, 193-227.

    Examples
    --------
    >>> P = [[0.1, 0.1], [0.3, 0.2], [0.8, 0.7], [0.5, 0.9], [0.2, 0.6], [0.7, 0.3]]
    >>> r = gibbs_pseudolikelihood(P, (0, 1, 0, 1), "geyer", r=0.25, sat=2, nx=4, ny=4)
    >>> [round(t, 9) for t in r.value]
    [3.071395607, -1.024667376]
    """
    P = [[float(v) for v in p] for p in points]
    x0, x1, y0, y1 = (float(v) for v in window)
    n = len(P)
    if interaction == "softcore" and sigma0 is None:
        sigma0 = min(math.dist(P[a], P[b]) for a in range(n) for b in range(n) if a != b)
    par = {"r": r, "sat": sat, "kappa": kappa, "sigma0": sigma0, "delta": delta, "rho": rho}
    near = (
        [sum(1 for k in range(n) if k != j and math.dist(P[j], P[k]) <= r) for j in range(n)]
        if interaction == "geyer"
        else None
    )
    dummy = [[x0 + (a + 0.5) * (x1 - x0) / nx, y0 + (b + 0.5) * (y1 - y0) / ny] for b in range(ny) for a in range(nx)]
    quad = P + dummy

    def tile(p):
        return min(int((p[0] - x0) / (x1 - x0) * nx), nx - 1), min(int((p[1] - y0) / (y1 - y0) * ny), ny - 1)

    tiles = [tile(p) for p in quad]
    counts = {}
    for t in tiles:
        counts[t] = counts.get(t, 0) + 1
    tarea = (x1 - x0) * (y1 - y0) / (nx * ny)
    w = [tarea / counts[t] for t in tiles]
    V = [_stat(interaction, u, P, i if i < n else None, par, near) for i, u in enumerate(quad)]
    if any(v == -math.inf for v in V[:n]):
        raise ValueError("the data violate the hard core")
    keep = [i for i in range(len(quad)) if V[i] > -math.inf]
    m = ssum(V[i] for i in keep) / len(keep)
    sd = math.sqrt(ssum((V[i] - m) ** 2 for i in keep) / len(keep)) or 1.0
    Z = {i: [1.0, (V[i] - m) / sd] for i in keep}
    y = {i: 1.0 if i < n else 0.0 for i in keep}
    th = [math.log(n / ((x1 - x0) * (y1 - y0))), 0.0]
    for _ in range(max_iter):
        lam = {i: math.exp(th[0] + th[1] * Z[i][1]) for i in keep}
        g = [ssum(Z[i][a] * (y[i] - w[i] * lam[i]) for i in keep) for a in range(2)]
        H = [[ssum(w[i] * lam[i] * Z[i][a] * Z[i][b] for i in keep) for b in range(2)] for a in range(2)]
        step = solve(H, g)
        th = [th[0] + step[0], th[1] + step[1]]
        if max(abs(t) for t in step) < 1e-12:
            break
    lam = {i: math.exp(th[0] + th[1] * Z[i][1]) for i in keep}
    H = [[ssum(w[i] * lam[i] * Z[i][a] * Z[i][b] for i in keep) for b in range(2)] for a in range(2)]
    C = [solve(H, [1.0, 0.0]), solve(H, [0.0, 1.0])]
    # back to the raw statistic: theta = A th with A = [[1, -m/sd], [0, 1/sd]]
    A = [[1.0, -m / sd], [0.0, 1.0 / sd]]
    theta = [th[0] - th[1] * m / sd, th[1] / sd]
    cov = [[ssum(A[a][k] * C[k][m] * A[b][m] for k in range(2) for m in range(2)) for b in range(2)] for a in range(2)]
    se = [math.sqrt(cov[0][0]), math.sqrt(cov[1][1])]
    extra = {
        "se": se,
        "beta": math.exp(theta[0]),
        "loglik": ssum(th[0] + th[1] * Z[i][1] for i in range(n)) - ssum(w[i] * lam[i] for i in keep),
        "n_quadrature": len(keep),
    }
    if interaction in ("strauss", "geyer"):
        extra["gamma"] = math.exp(theta[1])
    if interaction == "softcore":
        extra["sigma"] = theta[1] ** (kappa / 2.0) * sigma0 if theta[1] > 0 else float("nan")
        extra["sigma0"] = sigma0
    return DescriptiveResult(name="gibbs_pseudolikelihood", value=theta, extra=extra)


gibbsp = gibbs_pseudolikelihood


def cheatsheet() -> str:
    return "gibbs_pseudolikelihood(points, window, interaction) -> Geyer / soft-core / Diggle-Gratton / Strauss MPLE"
