# morie.fn -- function file (rootcoder007/morie)
r"""The Strauss process: a point pattern that repels itself.

A Poisson process is the null model in which points ignore each other.
Trees do not ignore each other -- they compete for light -- and neither
do cells, nests or retail outlets. The Strauss process is the smallest
departure from Poisson that says so: the density is

.. math:: f(x) \propto \beta^{n(x)}\,\gamma^{\,s_r(x)},

where :math:`n(x)` counts the points and :math:`s_r(x)` counts the pairs
closer together than the interaction radius :math:`r`. With
:math:`\gamma<1` close pairs are penalised and the pattern is regular;
:math:`\gamma=1` recovers Poisson; :math:`\gamma>1` would reward
clustering, and the density is then NOT integrable -- Kelly and Ripley
(1976) showed the model is undefined there. A fitted
:math:`\hat\gamma>1` is therefore a statement about the data, not a
usable model, and is reported with ``valid_density = False`` rather than
being silently clamped.

The normalising constant of that density is intractable, so fitting uses
Besag's pseudolikelihood -- the product of the conditional intensities
at the observed points -- which Baddeley and Turner's device turns into
an ordinary weighted Poisson regression over a quadrature scheme of the
data points plus a dummy grid:

.. math:: \log\lambda(u\mid x) = \log\beta + t_r(u)\log\gamma ,

with :math:`t_r(u)` the number of data points within :math:`r` of
:math:`u`. The quadrature weights are the tile-counting weights of
Baddeley and Turner (2000): each tile's area shared among the quadrature
points that fall in it.

If ``gamma`` is supplied the interaction is treated as known and only
the log density and the sufficient statistics are returned; if it is
``None`` both parameters are estimated. Both routes are always
computed and reported, because the comparison between an assumed and a
fitted interaction is usually the point of the analysis.

References
----------
Strauss, D. J. (1975) "A model for clustering", *Biometrika* **62**(2),
467-475, doi:10.1093/biomet/62.2.467.

Kelly, F. P. and Ripley, B. D. (1976) "A note on Strauss's model for
clustering", *Biometrika* **63**(2), 357-360,
doi:10.1093/biomet/63.2.357. Why ``gamma > 1`` is not a model.

Besag, J. (1977) "Some methods of statistical analysis for spatial
data", *Bulletin of the International Statistical Institute* **47**(2),
77-92. The pseudolikelihood.

Baddeley, A. and Turner, R. (2000) "Practical maximum pseudolikelihood
for spatial point patterns", *Australian & New Zealand Journal of
Statistics* **42**(3), 283-322, doi:10.1111/1467-842X.00128. The
quadrature device and the counting weights used here.

Ripley, B. D. (1988) *Statistical Inference for Spatial Processes*,
Cambridge University Press, Ch. 4.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["strauss_process"]

_EPS = 1e-12


def strauss_process(coords, r, gamma=None, window=None, nx=12, ny=12,
                    max_iter=100, tol=1e-11):
    r"""Fit or evaluate a Strauss process on a planar point pattern.

    Parameters
    ----------
    coords : array-like, shape ``(n, 2)``
        The point pattern.
    r : float
        Interaction radius. Pairs closer than this are the ones the
        model penalises.
    gamma : float, optional
        Interaction parameter, if known. When supplied the log density
        is evaluated at it; the pseudolikelihood fit is reported either
        way.
    window : sequence of four floats, optional
        ``(xmin, xmax, ymin, ymax)``. Defaults to the bounding box of
        the pattern, which biases the intensity upward -- pass the real
        sampling window when you have it.
    nx, ny : int
        Dummy grid dimensions for the quadrature scheme. More dummy
        points give a closer approximation to the pseudolikelihood at
        linear cost.

    Returns
    -------
    RichResult
        ``beta`` and ``gamma`` (the pseudolikelihood estimates), their
        standard errors from the Poisson information, the sufficient
        statistic ``n_close_pairs``, ``valid_density``, and the log
        pseudolikelihood.
    """
    P = [[float(v) for v in row] for row in k.mat(coords)]
    n = len(P)
    if n == 0:
        raise ValueError("strmkr: an empty pattern carries no information "
                         "about interaction")
    if any(len(p) != 2 for p in P):
        raise ValueError("strmkr: coords must be two-dimensional")
    rr = float(r)
    if rr <= 0.0:
        raise ValueError("strmkr: the interaction radius must be positive")
    nx = int(nx)
    ny = int(ny)
    if nx < 1 or ny < 1:
        raise ValueError("strmkr: the dummy grid must be at least 1 by 1")

    if window is None:
        xs = [p[0] for p in P]
        ys = [p[1] for p in P]
        win = [min(xs), max(xs), min(ys), max(ys)]
        if win[1] - win[0] <= _EPS:
            win[0] -= 0.5
            win[1] += 0.5
        if win[3] - win[2] <= _EPS:
            win[2] -= 0.5
            win[3] += 0.5
        window_source = "bounding box of the pattern"
    else:
        win = [float(v) for v in k.vec(window)]
        if len(win) != 4:
            raise ValueError("strmkr: window must be "
                             "(xmin, xmax, ymin, ymax)")
        if win[1] <= win[0] or win[3] <= win[2]:
            raise ValueError("strmkr: the window has non-positive area")
        window_source = "supplied"
    area = (win[1] - win[0]) * (win[3] - win[2])

    # sufficient statistic: pairs closer together than r
    npairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = P[i][0] - P[j][0]
            dy = P[i][1] - P[j][1]
            if math.sqrt(dx * dx + dy * dy) < rr:
                npairs += 1

    # ---- Baddeley-Turner quadrature: data points plus a dummy grid
    dummy = []
    for a in range(nx):
        for b in range(ny):
            dummy.append([win[0] + (a + 0.5) * (win[1] - win[0]) / nx,
                          win[2] + (b + 0.5) * (win[3] - win[2]) / ny])
    quad = [list(p) for p in P] + dummy
    isdata = [1.0] * n + [0.0] * len(dummy)

    def tile(p):
        a = int((p[0] - win[0]) / (win[1] - win[0]) * nx)
        b = int((p[1] - win[2]) / (win[3] - win[2]) * ny)
        return (min(max(a, 0), nx - 1), min(max(b, 0), ny - 1))

    counts = {}
    tiles = [tile(p) for p in quad]
    for t in tiles:
        counts[t] = counts.get(t, 0) + 1
    tile_area = area / float(nx * ny)
    w = [tile_area / counts[tiles[i]] for i in range(len(quad))]

    # t_r(u): data points within r of u, never counting u against itself
    tstat = []
    for i, u in enumerate(quad):
        c = 0
        for j in range(n):
            if i == j:
                continue
            dx = u[0] - P[j][0]
            dy = u[1] - P[j][1]
            if math.sqrt(dx * dx + dy * dy) < rr:
                c += 1
        tstat.append(float(c))

    # ---- weighted Poisson regression, log link, response z/w
    m = len(quad)
    X = [[1.0, tstat[i]] for i in range(m)]
    yq = [isdata[i] / w[i] for i in range(m)]
    beta = [math.log(max(n, 1) / area), 0.0]
    it = 0
    converged = False
    A = [[0.0, 0.0], [0.0, 0.0]]
    for it in range(1, max_iter + 1):
        eta = [X[i][0] * beta[0] + X[i][1] * beta[1] for i in range(m)]
        mu = [math.exp(max(-500.0, min(500.0, e))) for e in eta]
        # working weights w_i * mu_i, working response eta + (y-mu)/mu
        A = [[0.0, 0.0], [0.0, 0.0]]
        rhs = [0.0, 0.0]
        for i in range(m):
            ww = w[i] * mu[i]
            zi = eta[i] + (yq[i] - mu[i]) / max(mu[i], 1e-300)
            for a in range(2):
                rhs[a] += ww * X[i][a] * zi
                for b in range(2):
                    A[a][b] += ww * X[i][a] * X[i][b]
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if abs(det) < 1e-300:
            raise ValueError("strmkr: the pseudolikelihood information "
                             "matrix is singular -- no quadrature point has "
                             "a close neighbour, so gamma is not identified "
                             "at this radius")
        new = [(A[1][1] * rhs[0] - A[0][1] * rhs[1]) / det,
               (A[0][0] * rhs[1] - A[1][0] * rhs[0]) / det]
        shift = max(abs(new[0] - beta[0]), abs(new[1] - beta[1]))
        beta = new
        if shift < tol:
            converged = True
            break

    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    cov = [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]]
    se = [math.sqrt(max(cov[0][0], 0.0)), math.sqrt(max(cov[1][1], 0.0))]

    eta = [X[i][0] * beta[0] + X[i][1] * beta[1] for i in range(m)]
    mu = [math.exp(max(-500.0, min(500.0, e))) for e in eta]
    logpl = sum(w[i] * (yq[i] * eta[i] - mu[i]) for i in range(m))

    beta_hat = math.exp(beta[0])
    gamma_hat = math.exp(beta[1])
    # Poisson null: the same fit with the interaction term dropped
    logpl_pois = (n * math.log(max(n / area, 1e-300)) - n)
    out = {
        "estimate": [beta_hat, gamma_hat],
        "beta": beta_hat, "gamma": gamma_hat,
        "log_beta": beta[0], "log_gamma": beta[1],
        "se_log_beta": se[0], "se_log_gamma": se[1],
        "gamma_ci_lower": math.exp(beta[1] - 1.959963984540054 * se[1]),
        "gamma_ci_upper": math.exp(beta[1] + 1.959963984540054 * se[1]),
        "n_points": n, "n_close_pairs": npairs, "radius": rr,
        "area": area, "window": win, "window_source": window_source,
        "n_quadrature": m, "n_dummy": len(dummy),
        "log_pseudolikelihood": logpl,
        "log_pseudolikelihood_poisson": logpl_pois,
        "iterations": it, "converged": converged,
        "valid_density": bool(gamma_hat <= 1.0),
        "interaction": ("inhibition" if gamma_hat < 1.0 - 1e-8 else
                        ("none (Poisson)" if abs(gamma_hat - 1.0) <= 1e-8
                         else "attraction -- NOT a valid Strauss density")),
    }
    if gamma is not None:
        g = float(gamma)
        if g <= 0.0:
            raise ValueError("strmkr: gamma must be positive")
        out["gamma_given"] = g
        out["log_density_unnormalised"] = (n * math.log(beta_hat)
                                           + npairs * math.log(g))
        out["valid_density_given"] = bool(g <= 1.0)
    out["method"] = ("Strauss process fitted by maximum pseudolikelihood "
                     "through the Baddeley-Turner quadrature device "
                     "(Strauss 1975; Besag 1977; Baddeley & Turner 2000)")
    out["note"] = ("gamma < 1 is inhibition, gamma = 1 is Poisson, and "
                   "gamma > 1 is not an integrable density at all (Kelly & "
                   "Ripley 1976) -- valid_density says which case the fit "
                   "landed in instead of clamping it")
    return RichResult(payload=out)


def cheatsheet():
    return ("strmkr: strauss_process(coords, r, gamma) -> pseudolikelihood "
            "beta and gamma for the Strauss inhibition model (Strauss 1975; "
            "Baddeley & Turner 2000)")


# compact alias per ledger/NAMING.md
straussprocess = strauss_process
