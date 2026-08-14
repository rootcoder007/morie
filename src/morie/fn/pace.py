# morie.fn -- function file (rootcoder007/morie)
r"""PACE: functional principal components for sparsely observed curves.

**The problem.** Each subject is seen at only a handful of times, and
the times differ from subject to subject. Pre-smoothing each curve and
then running ordinary FPCA fails here: with three or four points per
subject there is nothing to smooth, and the integral
:math:`\int (X_i(t)-\mu(t))\phi_k(t)\,dt` that defines the score cannot
be approximated from them.

**The idea.** Borrow strength across subjects rather than within them.
Pool every observation into one scatter plot and smooth *that* -- the
combined design is dense even when each individual design is not.

* **Mean.** Smooth :math:`Y_{ij}` against :math:`t_{ij}` pooled over
  all subjects.
* **Covariance.** Form the raw covariances
  :math:`u_{ikl} = (Y_{ik}-\hat\mu(t_{ik}))(Y_{il}-\hat\mu(t_{il}))`
  and smooth them against :math:`(t_{ik}, t_{il})` over the product
  interval, **using only the off-diagonal terms** :math:`k \ne l`.
* **Why the diagonal is dropped.** :math:`\operatorname{var}(Y(t)) =
  \operatorname{var}(X(t)) + \sigma^2`, so the :math:`k = l` terms
  carry the measurement-error variance on top of the covariance and
  would bias the surface upward along its diagonal. Removing them is
  what makes the surface estimate the covariance of :math:`X` rather
  than of :math:`Y`.
* **The error variance falls out of the same gap.** What was discarded
  is exactly what identifies :math:`\sigma^2`: smoothing
  :math:`(Y_{ij}-\hat\mu(t_{ij}))^2 - \hat\Sigma(t_{ij},t_{ij})`
  against :math:`t_{ij}` recovers it.

**The scores, which is where the name comes from.** With the integral
unavailable, the score is estimated by *conditioning* instead --
Principal Analysis by Conditional Expectation. Under joint normality
of :math:`(\xi_{ik}, Y_i)`,

.. math:: \hat\xi_{ik} = \operatorname{E}[\xi_{ik} \mid Y_i]
          = \lambda_k \phi_{ik}^\top \Sigma_{Y_i}^{-1}
            (Y_i - \mu_i),

using :math:`\operatorname{cov}(\xi_{ik}, Y_{ij}) = \lambda_k
\phi_k(t_{ij})` and
:math:`\Sigma_{Y_i} = \Phi_i \Lambda \Phi_i^\top + \sigma^2 I`.
This is a shrinkage estimator: it is pulled toward zero when a subject
has few or noisy observations, which is the honest answer when the
data do not pin the score down, and it is exactly what an integral
approximation cannot deliver from four points.

Sources
-------
Yao, F., Müller, H.-G. & Wang, J.-L. (2005) "Functional Data Analysis
for Sparse Longitudinal Data", *Journal of the American Statistical
Association* 100(470), 577-590, doi:10.1198/016214504000001745. The
method: pooled smoothing of mean and covariance, removal of the
diagonal raw covariances, estimation of the error variance from the
diagonal gap, and the conditional-expectation (PACE) scores.

Wang, J.-L., Chiou, J.-M. & Müller, H.-G. (2016) "Functional Data
Analysis", *Annual Review of Statistics and Its Application* 3,
257-295, doi:10.1146/annurev-statistics-041715-033624; preprint
arXiv:1507.05135, p. 6, which states the raw covariances
:math:`u_{ikl}`, that "the diagonal raw covariances where :math:`k=l`
are removed from the 2D scatter plot prior to the smoothing step
because these include an additional term that is due to the variance
of the measurement errors", and the recovery of :math:`\sigma^2` by
smoothing :math:`(Y_{ij}-\hat\mu(t_{ij}))^2 - \hat\Sigma(t_{ij})`.

Fan, J. & Gijbels, I. (1996) *Local Polynomial Modelling and Its
Applications*, Chapman & Hall. The local linear smoother used for both
the pooled mean and the covariance surface.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["pace", "local_linear", "local_linear_2d"]

_KERNELS = ("epan", "gauss")


def _kweight(u, kernel):
    if kernel == "gauss":
        return math.exp(-0.5 * u * u)
    a = 1.0 - u * u
    return 0.75 * a if a > 0.0 else 0.0


def local_linear(t, y, at, bw, kernel="epan"):
    r"""Local linear smoother: the intercept of a weighted line fit.

    Local *linear* rather than local constant because the pooled
    design is irregular and a local constant fit is biased at the
    boundary, where sparse designs put a large share of their points.
    """
    if kernel not in _KERNELS:
        raise ValueError("pace: kernel must be epan or gauss, got %r"
                         % (kernel,))
    if bw <= 0:
        raise ValueError("pace: the bandwidth must be positive")
    out = []
    for t0 in at:
        s0 = s1 = s2 = b0 = b1 = 0.0
        for i in range(len(t)):
            d = t[i] - t0
            w = _kweight(d / bw, kernel)
            if w == 0.0:
                continue
            s0 += w
            s1 += w * d
            s2 += w * d * d
            b0 += w * y[i]
            b1 += w * d * y[i]
        det = s0 * s2 - s1 * s1
        if s0 <= 0.0:
            raise ValueError("pace: bandwidth %g leaves the point %g "
                             "with no data" % (bw, t0))
        if abs(det) < 1e-12:
            out.append(b0 / s0)          # only one distinct design point
        else:
            out.append((s2 * b0 - s1 * b1) / det)
    return out


def local_linear_2d(s, t, z, at_s, at_t, bw, kernel="epan"):
    r"""Local linear surface smoother, fitting :math:`a + b\,\Delta s +
    c\,\Delta t` at each target and returning :math:`a`.

    Points are bucketed into cells of side ``bw`` and each target scans
    only its own cell and the eight around it. Every point within the
    bandwidth lies in one of those nine cells, so this is exact, not an
    approximation -- it just avoids sweeping all pairs for every target,
    which is what makes the covariance surface affordable when there are
    hundreds of thousands of raw covariances.
    """
    if kernel not in _KERNELS:
        raise ValueError("pace: kernel must be epan or gauss, got %r"
                         % (kernel,))
    if bw <= 0:
        raise ValueError("pace: the bandwidth must be positive")
    # a Gaussian kernel has unbounded support, so its cell radius is
    # widened to where the weight is negligible rather than truncated
    reach = 1 if kernel == "epan" else 4
    s0m, t0m = min(s), min(t)
    buckets = {}
    for i in range(len(z)):
        key = (int((s[i] - s0m) / bw), int((t[i] - t0m) / bw))
        buckets.setdefault(key, []).append(i)
    out = []
    for sv in at_s:
        row = []
        for tv in at_t:
            ci = int((sv - s0m) / bw)
            cj = int((tv - t0m) / bw)
            X, W, Y = [], [], []
            for a in range(ci - reach, ci + reach + 1):
                for b in range(cj - reach, cj + reach + 1):
                    for i in buckets.get((a, b), ()):
                        ds, dt = s[i] - sv, t[i] - tv
                        w = (_kweight(ds / bw, kernel)
                             * _kweight(dt / bw, kernel))
                        if w == 0.0:
                            continue
                        X.append([1.0, ds, dt])
                        W.append(w)
                        Y.append(z[i])
            if not X:
                raise ValueError("pace: bandwidth %g leaves (%g, %g) "
                                 "with no data" % (bw, sv, tv))
            Xw = [[X[i][a] * W[i] for a in range(3)]
                  for i in range(len(X))]
            XtX = [[sum(Xw[i][a] * X[i][b] for i in range(len(X)))
                    for b in range(3)] for a in range(3)]
            Xty = [sum(Xw[i][a] * Y[i] for i in range(len(X)))
                   for a in range(3)]
            try:
                row.append(k.ridgesolve(XtX, Xty, 1e-10)[0])
            except Exception:                                # noqa: BLE001
                sw = sum(W)
                row.append(sum(W[i] * Y[i] for i in range(len(W))) / sw)
        out.append(row)
    return out


def _rule_of_thumb(ts):
    lo, hi = min(ts), max(ts)
    n = max(len(ts), 2)
    return max((hi - lo) * n ** (-0.2) / 2.0, (hi - lo) * 1e-3)


def pace(Y, argvals, K=2, n_grid=21, bw_mu=None, bw_cov=None,
         kernel="epan", shrink=True):
    r"""Sparse FPCA by conditional expectation.

    Parameters
    ----------
    Y : sequence of sequences
        ``Y[i]`` are subject *i*'s observations.
    argvals : sequence of sequences
        ``argvals[i]`` are the times at which they were taken. Ragged
        by design -- subjects need not share a schedule.
    K : int
        Number of components to keep.
    n_grid : int
        Size of the working grid on which the surface is built.
    bw_mu, bw_cov : float, optional
        Bandwidths; a rule of thumb from the pooled design is used
        when omitted.
    kernel : {"epan", "gauss"}
        Smoothing kernel.
    shrink : bool
        Use the conditional-expectation (PACE) scores. ``False`` uses
        the integral approximation instead, which is what fails on
        sparse designs -- kept so the two can be compared.

    Returns
    -------
    RichResult
        ``estimate`` (the scores), ``eigenvalues``, ``eigenfunctions``,
        ``mean``, ``sigma2``, ``fve``, ``fitted``, ``grid``.
    """
    if kernel not in _KERNELS:
        raise ValueError("pace: kernel must be epan or gauss, got %r"
                         % (kernel,))
    ys = [[float(v) for v in row] for row in Y]
    ts = [[float(v) for v in row] for row in argvals]
    n = len(ys)
    if n == 0:
        raise ValueError("pace: no subjects")
    if len(ts) != n:
        raise ValueError("pace: %d subjects but %d time vectors"
                         % (n, len(ts)))
    for i in range(n):
        if len(ys[i]) != len(ts[i]):
            raise ValueError("pace: subject %d has %d values and %d "
                             "times" % (i, len(ys[i]), len(ts[i])))
    pooled_t = [v for row in ts for v in row]
    pooled_y = [v for row in ys for v in row]
    if len(pooled_t) < 3:
        raise ValueError("pace: need at least three observations in "
                         "total")
    K = int(K)
    if K < 1:
        raise ValueError("pace: K must be at least 1")

    lo, hi = min(pooled_t), max(pooled_t)
    if hi <= lo:
        raise ValueError("pace: all observation times are identical")
    ng = int(n_grid)
    if ng < 3:
        raise ValueError("pace: n_grid must be at least 3")
    gr = [lo + (hi - lo) * i / float(ng - 1) for i in range(ng)]

    hmu = float(bw_mu) if bw_mu else _rule_of_thumb(pooled_t)
    hcov = float(bw_cov) if bw_cov else 1.5 * hmu

    # 1. pooled mean
    mu_g = local_linear(pooled_t, pooled_y, gr, hmu, kernel)

    # mu is evaluated on the grid once and interpolated, the same way
    # the eigenfunctions are: re-smoothing at every one of the pooled
    # observation times would cost O(N^2) kernel evaluations and buys
    # nothing, since the grid is finer than the smoother's resolution
    def _interp(vals, x):
        p = (x - lo) / ((hi - lo) / float(ng - 1))
        i0 = int(math.floor(p))
        if i0 < 0:
            i0 = 0
        if i0 > ng - 2:
            i0 = ng - 2
        w = p - i0
        return vals[i0] * (1.0 - w) + vals[i0 + 1] * w

    def mu_at(x):
        return _interp(mu_g, x)

    # 2. raw covariances, OFF-DIAGONAL ONLY
    cs, ct, cz = [], [], []
    diag_s, diag_z = [], []
    for i in range(n):
        m = len(ts[i])
        cen = [ys[i][j] - mu_at(ts[i][j]) for j in range(m)]
        for a in range(m):
            diag_s.append(ts[i][a])
            diag_z.append(cen[a] * cen[a])
            for b in range(m):
                if a == b:
                    continue                 # the measurement-error term
                cs.append(ts[i][a])
                ct.append(ts[i][b])
                cz.append(cen[a] * cen[b])
    if not cz:
        raise ValueError("pace: no off-diagonal pairs -- every subject "
                         "has a single observation, so the covariance "
                         "is not identified")
    G = local_linear_2d(cs, ct, cz, gr, gr, hcov, kernel)
    G = [[0.5 * (G[a][b] + G[b][a]) for b in range(ng)]
         for a in range(ng)]

    # 3. sigma^2 from the gap the diagonal left behind
    dsm = local_linear(diag_s, diag_z, gr, hmu, kernel)
    gaps = [dsm[a] - G[a][a] for a in range(ng)]
    sigma2 = sum(gaps) / float(ng)
    if sigma2 < 0.0:
        sigma2 = 0.0

    # 4. eigen-decomposition on the grid
    dt = (hi - lo) / float(ng - 1)
    vals, vecs = k.jacobi(G)
    order = sorted(range(ng), key=lambda i: -vals[i])
    lam, phi = [], []
    for idx in order[:K]:
        ev = max(vals[idx], 0.0) * dt          # trapezoid-free scaling
        f = [vecs[r][idx] for r in range(ng)]
        nrm = math.sqrt(max(sum(v * v for v in f) * dt, 1e-300))
        f = [v / nrm for v in f]
        lam.append(ev)
        phi.append(f)
    total = sum(max(v, 0.0) for v in vals) * dt
    fve = [(sum(lam[:j + 1]) / total if total > 0 else float("nan"))
           for j in range(len(lam))]

    def phi_at(j, x):
        p = (x - lo) / dt
        i0 = int(math.floor(p))
        if i0 < 0:
            i0 = 0
        if i0 > ng - 2:
            i0 = ng - 2
        w = p - i0
        return phi[j][i0] * (1.0 - w) + phi[j][i0 + 1] * w

    # 5. scores by conditional expectation
    scores, fitted = [], []
    for i in range(n):
        m = len(ts[i])
        cen = [ys[i][j] - mu_at(ts[i][j]) for j in range(m)]
        P = [[phi_at(j, ts[i][a]) for j in range(len(lam))]
             for a in range(m)]
        if shrink:
            S = [[sum(lam[j] * P[a][j] * P[b][j]
                      for j in range(len(lam)))
                  + (sigma2 if a == b else 0.0) for b in range(m)]
                 for a in range(m)]
            try:
                z = k.ridgesolve(S, cen, 1e-10)
            except Exception:                                # noqa: BLE001
                z = [0.0] * m
            xi = [lam[j] * sum(P[a][j] * z[a] for a in range(m))
                  for j in range(len(lam))]
        else:
            xi = []
            for j in range(len(lam)):
                if m < 2:
                    xi.append(0.0)
                    continue
                tot = 0.0
                for a in range(m - 1):
                    h = ts[i][a + 1] - ts[i][a]
                    tot += 0.5 * h * (cen[a] * P[a][j]
                                      + cen[a + 1] * P[a + 1][j])
                xi.append(tot)
        scores.append(xi)
        fitted.append([mu_g[g] + sum(xi[j] * phi[j][g]
                                     for j in range(len(lam)))
                       for g in range(ng)])

    return RichResult(payload={
        "estimate": scores, "scores": scores,
        "eigenvalues": lam, "eigenfunctions": phi,
        "mean": mu_g, "grid": gr, "sigma2": sigma2,
        "fve": fve, "fitted": fitted,
        "n": n, "K": len(lam), "n_grid": ng,
        "bw_mu": hmu, "bw_cov": hcov, "kernel": kernel,
        "shrink": shrink,
        "n_obs": len(pooled_t),
        "method": "PACE sparse FPCA with conditional-expectation "
                  "scores (Yao, Müller & Wang 2005)",
    })


def cheatsheet():
    return ("pace: sparse FPCA. Pool everyone's points and smooth "
            "THAT -- the combined design is dense even when no "
            "single subject's is. Drop the diagonal raw covariances "
            "before smoothing, because var(Y) = var(X) + sigma^2 and "
            "the diagonal carries the error variance; the gap it "
            "leaves is what identifies sigma^2. Scores come from "
            "CONDITIONING, not integration: xi = lambda * phi' "
            "Sigma_Y^-1 (Y - mu), which shrinks toward zero when a "
            "subject has few points -- the honest answer, and the "
            "one an integral over four points cannot give.")
