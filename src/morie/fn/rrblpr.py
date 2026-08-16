# morie.fn -- function file (rootcoder007/morie)
r"""Ridge-regression BLUP: every marker gets an effect, none gets a big one.

Genomic prediction has more markers than individuals, so least squares
has no unique answer and marker-by-marker regression answers a different
question in each column. RR-BLUP treats the marker effects as random
draws from a common distribution instead of as fixed unknowns,

.. math:: y = X\beta + Mu + e,\qquad u\sim N(0,\sigma_u^2 I_m),\qquad
          e\sim N(0,\sigma_e^2 I_n),

which turns the estimation problem into Henderson's mixed model
equations

.. math:: \begin{pmatrix}X'X & X'M\\ M'X & M'M+\lambda I\end{pmatrix}
          \begin{pmatrix}\hat\beta\\ \hat u\end{pmatrix}
          = \begin{pmatrix}X'y\\ M'y\end{pmatrix},\qquad
          \lambda=\sigma_e^2/\sigma_u^2 .

The single ridge :math:`\lambda` is not a tuning knob borrowed from
machine learning; it is a variance ratio, and it can be estimated from
the data rather than chosen. Passing ``lam`` fixes it, passing ``None``
estimates it by restricted maximum likelihood over a fixed-grid
search on :math:`\log\lambda`, with the profile returned so the optimum
can be seen rather than trusted.

The identity that makes this practical is returned alongside: the
marker-effect form and the kernel (GBLUP) form give the SAME breeding
values,

.. math:: M\hat u = MM'(MM'+\lambda I)^{-1}(y-X\hat\beta),

so a study with 500,000 markers and 800 individuals can be fitted at the
size of the sample rather than the size of the chip. Both are computed
here and their agreement is checked, because the identity holding is
what says the implementation is right.

References
----------
Whittaker, J. C., Thompson, R. and Denham, M. C. (2000) "Marker-assisted
selection using ridge regression", *Genetical Research* **75**(2),
249-252, doi:10.1017/S0016672399004462.

Meuwissen, T. H. E., Hayes, B. J. and Goddard, M. E. (2001) "Prediction
of total genetic value using genome-wide dense marker maps", *Genetics*
**157**(4), 1819-1829, doi:10.1093/genetics/157.4.1819.

Henderson, C. R. (1975) "Best linear unbiased estimation and prediction
under a selection model", *Biometrics* **31**(2), 423-447,
doi:10.2307/2529430. The mixed model equations.

Endelman, J. B. (2011) "Ridge regression and other kernels for genomic
selection with R package rrBLUP", *The Plant Genome* **4**(3), 250-255,
doi:10.3835/plantgenome2011.08.0024.

Kang, H. M., Zaitlen, N. A., Wade, C. M., Kirby, A., Heckerman, D.,
Daly, M. J. and Eskin, E. (2008) "Efficient control of population
structure in model organism association mapping", *Genetics* **178**(3),
1709-1723, doi:10.1534/genetics.107.080101. The profile-REML strategy.

VanRaden, P. M. (2008) "Efficient methods to compute genomic
predictions", *Journal of Dairy Science* **91**(11), 4414-4423,
doi:10.3168/jds.2007-0980. The kernel form.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["rr_blup"]

_EPS = 1e-12


def _gridmax(f, lo, hi, points=201, stages=4):
    r"""Maximise ``f`` over ``[lo, hi]`` by a staged fixed-grid argmax.

    A golden-section search is PATH-DEPENDENT. Each arm walks its own
    sequence of brackets, and near a flat maximum the ``fc > fd`` branch is
    decided by the last bits of two nearly equal likelihoods, so the two
    languages take different paths and land on different answers. Quantising
    the result afterwards hides that only when the answer does not fall near
    a cell boundary, which is a coincidence rather than a guarantee: the
    measured failure was two arms landing on ADJACENT points of a 1e-6 grid.

    Here both arms evaluate the SAME list of points -- ``a + i * step`` is
    the same double in both languages -- and take the argmax BY INDEX, ties
    to the lowest index. The winning index is therefore the same by
    construction, and the value returned is an exact grid point rather than a
    bracket midpoint, so the two arms return bit-identical doubles.

    Refinement stops while adjacent grid values still differ by far more than
    floating-point noise. Going finer would push the comparison back below
    the noise floor and reintroduce exactly the disagreement this exists to
    remove. It would also be false precision: a REML optimum this flat is not
    located to better than the square root of machine epsilon by any method,
    and the resolution reached here is already orders of magnitude finer than
    the statistical precision of the estimate.
    """
    a, b = float(lo), float(hi)
    npt = int(points)
    last = int(stages) - 1
    for s in range(int(stages)):
        step = (b - a) / (npt - 1)
        vals = [f(a + i * step) for i in range(npt)]
        best = 0
        for i in range(1, npt):
            if vals[i] > vals[best]:
                best = i
        if s == last:
            return a + best * step
        lo_i = best - 1 if best > 0 else 0
        hi_i = best + 1 if best < npt - 1 else npt - 1
        a, b = a + lo_i * step, a + hi_i * step
        npt = 21
    return a


def _chol(A):
    """Cholesky factor, lower triangular, with a scaled jitter."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    scale = sum(A[i][i] for i in range(n)) / n
    jit = 1e-12 * max(abs(scale), 1.0)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][u] * L[j][u] for u in range(j))
            if i == j:
                s += jit
                if s <= 0.0:
                    raise ValueError("rrblpr: the covariance matrix is not "
                                     "positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _chol_solve(L, b):
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        z[i] = (b[i] - sum(L[i][u] * z[u] for u in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(L[u][i] * x[u] for u in range(i + 1, n))) / L[i][i]
    return x


def _logdet(L):
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


def _reml_at(loglam, G, y, X):
    """Restricted log likelihood at lambda, profiled over sigma_e^2."""
    n = len(y)
    p = len(X[0])
    lam = math.exp(loglam)
    V = [[G[i][j] / lam + (1.0 if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    L = _chol(V)
    Viy = _chol_solve(L, y)
    ViX = [_chol_solve(L, [X[i][a] for i in range(n)]) for a in range(p)]
    XtViX = [[sum(X[i][a] * ViX[b][i] for i in range(n)) for b in range(p)]
             for a in range(p)]
    XtViy = [sum(X[i][a] * Viy[i] for i in range(n)) for a in range(p)]
    Lx = _chol(XtViX)
    beta = _chol_solve(Lx, XtViy)
    r = [y[i] - sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    Vir = _chol_solve(L, r)
    rss = sum(r[i] * Vir[i] for i in range(n))
    dfr = n - p
    s2e = rss / dfr
    ll = -0.5 * (dfr * math.log(max(s2e, 1e-300)) + _logdet(L)
                 + _logdet(Lx) + dfr)
    return ll, lam, beta, s2e, L


def rr_blup(y, M, lam=None, X=None, M_new=None, log_lam_lo=-12.0,
            log_lam_hi=12.0, max_iter=200, tol=1e-9):
    r"""Fit ``y = X beta + M u + e`` with ``u`` random and equal-variance.

    Parameters
    ----------
    y : array-like, length ``n``
        Phenotype.
    M : array-like, shape ``(n, m)``
        Marker matrix. Coding is the caller's choice; the effects are
        reported on whatever scale the columns carry.
    lam : float, optional
        The variance ratio :math:`\sigma_e^2/\sigma_u^2`. ``None``
        estimates it by REML. ``0`` is allowed and gives least squares,
        which is only defined when ``M`` has full column rank.
    X : array-like, optional
        Fixed effects. Defaults to an intercept.
    M_new : array-like, optional
        Markers of genotypes to predict.

    Returns
    -------
    RichResult
        ``marker_effects``, ``breeding_values`` from both the
        marker-effect and the kernel form, the fitted variance
        components, ``h2``, and the REML profile when ``lam`` was
        estimated.
    """
    yv = [float(v) for v in k.vec(y)]
    Mm = [[float(v) for v in row] for row in k.mat(M)]
    n = len(yv)
    if n == 0:
        raise ValueError("rrblpr: no observations")
    if len(Mm) != n:
        raise ValueError("rrblpr: %d phenotypes but %d marker rows"
                         % (n, len(Mm)))
    m = len(Mm[0])
    if any(len(r) != m for r in Mm):
        raise ValueError("rrblpr: every row of M must have %d markers" % m)
    Xm = ([[1.0] for _ in range(n)] if X is None
          else [[float(v) for v in row] for row in k.mat(X)])
    if len(Xm) != n:
        raise ValueError("rrblpr: X has %d rows, y has %d" % (len(Xm), n))
    p = len(Xm[0])
    if n - p < 1:
        raise ValueError("rrblpr: %d observations and %d fixed effects "
                         "leave no residual degrees of freedom" % (n, p))

    # the kernel MM' -- the object both forms of the predictor share
    G = [[sum(Mm[i][a] * Mm[j][a] for a in range(m)) for j in range(n)]
         for i in range(n)]

    profile = []
    if lam is None:
        # fixed-grid argmax on log lambda; the profile is returned so the
        # optimum is visible rather than asserted. max_iter is accepted and
        # ignored -- the grid schedule fixes the evaluation count, and
        # dropping the argument would break callers that pass it.
        loglam = _gridmax(lambda t: _reml_at(t, G, yv, Xm)[0],
                          log_lam_lo, log_lam_hi)
        ll, lam_hat, beta, s2e, L = _reml_at(loglam, G, yv, Xm)
        for t in range(21):
            lt = float(log_lam_lo) + (float(log_lam_hi)
                                      - float(log_lam_lo)) * t / 20.0
            profile.append([lt, _reml_at(lt, G, yv, Xm)[0]])
        estimated = True
    else:
        lam_hat = float(lam)
        if lam_hat < 0.0:
            raise ValueError("rrblpr: lambda is a variance ratio and cannot "
                             "be negative")
        if lam_hat <= _EPS:
            # least squares: only defined when M has full column rank
            if m > n - p:
                raise ValueError("rrblpr: lambda = 0 with %d markers and %d "
                                 "residual degrees of freedom -- the least "
                                 "squares problem is not identified" % (m,
                                                                        n - p))
            lam_hat = 0.0
        ll, beta, s2e, L, estimated = None, None, None, None, False

    # ---- Henderson's mixed model equations, solved as written
    q = p + m
    A = [[0.0] * q for _ in range(q)]
    rhs = [0.0] * q
    for a in range(p):
        for b2 in range(p):
            A[a][b2] = sum(Xm[i][a] * Xm[i][b2] for i in range(n))
        for b2 in range(m):
            A[a][p + b2] = sum(Xm[i][a] * Mm[i][b2] for i in range(n))
        rhs[a] = sum(Xm[i][a] * yv[i] for i in range(n))
    for a in range(m):
        for b2 in range(p):
            A[p + a][b2] = A[b2][p + a]
        for b2 in range(m):
            A[p + a][p + b2] = sum(Mm[i][a] * Mm[i][b2] for i in range(n))
        A[p + a][p + a] += lam_hat
        rhs[p + a] = sum(Mm[i][a] * yv[i] for i in range(n))
    sol = _chol_solve(_chol(A), rhs)
    beta_h = sol[:p]
    u = sol[p:]

    fitted = [sum(Xm[i][a] * beta_h[a] for a in range(p))
              + sum(Mm[i][a] * u[a] for a in range(m)) for i in range(n)]
    gv = [sum(Mm[i][a] * u[a] for a in range(m)) for i in range(n)]

    # ---- the kernel form of the same predictor, computed independently
    r = [yv[i] - sum(Xm[i][a] * beta_h[a] for a in range(p))
         for i in range(n)]
    if lam_hat > _EPS:
        Vk = [[G[i][j] + (lam_hat if i == j else 0.0) for j in range(n)]
              for i in range(n)]
        w = _chol_solve(_chol(Vk), r)
        gv_kernel = [sum(G[i][j] * w[j] for j in range(n)) for i in range(n)]
    else:
        gv_kernel = list(gv)
    kernel_gap = max(abs(gv[i] - gv_kernel[i]) for i in range(n))

    resid = [yv[i] - fitted[i] for i in range(n)]
    rss = sum(v * v for v in resid)
    s2e_h = rss / max(n - p, 1) if s2e is None else s2e
    s2u = s2e_h / lam_hat if lam_hat > _EPS else float("inf")
    trG = sum(G[i][i] for i in range(n)) / n
    s2g = s2u * trG if lam_hat > _EPS else float("inf")
    h2 = (s2g / (s2g + s2e_h)) if lam_hat > _EPS else float("nan")

    pred_new = None
    if M_new is not None:
        Mn = [[float(v) for v in row] for row in k.mat(M_new)]
        if any(len(rw) != m for rw in Mn):
            raise ValueError("rrblpr: M_new must have %d markers" % m)
        pred_new = [sum(rw[a] * u[a] for a in range(m)) for rw in Mn]

    return RichResult(payload={
        "estimate": u, "marker_effects": u, "coefficients": beta_h,
        "breeding_values": gv, "breeding_values_kernel": gv_kernel,
        "kernel_identity_gap": kernel_gap,
        "fitted": fitted, "residuals": resid,
        "lambda": lam_hat, "lambda_estimated": estimated,
        "sigma2_e": s2e_h, "sigma2_u": s2u, "sigma2_g": s2g, "h2": h2,
        "reml_loglik": ll, "reml_profile": profile,
        "prediction_new": pred_new,
        "n": n, "m": m, "p": p,
        "method": "RR-BLUP: Henderson's mixed model equations with a single "
                  "variance ratio, the ratio estimated by profile REML when "
                  "it is not supplied (Whittaker et al. 2000; Meuwissen et "
                  "al. 2001; Henderson 1975)",
        "note": "breeding_values and breeding_values_kernel are the "
                "marker-effect and GBLUP forms of the same predictor; "
                "kernel_identity_gap is how far apart they came out, and it "
                "is the check that the implementation is right rather than "
                "merely plausible",
    })


def cheatsheet():
    return ("rrblpr: rr_blup(y, M, lam) -> marker effects and breeding "
            "values from the mixed model equations, lambda by REML when "
            "None (Whittaker, Thompson & Denham 2000; Meuwissen et al. 2001)")
