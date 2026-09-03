# morie.fn -- function file (rootcoder007/morie)
r"""Counts and proportions that are correlated in space.

Disease counts by district, presence of a species at survey points,
tree failures along a road: the response is not Gaussian and the
residuals are not independent, and fixing only one of those gives a
model that is wrong in the other direction. The spatial generalised
linear mixed model fixes both at once,

.. math:: g(\mathbb E[y_i]) = x_i'\beta + u_i,\qquad
          u\sim N\!\left(0,\ \sigma^2 R(\phi)\right),

with :math:`R` a correlation function of distance. The random effect is
not a nuisance: it is the spatially structured part of the risk, and it
is what a map of the fitted ``u`` shows.

The likelihood integrates over ``u`` and has no closed form, so it is
approximated at the mode -- the Laplace approximation,

.. math:: \ell(\theta) \approx \ell(\hat\beta,\hat u\mid\theta)
          - \tfrac12\hat u'\Sigma^{-1}\hat u
          - \tfrac12\log|\Sigma| - \tfrac12\log|H| ,

with the inner mode found by penalised iteratively reweighted least
squares and the outer :math:`(\sigma^2,\phi)` by cycling golden-section
searches. The approximation is EXACT for the Gaussian family with an
identity link, where it reduces to generalised least squares, and that
reduction is checked here against a direct GLS solve rather than
assumed -- it is the only case where the answer is known independently.

Breslow and Clayton's warning applies and is not softened: for binary
data with few observations per correlated unit the Laplace and PQL
estimates of the variance component are biased downward. The fitted
``sigma2`` is reported with the number of observations so the reader can
judge, not quietly presented as unbiased.

References
----------
Diggle, P. J., Tawn, J. A. and Moyeed, R. A. (1998) "Model-based
geostatistics", *Journal of the Royal Statistical Society C* **47**(3),
299-350, doi:10.1111/1467-9876.00113.

Breslow, N. E. and Clayton, D. G. (1993) "Approximate inference in
generalized linear mixed models", *Journal of the American Statistical
Association* **88**(421), 9-25, doi:10.1080/01621459.1993.10594284.

Tierney, L. and Kadane, J. B. (1986) "Accurate approximations for
posterior moments and marginal densities", *Journal of the American
Statistical Association* **81**(393), 82-86,
doi:10.1080/01621459.1986.10478240. The Laplace approximation.

Rue, H., Martino, S. and Chopin, N. (2009) "Approximate Bayesian
inference for latent Gaussian models by using integrated nested Laplace
approximations", *Journal of the Royal Statistical Society B* **71**(2),
319-392, doi:10.1111/j.1467-9868.2008.00700.x.

Diggle, P. J. and Ribeiro, P. J. (2007) *Model-based Geostatistics*,
Springer, Ch. 4 and 9, doi:10.1007/978-0-387-48536-2.

McCulloch, C. E. and Searle, S. R. (2001) *Generalized, Linear, and
Mixed Models*, Wiley, Ch. 8.

Matern, B. (1986) *Spatial Variation*, 2nd ed., Springer,
doi:10.1007/978-1-4615-7892-5. The correlation families offered.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["spatial_glmm_fit"]

_EPS = 1e-12
_INVPHI = 0.6180339887498949


def _chol(A, rel_jitter=1e-10):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    scale = sum(A[i][i] for i in range(n)) / n
    jit = rel_jitter * max(abs(scale), 1.0)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][u] * L[j][u] for u in range(j))
            if i == j:
                s += jit
                if s <= 0.0:
                    return None
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _solve(L, b):
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        z[i] = (b[i] - sum(L[i][u] * z[u] for u in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(L[u][i] * x[u] for u in range(i + 1, n))) / L[i][i]
    return x


def _inv(L):
    n = len(L)
    M = [[0.0] * n for _ in range(n)]
    for a in range(n):
        e = [0.0] * n
        e[a] = 1.0
        col = _solve(L, e)
        for b in range(n):
            M[b][a] = col[b]
    return M


def _logdet(L):
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


def _corr(h, model, phi, kappa):
    if h <= 0.0:
        return 1.0
    if model == "exponential":
        return math.exp(-h / phi)
    if model == "gaussian":
        return math.exp(-(h / phi) ** 2)
    if model == "spherical":
        if h >= phi:
            return 0.0
        r = h / phi
        return 1.0 - 1.5 * r + 0.5 * r ** 3
    if model == "matern":
        # Matern with smoothness kappa, through the modified Bessel K
        z = math.sqrt(2.0 * kappa) * h / phi
        if z <= 0.0:
            return 1.0
        lg = k.lgamma(kappa)
        val = ((2.0 ** (1.0 - kappa)) / math.exp(lg)
               * (z ** kappa) * k.besselk(kappa, z))
        return max(min(val, 1.0), 0.0)
    raise ValueError("sgflrt: model must be exponential, gaussian, "
                     "spherical or matern, got %r" % (model,))


def _family(family, disp=1.0):
    r"""Return ``(linkinv, V, W, loglik)``.

    ``V`` is the variance function and ``W = V/\phi`` the IRLS weight.
    They coincide for the canonical links with a fixed dispersion, which
    is why writing the working response as ``eta + (y - mu) / W`` looks
    correct and is -- right up until a family with a real dispersion
    parameter arrives, where it is off by exactly that factor. The
    working response is ``eta + (y - mu) / V(mu)`` and nothing else.
    """
    if family == "poisson":
        return (lambda e: math.exp(max(-500.0, min(500.0, e))),
                lambda m: max(m, 1e-10),
                lambda m: max(m, 1e-10),
                lambda y, m: y * math.log(max(m, 1e-300)) - m
                - k.lgamma(y + 1.0))
    if family == "binomial":
        def inv(e):
            e = max(-500.0, min(500.0, e))
            return 1.0 / (1.0 + math.exp(-e))

        def vf(m):
            return max(m * (1.0 - m), 1e-10)
        return (inv, vf, vf,
                lambda y, m: (y * math.log(min(max(m, 1e-12), 1 - 1e-12))
                              + (1.0 - y)
                              * math.log(1.0 - min(max(m, 1e-12),
                                                   1 - 1e-12))))
    if family == "gaussian":
        # the Gaussian family HAS a dispersion parameter and the others do
        # not. Pinning it at one -- which is what a GLM weight of 1 does --
        # forces the residual variance to be exactly 1 and drives the
        # spatial variance to zero whenever the real residual is smaller.
        d = max(float(disp), 1e-300)
        return (lambda e: e, lambda m: 1.0, lambda m: 1.0 / d,
                lambda y, m: -0.5 * (math.log(2.0 * math.pi * d)
                                     + (y - m) ** 2 / d))
    raise ValueError("sgflrt: family must be poisson, binomial or gaussian, "
                     "got %r" % (family,))


def _laplace(y, X, Sig, family, inner_iter, tol, disp=1.0):
    r"""Inner Laplace mode and the approximated log likelihood.

    The random effect is carried as :math:`u = Lv` with :math:`LL'=\Sigma`
    and :math:`v\sim N(0,I)`, so the penalty is :math:`\tfrac12 v'v` and
    :math:`\Sigma^{-1}` is never formed. That is not a nicety: a spatial
    correlation matrix with any two nearby locations is close to singular,
    an explicitly inverted one is dominated by the jitter that made the
    inversion possible, and the fitted coefficients then miss the exact
    Gaussian answer by tenths rather than by 1e-10. The determinant term
    also disappears -- the Jacobian of the substitution cancels it.
    """
    n = len(y)
    p = len(X[0])
    inv, vfun, wfun, ll = _family(family, disp)
    L = _chol(Sig)
    if L is None:
        return None
    beta = [0.0] * p
    v = [0.0] * n
    for _ in range(int(inner_iter)):
        u = [sum(L[i][j] * v[j] for j in range(i + 1)) for i in range(n)]
        eta = [sum(X[i][a] * beta[a] for a in range(p)) + u[i]
               for i in range(n)]
        mu = [inv(e) for e in eta]
        w = [wfun(m) for m in mu]
        z = [eta[i] + (y[i] - mu[i]) / vfun(mu[i]) for i in range(n)]
        q = p + n
        A = [[0.0] * q for _ in range(q)]
        rhs = [0.0] * q
        WL = [[w[i] * L[i][j] for j in range(n)] for i in range(n)]
        for a in range(p):
            for b in range(p):
                A[a][b] = sum(w[i] * X[i][a] * X[i][b] for i in range(n))
            for j in range(n):
                A[a][p + j] = sum(X[i][a] * WL[i][j] for i in range(n))
            rhs[a] = sum(w[i] * X[i][a] * z[i] for i in range(n))
        for j in range(n):
            for b in range(p):
                A[p + j][b] = A[b][p + j]
            for j2 in range(n):
                A[p + j][p + j2] = (sum(L[i][j] * WL[i][j2]
                                        for i in range(n))
                                    + (1.0 if j == j2 else 0.0))
            rhs[p + j] = sum(L[i][j] * w[i] * z[i] for i in range(n))
        LA = _chol(A)
        if LA is None:
            return None
        sol = _solve(LA, rhs)
        nb, nv = sol[:p], sol[p:]
        shift = max(max(abs(nb[a] - beta[a]) for a in range(p)),
                    max(abs(nv[j] - v[j]) for j in range(n)))
        beta, v = nb, nv
        if shift < tol:
            break
    u = [sum(L[i][j] * v[j] for j in range(i + 1)) for i in range(n)]
    eta = [sum(X[i][a] * beta[a] for a in range(p)) + u[i] for i in range(n)]
    mu = [inv(e) for e in eta]
    w = [wfun(m) for m in mu]
    loglik = sum(ll(y[i], mu[i]) for i in range(n))
    pen = 0.5 * sum(t * t for t in v)
    WL = [[w[i] * L[i][j] for j in range(n)] for i in range(n)]
    H = [[sum(L[i][j] * WL[i][j2] for i in range(n))
          + (1.0 if j == j2 else 0.0) for j2 in range(n)]
         for j in range(n)]
    LH = _chol(H)
    if LH is None:
        return None
    lap = loglik - pen - 0.5 * _logdet(LH)
    return lap, beta, u, mu, eta, loglik, w, L, v


def _golden(f, lo, hi, iters=16):
    a, b = lo, hi
    c = b - (b - a) * _INVPHI
    d = a + (b - a) * _INVPHI
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if b - a < 1e-8:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * _INVPHI
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * _INVPHI
            fd = f(d)
    return 0.5 * (a + b)


def spatial_glmm_fit(y, X, coords, family="poisson", model="exponential",
                     sigma2=None, phi=None, kappa=1.5, nugget=0.0,
                     dispersion=None,
                     inner_iter=50, outer_cycles=3, tol=1e-10):
    r"""Fit a spatial GLMM by the Laplace approximation.

    Parameters
    ----------
    y : array-like, length ``n``
    X : array-like, shape ``(n, p)``
        Fixed-effect design. Include your own intercept column.
    coords : array-like, shape ``(n, d)``
    family : {'poisson', 'binomial', 'gaussian'}
    model : {'exponential', 'gaussian', 'spherical', 'matern'}
    sigma2, phi : float, optional
        Fix the variance and range instead of estimating them.
        ``sigma2 = 0`` removes the random effect and the fit reduces to
        an ordinary GLM exactly.
    outer_cycles : int
        Sweeps of the coordinate search over ``(sigma2, phi)``. Each
        sweep costs two golden-section searches, and each of those costs
        a full Laplace fit per evaluation, so this is the knob that
        decides the runtime. Three sweeps of a twenty-step search locate
        the optimum to about 1e-4 on the log scale, which is finer than
        the approximation error the Laplace step itself carries.
    nugget : float
        Added to the diagonal of the correlation matrix. A pure spatial
        correlation matrix is often near singular at close range; this
        is the standard remedy and is reported.

    Returns
    -------
    RichResult
        ``coefficients``, the fitted spatial effect ``u``, the variance
        parameters, and the Laplace log likelihood.
    """
    yv = [float(v) for v in k.vec(y)]
    n = len(yv)
    if n == 0:
        raise ValueError("sgflrt: no observations")
    Xm = [[float(v) for v in row] for row in k.mat(X)]
    if len(Xm) != n:
        raise ValueError("sgflrt: %d responses but %d design rows"
                         % (n, len(Xm)))
    p = len(Xm[0])
    C = [[float(v) for v in row] for row in k.mat(coords)]
    if len(C) != n:
        raise ValueError("sgflrt: %d responses but %d coordinate rows"
                         % (n, len(C)))
    d = len(C[0])
    if any(len(r) != d for r in C):
        raise ValueError("sgflrt: all coordinates must have the same "
                         "dimension")
    _family(family)
    if dispersion is not None and family != "gaussian":
        raise ValueError("sgflrt: only the gaussian family has a dispersion "
                         "parameter; poisson and binomial fix it at one")
    disp = 1.0 if dispersion is None else float(dispersion)
    if disp <= 0.0:
        raise ValueError("sgflrt: the dispersion must be positive")
    fit_disp = family == "gaussian" and dispersion is None
    nug = float(nugget)
    if nug < 0.0:
        raise ValueError("sgflrt: the nugget cannot be negative")
    if family == "binomial" and any(v not in (0.0, 1.0) for v in yv):
        raise ValueError("sgflrt: the binomial family here takes 0/1 "
                         "responses")
    if family == "poisson" and any(v < 0.0 or v != math.floor(v)
                                   for v in yv):
        raise ValueError("sgflrt: the Poisson family takes non-negative "
                         "counts")

    D = [[math.sqrt(sum((C[i][a] - C[j][a]) ** 2 for a in range(d)))
          for j in range(n)] for i in range(n)]
    dmax = max(max(r) for r in D)
    if dmax <= _EPS:
        raise ValueError("sgflrt: every location is the same point, so "
                         "there is no spatial structure to fit")
    dmin = min(D[i][j] for i in range(n) for j in range(n) if i != j)

    def corrmat(ph):
        R = [[_corr(D[i][j], model, ph, float(kappa)) for j in range(n)]
             for i in range(n)]
        for i in range(n):
            R[i][i] += nug
        return R

    def scaled(R, s2):
        return [[s2 * R[i][j] for j in range(n)] for i in range(n)]

    if sigma2 is not None and float(sigma2) <= _EPS:
        # no random effect: an ordinary GLM, and exactly that
        s2h, phh = 0.0, (float(phi) if phi is not None else dmax / 3.0)
        Sig = [[1e-10 if i == j else 0.0 for j in range(n)]
               for i in range(n)]
        res = _laplace(yv, Xm, Sig, family, inner_iter, tol, disp)
        at_bound = False
    else:
        lo_s, hi_s = math.log(1e-6), math.log(1e3)
        lo_p, hi_p = math.log(max(dmin, 1e-6) / 4.0), math.log(dmax * 4.0)
        s2h = 1.0 if sigma2 is None else float(sigma2)
        phh = dmax / 3.0 if phi is None else float(phi)
        for _ in range(int(outer_cycles)):
            if sigma2 is None:
                # the correlation matrix does not depend on sigma2, so it is
                # built once and reused across the whole variance search --
                # rebuilding it per step is what made the Matern model
                # unusable, since every entry costs a Bessel evaluation
                R = corrmat(phh)

                def fs(ls):
                    r = _laplace(yv, Xm, scaled(R, math.exp(ls)), family,
                                 inner_iter, tol, disp)
                    return -1e300 if r is None else r[0]
                ls_hat = _golden(fs, lo_s, hi_s)
                s2h = math.exp(ls_hat)
            if phi is None:
                def fp(lp):
                    r = _laplace(yv, Xm, scaled(corrmat(math.exp(lp)), s2h),
                                 family, inner_iter, tol, disp)
                    return -1e300 if r is None else r[0]
                phh = math.exp(_golden(fp, lo_p, hi_p))
            if fit_disp:
                Rd = corrmat(phh)
                Sd = scaled(Rd, s2h)

                def fdp(ld):
                    r = _laplace(yv, Xm, Sd, family, inner_iter, tol,
                                 math.exp(ld))
                    return -1e300 if r is None else r[0]
                disp = math.exp(_golden(fdp, math.log(1e-8), math.log(1e4)))
            if sigma2 is not None and phi is not None and not fit_disp:
                break
        # a variance pinned against its lower bound means the data carry no
        # spatial signal, and the range is then not identified at all --
        # reported rather than left as a number that looks like an estimate
        at_bound = sigma2 is None and s2h < math.exp(lo_s) * 1.01
        Sig = scaled(corrmat(phh), s2h)
        res = _laplace(yv, Xm, Sig, family, inner_iter, tol, disp)
    if res is None:
        raise ValueError("sgflrt: the penalised system is not positive "
                         "definite -- try a positive nugget, or a shorter "
                         "range")
    lap, beta, u, mu, eta, loglik, w, Lsig, vlat = res

    # standard errors for beta: the curvature after profiling out v, in the
    # same parameterisation the fit used
    WL = [[w[i] * Lsig[i][j] for j in range(n)] for i in range(n)]
    H = [[sum(Lsig[i][j] * WL[i][j2] for i in range(n))
          + (1.0 if j == j2 else 0.0) for j2 in range(n)]
         for j in range(n)]
    LH = _chol(H)
    XtWX = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in range(n))
             for b in range(p)] for a in range(p)]
    B = [[sum(Xm[i][a] * WL[i][j] for i in range(n)) for j in range(n)]
         for a in range(p)]
    HiB = [_solve(LH, B[a]) for a in range(p)]
    Ib = [[XtWX[a][b] - sum(B[a][j] * HiB[b][j] for j in range(n))
           for b in range(p)] for a in range(p)]
    LIb = _chol(Ib)
    covb = _inv(LIb) if LIb is not None else [[float("nan")] * p
                                              for _ in range(p)]
    se = [math.sqrt(max(covb[a][a], 0.0)) if covb[a][a] == covb[a][a]
          else float("nan") for a in range(p)]

    # the Gaussian identity-link case has a closed form; compute it so the
    # approximation can be compared with the answer rather than trusted
    gls_gap = float("nan")
    if family == "gaussian":
        V = [[Sig[i][j] + (disp if i == j else 0.0) for j in range(n)]
             for i in range(n)]
        Lv = _chol(V)
        if Lv is not None:
            Viy = _solve(Lv, yv)
            ViX = [_solve(Lv, [Xm[i][a] for i in range(n)])
                   for a in range(p)]
            A = [[sum(Xm[i][a] * ViX[b][i] for i in range(n))
                  for b in range(p)] for a in range(p)]
            rhs = [sum(Xm[i][a] * Viy[i] for i in range(n))
                   for a in range(p)]
            LA = _chol(A)
            if LA is not None:
                bg = _solve(LA, rhs)
                gls_gap = max(abs(bg[a] - beta[a]) for a in range(p))

    return RichResult(payload={
        "estimate": beta, "coefficients": beta, "std_error": se,
        "z": [beta[a] / se[a] if se[a] > _EPS else float("nan")
              for a in range(p)],
        "spatial_effect": u, "fitted": mu, "linear_predictor": eta,
        "sigma2": s2h, "phi": phh, "dispersion": disp,
        "sigma2_at_lower_bound": at_bound,
        "spatial_signal": not at_bound, "kappa": float(kappa), "nugget": nug,
        "loglik": loglik, "laplace_loglik": lap,
        "gls_identity_gap": gls_gap,
        "covariance": covb,
        "family": family, "model": model,
        "n": n, "p": p, "d": d,
        "min_distance": dmin, "max_distance": dmax,
        "method": "spatial GLMM by the Laplace approximation: penalised "
                  "IRLS for the joint mode of (beta, u), the variance and "
                  "range by cycling golden-section searches on the "
                  "approximated marginal likelihood (Diggle, Tawn & Moyeed "
                  "1998; Breslow & Clayton 1993)",
        "note": "the Laplace approximation is exact for the Gaussian "
                "identity-link case, and gls_identity_gap is how far the "
                "fit sits from the closed-form GLS answer there; for "
                "binary data with few observations per correlated unit the "
                "variance component is biased downward (Breslow & Clayton "
                "1993) and n is reported so that can be judged; "
                "sigma2_at_lower_bound means the data carry no spatial "
                "signal, and phi is then not identified whatever value it "
                "was left at",
    })


def cheatsheet():
    return ("sgflrt: spatial_glmm_fit(y, X, coords, family) -> spatial GLMM "
            "by Laplace, with the spatial random effect returned "
            "(Diggle, Tawn & Moyeed 1998; Breslow & Clayton 1993)")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
spatialglmmfit = spatial_glmm_fit
