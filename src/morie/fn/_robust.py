# morie.fn -- internal helpers (rootcoder007/morie)
"""Shared machinery for the robust-regression and robust-scale shelf.

The organising trade-off of this shelf is BREAKDOWN against
EFFICIENCY. Least squares has efficiency 1 at the normal model and
breakdown 0 -- one bad point moves it arbitrarily. The estimators
here buy breakdown with redescending loss functions and pay in
efficiency, and the constants scattered through them (1.345, 1.547,
4.685, 2.2191, 1.1926) are not folklore: each is the solution of a
calibration equation, stated at its definition below, and the tests
recompute the ones that have closed forms.

References
----------
Rousseeuw, P. J., & Yohai, V. (1984). "Robust Regression by Means of
S-Estimators." In *Robust and Nonlinear Time Series Analysis*, Lecture
Notes in Statistics 26, Springer, 256-272.

Rousseeuw, P. J., & Croux, C. (1993). "Alternatives to the Median
Absolute Deviation." *Journal of the American Statistical Association*
88(424), 1273-1283.

None of these PDFs is in the local library; they are cited from
bibliographic details, and the formulas here have not been
re-verified against them.
"""

from . import _array_core as np

__all__ = ["mad_scale", "huber_psi", "tukey_rho", "tukey_weight",
           "irls", "s_scale", "s_regression", "mm_regression",
           "prepare_design", "HUBER_C_95", "TUKEY_C_BREAKDOWN",
           "TUKEY_C_95", "QN_D", "SN_C"]

# Huber (1964, 1973): psi_c(u) = clip(u, -c, c). c = 1.345 gives 95%
# efficiency at the normal model -- it solves
# eff(c) = (int psi' dPhi)^2 / int psi^2 dPhi = 0.95.
HUBER_C_95 = 1.345

# Tukey biweight rho(u) = min(1, 1 - (1 - (u/c)^2)^3) (scaled to
# rho(inf) = 1). c = 1.5476 makes E_Phi[rho] = 1/2, which is what
# gives the S-estimator its 50% breakdown point (Rousseeuw and Yohai
# 1984, Table 1); c = 4.685 gives the M-step 95% normal efficiency
# (Yohai 1987).
TUKEY_C_BREAKDOWN = 1.5476
TUKEY_C_95 = 4.685

# Rousseeuw and Croux (1993): Qn = d * {|x_i - x_j|, i < j}_(k).
# d = 1/(sqrt(2) Phi^-1(5/8)) makes Qn consistent for sigma at the
# normal; the paper prints 2.2219 from a slightly different rounding,
# and the exact constant is what the code uses.
QN_D = 2.21914446598508    # = 1 / (sqrt(2) * qnorm(5/8))

# Sn = c * lowmed_i highmed_j |x_i - x_j|, c = 1.1926 for normal
# consistency (Rousseeuw and Croux 1993, Sec. 2).
SN_C = 1.1926


def mad_scale(r):
    r"""Median absolute deviation about the median, scaled by
    1/Phi^-1(3/4) = 1.4826 for consistency at the normal."""
    r = np.asarray(r, dtype=float).ravel()
    m = float(np.median(np.abs(r - np.median(r))))
    return 1.482602218505602 * m


def huber_psi(u, c=HUBER_C_95):
    return np.clip(u, -c, c)


def tukey_rho(u, c):
    r"""Biweight rho scaled so rho(inf) = 1."""
    v = np.clip(np.asarray(u, dtype=float) / c, -1.0, 1.0)
    return 1.0 - (1.0 - v ** 2) ** 3


def tukey_weight(u, c):
    r"""w(u) = psi(u)/u for the biweight: (1 - (u/c)^2)^2 inside,
    zero outside. The REDESCENDING part -- weight exactly zero beyond
    c -- is what buys breakdown: a gross outlier gets no vote at
    all, where Huber's psi still gives it a bounded but non-zero
    one."""
    u = np.asarray(u, dtype=float)
    v = u / c
    w = (1.0 - v ** 2) ** 2
    return np.where(np.abs(v) < 1.0, w, 0.0)


def irls(X, y, weight_fn, scale, beta0, max_iter=100, tol=1e-10):
    """Iteratively reweighted least squares for a fixed scale."""
    beta = np.asarray(beta0, dtype=float).copy()
    for _ in range(int(max_iter)):
        r = y - X @ beta
        w = weight_fn(r / scale)
        if not np.any(w > 0):
            return beta, False
        W = w[:, None]
        A = X.T @ (X * W)
        b = X.T @ (w * y)
        new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol * (1 + np.max(np.abs(beta))):
            return new, True
        beta = new
    return beta, False


def s_scale(r, c=TUKEY_C_BREAKDOWN, b=0.5, max_iter=200, tol=1e-12):
    r"""The M-scale: the s solving
    :math:`\frac1n\sum_i \rho(r_i/s) = b` with the biweight rho.

    With b = E_Phi[rho] = 1/2 at c = 1.5476 the resulting scale
    (and the S-estimator built on it) has breakdown point 50% -- the
    estimator survives until half the data are bad. The fixed-point
    iteration s^2 <- s^2 * mean(rho(r/s))/b is monotone and
    convergent for the biweight.
    """
    r = np.asarray(r, dtype=float).ravel()
    s = mad_scale(r)
    if s <= 0:
        s = float(np.mean(np.abs(r))) or 1.0
    for _ in range(int(max_iter)):
        m = float(np.mean(tukey_rho(r / s, c)))
        if m <= 0:
            return 0.0
        new = s * np.sqrt(m / b)
        if abs(new - s) < tol * s:
            return float(new)
        s = new
    return float(s)


def cheatsheet():
    return ("_robust: every magic constant is a calibration equation -- "
            "1.345 (Huber 95%), 1.5476 (biweight 50% breakdown), "
            "4.685 (biweight 95%), 2.2191 (Qn), 1.1926 (Sn)")


def prepare_design(X, y, intercept=True):
    """Common design handling: coerce, orient, optionally prepend 1s."""
    yv = np.asarray(y, dtype=float).ravel()
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.shape[0] != yv.size:
        A = A.T
    if A.shape[0] != yv.size:
        raise ValueError(
            f"X has {A.shape[0]} rows for {yv.size} responses.")
    if intercept and not np.any(np.all(np.isclose(A, 1.0), axis=0)):
        A = np.column_stack([np.ones(yv.size), A])
    return A, yv


def s_regression(X, y, n_subsets=200, seed=0, c=TUKEY_C_BREAKDOWN, b=0.5):
    r"""The S-estimator of Rousseeuw and Yohai (1984): the beta whose
    residuals have the SMALLEST M-scale,

    .. math:: \hat\beta_S = \arg\min_\beta s(r_1(\beta), \dots),
              \qquad \frac1n\sum_i \rho\!\left(\frac{r_i}
              {s}\right) = b .

    The objective is non-convex, so the standard strategy is random
    p-subsets: fit exactly through p points, compute the residual
    M-scale, keep the best, then refine by IRLS at the winning scale.
    With the biweight at c = 1.5476 and b = 1/2 the breakdown point
    is 50% -- but the price is normal efficiency of only 28.7%, which
    is why the S-estimate is a STARTING POINT for the MM step rather
    than an endpoint.
    """
    n, p = X.shape
    if n <= p:
        raise ValueError(f"need more observations than parameters, "
                         f"got n = {n}, p = {p}.")
    rng = np.random.default_rng(seed)
    best_s = np.inf
    best_beta = None
    for _ in range(int(n_subsets)):
        idx = rng.choice(n, p, replace=False)
        sub = X[idx]
        if np.linalg.matrix_rank(sub) < p:
            continue
        try:
            beta = np.linalg.solve(sub, y[idx])
        except np.linalg.LinAlgError:
            continue
        sc = s_scale(y - X @ beta, c=c, b=b)
        if 0 < sc < best_s:
            best_s = sc
            best_beta = beta
    if best_beta is None:
        raise ValueError("no non-singular p-subset was found; the design "
                         "is rank-deficient.")
    # local improvement: IRLS at the current scale, re-solving the
    # scale as beta moves
    beta = best_beta
    for _ in range(50):
        beta_new, _ = irls(X, y, lambda u: tukey_weight(u, c), best_s, beta,
                           max_iter=1)
        sc = s_scale(y - X @ beta_new, c=c, b=b)
        if sc >= best_s - 1e-12:
            break
        best_s, beta = sc, beta_new
    return beta, float(best_s)


def mm_regression(X, y, n_subsets=200, seed=0, c_eff=TUKEY_C_95):
    r"""Yohai's (1987) MM-estimator: an S-estimate supplies the scale
    (and the 50% breakdown), then an M-step with the biweight at
    c = 4.685 supplies 95% normal efficiency, iterating from the
    S-estimate WITHOUT updating the scale. Keeping the scale fixed is
    what lets the M-step inherit the S-scale's breakdown -- update it
    and the high-efficiency rho's larger c lets outliers back into
    the scale, and the breakdown drops."""
    beta_s, scale = s_regression(X, y, n_subsets=n_subsets, seed=seed)
    beta, conv = irls(X, y, lambda u: tukey_weight(u, c_eff), scale, beta_s)
    return beta, scale, beta_s, conv