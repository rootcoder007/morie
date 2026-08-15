# morie.fn -- function file (rootcoder007/morie)
r"""Bayesian confidence intervals for the cross-validated smoothing spline.

For the model :math:`Y(t_i) = g(t_i) + \epsilon_i`, :math:`\epsilon \sim
(0, \sigma^2 I)`, the smoothing spline :math:`\hat g_{n,\lambda}` is the
minimiser of

.. math:: n^{-1} \sum_i (y_i - g(t_i))^2
          + \lambda \int_0^1 (g^{(m)}(t))^2 \, dt,

and is linear in the data: :math:`\hat g = A(\lambda) y`, where
:math:`a_{ij}(\lambda) = \partial \hat g_{n,\lambda}(t_i) / \partial
y_j` -- "which is the source of the terminology *influence matrix*".

The whole method rests on the paper's **Theorem 1**: under the prior of
its Section 2, the posterior covariance matrix of the fitted values is

.. math:: \mathrm{cov}(\hat g_{n,\lambda} \mid Y) = \sigma^2 A(\lambda),

so the interval at a design point uses the *diagonal* of the influence
matrix. With :math:`\sigma` unknown the paper estimates it by

.. math:: \hat\sigma^2(\lambda) = \frac{\mathrm{RSS}(\lambda)}
                                       {n(1 - a(\lambda))},
          \qquad n(1 - a(\lambda)) = \mathrm{EDF}(\lambda)
                                   = \mathrm{Tr}(I - A(\lambda)),

:math:`n a(\lambda) = \mathrm{Tr}\,A(\lambda)` being the equivalent
degrees of freedom for signal, and forms

.. math:: \hat g_{n,\hat\lambda}(t_i) \pm 1.96\,
          \hat\sigma(\hat\lambda) \sqrt{a_{ii}(\hat\lambda)} .

The paper raises, and this implementation exposes as ``quantile``,
"a conceptual question whether 1.96 or the 0.025 point of the *t*
distribution with EDF(:math:`\lambda`) degrees of freedom should be
used"; it reports that for :math:`n = 32` the *t* point "would most
likely have improved the confidence intervals obtained here somewhat",
so ``quantile="t"`` is offered and is the better default at small
:math:`n`. ``"normal"` reproduces the paper's own Monte Carlo exactly.

:math:`\lambda` is chosen by generalised cross-validation, eq. (2.16):

.. math:: V(\lambda) = \frac{n^{-1} \| (I - A(\lambda)) y \|^2}
                            {\left[ n^{-1}
                             \mathrm{Tr}(I - A(\lambda)) \right]^2}.

**What the coverage statement actually says.** These are *not* pointwise
confidence intervals in the frequentist sense, and the paper never
claims they are. The Monte Carlo studies "to what extent the resulting
95 per cent confidence intervals can be expected to cover about 95 per
cent of the true (but in practice unknown) values of :math:`g(t_i)`" --
coverage averaged *across the function*, not at any fixed point.
``coverage`` in the result reports exactly that quantity when the truth
is supplied, and is documented as an across-the-function rate.

The influence matrix is built from the Reinsch/Green-Silverman
formulation of the natural cubic spline (:math:`m = 2`), which gives
:math:`A(\lambda) = (I + \lambda K)^{-1}` with :math:`K = Q R^{-1} Q'`
for the usual banded :math:`Q` and :math:`R`. Because :math:`Q'` kills
constants and linear terms, :math:`K` does too, and hence :math:`A`
reproduces any straight line exactly at every :math:`\lambda` -- the
identity the tests use as an anchor.

References
----------
Wahba, G. (1983) "Bayesian 'confidence intervals' for the
cross-validated smoothing spline", *Journal of the Royal Statistical
Society, Series B (Methodological)* 45(1), 133-150,
doi:10.1111/j.2517-6161.1983.tb01239.x -- Theorem 1 (eq. 2.4) for the
posterior covariance, eq. (2.16) for GCV, and Section 3 for
:math:`\hat\sigma^2(\lambda) = \mathrm{RSS}(\lambda)/n(1-a(\lambda))`
and the :math:`\pm 1.96 \hat\sigma \sqrt{a_{ii}}` intervals.

Craven, P. and Wahba, G. (1979) "Smoothing noisy data with spline
functions: estimating the correct degree of smoothing by the method of
generalized cross-validation", *Numerische Mathematik* 31(4), 377-403,
doi:10.1007/BF01404567 -- the GCV criterion the 1983 paper refers to.

Green, P. J. and Silverman, B. W. (1994) *Nonparametric Regression and
Generalized Linear Models: A Roughness Penalty Approach*, Monographs on
Statistics and Applied Probability 58, Chapman & Hall, London,
ISBN 978-0-412-30040-0 -- Sec. 2.1.2 for the banded Q and R that build
the roughness matrix used here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from . import _stats_core as _st
from ._richresult import RichResult

__all__ = ["funBand", "functional_band", "influence_matrix", "gcv_score",
           "cheatsheet"]


def _qr_bands(x):
    r"""The banded :math:`Q` (n x n-2) and :math:`R` (n-2 x n-2) of the
    natural cubic spline, Green & Silverman Sec. 2.1.2."""
    n = len(x)
    if n < 4:
        raise ValueError("funBand: a cubic smoothing spline needs at least "
                         "four distinct design points, got %d" % n)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    if any(v <= 0.0 for v in h):
        raise ValueError("funBand: the design points must be strictly "
                         "increasing and distinct")
    m = n - 2
    Q = [[0.0] * m for _ in range(n)]
    R = [[0.0] * m for _ in range(m)]
    for j in range(m):          # column j corresponds to interior knot j+1
        Q[j][j] = 1.0 / h[j]
        Q[j + 1][j] = -1.0 / h[j] - 1.0 / h[j + 1]
        Q[j + 2][j] = 1.0 / h[j + 1]
        R[j][j] = (h[j] + h[j + 1]) / 3.0
        if j + 1 < m:
            R[j][j + 1] = R[j + 1][j] = h[j + 1] / 6.0
    return Q, R


def _roughness(x):
    r""":math:`K = Q R^{-1} Q'`, symmetric positive semi-definite with a
    two-dimensional null space spanned by the constant and the identity."""
    Q, R = _qr_bands(x)
    n = len(x)
    m = n - 2
    # solve R Z = Q' column by column, i.e. Z = R^-1 Q'
    Z = []
    for i in range(n):
        Z.append(k.cholsolve(R, [Q[i][j] for j in range(m)]))
    K = [[sum(Q[i][t] * Z[j][t] for t in range(m)) for j in range(n)]
         for i in range(n)]
    # symmetrise against round-off: K is symmetric by construction
    for i in range(n):
        for j in range(i + 1, n):
            v = 0.5 * (K[i][j] + K[j][i])
            K[i][j] = K[j][i] = v
    return K


def influence_matrix(x, lam):
    r"""The influence matrix :math:`A(\lambda) = (I + \lambda K)^{-1}`.

    :math:`a_{ij} = \partial \hat g(t_i)/\partial y_j`, the matrix whose
    diagonal Theorem 1 turns into posterior variances.

    Computed spectrally: :math:`K = U D U^\top` once, then
    :math:`A = U (I + \lambda D)^{-1} U^\top`, so each eigenvalue is
    damped by the scalar :math:`1/(1 + \lambda d_i)`. Factorising
    :math:`I + \lambda K` directly instead loses positive definiteness to
    round-off once :math:`\lambda \|K\|` reaches about :math:`10^{12}`,
    and the limiting case :math:`\lambda \to \infty` -- where the trace
    must fall to exactly two -- is precisely where that happens. The
    spectral form is exact at every :math:`\lambda`: as :math:`\lambda`
    grows every factor with :math:`d_i > 0` vanishes and only the
    two-dimensional null space of :math:`K`, spanned by the constant and
    the linear term, survives. That limit is the least-squares line.
    """
    xs = [float(v) for v in k.vec(x)]
    n = len(xs)
    lm = float(lam)
    if lm < 0.0:
        raise ValueError("funBand: lambda must be non-negative")
    K = _roughness(xs)
    # k.jacobi returns the eigenvectors as COLUMNS of U, so the spectral
    # sum contracts over the column index: K = sum_t d_t U[.,t] U[.,t]'.
    d, U = k.jacobi(K)
    # K annihilates constants and linear terms exactly, so two of its
    # eigenvalues are ZERO by construction. Jacobi returns them as O(1e-10)
    # relative to the largest, and at lambda = 1e15 that residue multiplies
    # up to damp the null space as well -- the straight line stops being
    # reproduced and the trace falls below two. Rounding sub-tolerance
    # eigenvalues to exact zeros restores the algebraic property at every
    # lambda; the tolerance is the usual rank cut, n * eps * max|d|.
    dmax = max(abs(v) for v in d) if d else 0.0
    tol = n * 2.220446049250313e-16 * dmax
    w = [1.0 if abs(v) <= tol else 1.0 / (1.0 + lm * (v if v > 0.0 else 0.0))
         for v in d]
    A = [[sum(U[i][t] * w[t] * U[j][t] for t in range(n))
          for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = 0.5 * (A[i][j] + A[j][i])
            A[i][j] = A[j][i] = v
    return A


def gcv_score(y, A):
    r"""Wahba (1983) eq. (2.16):
    :math:`V(\lambda) = n^{-1}\|(I-A)y\|^2 / [n^{-1}\mathrm{Tr}(I-A)]^2`."""
    n = len(y)
    fit = k.matvec(A, y)
    rss = sum((y[i] - fit[i]) ** 2 for i in range(n))
    tr_ia = float(n) - sum(A[i][i] for i in range(n))
    if abs(tr_ia) < 1e-12:
        return float("inf")
    return (rss / n) / ((tr_ia / n) ** 2)


def funBand(Y, alpha=0.05, x=None, lam=None, quantile="t", truth=None,
            n_lambda=40, log_lambda_range=(-8.0, 8.0)):
    r"""Smoothing-spline fit with Wahba's Bayesian confidence intervals.

    Parameters
    ----------
    Y : array-like
        The observations :math:`y_i`.
    alpha : float
        One minus the nominal level; ``0.05`` gives the paper's 95 per
        cent intervals.
    x : array-like, optional
        Design points, strictly increasing. Defaults to
        :math:`t_i = i/n`, the equally spaced design of the paper's
        Monte Carlo study.
    lam : float, optional
        The smoothing parameter. ``None`` selects it by minimising the
        GCV function (2.16) over a log-spaced grid.
    quantile : {"t", "normal"}
        ``"normal"`` uses 1.96 (at the default level), which is what the
        paper's own experiments used; ``"t"`` uses the 0.025 point of the
        t distribution on EDF degrees of freedom, which the paper says
        "would most likely have improved the confidence intervals" at
        small n.
    truth : array-like, optional
        The true :math:`g(t_i)`, available in simulation. When given,
        ``coverage`` reports the fraction of the n intervals containing
        it -- the ACROSS-THE-FUNCTION rate the paper studies, not a
        pointwise coverage probability.

    Returns
    -------
    RichResult
        ``estimate`` is the fitted curve; ``lower``/``upper`` the band,
        ``diag_A`` the posterior variances divided by
        :math:`\hat\sigma^2`, ``edf_signal`` = Tr A and ``edf_error``
        = Tr(I - A).
    """
    y = [float(v) for v in k.vec(Y)]
    n = len(y)
    if n < 4:
        raise ValueError("funBand: need at least four observations, got %d"
                         % n)
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("funBand: alpha must lie in (0, 1), got %g" % a)
    if x is None:
        xs = [(i + 1.0) / n for i in range(n)]
    else:
        xs = [float(v) for v in k.vec(x)]
        if len(xs) != n:
            raise ValueError("funBand: %d observations but %d design points"
                             % (n, len(xs)))
    if quantile not in ("t", "normal"):
        raise ValueError("funBand: quantile must be 't' or 'normal', got %r"
                         % (quantile,))

    if lam is None:
        lo, hi = (float(v) for v in log_lambda_range)
        grid = [10.0 ** (lo + (hi - lo) * t / (int(n_lambda) - 1.0))
                for t in range(int(n_lambda))]
        best = None
        for lm in grid:
            A = influence_matrix(xs, lm)
            v = gcv_score(y, A)
            if best is None or v < best[0]:
                best = (v, lm, A)
        gcv, lam_used, A = best
    else:
        lam_used = float(lam)
        A = influence_matrix(xs, lam_used)
        gcv = gcv_score(y, A)

    fit = k.matvec(A, y)
    resid = [y[i] - fit[i] for i in range(n)]
    rss = sum(v * v for v in resid)
    tr_a = sum(A[i][i] for i in range(n))
    edf_err = float(n) - tr_a
    if edf_err <= 0.0:
        raise ValueError("funBand: the fit has no residual degrees of "
                         "freedom; lambda is too small for these data")
    sigma2 = rss / edf_err
    sigma = math.sqrt(sigma2)
    diag = [A[i][i] for i in range(n)]

    if quantile == "normal":
        z = _st.norm.ppf(1.0 - a / 2.0)
    else:
        z = _st.t.ppf(1.0 - a / 2.0, edf_err)
    half = [z * sigma * math.sqrt(v if v > 0.0 else 0.0) for v in diag]
    lower = [fit[i] - half[i] for i in range(n)]
    upper = [fit[i] + half[i] for i in range(n)]

    cover = None
    if truth is not None:
        g = [float(v) for v in k.vec(truth)]
        if len(g) != n:
            raise ValueError("funBand: %d observations but %d true values"
                             % (n, len(g)))
        cover = sum(1 for i in range(n)
                    if lower[i] <= g[i] <= upper[i]) / float(n)

    return RichResult(payload={
        "estimate": list(fit),
        "fitted": list(fit),
        "lower": lower,
        "upper": upper,
        "half_width": half,
        "residuals": resid,
        "diag_A": diag,
        "posterior_variance": [sigma2 * v for v in diag],
        "sigma2": sigma2,
        "sigma": sigma,
        "lambda": lam_used,
        "gcv": gcv,
        "edf_signal": tr_a,
        "edf_error": edf_err,
        "rss": rss,
        "multiplier": z,
        "quantile": quantile,
        "coverage": cover,
        "alpha": a,
        "n": n,
        "x": xs,
        "method": ("Bayesian confidence intervals for the cross-validated "
                   "smoothing spline, Wahba (1983) Theorem 1 with GCV "
                   "eq. (2.16)"),
        "note": ("cov(g_hat | Y) = sigma^2 A(lambda), so the band uses the "
                 "DIAGONAL of the influence matrix; coverage is measured "
                 "ACROSS THE FUNCTION -- the fraction of the n true values "
                 "covered -- and is not a pointwise coverage probability"),
    })


# the descriptive name kept as an alias, per the naming rules
functional_band = funBand
functionalband = funBand


def cheatsheet():
    return ("funBand: smoothing-spline band. Theorem 1 of Wahba (1983): "
            "cov(g_hat|Y) = sigma^2 A(lambda), so the interval at t_i is "
            "g_hat +- z sigma_hat sqrt(a_ii) using the DIAGONAL of the "
            "influence matrix. sigma_hat^2 = RSS/Tr(I-A); lambda by GCV "
            "eq. (2.16) V = n^-1||(I-A)y||^2 / [n^-1 Tr(I-A)]^2. Coverage "
            "is ACROSS THE FUNCTION, not pointwise. A reproduces straight "
            "lines exactly at every lambda.")
