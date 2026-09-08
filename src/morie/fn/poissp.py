# morie.fn -- function file (rootcoder007/morie)
r"""Poisson areal regression with a conditionally autoregressive effect.

The count in area :math:`i` is modelled as

.. math:: Y_i \mid \mu_i \sim \mathrm{Poisson}(E_i \mu_i), \qquad
          \log \mu_i = x_i^\top \beta + u_i,

with :math:`E_i` a known *offset* -- the expected count under the null,
usually an internally standardised population -- so that
:math:`\exp(x_i^\top\beta + u_i)` is a relative risk rather than a rate.
The offset enters the linear predictor with coefficient fixed at one;
it is not estimated.

The spatial term is the **proper CAR** of the chapter: conditionally,

.. math:: u_i \mid u_{-i} \sim
          N\!\left(\rho \frac{\sum_j w_{ij} u_j}{w_{i+}},
                   \frac{1}{\tau\, w_{i+}}\right),

which by Brook's lemma is jointly :math:`u \sim N(0, Q^{-1})` with
precision :math:`Q = \tau (D_w - \rho W)`, :math:`D_w =
\mathrm{diag}(w_{i+})`. Two facts about this parametrisation are worth
stating because they are what the anchors check:

* :math:`\rho = 0` leaves :math:`Q = \tau D_w`, so the effects are
  independent with variance :math:`1/(\tau w_{i+})` -- *not* a common
  variance, unless every area has the same number of neighbours.
* :math:`\rho = 1` gives the **intrinsic** CAR: :math:`Q\mathbf{1} = 0`,
  the precision is singular, the prior is improper and identifies only
  contrasts. The intercept and the spatial effect are then confounded,
  which is why ``constrain`` imposes :math:`\sum_i u_i = 0`. The book
  recommends exactly this prior for areal disease mapping.

Propriety needs :math:`\rho \in (1/\lambda_{\min}, 1/\lambda_{\max})`
for the eigenvalues of :math:`D_w^{-1/2} W D_w^{-1/2}`; since that
matrix is a symmetric stochastic-similar matrix, :math:`\lambda_{\max} =
1` and the upper end is always one. ``rho_bounds`` returns the interval
and the fit refuses a value outside it.

**Fitting.** For fixed :math:`(\tau, \rho)` the joint mode of
:math:`(\beta, u)` solves the penalised score equations

.. math:: X^\top (y - m) = 0, \qquad (y - m) - Q u = 0,
          \qquad m_i = E_i e^{x_i^\top\beta + u_i},

by Newton-Raphson on the joint system, which for the canonical log link
is iteratively reweighted least squares with weight :math:`m` and a
:math:`Q` block added to the :math:`u`-:math:`u` corner of the Hessian.
That the *fixed-effect* score is unpenalised is the second anchor: at
convergence :math:`X^\top(y - m)` is zero to machine precision, so an
intercept column forces :math:`\sum_i y_i = \sum_i m_i` exactly.

:math:`\tau` is chosen by maximising the Laplace approximation to the
marginal likelihood, which for this model is

.. math:: \ell(\tau) = \ell_{\text{Poisson}}(\hat\beta, \hat u)
          - \tfrac12 \hat u^\top Q \hat u
          + \tfrac12 \log|Q|_{+} - \tfrac12 \log|Q + \hat M|,

with :math:`\hat M = \mathrm{diag}(\hat m)` and :math:`|\cdot|_+` the
generalised determinant (the product of the nonzero eigenvalues) under
the intrinsic model, whose rank is :math:`n - 1`.

References
----------
Banerjee, S., Carlin, B. P. and Gelfand, A. E. (2014) *Hierarchical
Modeling and Analysis for Spatial Data*, 2nd edn, Monographs on
Statistics and Applied Probability 135, Chapman & Hall/CRC, Boca Raton,
ISBN 978-1-4398-1917-3 -- Ch. 4 (areal data models; the proper and
intrinsic CAR, eqs. 4.13-4.16) and Ch. 6 (hierarchical modelling for
areal counts; the Poisson-CAR disease-mapping model).

Besag, J., York, J. and Mollie, A. (1991) "Bayesian image restoration,
with two applications in spatial statistics", *Annals of the Institute
of Statistical Mathematics* 43(1), 1-20, doi:10.1007/BF00116466 -- the
intrinsic CAR and the convolution prior the chapter builds on.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["poissp", "poisson_spatial_glm", "car_precision", "rho_bounds",
           "cheatsheet"]


def _adjacency(W):
    """Validate the weight matrix: square, symmetric, non-negative, no loops."""
    A = [[float(v) for v in row] for row in k.mat(W)]
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError("poissp: W must be a square weight matrix")
    for i in range(n):
        if A[i][i] != 0.0:
            raise ValueError("poissp: W must have a zero diagonal; area %d "
                             "is its own neighbour" % i)
        for j in range(n):
            if A[i][j] < 0.0:
                raise ValueError("poissp: weights must be non-negative")
            if abs(A[i][j] - A[j][i]) > 1e-12:
                raise ValueError("poissp: W must be symmetric; w[%d][%d] "
                                 "and w[%d][%d] differ" % (i, j, j, i))
    return A, n


def car_precision(W, tau=1.0, rho=1.0):
    r"""The proper-CAR precision :math:`Q = \tau (D_w - \rho W)`.

    With ``rho=1`` this is the intrinsic CAR precision, singular by
    construction: every row sums to zero.
    """
    A, n = _adjacency(W)
    d = [sum(A[i]) for i in range(n)]
    t, r = float(tau), float(rho)
    if t <= 0.0:
        raise ValueError("poissp: tau must be positive")
    return [[t * ((d[i] if i == j else 0.0) - r * A[i][j])
             for j in range(n)] for i in range(n)]


def rho_bounds(W):
    r"""The propriety interval for :math:`\rho`.

    :math:`Q` is positive definite exactly when :math:`\rho` lies
    strictly between the reciprocals of the extreme eigenvalues of
    :math:`D_w^{-1/2} W D_w^{-1/2}`; the upper bound is one for any
    connected graph.
    """
    A, n = _adjacency(W)
    d = [sum(A[i]) for i in range(n)]
    if any(v <= 0.0 for v in d):
        raise ValueError("poissp: every area needs at least one neighbour")
    S = [[A[i][j] / math.sqrt(d[i] * d[j]) for j in range(n)]
         for i in range(n)]
    ev, _ = k.jacobi(S)
    lo, hi = min(ev), max(ev)
    return {"lower": 1.0 / lo if lo < 0 else float("-inf"),
            "upper": 1.0 / hi if hi > 0 else float("inf"),
            "eigenvalues": list(ev)}


def _logdet_pd(A, ridge=0.0):
    """log|A| for a symmetric positive-definite A, via Cholesky."""
    n = len(A)
    M = [[A[i][j] + (ridge if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    L = k.chol(M)
    return 2.0 * sum(math.log(L[i][i]) for i in range(n))


def _logdet_gen(A, rank_deficit=0):
    """Generalised log-determinant: the sum of logs of the nonzero
    eigenvalues, dropping the ``rank_deficit`` smallest in magnitude."""
    ev, _ = k.jacobi(A)
    vals = sorted((abs(v) for v in ev), reverse=True)
    keep = vals[:len(vals) - int(rank_deficit)] if rank_deficit else vals
    return sum(math.log(v) for v in keep if v > 1e-300)


def _constraint_weight(m):
    """Weight of the sum(u) = 0 penalty: large relative to the data
    information, so the constrained mode is the unconstrained mode of the
    penalised problem."""
    return 1e8 * max(1.0, max(m) if m else 1.0)


def _joint_hessian(X, m, Q, constrain):
    """The negative joint Hessian [[X'MX, X'M], [MX, M + Q]], plus the
    sum-to-zero penalty when the model is intrinsic."""
    n = len(m)
    p = len(X[0])
    dim = p + n
    H = [[0.0] * dim for _ in range(dim)]
    for a in range(p):
        for b in range(p):
            H[a][b] = sum(X[i][a] * m[i] * X[i][b] for i in range(n))
        for i in range(n):
            H[a][p + i] = H[p + i][a] = X[i][a] * m[i]
    for i in range(n):
        for j in range(n):
            H[p + i][p + j] = Q[i][j] + (m[i] if i == j else 0.0)
    if constrain:
        big = _constraint_weight(m)
        for i in range(n):
            for j in range(n):
                H[p + i][p + j] += big
    return H


def _fit_mode(y, X, off, Q, constrain, iters, tol, ridge):
    """Joint Newton-Raphson for (beta, u) at fixed (tau, rho).

    The fixed effects are unpenalised, so their score X'(y - m) is zero
    at convergence; only the u-block carries the Q penalty.
    """
    n = len(y)
    p = len(X[0])
    beta = [0.0] * p
    # Start beta at the offset-only MLE of the intercept when there is
    # one, which is the exact closed form log(sum y / sum E) and puts
    # the first Newton step in the right basin for sparse counts.
    tot_y, tot_e = sum(y), sum(off)
    if p and all(abs(X[i][0] - 1.0) < 1e-12 for i in range(n)):
        beta[0] = math.log(tot_y / tot_e) if tot_y > 0 and tot_e > 0 else 0.0
    u = [0.0] * n
    dim = p + n
    for _ in range(int(iters)):
        eta = [sum(X[i][a] * beta[a] for a in range(p)) + u[i]
               for i in range(n)]
        m = [off[i] * math.exp(eta[i]) for i in range(n)]
        # score
        r = [y[i] - m[i] for i in range(n)]
        g = [sum(X[i][a] * r[i] for i in range(n)) for a in range(p)]
        Qu = k.matvec(Q, u)
        g += [r[i] - Qu[i] for i in range(n)]
        H = _joint_hessian(X, m, Q, constrain)
        if constrain:
            big = _constraint_weight(m)
            su = sum(u)
            for i in range(n):
                g[p + i] -= big * su
        step = k.ridgesolve(H, g, ridge)
        mx = 0.0
        for a in range(p):
            beta[a] += step[a]
        for i in range(n):
            u[i] += step[p + i]
        for v in step:
            if abs(v) > mx:
                mx = abs(v)
        if mx < tol:
            break
    eta = [sum(X[i][a] * beta[a] for a in range(p)) + u[i] for i in range(n)]
    m = [off[i] * math.exp(eta[i]) for i in range(n)]
    return beta, u, m, eta


def _poisson_loglik(y, m):
    """Full Poisson log-likelihood including the log(y!) term."""
    tot = 0.0
    for i in range(len(y)):
        tot += y[i] * math.log(m[i]) - m[i] - math.lgamma(y[i] + 1.0)
    return tot


def poissp(counts, X=None, offset=None, W=None, rho=1.0, tau=None,
           constrain=None, iters=100, tol=1e-11, ridge=1e-10,
           tau_grid=None, level=0.95):
    r"""Fit the Poisson-CAR areal model.

    Parameters
    ----------
    counts : array-like
        Observed counts :math:`y_i`, one per area.
    X : array-like, optional
        Covariates. An intercept column is prepended; pass ``None`` for
        an intercept-only model.
    offset : array-like, optional
        Expected counts :math:`E_i`, entering with coefficient one.
        Defaults to all ones.
    W : array-like, optional
        Symmetric non-negative adjacency/weight matrix with a zero
        diagonal. ``None`` drops the spatial term entirely and the fit
        reduces to an ordinary Poisson GLM with offset.
    rho : float
        The CAR propriety parameter. ``1.0`` (the default, and the
        book's recommendation for disease mapping) is the intrinsic CAR.
    tau : float, optional
        The CAR precision. ``None`` estimates it by maximising the
        Laplace approximation to the marginal likelihood over
        ``tau_grid``.
    constrain : bool, optional
        Impose :math:`\sum_i u_i = 0`. Defaults to ``True`` exactly when
        the model is intrinsic (``rho == 1``), where it is required for
        identification.

    Returns
    -------
    RichResult
        ``estimate`` is the vector of fixed effects; ``u`` the fitted
        spatial effects, ``relative_risk`` the :math:`\exp(\eta)` per
        area, ``score_beta`` the fixed-effect score at the mode (zero to
        machine precision -- the model's own falsifiable check).
    """
    y = [float(v) for v in k.vec(counts)]
    n = len(y)
    if n == 0:
        raise ValueError("poissp: no observations")
    if any(v < 0 for v in y):
        raise ValueError("poissp: counts must be non-negative")
    if any(abs(v - round(v)) > 1e-9 for v in y):
        raise ValueError("poissp: counts must be integers")
    off = [1.0] * n if offset is None else [float(v) for v in k.vec(offset)]
    if len(off) != n:
        raise ValueError("poissp: %d counts but %d offsets" % (n, len(off)))
    if any(v <= 0.0 for v in off):
        raise ValueError("poissp: offsets must be positive")
    Xd = k.design(X, n)
    if len(Xd) != n:
        raise ValueError("poissp: %d counts but %d covariate rows"
                         % (n, len(Xd)))

    if W is None:
        Q = [[0.0] * n for _ in range(n)]
        # no spatial term: pin u at zero by an enormous precision
        for i in range(n):
            Q[i][i] = 1e12
        rho_used, tau_used, spatial = None, None, False
        constrain = False
    else:
        spatial = True
        rho_used = float(rho)
        if constrain is None:
            constrain = abs(rho_used - 1.0) < 1e-12
        if abs(rho_used - 1.0) > 1e-12:
            b = rho_bounds(W)
            if not (b["lower"] < rho_used < b["upper"]):
                raise ValueError(
                    "poissp: rho = %g is outside the propriety interval "
                    "(%g, %g); the CAR prior would be improper"
                    % (rho_used, b["lower"], b["upper"]))
        if tau is None:
            grid = (tau_grid if tau_grid is not None else
                    [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0,
                     50.0, 100.0])
            best = None
            for t in grid:
                Qt = car_precision(W, t, rho_used)
                bb, uu, mm, _ = _fit_mode(y, Xd, off, Qt, constrain,
                                          iters, tol, ridge)
                lap = _laplace(y, uu, mm, Qt, constrain)
                if best is None or lap > best[0]:
                    best = (lap, t)
            tau_used = best[1]
        else:
            tau_used = float(tau)
            if tau_used <= 0.0:
                raise ValueError("poissp: tau must be positive")
        Q = car_precision(W, tau_used, rho_used)

    beta, u, m, eta = _fit_mode(y, Xd, off, Q, constrain, iters, tol, ridge)
    p = len(Xd[0])
    score = [sum(Xd[i][a] * (y[i] - m[i]) for i in range(n))
             for a in range(p)]

    # Fixed-effect covariance: the (beta, beta) block of the INVERSE joint
    # Hessian, obtained by solving H z = e_a rather than by forming the
    # Schur complement A - B D^-1 B' explicitly. With a large CAR
    # precision the two terms of that difference agree to many digits and
    # subtracting them loses every one of them -- the anchor with
    # tau -> infinity turns the resulting matrix indefinite. Solving the
    # joint system never forms the difference.
    H = _joint_hessian(Xd, m, Q, constrain)
    cov = []
    for a in range(p):
        e = [1.0 if t == a else 0.0 for t in range(p + n)]
        z = k.ridgesolve(H, e, ridge)
        cov.append(z[:p])
    se = [math.sqrt(cov[a][a]) if cov[a][a] > 0 else float("nan")
          for a in range(p)]
    z = k.qnorm(0.5 + 0.5 * float(level))
    lo = [beta[a] - z * se[a] for a in range(p)]
    hi = [beta[a] + z * se[a] for a in range(p)]

    ll = _poisson_loglik(y, m)
    dev = 2.0 * sum((y[i] * math.log(y[i] / m[i]) if y[i] > 0 else 0.0)
                    - (y[i] - m[i]) for i in range(n))

    return RichResult(payload={
        "estimate": list(beta),
        "beta": list(beta),
        "se": se,
        "lower": lo,
        "upper": hi,
        "u": list(u),
        "eta": list(eta),
        "fitted": list(m),
        "relative_risk": [math.exp(v) for v in eta],
        "score_beta": score,
        "loglik": ll,
        "deviance": dev,
        "tau": tau_used,
        "rho": rho_used,
        "spatial": spatial,
        "constrained": bool(constrain),
        "n": n,
        "p": p,
        "level": float(level),
        "method": ("Poisson areal regression with a %s CAR effect, "
                   "Banerjee, Carlin & Gelfand (2014) Ch. 4 and 6"
                   % ("intrinsic" if spatial and
                      abs((rho_used or 0.0) - 1.0) < 1e-12 else
                      ("proper" if spatial else "no"))),
        "note": ("the offset enters with coefficient fixed at one, so "
                 "exp(eta) is a relative risk; score_beta is zero at the "
                 "mode because the fixed effects are unpenalised"),
    })


def _laplace(y, u, m, Q, constrain):
    """Laplace approximation to the marginal log-likelihood at the mode."""
    n = len(y)
    quad = 0.0
    Qu = k.matvec(Q, u)
    for i in range(n):
        quad += u[i] * Qu[i]
    deficit = 1 if constrain else 0
    ldQ = _logdet_gen(Q, deficit)
    MQ = [[Q[i][j] + (m[i] if i == j else 0.0) for j in range(n)]
          for i in range(n)]
    ldH = _logdet_pd(MQ, 1e-12)
    return _poisson_loglik(y, m) - 0.5 * quad + 0.5 * ldQ - 0.5 * ldH


# the descriptive name kept as an alias, per the naming rules
poisson_spatial_glm = poissp


def cheatsheet():
    return ("poissp: Poisson areal regression, log mu = X beta + u with a "
            "known offset E entering at coefficient one, and u ~ CAR with "
            "precision tau(D_w - rho W). rho=1 is the INTRINSIC CAR: Q1=0, "
            "improper, needs sum(u)=0. The fixed-effect score X'(y-m) is "
            "zero at the mode, so an intercept forces sum(y)=sum(fitted). "
            "tau by Laplace marginal likelihood. BCG (2014) Ch. 4, 6.")
