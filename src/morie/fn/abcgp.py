# morie.fn -- function file (rootcoder007/morie)
r"""ABC with a Gaussian-process surrogate.

Two published methods answer to this name and they are not the same
algorithm, so both are here and the caller picks.

**Wilkinson (2014), the default.** Emulate the *log-likelihood* with a
GP and use the emulator, not the simulator, to trace out the posterior.
Sec. 2.1: the GABC likelihood

.. math:: \pi_{GABC}(D\mid\theta)=\int \pi(D\mid X)\pi(X\mid\theta)\,dX

is estimated by the unbiased Monte Carlo sum of his eq. (1),

.. math:: \hat\pi_{GABC}(D\mid\theta)=\frac1M\sum_{i=1}^{M}\pi(D\mid X_i)

evaluated with the log-sum-exp trick he gives in footnote 1, and the GP
is placed on :math:`l(\theta)=\log\pi_{GABC}(D\mid\theta)` rather than
on the likelihood itself because the likelihood "varies from 0 to very
small values, and is required to be positive". The prior mean is his
quadratic

.. math:: m_\beta(\theta)=\beta_0+\theta^T\beta_1
                          +\mathrm{diag}(\theta\theta^T)\beta_2

-- the square is there because :math:`l(\theta)\to-\infty` as
:math:`\theta\to\pm\infty`, so it is what makes the emulator behave
outside the design -- and the covariance is

.. math:: c_\psi(\theta_i,\theta_j)
          =\tau^2 c_\lambda(\theta_i,\theta_j)+v^2 1_{i=j},

the nugget :math:`v^2` being the *sampling* variance of
:math:`\hat l_M(\theta)`, not a numerical fudge: the training values are
noisy Monte Carlo estimates, and he estimates :math:`v^2` by the
bootstrap variance of the terms in the log-likelihood estimate.

Sec. 3, sequential history matching (Craig et al. 1997): a wave builds
a GP, rules parameter values *implausible* by his eq. (3),

.. math:: m + 3\sigma < \max_{\theta\in E}\hat l_M(\theta) - T,

and the next wave designs only inside what is left. :math:`T = 10` is
his choice, justified because :math:`e^{10} > 10^4` so discarding
:math:`\theta` with :math:`l(\theta) < l(\hat\theta) - 10` costs little.
The design is a Sobol sequence pushed through the prior's inverse CDF
(Sec. 2.2), not a random sample.

**Meeds & Welling (2014), GPS-ABC.** A Metropolis-Hastings sampler in
which the accept probability is itself a random variable, because
:math:`\hat\mu_\theta` and :math:`\hat\Sigma_\theta` come from finitely
many simulations. Their eq. (11) says the mean's uncertainty is

.. math:: \mu_\theta\sim N(\hat\mu_\theta,\hat\Sigma_\theta/S),

so :math:`M` draws give :math:`M` values of the accept probability by
eq. (12), the decision threshold is :math:`\tau=\mathrm{median}(\alpha)`
-- which minimises the error because eq. (16) is a mean absolute
deviation -- and eq. (15)-(16) give the probability that the accept or
reject decision was wrong. Algorithm 2 keeps simulating until that error
falls below :math:`\xi`; Algorithm 3 replaces the simulations entirely
with :math:`J` independent GP surrogates over the summary statistics, so
that nothing is thrown away between steps.

Their synthetic likelihood is eq. (9),

.. math:: \pi_\epsilon(y\mid\theta)=N(y;\hat\mu_\theta,
                                      \hat\Sigma_\theta+\epsilon^2 I),

which admits :math:`\epsilon\to 0`: the tolerance bias goes away and is
traded for the bias of assuming the simulator output is normal.

``method="synthetic"`` is their Algorithm 1, the fixed-:math:`S`
baseline both adaptive methods are built on, and is kept because it is
the only one of the four that runs no surrogate at all and so is the
thing the others must agree with when :math:`S` is large.

References
----------
Wilkinson, R. D. (2014) "Accelerating ABC methods using Gaussian
processes", *Proceedings of the 17th International Conference on
Artificial Intelligence and Statistics (AISTATS)*, PMLR 33, 1015-1023.

Meeds, E. & Welling, M. (2014) "GPS-ABC: Gaussian process surrogate
approximate Bayesian computation", *Proceedings of the 30th Conference
on Uncertainty in Artificial Intelligence (UAI)*, 593-602;
arXiv:1401.2838.

Wood, S. N. (2010) "Statistical inference for noisy nonlinear ecological
dynamic systems", *Nature* 466, 1102-1104, doi:10.1038/nature09319 --
the synthetic likelihood of Wilkinson's eq. (2) and Meeds & Welling's
eq. (9).

Craig, P. S., Goldstein, M., Seheult, A. H. & Smith, J. A. (1997)
"Pressure matching for hydrocarbon reservoirs: a case study in the use
of Bayes linear strategies for large computer experiments", in Gatsonis,
C. et al. (eds.), *Case Studies in Bayesian Statistics III*, Lecture
Notes in Statistics 121, Springer, 37-93,
doi:10.1007/978-1-4612-2290-3_2 -- sequential history matching.

Sobol, I. M. (1967) "On the distribution of points in a cube and the
approximate evaluation of integrals", *USSR Computational Mathematics
and Mathematical Physics* 7(4), 86-112,
doi:10.1016/0041-5553(67)90144-9, with the direction-number
construction of Bratley, P. & Fox, B. L. (1988) "Algorithm 659:
implementing Sobol's quasirandom sequence generator", *ACM Transactions
on Mathematical Software* 14(1), 88-100, doi:10.1145/42288.214372.

Rasmussen, C. E. & Williams, C. K. I. (2006) *Gaussian Processes for
Machine Learning*, MIT Press, doi:10.7551/mitpress/3206.001.0001 -- the
GP posterior Wilkinson refers to for details.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = [
    "abc_gp_emulator", "gabc_log_likelihood", "synthetic_log_likelihood",
    "gp_fit", "gp_predict", "implausible", "history_match",
    "sobol_sequence", "design_from_prior", "gps_abc", "synthetic_abc",
]

_METHODS = ("wilkinson", "gps", "adaptive", "synthetic")
_KERNELS = ("sqexp", "matern32", "matern52")

# Bratley & Fox (1988) Table 1: primitive polynomials as (degree, encoded
# coefficients) and the odd initial direction numbers m_1..m_degree.
# Dimension 1 is the plain van der Corput sequence in base 2 and needs no
# polynomial. Eight dimensions is well beyond what an ABC design uses.
_SOBOL_POLY = [
    None,                       # dimension 1: no polynomial, m_k = 1
    (1, 0, [1]),
    (2, 1, [1, 3]),
    (3, 1, [1, 3, 1]),
    (3, 2, [1, 1, 1]),
    (4, 1, [1, 1, 3, 3]),
    (4, 4, [1, 3, 5, 13]),
    (5, 2, [1, 1, 5, 5, 17]),
]


def _lse(values):
    """log(sum(exp(v))) with Wilkinson's footnote-1 shift."""
    vals = [v for v in values if v == v]          # drop nan
    if not vals:
        return float("-inf")
    a = max(vals)
    if a == float("-inf"):
        return float("-inf")
    return a + math.log(sum(math.exp(v - a) for v in vals))


# --------------------------------------------------------------------
# design
# --------------------------------------------------------------------

def sobol_sequence(n, dim, skip=0):
    """The first `n` points of the Sobol sequence in [0, 1)^dim.

    Sobol (1967) with the direction numbers and recurrence of Bratley &
    Fox (1988). Wilkinson Sec. 2.2 uses this rather than a maxi-min
    Latin hypercube for one stated reason: a Sobol design "can be
    extended when required", which is exactly what the history-matching
    waves need when they add points to an existing ensemble.

    The sequence starts at the origin, because the defining property --
    the first 2^m points form a (0, m, s)-net in base 2, one point in
    every elementary box of volume 2^-m -- counts from index 0. In one
    dimension the points are 0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625,
    0.125, which is the base-2 radical inverse in Gray-code order
    (Antonov-Saleev), and the first eight are exactly the eight dyadic
    eighths.

    `skip` drops that many leading points. :func:`design_from_prior`
    skips the origin, since an inverse-CDF map sends it to the corner of
    the prior box; a later history-matching wave sets `skip` to continue
    the same sequence rather than restart it.
    """
    n = int(n)
    dim = int(dim)
    if n < 1:
        raise ValueError("sobol_sequence: n must be at least 1, got %r" % n)
    if not 1 <= dim <= len(_SOBOL_POLY):
        raise ValueError(
            "sobol_sequence: dim must be between 1 and %d (the direction "
            "numbers tabulated here), got %r" % (len(_SOBOL_POLY), dim))
    total = n + int(skip)
    bits = max(1, int(math.ceil(math.log(total + 1, 2))) + 1)

    # v[d][k] are the direction numbers, held as integers scaled by 2^bits.
    v = []
    for d in range(dim):
        entry = _SOBOL_POLY[d]
        if entry is None:
            m = [1] * bits           # dimension 1 is plain base-2 radical
        else:
            degree, coeff, m_init = entry
            m = list(m_init)
            for k in range(degree, bits):
                # m_k = 2^d m_{k-d} xor m_{k-d} xor (2^j a_j m_{k-j})
                val = m[k - degree]
                val ^= val << degree
                for j in range(1, degree):
                    if (coeff >> (degree - 1 - j)) & 1:
                        val ^= m[k - j] << j
                m.append(val)
        v.append([m[k] << (bits - 1 - k) for k in range(bits)])

    out = []
    x = [0] * dim
    for i in range(total):
        if i >= skip:
            out.append([xd / float(1 << bits) for xd in x])
        # index of the rightmost zero bit of i, the Gray-code recurrence
        c = 0
        value = i
        while value & 1:
            value >>= 1
            c += 1
        for d in range(dim):
            x[d] ^= v[d][c]
    return np.asarray(out, dtype=float)


def design_from_prior(n, prior_ppf, dim=None, skip=1):
    """Sobol points pushed through the prior's inverse CDF, Sec. 2.2.

    `prior_ppf` is either a list of per-parameter quantile functions --
    Wilkinson's independent-prior case, "apply the inverse cumulative
    density function to each parameter" -- or a pair ``(lo, hi)``, the
    linear transformation he gives for a product of uniforms.
    """
    if isinstance(prior_ppf, tuple) and len(prior_ppf) == 2 \
            and not callable(prior_ppf[0]):
        lo = [float(v) for v in np.atleast_1d(np.asarray(prior_ppf[0]))]
        hi = [float(v) for v in np.atleast_1d(np.asarray(prior_ppf[1]))]
        if len(lo) != len(hi):
            raise ValueError("design_from_prior: lo and hi differ in length")
        u = sobol_sequence(n, len(lo), skip=skip)
        return np.asarray([[lo[j] + (hi[j] - lo[j]) * row[j]
                            for j in range(len(lo))]
                           for row in u.tolist()], dtype=float)
    fns = list(prior_ppf)
    if dim is not None and int(dim) != len(fns):
        raise ValueError("design_from_prior: dim=%r but %d quantile "
                         "functions given" % (dim, len(fns)))
    u = sobol_sequence(n, len(fns), skip=skip)
    return np.asarray([[float(fns[j](row[j])) for j in range(len(fns))]
                       for row in u.tolist()], dtype=float)


# --------------------------------------------------------------------
# the ABC likelihood being emulated
# --------------------------------------------------------------------

def gabc_log_likelihood(sim, obs, theta, n_sim=50, epsilon=1.0,
                        summary=None, kernel="gaussian", seed=0,
                        bootstrap=25):
    r"""Wilkinson eq. (1): log of the Monte Carlo GABC likelihood.

    Returns ``(log_lik, nugget_variance)``. The second value is the
    bootstrap variance of the estimate, which Sec. 2.1 uses as the GP's
    nugget :math:`v^2` -- "which helps avoid non-identifiability in the
    estimation of the other GP parameters". Without it the GP would be
    fitted as if the training values were exact, and they are not.

    `kernel` is the acceptance kernel :math:`\pi(D\mid X)`. "uniform"
    recovers plain rejection-ABC, where
    :math:`\pi(D\mid X)\propto 1_{\rho(D,X)\le\epsilon}`.
    """
    if kernel not in ("gaussian", "uniform"):
        raise ValueError("gabc_log_likelihood: kernel must be 'gaussian' "
                         "or 'uniform', got %r" % (kernel,))
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("gabc_log_likelihood: epsilon must be positive")
    d_obs = _summarise(obs, summary)
    terms = []
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_sim)):
        x = sim(theta, rng)
        s = _summarise(x, summary)
        if len(s) != len(d_obs):
            raise ValueError(
                "gabc_log_likelihood: simulator summary has length %d but "
                "the observed summary has length %d" % (len(s), len(d_obs)))
        rho = math.sqrt(sum((a - b) ** 2 for a, b in zip(s, d_obs)))
        if kernel == "uniform":
            terms.append(0.0 if rho <= eps else float("-inf"))
        else:
            terms.append(-0.5 * (rho / eps) ** 2)
    m = len(terms)
    log_lik = _lse(terms) - math.log(m)

    # Bootstrap variance of the SAME statistic, Sec. 2.1.
    reps = int(bootstrap)
    if reps < 2 or m < 2:
        return log_lik, 0.0
    boot = []
    for _ in range(reps):
        idx = [int(float(rng.uniform()) * m) for _ in range(m)]
        idx = [min(i, m - 1) for i in idx]
        boot.append(_lse([terms[i] for i in idx]) - math.log(m))
    finite = [b for b in boot if b > float("-inf")]
    if len(finite) < 2:
        return log_lik, 0.0
    mu = sum(finite) / len(finite)
    var = sum((b - mu) ** 2 for b in finite) / (len(finite) - 1)
    return log_lik, float(var)


def synthetic_log_likelihood(draws, obs, epsilon=0.0, summary=None):
    r"""Meeds & Welling eq. (9) / Wilkinson eq. (2).

    :math:`N(y;\hat\mu_\theta,\hat\Sigma_\theta+\epsilon^2 I)` with
    :math:`\hat\mu,\hat\Sigma` the sample mean and covariance of the
    simulated summaries. Returns ``(log_lik, mu, cov)``; the moments are
    returned because both adaptive algorithms need them again.

    :math:`\epsilon = 0` is allowed and is the point of eq. (9): the
    tolerance bias is removed, at the cost of the normal assumption.
    """
    rows = [_summarise(x, summary) for x in draws]
    S = len(rows)
    if S < 2:
        raise ValueError("synthetic_log_likelihood: need at least 2 "
                         "simulations to form a covariance, got %d" % S)
    J = len(rows[0])
    y = _summarise(obs, summary)
    if len(y) != J:
        raise ValueError("synthetic_log_likelihood: %d observed summaries "
                         "but %d simulated" % (len(y), J))
    mu = [sum(r[j] for r in rows) / S for j in range(J)]
    cov = [[sum((r[a] - mu[a]) * (r[b] - mu[b]) for r in rows) / (S - 1)
            for b in range(J)] for a in range(J)]
    e2 = float(epsilon) ** 2
    for j in range(J):
        cov[j][j] += e2
    return _mvn_logpdf(y, mu, cov), mu, cov


def _summarise(x, summary):
    v = summary(x) if summary is not None else x
    return [float(t) for t in np.atleast_1d(np.asarray(v, dtype=float))]


def _chol(a, jitter=1e-12):
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    s = jitter
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _chol_solve(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def _mvn_logpdf(y, mu, cov):
    n = len(y)
    L = _chol(cov)
    diff = [y[i] - mu[i] for i in range(n)]
    alpha = _chol_solve(L, diff)
    quad = sum(diff[i] * alpha[i] for i in range(n))
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    return -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))


# --------------------------------------------------------------------
# the emulator
# --------------------------------------------------------------------

def _corr(a, b, lengthscale, kernel):
    r2 = 0.0
    for i in range(len(a)):
        d = (a[i] - b[i]) / lengthscale[i]
        r2 += d * d
    if kernel == "sqexp":
        return math.exp(-0.5 * r2)
    r = math.sqrt(r2)
    if kernel == "matern32":
        s = math.sqrt(3.0) * r
        return (1.0 + s) * math.exp(-s)
    s = math.sqrt(5.0) * r
    return (1.0 + s + s * s / 3.0) * math.exp(-s)


def _basis(theta):
    """Wilkinson's quadratic mean, m(t) = b0 + t'b1 + diag(tt')b2."""
    return [1.0] + list(theta) + [t * t for t in theta]


def gp_fit(design, values, nugget=None, lengthscale=None, kernel="sqexp",
           tau2=None):
    r"""Fit the GP of Sec. 2.1 to an ensemble
    :math:`E=\{(\theta_i,\hat l_M(\theta_i))\}`.

    :math:`\beta` and :math:`\tau^2` are integrated out under the
    conjugate improper prior :math:`\pi(\beta,\tau^2)\propto1/\tau^2`
    the paper uses, which is what makes the posterior a multivariate
    :math:`t` rather than a normal; the length-scales are plugged in at
    their maximum-likelihood values, again as stated.

    `nugget` is :math:`v^2` from :func:`gabc_log_likelihood`, per point.
    """
    if kernel not in _KERNELS:
        raise ValueError("gp_fit: kernel must be one of %r, got %r"
                         % (_KERNELS, kernel))
    X = [[float(v) for v in row]
         for row in np.atleast_2d(np.asarray(design, dtype=float)).tolist()]
    y = [float(v) for v in np.atleast_1d(np.asarray(values, dtype=float))]
    n = len(X)
    if n != len(y):
        raise ValueError("gp_fit: %d design points but %d values"
                         % (n, len(y)))
    if n < 3:
        raise ValueError("gp_fit: need at least 3 design points, got %d" % n)
    p = len(X[0])
    if lengthscale is None:
        lengthscale = _mle_lengthscale(X, y, nugget, kernel)
    ls = [float(v) for v in np.atleast_1d(np.asarray(lengthscale,
                                                     dtype=float))]
    if len(ls) == 1:
        ls = ls * p
    if any(v <= 0.0 for v in ls):
        raise ValueError("gp_fit: length-scales must be positive")
    nug = _as_nugget(nugget, n)

    A = [[_corr(X[i], X[j], ls, kernel) + (nug[i] if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    L = _chol(A)
    H = [_basis(x) for x in X]
    q = len(H[0])
    if n <= q:
        raise ValueError(
            "gp_fit: the quadratic mean has %d coefficients and only %d "
            "design points; add points or the mean is not identified"
            % (q, n))
    Ainv_y = _chol_solve(L, y)
    Ainv_H = [_chol_solve(L, [H[i][k] for i in range(n)]) for k in range(q)]
    HtAinvH = [[sum(H[i][a] * Ainv_H[b][i] for i in range(n))
                for b in range(q)] for a in range(q)]
    HtAinvy = [sum(H[i][a] * Ainv_y[i] for i in range(n)) for a in range(q)]
    Lh = _chol(HtAinvH)
    beta = _chol_solve(Lh, HtAinvy)
    resid = [y[i] - sum(H[i][k] * beta[k] for k in range(q))
             for i in range(n)]
    Ainv_r = _chol_solve(L, resid)
    if tau2 is None:
        tau2 = sum(resid[i] * Ainv_r[i] for i in range(n)) / float(n - q)
    return {"design": X, "values": y, "beta": beta, "tau2": float(tau2),
            "lengthscale": ls, "kernel": kernel, "nugget": nug,
            "chol": L, "Ainv_r": Ainv_r, "Ainv_H": Ainv_H, "H": H,
            "HtAinvH_chol": Lh, "n": n, "q": q, "dim": p}


def _as_nugget(nugget, n):
    if nugget is None:
        return [1e-8] * n
    v = np.atleast_1d(np.asarray(nugget, dtype=float)).tolist()
    if len(v) == 1:
        v = v * n
    if len(v) != n:
        raise ValueError("gp_fit: %d nugget values for %d points"
                         % (len(v), n))
    return [max(float(t), 1e-12) for t in v]


def _profile_nll(X, y, ls, nug, kernel):
    n = len(X)
    A = [[_corr(X[i], X[j], ls, kernel) + (nug[i] if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    L = _chol(A)
    H = [_basis(x) for x in X]
    q = len(H[0])
    if n <= q:
        return float("inf")
    Ainv_y = _chol_solve(L, y)
    Ainv_H = [_chol_solve(L, [H[i][k] for i in range(n)]) for k in range(q)]
    HtAinvH = [[sum(H[i][a] * Ainv_H[b][i] for i in range(n))
                for b in range(q)] for a in range(q)]
    HtAinvy = [sum(H[i][a] * Ainv_y[i] for i in range(n)) for a in range(q)]
    try:
        beta = _chol_solve(_chol(HtAinvH), HtAinvy)
    except (ValueError, ZeroDivisionError):
        return float("inf")
    resid = [y[i] - sum(H[i][k] * beta[k] for k in range(q))
             for i in range(n)]
    Ainv_r = _chol_solve(L, resid)
    s2 = sum(resid[i] * Ainv_r[i] for i in range(n)) / float(n - q)
    if s2 <= 0.0:
        return float("inf")
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    return 0.5 * (logdet + (n - q) * math.log(s2))


def _mle_lengthscale(X, y, nugget, kernel):
    """Plug-in maximum likelihood for lambda, over a log grid.

    Wilkinson estimates the length-scales by maximum likelihood and
    plugs them in. A coordinate search over a log-spaced grid is used
    rather than a gradient method so that the R arm executes the same
    arithmetic and the two agree exactly rather than approximately.
    """
    n = len(X)
    p = len(X[0])
    nug = _as_nugget(nugget, n)
    spans = []
    for j in range(p):
        col = [row[j] for row in X]
        s = max(col) - min(col)
        spans.append(s if s > 0.0 else 1.0)
    ls = [0.5 * s for s in spans]
    best = _profile_nll(X, y, ls, nug, kernel)
    grid = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for _ in range(3):
        improved = False
        for j in range(p):
            for g in grid:
                trial = list(ls)
                trial[j] = g * spans[j]
                val = _profile_nll(X, y, trial, nug, kernel)
                if val < best - 1e-12:
                    best, ls, improved = val, trial, True
        if not improved:
            break
    return ls


def gp_predict(fit, theta):
    r"""Posterior mean and standard deviation at `theta`.

    The variance carries the extra term from having integrated
    :math:`\beta` out; dropping it would make the emulator look more
    certain than it is exactly where it matters, outside the design,
    and the implausibility rule of eq. (3) is a statement about
    :math:`m+3\sigma`.
    """
    X = fit["design"]
    ls, kern = fit["lengthscale"], fit["kernel"]
    t = [float(v) for v in np.atleast_1d(np.asarray(theta, dtype=float))]
    if len(t) != fit["dim"]:
        raise ValueError("gp_predict: theta has %d entries, design has %d"
                         % (len(t), fit["dim"]))
    k = [_corr(t, X[i], ls, kern) for i in range(fit["n"])]
    h = _basis(t)
    mean = sum(h[j] * fit["beta"][j] for j in range(fit["q"])) \
        + sum(k[i] * fit["Ainv_r"][i] for i in range(fit["n"]))
    Ainv_k = _chol_solve(fit["chol"], k)
    var = 1.0 - sum(k[i] * Ainv_k[i] for i in range(fit["n"]))
    # h - H' A^-1 k, the correction for the estimated mean coefficients.
    # H, not A^-1 H: contracting A^-1 k against A^-1 H instead inflates
    # this term by a factor of A^-1 and makes the emulator look wildly
    # uncertain at points it has actually visited, which switches off
    # the implausibility rule of eq. (3).
    H = fit["H"]
    hh = [h[j] - sum(H[i][j] * Ainv_k[i] for i in range(fit["n"]))
          for j in range(fit["q"])]
    w = _chol_solve(fit["HtAinvH_chol"], hh)
    var += sum(hh[j] * w[j] for j in range(fit["q"]))
    var = fit["tau2"] * max(var, 0.0)
    return mean, math.sqrt(var)


# --------------------------------------------------------------------
# history matching
# --------------------------------------------------------------------

def implausible(fit, theta, threshold=10.0, n_sd=3.0):
    """Wilkinson eq. (3). True means `theta` is ruled out.

    ``m + 3 sigma < max_E l_hat - T``. The 3 is his: "so that the
    estimated probability of l(theta) exceeding m + 3 sigma is less than
    0.003". T = 10 is his default, chosen because exp(10) > 1e4.
    """
    m, sd = gp_predict(fit, theta)
    return bool(m + float(n_sd) * sd < max(fit["values"]) - float(threshold))


def history_match(sim, obs, prior_ppf, n_waves=3, n_design=32, n_sim=50,
                  epsilon=1.0, summary=None, threshold=10.0, n_sd=3.0,
                  kernel="sqexp", accept_kernel="gaussian", seed=0):
    """Sec. 3.2: waves of design, emulate, rule out, redesign.

    Each wave extends the design *inside what the previous wave did not
    rule implausible*, which is the whole point -- a single GP over the
    prior support cannot model a log-likelihood ranging over "-5 to
    -10^3", so the space is cut down before the emulator is asked to be
    accurate on it.
    """
    ensemble_x, ensemble_y, ensemble_v = [], [], []
    waves = []
    fit = None
    for w in range(int(n_waves)):
        cand = design_from_prior(int(n_design) * 4, prior_ppf,
                                 skip=1 + w * int(n_design) * 4)
        rows = [list(r) for r in cand.tolist()]
        if fit is not None:
            keep = [r for r in rows
                    if not implausible(fit, r, threshold, n_sd)]
            ruled = len(rows) - len(keep)
            rows = keep if keep else rows[:int(n_design)]
        else:
            ruled = 0
        rows = rows[:int(n_design)]
        for i, th in enumerate(rows):
            ll, v = gabc_log_likelihood(
                sim, obs, th, n_sim=n_sim, epsilon=epsilon, summary=summary,
                kernel=accept_kernel, seed=int(seed) + 1000 * w + i)
            if ll > float("-inf"):
                ensemble_x.append(th)
                ensemble_y.append(ll)
                ensemble_v.append(v)
        if len(ensemble_x) < 3:
            raise ValueError(
                "history_match: wave %d left %d usable points; every "
                "simulation was rejected, so epsilon is too small for this "
                "simulator" % (w, len(ensemble_x)))
        fit = gp_fit(ensemble_x, ensemble_y, nugget=ensemble_v,
                     kernel=kernel)
        waves.append({"wave": w, "n_ensemble": len(ensemble_x),
                      "ruled_implausible": ruled,
                      "max_log_lik": max(ensemble_y)})
    return fit, waves


# --------------------------------------------------------------------
# Meeds & Welling
# --------------------------------------------------------------------

def _alpha_terms(log_prior, theta, theta_p, ll, ll_p, log_q, log_q_p):
    return min(0.0, (log_prior(theta_p) + ll_p + log_q_p)
               - (log_prior(theta) + ll + log_q))


def _expected_error(alphas, tau, n_grid=101):
    """Eqs. (13)-(16): the probability the accept/reject call was wrong.

    Integrating eq. (15) over u on a grid; the paper says "estimated by
    Monte Carlo or grid values of u" and a grid is deterministic.
    """
    M = len(alphas)
    total = 0.0
    for i in range(n_grid):
        u = (i + 0.5) / n_grid
        if u <= tau:
            err = sum(1 for a in alphas if a < u) / float(M)
        else:
            err = sum(1 for a in alphas if a >= u) / float(M)
        total += err
    return total / n_grid


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def synthetic_abc(sim, obs, log_prior, theta0, n_iter=200, n_sim=20,
                  epsilon=0.0, proposal_sd=0.5, summary=None, seed=0):
    """Meeds & Welling Algorithm 1: fixed S, no surrogate, no adaptation.

    The reference the other two must reproduce as S grows.
    """
    return _mw_sampler(sim, obs, log_prior, theta0, n_iter, n_sim, epsilon,
                       proposal_sd, summary, seed, adaptive=False,
                       xi=None, delta_s=0, n_alpha=0)


def gps_abc(sim, obs, log_prior, theta0, n_iter=200, n_sim=10, epsilon=0.0,
            proposal_sd=0.5, summary=None, seed=0, xi=0.05, delta_s=10,
            n_alpha=64, max_sim=400):
    """Meeds & Welling Algorithm 2, the adaptive synthetic-likelihood step.

    Simulations are added in blocks of `delta_s` until the expected
    decision error E(alpha) of eq. (16) drops below `xi`, then the
    decision is taken against tau = median(alpha). `max_sim` bounds the
    work per step; hitting it is reported rather than hidden, because a
    step that never reached `xi` made its decision under more
    uncertainty than was asked for.
    """
    return _mw_sampler(sim, obs, log_prior, theta0, n_iter, n_sim, epsilon,
                       proposal_sd, summary, seed, adaptive=True, xi=xi,
                       delta_s=delta_s, n_alpha=n_alpha, max_sim=max_sim)


def _mw_sampler(sim, obs, log_prior, theta0, n_iter, n_sim, epsilon,
                proposal_sd, summary, seed, adaptive, xi, delta_s,
                n_alpha, max_sim=None):
    rng = np.random.default_rng(int(seed))
    theta = [float(v) for v in np.atleast_1d(np.asarray(theta0,
                                                        dtype=float))]
    p = len(theta)
    sd = np.atleast_1d(np.asarray(proposal_sd, dtype=float)).tolist()
    if len(sd) == 1:
        sd = sd * p
    chain = [list(theta)]
    n_accept = 0
    unresolved = 0
    sims_used = 0
    for it in range(int(n_iter)):
        prop = [theta[j] + sd[j] * rng.standard_normal() for j in range(p)]
        if log_prior(prop) == float("-inf"):
            chain.append(list(theta))
            continue
        S = int(n_sim)
        while True:
            cur = [sim(theta, rng) for _ in range(S)]
            new = [sim(prop, rng) for _ in range(S)]
            sims_used += 2 * S
            ll_c, mu_c, cov_c = synthetic_log_likelihood(
                cur, obs, epsilon=epsilon, summary=summary)
            ll_p, mu_p, cov_p = synthetic_log_likelihood(
                new, obs, epsilon=epsilon, summary=summary)
            if not adaptive:
                # Algorithm 1: eq. (10), the plug-in accept probability.
                loga = _alpha_terms(log_prior, theta, prop, ll_c, ll_p,
                                    0.0, 0.0)
                tau = math.exp(loga)
                break
            # eq. (11): mu | data ~ N(mu_hat, Sigma_hat / S)
            y = _summarise(obs, summary)
            alphas = []
            for _ in range(int(n_alpha)):
                mc = _draw_mean(mu_c, cov_c, S, rng)
                mp = _draw_mean(mu_p, cov_p, S, rng)
                a = min(0.0, (log_prior(prop) + _mvn_logpdf(y, mp, cov_p))
                        - (log_prior(theta) + _mvn_logpdf(y, mc, cov_c)))
                alphas.append(math.exp(a))
            tau = _median(alphas)
            err = _expected_error(alphas, tau)
            if err < float(xi):
                break
            if max_sim is not None and S >= int(max_sim):
                unresolved += 1
                break
            S += int(delta_s)
        if float(rng.uniform()) <= tau:
            theta = prop
            n_accept += 1
        chain.append(list(theta))
    return {"chain": chain, "acceptance_rate": n_accept / float(n_iter),
            "n_simulations": sims_used, "unresolved_steps": unresolved}


def _draw_mean(mu, cov, S, rng):
    """One draw from eq. (11), N(mu_hat, Sigma_hat / S)."""
    n = len(mu)
    scaled = [[cov[i][j] / float(S) for j in range(n)] for i in range(n)]
    L = _chol(scaled)
    z = [rng.standard_normal() for _ in range(n)]
    return [mu[i] + sum(L[i][k] * z[k] for k in range(i + 1))
            for i in range(n)]


# --------------------------------------------------------------------
# front end
# --------------------------------------------------------------------

def abc_gp_emulator(sim, obs, X_grid=None, kernel="sqexp",
                    method="wilkinson", prior_ppf=None, log_prior=None,
                    theta0=None, n_sim=50, epsilon=1.0, summary=None,
                    n_waves=3, n_design=32, threshold=10.0, n_sd=3.0,
                    accept_kernel="gaussian", n_iter=200, proposal_sd=0.5,
                    xi=0.05, delta_s=10, n_alpha=64, seed=0):
    """ABC with a GP surrogate; four published routes, `method` picks.

    ``"wilkinson"`` (default) emulates the log-likelihood and returns
    the posterior evaluated on `X_grid`, which is where an emulator
    earns its keep: the grid costs GP evaluations, not simulator runs.
    ``"gps"`` and ``"adaptive"`` are Meeds & Welling's samplers and
    return a chain. ``"synthetic"`` is their fixed-S baseline.

    Wilkinson is the default because it is the method that answers the
    question the name asks -- a surrogate *likelihood* over the whole
    parameter space -- while GPS-ABC is a sampler that happens to use
    one.

    Examples
    --------
    A normal simulator with known mean, recovered on a grid::

        def sim(theta, rng):
            return [theta[0] + rng.standard_normal() for _ in range(20)]
        r = abc_gp_emulator(sim, [2.0] * 20, X_grid=[[t / 10.0]
                            for t in range(-20, 21)],
                            prior_ppf=([-2.0], [2.0]))
        r["estimate"]           # posterior mode, near 2.0
    """
    if method not in _METHODS:
        raise ValueError("abc_gp_emulator: method must be one of %r, got %r"
                         % (_METHODS, method))
    if not callable(sim):
        raise ValueError("abc_gp_emulator: sim must be a callable "
                         "simulator sim(theta, rng)")

    if method == "wilkinson":
        if prior_ppf is None:
            raise ValueError(
                "abc_gp_emulator: method='wilkinson' needs prior_ppf, "
                "either (lo, hi) or per-parameter quantile functions -- "
                "the Sobol design of Sec. 2.2 is defined on the prior "
                "support and there is no default support to guess")
        fit, waves = history_match(
            sim, obs, prior_ppf, n_waves=n_waves, n_design=n_design,
            n_sim=n_sim, epsilon=epsilon, summary=summary,
            threshold=threshold, n_sd=n_sd, kernel=kernel,
            accept_kernel=accept_kernel, seed=seed)
        if X_grid is None:
            grid = fit["design"]
        else:
            grid = [[float(v) for v in np.atleast_1d(np.asarray(r,
                                                                dtype=float))]
                    for r in np.atleast_2d(np.asarray(X_grid,
                                                      dtype=float)).tolist()]
        means, sds = [], []
        for th in grid:
            m, s = gp_predict(fit, th)
            means.append(m)
            sds.append(s)
        top = max(range(len(means)), key=lambda i: means[i])
        lse = _lse(means)
        post = [math.exp(m - lse) for m in means]
        return RichResult(payload={
            "estimate": grid[top],
            "grid": grid,
            "log_likelihood": means,
            "log_likelihood_sd": sds,
            "posterior": post,
            "waves": waves,
            "ensemble_size": fit["n"],
            "lengthscale": fit["lengthscale"],
            "tau2": fit["tau2"],
            "beta": fit["beta"],
            "n_simulations": sum(w["n_ensemble"] for w in waves) * int(n_sim),
            "method": "ABC GP emulator, Wilkinson (2014) with sequential "
                      "history matching",
        })

    if log_prior is None or theta0 is None:
        raise ValueError(
            "abc_gp_emulator: method=%r is a Metropolis-Hastings sampler "
            "and needs log_prior and theta0" % (method,))
    if method == "synthetic":
        out = synthetic_abc(sim, obs, log_prior, theta0, n_iter=n_iter,
                            n_sim=n_sim, epsilon=epsilon,
                            proposal_sd=proposal_sd, summary=summary,
                            seed=seed)
        label = "synthetic-likelihood ABC-MH, Meeds & Welling Algorithm 1"
    else:
        out = gps_abc(sim, obs, log_prior, theta0, n_iter=n_iter,
                      n_sim=n_sim, epsilon=epsilon,
                      proposal_sd=proposal_sd, summary=summary, seed=seed,
                      xi=xi, delta_s=delta_s, n_alpha=n_alpha)
        label = ("GPS-ABC adaptive MH, Meeds & Welling (2014) "
                 "Algorithm 2, eqs. 11-16")
    chain = out["chain"]
    p = len(chain[0])
    burn = len(chain) // 2
    kept = chain[burn:]
    est = [sum(r[j] for r in kept) / len(kept) for j in range(p)]
    payload = dict(out)
    payload.update({"estimate": est, "posterior_mean": est,
                    "burn_in": burn, "method": label})
    return RichResult(payload=payload)


def cheatsheet():
    return ("abcgp: GP surrogate ABC. wilkinson = GP on log GABC "
            "likelihood (eq.1) + history matching (m+3s < max l - T, "
            "T=10) on a Sobol design; gps/adaptive = Meeds-Welling "
            "randomized-acceptance MH (eq.11 mu~N(mu_hat,S_hat/S), "
            "tau=median(alpha), stop when E(alpha)<xi); synthetic = "
            "their fixed-S Algorithm 1.")


# compact alias per ledger/NAMING.md
abcgpemulator = abc_gp_emulator
