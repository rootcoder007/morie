# morie.fn -- function file (rootcoder007/morie)
"""Gibbs samplers for Bayesian regression and mixtures: linear regression with a half-Cauchy prior
on the residual scale, the finite normal mixture, the contaminated-normal outlier model of Box and
Tiao, the BayesA and BayesB marker-effect models of genomic prediction, and the reliability of
genomic predictions. All draws come from the Philox stream (``seed``); gamma variates use the
Marsaglia-Tsang method on that stream, so the R arm reproduces the chains."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._rrng_core import qnorm

__all__ = [
    "bayes_linear_halfcauchy",
    "finite_mixture_gibbs",
    "contaminated_normal_outliers",
    "bayes_a",
    "bayes_b",
    "genomic_reliability",
]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


class _St:
    """Philox uniforms consumed in blocks; normals by inversion, gammas by Marsaglia-Tsang."""

    def __init__(self, seed):
        self.seed = seed
        self.block = 0
        self.u = []
        self.k = 0

    def unif(self):
        if self.k >= len(self.u):
            self.u = [float(v) for v in random_uniform(4096, seed=self.seed, stream=self.block)]
            self.block += 1
            self.k = 0
        v = self.u[self.k]
        self.k += 1
        return v

    def norm(self):
        return qnorm(self.unif())

    def gamma(self, shape):
        if shape < 1.0:
            g = self.gamma(shape + 1.0)
            return g * self.unif() ** (1.0 / shape)
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.norm()
            v = (1.0 + c * x) ** 3
            if v <= 0:
                continue
            u = self.unif()
            if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
                return d * v


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    return L


def _summ(draws):
    m = len(draws)
    k = len(draws[0])
    mean = [ssum(d[a] for d in draws) / m for a in range(k)]
    sd = [math.sqrt(ssum((d[a] - mean[a]) ** 2 for d in draws) / (m - 1)) for a in range(k)]
    return mean, sd


def bayes_linear_halfcauchy(y, X, prior_var=1e6, scale=25.0, ndraw=2000, burn_in=500, seed=0):
    r"""Bayesian linear regression with ``beta ~ N(0, prior_var I)`` and ``sigma ~ half-Cauchy(0, scale)``.

    Gibbs sampler using the auxiliary-variable representation of Wand et al.
    (2011): ``sigma^2 | a ~ IG(1/2, 1/a)``, ``a ~ IG(1/2, 1/scale^2)``, so that
    ``beta | sigma^2`` is normal with covariance ``(X'X / sigma^2 + I /
    prior_var)^{-1}``, ``sigma^2 | beta, a ~ IG((n + 1)/2, SSE/2 + 1/a)`` and ``a |
    sigma^2 ~ IG(1, 1/sigma^2 + 1/scale^2)``. ``X`` should include the intercept.

    References
    ----------
    Gelman, A. (2006). Prior distributions for variance parameters in
    hierarchical models. *Bayesian Analysis* 1, 515-534.

    Wand, M. P., Ormerod, J. T., Padoan, S. A. and Fruhwirth, R. (2011). Mean
    field variational Bayes for elaborate distributions. *Bayesian Analysis*
    6, 847-900.

    Examples
    --------
    >>> r = bayes_linear_halfcauchy([1.0, 2.1, 2.9, 4.2], [[1, 0], [1, 1], [1, 2], [1, 3]], ndraw=200, burn_in=50)
    >>> round(r.beta[1], 1)
    1.1
    """
    yv, Xm = _vec(y), _mat(X)
    n, k = len(Xm), len(Xm[0])
    st = _St(seed)
    xtx = [[ssum(r[a] * r[b] for r in Xm) for b in range(k)] for a in range(k)]
    xty = [ssum(r[a] * v for r, v in zip(Xm, yv)) for a in range(k)]
    s2, a = 1.0, 1.0
    keep = []
    for it in range(ndraw + burn_in):
        M = inverse([[xtx[i][j] / s2 + (1.0 / prior_var if i == j else 0.0) for j in range(k)] for i in range(k)])
        mu = [ssum(M[i][j] * xty[j] / s2 for j in range(k)) for i in range(k)]
        L = _chol(M)
        z = [st.norm() for _ in range(k)]
        beta = [mu[i] + ssum(L[i][j] * z[j] for j in range(i + 1)) for i in range(k)]
        sse = ssum((v - ssum(r[j] * beta[j] for j in range(k))) ** 2 for r, v in zip(Xm, yv))
        s2 = (sse / 2 + 1.0 / a) / st.gamma((n + 1) / 2.0)
        a = (1.0 / s2 + 1.0 / scale**2) / st.gamma(1.0)
        if it >= burn_in:
            keep.append(beta + [s2])
    mean, sd = _summ(keep)
    return RichResult(payload={"beta": mean[:k], "beta_sd": sd[:k], "sigma2": mean[k], "draws": keep})


def finite_mixture_gibbs(y, K, ndraw=2000, burn_in=500, seed=0, alpha=1.0):
    r"""Bayesian finite normal mixture ``y_i ~ sum_k pi_k N(mu_k, sigma_k^2)`` by Gibbs sampling.

    Priors: ``mu_k ~ N(m0, s0^2)`` with ``m0`` the sample mean and ``s0^2`` its
    sample variance times 10, ``sigma_k^2 ~ IG(2, s^2)`` (``s^2`` the sample
    variance), ``pi ~ Dirichlet(alpha)``. Allocations, weights, means and
    variances are drawn from their full conditionals (Diebolt and Robert
    1994); label switching is removed by ordering each draw by ``mu``.

    References
    ----------
    Diebolt, J. and Robert, C. P. (1994). Estimation of finite mixture
    distributions through Bayesian sampling. *JRSS B* 56, 363-375.

    Examples
    --------
    >>> y = [0.1, -0.2, 0.3, 0.0, 5.1, 4.9, 5.2, 4.8]
    >>> r = finite_mixture_gibbs(y, 2, ndraw=200, burn_in=50)
    >>> [round(m) for m in r.mu]
    [0, 5]
    """
    yv = _vec(y)
    n = len(yv)
    m0 = ssum(yv) / n
    s2y = ssum((v - m0) ** 2 for v in yv) / (n - 1)
    s0 = 10 * s2y
    st = _St(seed)
    srt = sorted(yv)
    mu = [srt[int((k + 0.5) * n / K)] for k in range(K)]
    sig = [s2y] * K
    pi = [1.0 / K] * K
    keep = []
    for it in range(ndraw + burn_in):
        z = []
        for v in yv:
            w = [pi[k] * math.exp(-0.5 * (v - mu[k]) ** 2 / sig[k]) / math.sqrt(sig[k]) for k in range(K)]
            t = st.unif() * ssum(w)
            acc, c = 0.0, K - 1
            for k in range(K):
                acc += w[k]
                if acc >= t:
                    c = k
                    break
            z.append(c)
        cnt = [sum(1 for c in z if c == k) for k in range(K)]
        g = [st.gamma(alpha + cnt[k]) for k in range(K)]
        pi = [v / ssum(g) for v in g]
        for k in range(K):
            yk = [v for v, c in zip(yv, z) if c == k]
            prec = 1.0 / s0 + cnt[k] / sig[k]
            mk = (m0 / s0 + ssum(yk) / sig[k]) / prec
            mu[k] = mk + st.norm() / math.sqrt(prec)
            sig[k] = (s2y + 0.5 * ssum((v - mu[k]) ** 2 for v in yk)) / st.gamma(2.0 + cnt[k] / 2.0)
        order = sorted(range(K), key=lambda k: (mu[k], k))
        if it >= burn_in:
            keep.append([mu[k] for k in order] + [sig[k] for k in order] + [pi[k] for k in order])
    mean, sd = _summ(keep)
    return RichResult(payload={"mu": mean[:K], "sigma2": mean[K : 2 * K], "pi": mean[2 * K :], "draws": keep})


def contaminated_normal_outliers(y, eps=0.05, k=5.0, ndraw=2000, burn_in=500, seed=0):
    r"""Box-Tiao contaminated-normal outlier model: posterior outlier probabilities.

    ``y_i ~ (1 - eps) N(mu, sigma^2) + eps N(mu, k^2 sigma^2)`` with ``p(mu,
    sigma^2) propto 1/sigma^2``; Gibbs over the outlier indicators ``delta_i``,
    ``mu | delta, sigma^2`` (normal, weights ``1`` or ``1/k^2``) and ``sigma^2 |
    delta, mu ~ Q / chi^2_n`` with ``Q`` the weighted sum of squares. Returns
    ``P(delta_i = 1 | y)`` (averaged conditional probabilities) and the
    posterior means of ``mu`` and ``sigma^2``.

    References
    ----------
    Box, G. E. P. and Tiao, G. C. (1968). A Bayesian approach to some outlier
    problems. *Biometrika* 55, 119-129.

    Examples
    --------
    >>> r = contaminated_normal_outliers([0.2, -0.1, 0.4, 0.0, 0.3, 6.0], ndraw=300, burn_in=50)
    >>> r.outlier_prob[-1] > 0.9
    True
    """
    yv = _vec(y)
    n = len(yv)
    st = _St(seed)
    mu = sorted(yv)[n // 2]
    s2 = ssum((v - mu) ** 2 for v in yv) / n
    prob = [0.0] * n
    keep = []
    for it in range(ndraw + burn_in):
        delta = []
        pr = []
        for v in yv:
            a = (1 - eps) * math.exp(-0.5 * (v - mu) ** 2 / s2)
            b = eps / k * math.exp(-0.5 * (v - mu) ** 2 / (k * k * s2))
            p = b / (a + b)
            pr.append(p)
            delta.append(st.unif() < p)
        w = [1.0 / (k * k) if d else 1.0 for d in delta]
        sw = ssum(w)
        mu = ssum(wi * v for wi, v in zip(w, yv)) / sw + st.norm() * math.sqrt(s2 / sw)
        Q = ssum(wi * (v - mu) ** 2 for wi, v in zip(w, yv))
        s2 = Q / (2.0 * st.gamma(n / 2.0))
        if it >= burn_in:
            keep.append([mu, s2])
            prob = [a + b for a, b in zip(prob, pr)]
    mean, _ = _summ(keep)
    return RichResult(payload={"outlier_prob": [v / ndraw for v in prob], "mu": mean[0], "sigma2": mean[1]})


def _marker_setup(y, M):
    yv, Mm = _vec(y), _mat(M)
    n, p = len(Mm), len(Mm[0])
    cols = [[r[j] for r in Mm] for j in range(p)]
    mm = [ssum(v * v for v in c) for c in cols]
    return yv, cols, mm, n, p


def bayes_a(y, M, nu=4.012, s2=None, ndraw=2000, burn_in=500, seed=0):
    r"""BayesA (Meuwissen, Hayes and Goddard 2001): marker effects with locus-specific variances.

    ``y = 1 mu + sum_j m_j u_j + e``, ``u_j | sigma_j^2 ~ N(0, sigma_j^2)``,
    ``sigma_j^2 ~ nu S^2 / chi^2_nu`` (scaled inverse chi-square),
    ``e ~ N(0, sigma_e^2 I)`` with a flat prior on ``mu`` and on
    ``sigma_e^2``. Single-site Gibbs: ``u_j`` normal with precision ``m_j'm_j /
    sigma_e^2 + 1/sigma_j^2``; ``sigma_j^2 = (nu S^2 + u_j^2) / chi^2_{nu + 1}``;
    ``sigma_e^2 = e'e / chi^2_{n - 2}``. ``S^2`` defaults to ``var(y) (nu - 2) / (nu
    p)``.

    References
    ----------
    Meuwissen, T. H. E., Hayes, B. J. and Goddard, M. E. (2001). Prediction of
    total genetic value using genome-wide dense marker maps. *Genetics* 157,
    1819-1829.

    Examples
    --------
    >>> r = bayes_a([1.2, 0.3, 2.1, 1.6, 0.4, 1.9], [[1, 0], [0, 1], [2, 1], [1, 1], [0, 0], [2, 0]], ndraw=100, burn_in=20)
    >>> len(r.effects)
    2
    """
    return _bayes_ab(y, M, nu, s2, 1.0, ndraw, burn_in, seed)


def bayes_b(y, M, pi=0.95, nu=4.012, s2=None, ndraw=2000, burn_in=500, seed=0):
    r"""BayesB (Meuwissen et al. 2001): a fraction ``pi`` of markers has no effect.

    As :func:`bayes_a`, but ``sigma_j^2 = 0`` with probability ``pi``. The
    indicator ``delta_j`` is drawn with ``u_j`` integrated out: given
    ``sigma_j^2`` (drawn from its full conditional when ``delta_j = 1``, from
    the prior otherwise) the likelihood ratio of the corrected phenotype
    ``r_j`` is ``(1 + m_j'm_j sigma_j^2 / sigma_e^2)^{-1/2} exp((m_j'r_j)^2
    sigma_j^2 / (2 sigma_e^2 (sigma_e^2 + m_j'm_j sigma_j^2)))``; ``S^2``
    defaults to ``var(y) (nu - 2) / (nu p (1 - pi))``. Returns inclusion
    probabilities as well.

    References
    ----------
    Meuwissen, T. H. E., Hayes, B. J. and Goddard, M. E. (2001). *Genetics*
    157, 1819-1829.

    Habier, D., Fernando, R. L., Kizilkaya, K. and Garrick, D. J. (2011).
    Extension of the Bayesian alphabet for genomic selection. *BMC
    Bioinformatics* 12, 186.

    Examples
    --------
    >>> r = bayes_b([1.2, 0.3, 2.1, 1.6, 0.4, 1.9], [[1, 0], [0, 1], [2, 1], [1, 1], [0, 0], [2, 0]], ndraw=100, burn_in=20)
    >>> all(0 <= q <= 1 for q in r.inclusion)
    True
    """
    return _bayes_ab(y, M, nu, s2, 1.0 - pi, ndraw, burn_in, seed)


def _bayes_ab(y, M, nu, s2, incl, ndraw, burn_in, seed):
    yv, cols, mm, n, p = _marker_setup(y, M)
    my = ssum(yv) / n
    vy = ssum((v - my) ** 2 for v in yv) / (n - 1)
    S2 = vy * (nu - 2) / (nu * p * incl) if s2 is None else float(s2)
    st = _St(seed)
    u = [0.0] * p
    sj = [S2] * p
    delta = [1] * p
    mu = my
    se = vy / 2
    e = [v - mu for v in yv]
    keep_u, keep_d = [], []
    for it in range(ndraw + burn_in):
        for i in range(n):
            e[i] += mu
        mu = ssum(e) / n + st.norm() * math.sqrt(se / n)
        for i in range(n):
            e[i] -= mu
        for j in range(p):
            c = cols[j]
            rhs = ssum(c[i] * e[i] for i in range(n)) + mm[j] * u[j]
            if incl < 1.0:
                cand = sj[j] if delta[j] else nu * S2 / (2.0 * st.gamma(nu / 2.0))
                lr = -0.5 * math.log(1 + mm[j] * cand / se) + rhs * rhs * cand / (2 * se * (se + mm[j] * cand))
                p1 = incl / (incl + (1 - incl) * math.exp(-lr)) if lr > -700 else 0.0
                delta[j] = 1 if st.unif() < p1 else 0
                if delta[j]:
                    sj[j] = cand
            else:
                st.unif()
            if delta[j]:
                prec = mm[j] / se + 1.0 / sj[j]
                new = rhs / se / prec + st.norm() / math.sqrt(prec)
                sj[j] = (nu * S2 + new * new) / (2.0 * st.gamma((nu + 1) / 2.0))
            else:
                new = 0.0
                st.unif()
                st.unif()
            for i in range(n):
                e[i] += c[i] * (u[j] - new)
            u[j] = new
        se = ssum(v * v for v in e) / (2.0 * st.gamma((n - 2) / 2.0))
        if it >= burn_in:
            keep_u.append(list(u) + [mu, se])
            keep_d.append(list(delta))
    mean, sd = _summ(keep_u)
    inc = [ssum(d[j] for d in keep_d) / ndraw for j in range(p)]
    return RichResult(
        payload={"effects": mean[:p], "effects_sd": sd[:p], "mu": mean[p], "sigma2_e": mean[p + 1], "inclusion": inc}
    )


def genomic_reliability(pev, sigma2_a):
    r"""Reliability ``r^2 = 1 - PEV / sigma_a^2`` and accuracy ``r`` of genomic breeding values.

    ``pev`` is the prediction error variance (the posterior variance of each
    breeding value) and ``sigma_a^2`` the additive genetic variance.

    References
    ----------
    Henderson, C. R. (1975). Best linear unbiased estimation and prediction
    under a selection model. *Biometrics* 31, 423-447.

    Examples
    --------
    >>> r = genomic_reliability([0.2, 0.5], 0.8)
    >>> r.reliability, [round(v, 12) for v in r.accuracy]
    ([0.75, 0.375], [0.866025403784, 0.612372435696])
    """
    pv = _vec(pev)
    rel = [1 - v / sigma2_a for v in pv]
    return RichResult(payload={"reliability": rel, "accuracy": [math.sqrt(max(v, 0.0)) for v in rel]})


def cheatsheet() -> str:
    return (
        "bayes_linear_halfcauchy / finite_mixture_gibbs / contaminated_normal_outliers / bayes_a / bayes_b / "
        "genomic_reliability -> Bayesian regression samplers."
    )
