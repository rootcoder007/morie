# morie.fn -- Bayesian-nonparametrics core (rootcoder007/morie)
"""Shared machinery for the Ghosal & van der Vaart shelf.

Everything follows Ghosal, S. & van der Vaart, A. (2017) *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge Series in Statistical
and Probabilistic Mathematics 44, Cambridge University Press,
ISBN 978-0-521-87826-5.

Full title as printed: *Fundamentals of
Nonparametric Bayesian Inference*, CUP -- equations checked against
the library PDF:

* normalization construction, eq. (3.1) p.29;
* stick breaking, eq. (3.2) p.30; discrete hazard eq. (3.3) p.31;
* countable Dirichlet process: marginals eq. (3.4) p.31, posterior
  eq. (3.5)-(3.6) p.32, posterior moments from eq. (3.7) p.32 (and
  the standard finite-Dirichlet variance/covariance forms it heads);
* Polya trees, sec. 3.7: splitting variables Beta(alpha_e0, alpha_e1),
  set masses as products down the tree, posterior updating
  alpha_e -> alpha_e + N_e;
* Bernstein/Feller operator, sec. 2.3.4;
* conjugate Gaussian-process regression, sec. 2.4 (posterior mean by
  kernel ridge identity).
"""

from __future__ import annotations

import math

from . import _array_core as np


def _flat(x):
    if hasattr(x, "_flat"):
        return [float(v) for v in x._flat()]
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


# ------------------------------------------------- constructions
def normalize_weights(Y):
    """eq. (3.1): p_k = Y_k / sum_j Y_j for nonnegative Y."""
    ys = _flat(Y)
    if any(v < 0 for v in ys):
        raise ValueError("weights must be nonnegative")
    tot = sum(ys)
    if tot <= 0:
        raise ValueError("total mass must be positive")
    return [v / tot for v in ys]


def stick_breaking(V):
    """eq. (3.2): p_j = (prod_{l<j} (1 - V_l)) V_j."""
    vs = _flat(V)
    if any(v < 0 or v > 1 for v in vs):
        raise ValueError("stick fractions must lie in [0, 1]")
    out = []
    left = 1.0
    for v in vs:
        out.append(left * v)
        left *= (1.0 - v)
    return out


def discrete_hazard(p):
    """eq. (3.3): V_j = p_j / (1 - sum_{l<j} p_l) = P(X=j | X>=j)."""
    ps = _flat(p)
    out = []
    cum = 0.0
    for pj in ps:
        denom = 1.0 - cum
        out.append(pj / denom if denom > 1e-300 else 1.0)
        cum += pj
    return out


# --------------------------------------- countable Dirichlet process
def cdp_posterior_params(alpha, counts, alpha_tail=0.0):
    """alpha -> alpha + N updating of eq. (3.5)-(3.6): returns the
    updated cell parameters and the tail parameter
    sum_{j>k} alpha_j + n - sum_{j<=k} N_j."""
    a = _flat(alpha)
    N = _flat(counts)
    if len(a) != len(N):
        raise ValueError("alpha and counts must align")
    n = sum(N)
    upd = [ai + Ni for ai, Ni in zip(a, N)]
    tail = float(alpha_tail) + n - sum(N[:len(a)])
    return upd, tail


def cdp_posterior_mean(alpha, counts, j, alpha_total, n=None):
    """eq. (3.7): E(p_j | X) = (alpha_j + N_j) / (sum_l alpha_l + n)."""
    a = _flat(alpha)
    N = _flat(counts)
    n = sum(N) if n is None else float(n)
    return (a[j] + N[j]) / (float(alpha_total) + n)


def cdp_posterior_var(alpha, counts, j, alpha_total, n=None):
    """Finite-Dirichlet variance at the (3.7) posterior:
    var(p_j|X) = m_j(1-m_j)/(A+n+1) with m_j the posterior mean and
    A = sum_l alpha_l (the form (3.7) heads on p. 32-33)."""
    a = _flat(alpha)
    N = _flat(counts)
    n = sum(N) if n is None else float(n)
    A = float(alpha_total)
    m = (a[j] + N[j]) / (A + n)
    return m * (1.0 - m) / (A + n + 1.0)


def cdp_posterior_cov(alpha, counts, j, jp, alpha_total, n=None):
    """Finite-Dirichlet covariance at the posterior:
    cov(p_j, p_j') = -m_j m_j' / (A + n + 1)."""
    a = _flat(alpha)
    N = _flat(counts)
    n = sum(N) if n is None else float(n)
    A = float(alpha_total)
    mj = (a[j] + N[j]) / (A + n)
    mp = (a[jp] + N[jp]) / (A + n)
    return -mj * mp / (A + n + 1.0)


# ----------------------------------------------------- Polya trees
def _bits(x, depth):
    """First ``depth`` binary digits of x in [0,1) (the dyadic path)."""
    out = []
    v = min(max(float(x), 0.0), 1.0 - 1e-15)
    for _ in range(depth):
        v *= 2.0
        b = int(v)
        out.append(b)
        v -= b
    return out


def pt_set_mass_mean(alphas):
    """E P(A_eps) = prod_j alpha_(eps_1..eps_j) /
    (alpha_(..0) + alpha_(..1)) down the branch. ``alphas`` is a list
    of (alpha_taken, alpha_other) pairs along the path."""
    m = 1.0
    for a_take, a_other in alphas:
        m *= a_take / (a_take + a_other)
    return m


def pt_density_posterior(x, a_of_level, counts_along_path, n, depth):
    """Posterior mean density of a canonical (evenly split) Polya tree
    with level parameters a_m at the point x:
    E(p(x)|X) = prod_{m<=depth} (2 a_m + 2 N_path,m) /
                                (2 a_m + N_parent,m)
    where N_path,m counts observations whose first m digits match x
    and N_parent,m those matching the first m-1 digits (sec. 3.7,
    posterior updating alpha_e -> alpha_e + N_e)."""
    dens = 1.0
    for m in range(1, depth + 1):
        a = float(a_of_level(m))
        N_here = counts_along_path[m - 1]
        N_parent = counts_along_path[m - 2] if m >= 2 else n
        dens *= (2.0 * a + 2.0 * N_here) / (2.0 * a + N_parent)
    return dens


def pt_path_counts(x, data, depth):
    """N_path,m for m = 1..depth: how many data points share x's first
    m dyadic digits."""
    bx = _bits(x, depth)
    out = []
    for m in range(1, depth + 1):
        c = 0
        for d in data:
            if _bits(d, m) == bx[:m]:
                c += 1
        out.append(c)
    return out


# ---------------------------------------------- Bernstein / Feller
def bernstein_feller_cdf(F, x, K):
    """sec. 2.3.4: F_K(x) = sum_{k=0}^K F(k/K) C(K,k) x^k (1-x)^(K-k)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0, 1]")
    tot = 0.0
    for k in range(K + 1):
        tot += F(k / K) * math.comb(K, k) * x ** k \
            * (1.0 - x) ** (K - k)
    return tot


# ------------------------------------------- conjugate GP regression
def gp_regression_posterior_mean(x, y, xstar, kernel, sigma2):
    """sec. 2.4: with f ~ GP(0, k) and Gaussian noise, the posterior
    mean at x* is k(x*, X)[K + sigma^2 I]^{-1} y."""
    xs = _flat(x)
    ys = _flat(y)
    n = len(xs)
    K = [[kernel(xs[i], xs[j]) + (sigma2 if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    w = np.linalg.solve(np.marr(K), np.marr(ys))
    wl = [float(v) for v in w._flat()]
    return [sum(kernel(xq, xs[i]) * wl[i] for i in range(n))
            for xq in _flat(xstar)]


def rbf_kernel(length=1.0, var=1.0):
    def k(a, b):
        d = (a - b) / length
        return var * math.exp(-0.5 * d * d)
    return k
