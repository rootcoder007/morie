# morie.fn -- internal helpers (rootcoder007/morie)
"""Bayesian nonparametrics: DP, Polya trees, predictive recursion.

Spec: Ghosal, S. and van der Vaart, A., *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge. Section numbers below
are the book's own and were checked against its table of contents.

Two facts drive most of what follows.

* The Dirichlet process is CONJUGATE (Sec. 4.1.3): with
  ``F ~ DP(alpha G0)`` and observations ``X_1..X_n``, the posterior
  is ``DP(alpha G0 + sum delta_{X_i})``. The whole predictive
  machinery is a consequence.
* Posterior contraction rates are stated as ``eps_n`` such that
  ``Pi(d(p, p0) > M eps_n | X) -> 0``; for an s-smooth density in
  dimension d the benchmark is ``n^{-s/(2s+d)}``, which is the
  minimax rate. A prior does not beat it, and a good one attains it.
"""

import numpy as np

__all__ = ["dp_predictive", "stick_breaking", "predictive_recursion",
           "polya_tree_density", "hellinger", "minimax_rate",
           "dyadic_bins", "whittle_loglik"]


def dp_predictive(x, alpha=1.0, g0=None, grid=None):
    r"""Polya-urn predictive distribution of a Dirichlet process
    (Sec. 4.1.4):

    .. math:: p(X_{n+1} \in \cdot \mid X_1,\dots,X_n)
              = \frac{\alpha}{\alpha + n} G_0
              + \frac{1}{\alpha + n}\sum_{k} n_k\,
                \delta_{X_k^{*}},

    a mixture of the base measure and the empirical measure of the
    DISTINCT values, weighted by their multiplicities. The discrete
    part is why a DP draw is almost surely discrete and why ties
    appear with positive probability however continuous ``G0`` is.
    """
    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    a = float(alpha)
    if a <= 0:
        raise ValueError(f"alpha must be positive, got {a}.")
    vals, counts = np.unique(xv, return_counts=True)
    g = np.linspace(xv.min() - 3, xv.max() + 3, 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    base = (np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi)) if g0 is None else \
        np.asarray([float(g0(v)) for v in g])
    return {
        "grid": g,
        "base_weight": a / (a + n),
        "atom_weight": n / (a + n),
        "atoms": vals,
        "atom_probs": counts / (a + n),
        "base_density": base * a / (a + n),
        "n_distinct": int(vals.size),
    }


def stick_breaking(alpha, K, rng):
    r"""Truncated stick-breaking weights,
    :math:`w_k = V_k\prod_{j<k}(1 - V_j)` with
    :math:`V_k \sim \text{Beta}(1, \alpha)`, the last stick taking
    the remaining mass so the weights sum to one exactly."""
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}.")
    v = rng.beta(1.0, float(alpha), size=K)
    v[-1] = 1.0                              # truncation: close the stick
    w = v * np.concatenate([[1.0], np.cumprod(1.0 - v[:-1])])
    return w


def predictive_recursion(x, theta_grid, kernel, f0=None, weights=None):
    r"""Newton's predictive recursion (Sec. 5.4):

    .. math:: \hat f_i(\theta) = (1 - w_i)\hat f_{i-1}(\theta)
              + w_i\,\frac{\psi(X_i;\theta)\hat f_{i-1}(\theta)}
                          {\int \psi(X_i;t)\hat f_{i-1}(t)\,dt}.

    Each step is a convex combination of the current estimate and the
    "baseline posterior" that treats the current estimate as the
    prior and the kernel as the likelihood. It is a single sweep --
    no MCMC, no iteration to convergence -- which is what makes it
    cheap, and it is ORDER DEPENDENT for exactly the same reason.

    The weights must satisfy :math:`\sum w_i = \infty` and
    :math:`\sum w_i^2 < \infty` for convergence; ``w_i = (i+1)^{-2/3}``
    is the usual choice and the default here.
    """
    xv = np.asarray(x, dtype=float).ravel()
    th = np.asarray(theta_grid, dtype=float).ravel()
    n = xv.size
    f = np.ones(th.size) / (th.max() - th.min()) if f0 is None else \
        np.asarray(f0, dtype=float).ravel()
    if f.size != th.size:
        raise ValueError("f0 must match theta_grid.")
    # (i + 2)^{-2/3}, not (i + 1)^{-2/3}: the latter starts at w = 1,
    # which replaces the initial estimate outright instead of forming
    # a convex combination with it, and is outside the (0, 1) the
    # convergence conditions require
    w = np.array([(i + 2.0) ** (-2.0 / 3.0) for i in range(n)]) \
        if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.size != n:
        raise ValueError(f"weights has {w.size} entries for {n} observations.")
    if np.any((w <= 0) | (w >= 1)):
        raise ValueError("weights must lie strictly in (0, 1).")
    for i in range(n):
        like = kernel(xv[i], th)
        denom = float(np.trapezoid(like * f, th))
        if denom <= 0:
            continue
        f = (1.0 - w[i]) * f + w[i] * like * f / denom
    mass = float(np.trapezoid(f, th))
    return f / mass if mass > 0 else f


def dyadic_bins(level, lo, hi):
    """Edges of the level-m dyadic partition of [lo, hi]."""
    return np.linspace(lo, hi, 2 ** int(level) + 1)


def polya_tree_density(x, grid, levels=6, a_fn=None, lo=None, hi=None):
    r"""Posterior mean density under a Polya tree prior
    (Sec. 3.7, 7.2.3).

    On the level-:math:`m` dyadic partition the prior puts
    :math:`\text{Beta}(a_m, a_m)` splits at every node; conjugacy
    makes the posterior split parameters :math:`a_m + n_{\text{left}}`
    and :math:`a_m + n_{\text{right}}`, so the posterior mean density
    is a product of expected splits along the path to each cell.

    The choice of :math:`a_m` decides everything. :math:`a_m = m^2`
    makes the prior put mass on ABSOLUTELY CONTINUOUS distributions
    and yields near-optimal rates; a constant :math:`a_m` gives a
    prior supported on distributions singular to Lebesgue measure,
    which is why the default here is :math:`m^2` rather than 1.
    """
    xv = np.asarray(x, dtype=float).ravel()
    g = np.atleast_1d(np.asarray(grid, dtype=float))
    m = int(levels)
    if m < 1:
        raise ValueError(f"levels must be at least 1, got {m}.")
    a = (lambda k: float(k) ** 2) if a_fn is None else a_fn
    a0 = float(np.min(xv)) if lo is None else float(lo)
    a1 = float(np.max(xv)) if hi is None else float(hi)
    if a1 <= a0:
        raise ValueError(f"need lo < hi, got {(a0, a1)}.")
    span = a1 - a0
    dens = np.ones(g.size) / span
    for lev in range(1, m + 1):
        edges = dyadic_bins(lev, a0, a1)
        counts, _ = np.histogram(xv, bins=edges)
        am = a(lev)
        idx = np.clip(np.searchsorted(edges, g, side="right") - 1,
                      0, 2 ** lev - 1)
        # the split probability at each node, from Beta conjugacy
        left = counts[0::2]
        right = counts[1::2]
        p_left = (am + left) / (2 * am + left + right)
        share = np.where(idx % 2 == 0, p_left[idx // 2], 1 - p_left[idx // 2])
        dens = dens * 2.0 * share
    outside = (g < a0) | (g > a1)
    dens = np.where(outside, 0.0, dens)
    return dens


def hellinger(p, q, grid):
    r""":math:`d_H(p,q)^2 = \tfrac12\int(\sqrt p - \sqrt q)^2`.

    The metric the contraction theorems of Chapters 6-9 are stated
    in, because it is bounded and behaves well under products --
    unlike the :math:`L_1` distance, testing arguments in Hellinger
    distance carry directly to n-fold product measures.
    """
    pv = np.clip(np.asarray(p, dtype=float), 0, None)
    qv = np.clip(np.asarray(q, dtype=float), 0, None)
    g = np.asarray(grid, dtype=float)
    return float(np.sqrt(0.5 * np.trapezoid((np.sqrt(pv) - np.sqrt(qv))**2, g)))


def minimax_rate(n, s, d=1):
    r""":math:`\varepsilon_n = n^{-s/(2s+d)}`, the minimax rate for an
    s-smooth density in dimension d, and the benchmark every
    contraction theorem in the book is measured against."""
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}.")
    s = float(s)
    if s <= 0:
        raise ValueError(f"smoothness must be positive, got {s}.")
    return float(n ** (-s / (2.0 * s + float(d))))


def whittle_loglik(series, spectral_density, freqs=None):
    r"""Whittle log-likelihood (Sec. 7.3.3, 9.5.2):

    .. math:: \log L_W(f) = -\sum_j
              \Big[\log f(\omega_j) + \frac{I(\omega_j)}{f(\omega_j)}\Big],

    with :math:`I` the periodogram. It replaces the exact Gaussian
    likelihood, whose covariance determinant costs :math:`O(n^3)`,
    with a sum over Fourier frequencies at which the periodogram
    ordinates are asymptotically INDEPENDENT exponentials -- that
    independence is the whole point, and it is what makes a
    nonparametric prior on f tractable.
    """
    y = np.asarray(series, dtype=float).ravel()
    n = y.size
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    fft = np.fft.rfft(y - y.mean())
    per = (np.abs(fft) ** 2) / n
    w = np.fft.rfftfreq(n, d=1.0) * 2 * np.pi
    keep = (w > 0) & (w < np.pi)
    w, per = w[keep], per[keep]
    if freqs is not None:
        w = np.asarray(freqs, dtype=float).ravel()
    f = np.asarray([float(spectral_density(v)) for v in w])
    if np.any(f <= 0):
        raise ValueError("the spectral density must be strictly positive.")
    return float(-np.sum(np.log(f) + per / f)), w, per


def cheatsheet():
    return "_ghosal: DP conjugacy, predictive recursion (5.4), Polya trees, Hellinger rates"
