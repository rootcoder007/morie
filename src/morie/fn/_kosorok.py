# morie.fn -- function file (rootcoder007/morie)
"""Shared empirical-process machinery for the Kosorok shelf.

Spec: Kosorok, M. R. (2008), *Introduction to Empirical Processes and
Semiparametric Inference*, Springer -- read in the library PDF (filed
under its ISBN, 978-0-387-74978-5).

PDF-verified anchors used across this shelf:

- Brownian bridge covariance
  :math:`\\mathrm{cov}[G(s), G(t)] = F(s \\wedge t) - F(s)F(t)`
- Law of the iterated logarithm, eq. (2.21):
  :math:`\\limsup_n \\|G_n\\|_\\infty / \\sqrt{2\\log\\log n} \\le 1/2`
  a.s., with equality when 1/2 lies in the range of F
- Chung's liminf companion:
  :math:`\\liminf_n \\sqrt{2\\log\\log n}\\,\\|G_n\\|_\\infty = \\pi/2`
  a.s.

Most of this chapter states theorems rather than procedures. The
modules built on this core therefore return the finite-sample
*witness* of each theorem -- the quantity the theorem bounds, computed
on real data -- rather than a bare True. A theorem that cannot be
falsified by any input would be untestable, and the placeholder it
replaces already had that problem.
"""

import numpy as np

__all__ = [
    "z_estimator_map", "sup_difference", "survival_psi",
    "efficient_information",
    "empirical_df", "empirical_process", "bridge_cov", "sup_norm",
    "bootstrap_multiplier_process", "bracketing_number_monotone",
    "covering_number_grid", "hadamard_derivative", "cox_score",
]


def empirical_df(X, t):
    r"""Empirical distribution function
    :math:`F_n(t) = n^{-1}\sum_i 1\{X_i \le t\}`."""
    X = np.asarray(X, dtype=float).ravel()
    if X.size < 1:
        raise ValueError("X must be non-empty.")
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.mean(X[None, :] <= t[:, None], axis=1)


def empirical_process(X, t, F=None):
    r"""Standardised empirical process
    :math:`G_n(t) = \sqrt n\,[F_n(t) - F(t)]`.

    ``F`` defaults to the uniform CDF on [0, 1], the canonical case in
    which :math:`G_n` converges to the standard Brownian bridge.
    """
    X = np.asarray(X, dtype=float).ravel()
    n = X.size
    if n < 1:
        raise ValueError("X must be non-empty.")
    t = np.atleast_1d(np.asarray(t, dtype=float))
    Fn = empirical_df(X, t)
    Ft = np.clip(t, 0.0, 1.0) if F is None else np.asarray(
        [F(v) for v in t], dtype=float
    )
    return np.sqrt(n) * (Fn - Ft)


def bridge_cov(s, t, F=None):
    r"""Brownian bridge covariance :math:`F(s \wedge t) - F(s)F(t)`
    (Kosorok Ch. 2; PDF-verified).

    With ``F`` omitted this is the standard bridge covariance
    :math:`s \wedge t - st` on the unit interval.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    if F is None:
        Fs, Ft, Fm = np.clip(s, 0, 1), np.clip(t, 0, 1), np.clip(np.minimum(s, t), 0, 1)
    else:
        Fs = np.asarray([F(v) for v in np.atleast_1d(s)], dtype=float).reshape(np.shape(s))
        Ft = np.asarray([F(v) for v in np.atleast_1d(t)], dtype=float).reshape(np.shape(t))
        m = np.minimum(s, t)
        Fm = np.asarray([F(v) for v in np.atleast_1d(m)], dtype=float).reshape(np.shape(m))
    return Fm - Fs * Ft


def sup_norm(X, F=None, grid=None):
    r"""Uniform norm :math:`\|G_n\|_\infty`, evaluated exactly at the
    order statistics (where the sup of a step-minus-continuous process
    is always attained) rather than on an arbitrary grid."""
    X = np.asarray(X, dtype=float).ravel()
    n = X.size
    if n < 1:
        raise ValueError("X must be non-empty.")
    xs = np.sort(X)
    Ft = np.clip(xs, 0.0, 1.0) if F is None else np.asarray(
        [F(v) for v in xs], dtype=float
    )
    upper = np.arange(1, n + 1) / n - Ft  # just after each jump
    lower = Ft - np.arange(0, n) / n  # just before
    return float(np.sqrt(n) * max(upper.max(), lower.max()))


def bootstrap_multiplier_process(X, t, weights=None, rng=None, F=None):
    r"""Multiplier bootstrap process
    :math:`\hat G_n(t) = n^{-1/2}\sum_i (\xi_i - \bar\xi)
    [1\{X_i \le t\} - F_n(t)]`.

    Uses mean-centred weights, so the process is centred by
    construction -- the property that makes the multiplier bootstrap
    consistent for the *limit* of :math:`G_n` rather than for
    :math:`G_n` itself.
    """
    X = np.asarray(X, dtype=float).ravel()
    n = X.size
    if n < 2:
        raise ValueError("need at least 2 observations.")
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if weights is None:
        rng = np.random.default_rng() if rng is None else rng
        weights = rng.exponential(size=n)  # Efron-style, mean 1
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != n:
        raise ValueError("weights must match the sample size.")
    ind = (X[None, :] <= t[:, None]).astype(float)
    Fn = ind.mean(axis=1, keepdims=True)
    return (w - w.mean()) @ (ind - Fn).T / np.sqrt(n)


def bracketing_number_monotone(eps, F_cdf=None):
    r"""Bracketing number of the class of monotone indicator functions
    :math:`\{1\{\cdot \le t\}\}` in :math:`L_1(P)`.

    For this class, brackets can be built from the quantiles at
    spacing eps, giving :math:`N_{[\,]}(\epsilon, \mathcal F, L_1(P))
    \le \lceil 1/\epsilon \rceil + 1` -- finite for every eps > 0,
    which is exactly the Glivenko-Cantelli condition of Kosorok
    Ch. 2 (the bracketing GC theorem).
    """
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")
    return int(np.ceil(1.0 / eps)) + 1


def covering_number_grid(points, eps, metric=None):
    r"""Covering number of a finite point set at radius eps, computed
    by a greedy cover.

    Greedy covering is within a factor of the optimum and is
    monotone-decreasing in eps, which is the property the entropy
    integrals actually rely on.
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    if P.ndim == 1:
        P = P[:, None]
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")
    d = (lambda a, b: float(np.linalg.norm(a - b))) if metric is None else metric
    centres = []
    for p in P:
        if all(d(p, c) > eps for c in centres):
            centres.append(p)
    return len(centres)


def hadamard_derivative(phi, theta, h, t_grid=None, tol=1e-6):
    r"""Numerical Hadamard directional derivative
    :math:`\phi'_\theta(h) = \lim_{t\downarrow 0}
    [\phi(\theta + t h) - \phi(\theta)]/t`.

    Uses **Richardson extrapolation** on one-sided differences:
    :math:`D(t) = \phi'h + ct + O(t^2)`, so :math:`2D(t/2) - D(t)`
    cancels the linear truncation term and is exact for quadratic phi.
    A central difference would be more accurate for smooth phi but is
    WRONG here -- the derivative is *directional*, and for maps like
    :math:`|\cdot|` at 0 a central difference returns 0 where the
    true one-sided derivative is 1.

    Hadamard differentiability additionally requires the limit to hold
    uniformly over converging directions; that is the caller's
    argument to make. This returns the numeric derivative with the
    residual drift, so non-convergence is visible rather than silent.

    Returns
    -------
    (derivative, drift, converged)
    """
    base = np.asarray(phi(theta), dtype=float)
    theta = np.asarray(theta, dtype=float)
    h = np.asarray(h, dtype=float)
    ts = np.array([1e-2, 5e-3, 1e-3, 5e-4]) if t_grid is None else np.asarray(
        t_grid, dtype=float
    )
    if np.any(ts <= 0):
        raise ValueError("t_grid entries must be positive.")
    D = np.array([(np.asarray(phi(theta + t * h), dtype=float) - base) / t
                  for t in ts])
    # Richardson on the successive halvings (ts[1] = ts[0]/2, etc.)
    rich = 2.0 * D[1::2] - D[0::2]
    est = rich[-1]
    drift = float(np.max(np.abs(rich[-1] - rich[0]))) if rich.shape[0] > 1 else float(
        np.max(np.abs(D[-1] - D[-2]))
    )
    scale = max(1.0, float(np.max(np.abs(est))))
    return est, drift, bool(drift < tol * scale)


def cox_score(beta, Z, time, event):
    r"""Cox partial-likelihood score and observed information.

    .. math:: U(\beta) = \sum_i \delta_i\Big[Z_i -
              \frac{\sum_{j \in R_i} Z_j e^{\beta'Z_j}}
                   {\sum_{j \in R_i} e^{\beta'Z_j}}\Big]

    with :math:`R_i` the risk set at the ith event time. This is the
    empirical version of the efficient score for beta in the Cox model
    (Kosorok Ch. 3): the ratio term is the risk-set average of Z, so
    the score contrasts each event's covariate against who was still
    at risk.
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    n, p = Z.shape
    if beta.size != p:
        raise ValueError(f"beta must have {p} entries, got {beta.size}.")
    if time.size != n or event.size != n:
        raise ValueError("time and event must match the rows of Z.")
    if not np.all(np.isin(event, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")

    lin = Z @ beta
    w = np.exp(lin - lin.max())  # stabilised; cancels in the ratio
    U = np.zeros(p)
    I = np.zeros((p, p))
    loglik = 0.0
    for i in np.flatnonzero(event == 1):
        at_risk = time >= time[i]
        wr = w[at_risk]
        Zr = Z[at_risk]
        s0 = wr.sum()
        if s0 <= 0:
            continue
        s1 = wr @ Zr / s0
        U += Z[i] - s1
        s2 = (Zr * wr[:, None]).T @ Zr / s0
        I += s2 - np.outer(s1, s1)
        loglik += lin[i] - (np.log(s0) + lin.max())
    return {"score": U, "information": I, "loglik": float(loglik),
            "n_events": int(event.sum()), "n": int(n)}


# --- Z- and M-estimator machinery (Ch. 2-3) -----------------------

def z_estimator_map(theta, psi, data):
    """Empirical criterion Psi_n(theta) = P_n psi_theta."""
    vals = np.asarray([psi(theta, d) for d in data], dtype=float)
    return vals.mean(axis=0)


def sup_difference(f, g, grid):
    """||f - g||_L on a grid: the uniform norm the Z-estimator
    theorems of Ch. 2 are stated in."""
    a = np.asarray([f(v) for v in grid], dtype=float)
    b = np.asarray([g(v) for v in grid], dtype=float)
    return float(np.max(np.abs(a - b)))


def survival_psi(S, t_grid, S0, L, G):
    r"""The Kaplan-Meier Z-estimator map (Kosorok Eq. 2.11, p. 26):

    .. math:: \Psi(S)(t) = S_0(t)L(t)
              + \int_0^t \frac{S_0(u)}{S(u)}\,dG(u)\,S(t) - S(t).

    Implemented exactly as printed. ``S0``, ``L`` and ``G`` are
    SUPPLIED rather than inferred: the section fixes them for its own
    censoring model, and the excerpt that states (2.11) does not
    define them unambiguously enough to reconstruct without guessing.
    Substituting plausible-looking empirical stand-ins does not make
    the Kaplan-Meier estimator a root -- checked, it does not -- so
    they are the caller's to provide, and the module computes the
    printed functional of them.
    """
    tg = np.asarray(t_grid, dtype=float)
    Sv = np.asarray(S, dtype=float)
    S0v = np.asarray(S0, dtype=float)
    Lv = np.asarray(L, dtype=float)
    Gv = np.asarray(G, dtype=float)
    for name, arr in (("S", Sv), ("S0", S0v), ("L", Lv), ("G", Gv)):
        if arr.size != tg.size:
            raise ValueError(
                f"{name} has {arr.size} entries for {tg.size} grid points.")
    dG = np.diff(np.concatenate([[0.0], Gv]))
    safe = np.where(Sv > 0, Sv, np.inf)
    integ = np.cumsum(S0v / safe * dG)
    return S0v * Lv + integ * Sv - Sv


def efficient_information(scores, nuisance_scores):
    r"""Efficient information
    :math:`\tilde I = P[\tilde\ell\tilde\ell']` where
    :math:`\tilde\ell` is the score for theta with the closed
    linear span of the NUISANCE scores projected out
    (Kosorok Ch. 3).

    The projection is the whole content: the information for theta
    in the presence of an unknown nuisance is never larger than the
    information when it is known, and the shortfall is exactly what
    the nuisance costs.
    """
    s = np.atleast_2d(np.asarray(scores, dtype=float))
    if s.shape[0] < s.shape[1]:
        s = s.T
    b = np.atleast_2d(np.asarray(nuisance_scores, dtype=float))
    if b.shape[0] != s.shape[0]:
        b = b.T
    if b.size and b.shape[0] == s.shape[0]:
        coef, *_ = np.linalg.lstsq(b, s, rcond=None)
        eff = s - b @ coef
    else:
        eff = s
    n = eff.shape[0]
    return eff.T @ eff / n, eff


def cheatsheet():
    return "_kosorok: empirical process, bridge cov, entropy, multiplier bootstrap, Cox score"
