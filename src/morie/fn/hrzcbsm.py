# morie.fn -- function file (rootcoder007/morie)
"""Maximum-score estimator with choice-based samples."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_choice_based_sms", "choice_based_optimal_shares"]


def choice_based_optimal_shares(pi1):
    r"""Sampling shares that minimise the asymptotic variance
    (Horowitz Sec. 4.4.1):

    .. math:: q_0 = \frac{\pi_0^{1/2}}{\pi_0^{1/2} + \pi_1^{1/2}},
              \qquad
              q_1 = \frac{\pi_1^{1/2}}{\pi_0^{1/2} + \pi_1^{1/2}}.

    The designer of a choice-based sample picks :math:`n_1` and
    :math:`n_0`, and the asymptotic distribution depends on them only
    through :math:`\pi_1/q_1 + \pi_0/q_0`. Minimising that factor
    gives the square-root rule above: the optimum is NOT
    :math:`q_j = \pi_j` (a random sample) and not an even split
    either, except at :math:`\pi_1 = 1/2` where the two coincide.

    Parameters
    ----------
    pi1 : float in (0, 1)
        Population share with Y = 1.

    Returns
    -------
    RichResult
        keys: ``q1``, ``q0``, ``factor`` (the minimised
        :math:`\pi_1/q_1 + \pi_0/q_0`), ``factor_at_random_sample``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 4.4.1 (choice-based samples), the
    variance-minimising choice of q_0 and q_1, printed p. 123.
    """
    p1 = float(pi1)
    if not 0.0 < p1 < 1.0:
        raise ValueError(f"pi1 must lie in (0, 1), got {p1}.")
    p0 = 1.0 - p1
    denom = np.sqrt(p0) + np.sqrt(p1)
    q0 = float(np.sqrt(p0) / denom)
    q1 = float(np.sqrt(p1) / denom)
    return RichResult(payload={
        "q1": q1, "q0": q0,
        "factor": float(p1 / q1 + p0 / q0),
        "factor_at_random_sample": 2.0,
        "method": "q_j proportional to sqrt(pi_j); minimises pi_1/q_1 + pi_0/q_0"})


def horowitz_choice_based_sms(x, y, sampling_weights, smoothed=True, h=None,
                              n_restarts=8, seed=0):
    r"""Maximum-score estimator for a choice-based sample (Horowitz
    Sec. 4.4.1), equations (4.33) and (4.35):

    .. math:: S_{n,CB}(b) = \frac{\pi_1}{n_1}\sum_i Y_i\,
              \mathbf 1\{X_i'b \ge 0\}
              - \frac{\pi_0}{n_0}\sum_i (1 - Y_i)\,
              \mathbf 1\{X_i'b \ge 0\},

    maximised subject to :math:`|b_1| = 1`; the smoothed form (4.35)
    replaces each indicator with :math:`K(X_i'b/h_n)`.

    A choice-based sample is stratified ON Y: the fraction with
    :math:`Y = 1` is fixed by design and X is drawn conditional on Y.
    Estimators built for random samples are INCONSISTENT here except
    in special cases, and the reweighting is what repairs that -- the
    two terms are divided by the realised counts :math:`n_1, n_0` and
    multiplied by the POPULATION shares :math:`\pi_1, \pi_0`, which
    the method assumes are known from an external source such as a
    census. Passing the sample shares in place of the population
    shares silently reproduces the uncorrected estimator.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates; the first column carries the scale normalisation.
    y : array-like of {0, 1}, shape (n,)
        Binary response.
    sampling_weights : float or sequence of two floats
        The population aggregate share ``pi1``, or the pair
        ``(pi0, pi1)``. These are POPULATION shares, not sample ones.
    smoothed : bool, default True
        Use (4.35) rather than the discontinuous (4.33).
    h : float, optional
        Bandwidth for the smoothed form; ``n**(-1/5)`` otherwise.
    n_restarts : int, default 8
        Restarts, since (4.33) is a step function of b.
    seed : int, default 0
        RNG seed for the restarts.

    Returns
    -------
    RichResult
        keys: ``beta``, ``score``, ``pi1``, ``pi0``, ``n1``, ``n0``,
        ``smoothed``, ``bandwidth``, ``rate_exponent``,
        ``standard_errors_valid``, ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 4.4.1, eqs. (4.33)-(4.35) and
    Theorems 4.7-4.8.
    """
    from . import _stats_core as stats

    from ._horowitz import optimize_scale_normalized

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    if not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError("y must be binary 0/1.")
    n, d = X.shape
    if d < 2:
        raise ValueError(f"need at least 2 covariates, got {d}.")

    w = np.atleast_1d(np.asarray(sampling_weights, dtype=float)).ravel()
    if w.size == 1:
        pi1 = float(w[0])
        pi0 = 1.0 - pi1
    elif w.size == 2:
        pi0, pi1 = float(w[0]), float(w[1])
    else:
        raise ValueError("sampling_weights must be pi1 or (pi0, pi1).")
    if not (0.0 < pi1 < 1.0 and 0.0 < pi0 < 1.0):
        raise ValueError(f"aggregate shares must lie in (0, 1), got {(pi0, pi1)}.")

    n1 = int(np.sum(yv == 1.0))
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("need both response categories present.")

    hh = float(n ** (-0.2)) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")

    def score(b):
        v = X @ b
        # (4.35)'s K is the INTEGRAL of a kernel -- a smooth CDF
        # standing in for the indicator, not a density
        ind = stats.norm.cdf(v / hh) if smoothed else (v >= 0.0).astype(float)
        return (pi1 / n1) * float(np.sum(yv * ind)) - \
               (pi0 / n0) * float(np.sum((1.0 - yv) * ind))

    beta, negval = optimize_scale_normalized(lambda b: -score(b), d,
                                             n_restarts=n_restarts, seed=seed)
    return RichResult(payload={
        "beta": beta, "score": -negval, "pi1": pi1, "pi0": pi0,
        "n1": n1, "n0": n0, "smoothed": bool(smoothed),
        "bandwidth": hh if smoothed else None,
        "rate_exponent": -0.4 if smoothed else -1.0 / 3.0,
        "standard_errors_valid": bool(smoothed),
        "n": int(n), "d": int(d),
        "method": "Choice-based max score (4.33)/(4.35); population shares reweight the strata"})


def cheatsheet():
    return "hrzcbsm: pi_j are POPULATION shares; sample shares reproduce the uncorrected estimator"
