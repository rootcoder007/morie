# morie.fn -- function file (rootcoder007/morie)
"""Mean functional of a normalized random measure."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["normalized_random_measure"]


def normalized_random_measure(y, alpha=1.0, tau=1.0, mu0=0.0, sigma0=1.0):
    """Prior and posterior law of the mean of a normalized random measure.

    An NRMI is a completely random measure divided by its own total
    mass.  Regazzini, Lijoi & Prunster asked what the *linear
    functional* ``M = int x P~(dx)`` then looks like; for the gamma CRM,
    whose normalization is the Dirichlet process, the first two moments
    are available in closed form:

        E[M]   = mu0,
        Var[M] = sigma0^2 / (alpha + 1),

    and after observing ``y_1, ..., y_n`` the posterior is again a
    Dirichlet process with updated mass ``alpha + n`` and base measure
    ``(alpha P0 + sum_i delta_{y_i}) / (alpha + n)``, so

        E[M | y]   = (alpha mu0 + sum_i y_i) / (alpha + n),
        Var[M | y] = s2_post / (alpha + n + 1),

    with ``s2_post`` the variance of that updated base measure.  Because
    normalization divides the scale out, none of these depend on
    ``tau``; the invariance is asserted as an anchor.

    Parameters
    ----------
    y : array-like
        Observed values.
    alpha : float, default 1.0
        Total mass of the CRM, positive.
    tau : float, default 1.0
        Scale of the CRM, positive; carried through only to the
        unnormalized total mass.
    mu0 : float, default 0.0
        Mean of the base measure ``P0``.
    sigma0 : float, default 1.0
        Standard deviation of the base measure, non-negative.

    Returns
    -------
    RichResult
        ``estimate`` (``E[M | y]``), ``prior_mean``, ``prior_var``,
        ``post_mean``, ``post_var``, ``post_mass``, ``total_mass``,
        ``alpha``, ``tau``, ``n``.

    References
    ----------
    Regazzini, E., Lijoi, A. & Prunster, I. (2003).  Distributional
    results for means of normalized random measures with independent
    increments.  Annals of Statistics, 31(2), 560--585.
    doi:10.1214/aos/1051027881
    """
    a = float(alpha)
    t = float(tau)
    if a <= 0.0:
        raise ValueError("normalized_random_measure: alpha must be positive")
    if t <= 0.0:
        raise ValueError("normalized_random_measure: tau must be positive")
    s0 = float(sigma0)
    if s0 < 0.0:
        raise ValueError("normalized_random_measure: sigma0 must be non-negative")
    m0 = float(mu0)
    v = C.vec(y)
    n = len(v)
    if n == 0:
        raise ValueError("normalized_random_measure: y is empty")
    prior_mean = m0
    prior_var = s0 * s0 / (a + 1.0)
    mass = a + n
    w0 = a / mass
    post_mean = (a * m0 + sum(v)) / mass
    # second moment of the mixture (alpha P0 + sum delta_y) / (alpha + n)
    m2 = w0 * (s0 * s0 + m0 * m0) + sum(x * x for x in v) / mass
    s2_post = m2 - post_mean * post_mean
    if s2_post < 0.0:
        s2_post = 0.0
    return RichResult(payload={
        "estimate": post_mean, "prior_mean": prior_mean, "prior_var": prior_var,
        "post_mean": post_mean, "post_var": s2_post / (mass + 1.0),
        "post_base_var": s2_post, "post_mass": mass, "total_mass": a * t,
        "alpha": a, "tau": t, "n": n,
        "method": "Mean functional of a normalized random measure"})


def cheatsheet():
    return "nrgmwd: Mean functional of a normalized random measure"


normalizedrandommeasure = normalized_random_measure
