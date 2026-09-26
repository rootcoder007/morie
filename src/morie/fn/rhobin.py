"""Lower bound on the equicorrelation of exchangeable binary data."""

import math

from ._richresult import RichResult

__all__ = ["binary_equicorrelation_bound"]


def binary_equicorrelation_bound(mu, n):
    r"""Smallest equicorrelation of ``n`` exchangeable binary variables.

    Any ``n`` equicorrelated variables need :math:`\rho > -1/(n-1)`.
    For binary variables with common mean :math:`\mu` and a joint
    distribution symmetric in its coordinates the bound is larger
    (Gilliland & Schabenberger 2001; Schabenberger & Gotway 2005,
    Sec. 6.3.3, p. 356, Figure 6.7). An exchangeable law is a law for
    :math:`S = \sum Y_i` on :math:`\{0, \dots, n\}` with
    :math:`E[S] = n\mu`, and

    .. math::

        \rho = \frac{E[S(S-1)]/\{n(n-1)\} - \mu^2}{\mu(1-\mu)},

    so the smallest :math:`\rho` minimises :math:`E[S(S-1)]` at fixed
    mean: all mass on the two integers either side of :math:`n\mu`. The
    bound equals :math:`-1/(n-1)` exactly when :math:`n\mu` is an
    integer.

    Parameters
    ----------
    mu : float
        Common success probability, in (0, 1).
    n : int
        Number of variables, at least 2.

    Returns
    -------
    RichResult
        ``rho_lower`` (the binary bound), ``rho_min_any``
        (:math:`-1/(n-1)`), ``support`` and ``probabilities`` of the
        extremal law of :math:`S`.

    References
    ----------
    Gilliland, D. & Schabenberger, O. (2001). Limits on pairwise
    association for equi-correlated binary variables. Journal of Applied
    Statistical Science 10, 279-285. Schabenberger, O. & Gotway, C. A.
    (2005). Statistical Methods for Spatial Data Analysis. Chapman &
    Hall/CRC, p. 356.
    """
    mu = float(mu)
    n = int(n)
    if n < 2:
        raise ValueError("`n` must be at least 2")
    if not 0 < mu < 1:
        raise ValueError("`mu` must lie in (0, 1)")
    m = n * mu
    lo = math.floor(m)
    if m == lo:
        support, probs = [int(lo)], [1.0]
    else:
        p_hi = m - lo
        support, probs = [int(lo), int(lo) + 1], [1.0 - p_hi, p_hi]
    e2 = sum(p * s * (s - 1) for s, p in zip(support, probs)) / (n * (n - 1))
    rho = (e2 - mu * mu) / (mu * (1.0 - mu))
    return RichResult(
        title="Lower bound on binary equicorrelation",
        summary_lines=[("mu", mu), ("n", n), ("rho_lower", rho)],
        payload={
            "rho_lower": rho,
            "rho_min_any": -1.0 / (n - 1),
            "support": support,
            "probabilities": probs,
            "mu": mu,
            "n": n,
        },
    )


def cheatsheet():
    return "rhobin: smallest equicorrelation of n exchangeable binary variables with mean mu"
