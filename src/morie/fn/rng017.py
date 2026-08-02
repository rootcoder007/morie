# morie.fn -- function file (rootcoder007/morie)
"""Ensemble autocorrelation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_acf_ensemble_estimate"]


def rangayyan_ch3_acf_ensemble_estimate(x_k, t1, tau, M=None):
    r"""Ensemble autocorrelation (Rangayyan Ch. 3):

    .. math:: \phi_{xx}(t_1, t_1+\tau) = \lim_{M\to\infty}
              \frac1M \sum_{k=1}^{M} x_k(t_1)\,x_k(t_1+\tau).

    A function of BOTH times, not just the lag. It reduces to a
    function of tau alone precisely when the process is
    wide-sense stationary -- which the caller must establish, so this
    reports the value at the requested (t1, tau) rather than implying
    stationarity by returning a lag-only curve.

    Parameters
    ----------
    x_k : array-like, shape (M, T)
        Realisations.
    t1 : int
        First time index.
    tau : int
        Lag.
    M : int, optional
        Realisation count.

    Returns
    -------
    RichResult
        keys: ``acf``, ``t1``, ``tau``, ``M``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (ensemble autocorrelation).
    """
    X = np.atleast_2d(np.asarray(x_k, dtype=float))
    m, T = X.shape
    if M is not None and int(M) != m:
        raise ValueError(f"M = {M} does not match the {m} rows of x_k.")
    t1 = int(t1)
    tau = int(tau)
    if not 0 <= t1 < T:
        raise ValueError(f"t1 must lie in 0..{T - 1}, got {t1}.")
    if not 0 <= t1 + tau < T:
        raise ValueError(f"t1 + tau must lie in 0..{T - 1}, got {t1 + tau}.")
    return RichResult(payload={"acf": float(np.mean(X[:, t1] * X[:, t1 + tau])),
                               "t1": t1, "tau": tau, "M": int(m),
                               "method": "phi(t1, t1+tau) across realisations; two-time, not lag-only"})


def cheatsheet():
    return "rng017: two-time function; collapses to lag-only only under WSS"
