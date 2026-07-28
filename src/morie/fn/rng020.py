# morie.fn -- function file (rootcoder007/morie)
"""Time-averaged autocorrelation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_time_averaged_acf"]


def rangayyan_ch3_time_averaged_acf(x_k, tau, T=None):
    r"""Time-averaged autocorrelation (Rangayyan Ch. 3):

    .. math:: \phi_{xx}(\tau, k) = \lim_{T\to\infty} \frac1T
              \int_{-T/2}^{T/2} x_k(t)\,x_k(t+\tau)\,dt.

    The lag-domain counterpart of :mod:`morie.fn.rng019`: one
    realisation, averaged over time. Under ergodicity it converges to
    the ensemble autocorrelation of :mod:`morie.fn.rng017`.

    Parameters
    ----------
    x_k : array-like, shape (T,) or (M, T)
        Realisation(s).
    tau : int
        Lag.
    T : int, optional
        Length check.

    Returns
    -------
    RichResult
        keys: ``acf``, ``tau``, ``n_used``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (time-averaged autocorrelation).
    """
    X = np.atleast_2d(np.asarray(x_k, dtype=float))
    m, n = X.shape
    tau = int(tau)
    if not 0 <= tau < n:
        raise ValueError(f"tau must lie in 0..{n - 1}, got {tau}.")
    if T is not None and int(T) != n:
        raise ValueError(f"T = {T} does not match the {n} samples.")
    vals = np.array([float(np.mean(X[k, : n - tau] * X[k, tau:])) for k in range(m)])
    return RichResult(payload={"acf": float(vals[0]) if m == 1 else vals,
                               "tau": tau, "n_used": int(n - tau),
                               "method": "time-averaged phi(tau); -> ensemble ACF under ergodicity"})


def cheatsheet():
    return "rng020: one record over time; matches rng017 under ergodicity"
