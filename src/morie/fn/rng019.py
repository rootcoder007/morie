# morie.fn -- function file (rootcoder007/morie)
"""Time-average mean."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_time_average_mean"]


def rangayyan_ch3_time_average_mean(x_k, T=None, dt=1.0):
    r"""Time-average mean of one realisation (Rangayyan Ch. 3):

    .. math:: \mu_x(k) = \lim_{T\to\infty} \frac1T
              \int_{-T/2}^{T/2} x_k(t)\, dt.

    Averaging ALONG one record. It equals the ensemble mean only for
    an ergodic process; comparing the two is the practical test of
    ergodicity, so when several realisations are supplied their spread
    is returned for exactly that comparison.

    Parameters
    ----------
    x_k : array-like, shape (T,) or (M, T)
        One realisation, or several.
    T : int, optional
        Length check.
    dt : float, default 1.0
        Sample interval (the discrete average is dt-invariant).

    Returns
    -------
    RichResult
        keys: ``time_mean`` (per realisation), ``spread_across_k``,
        ``T``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (time averages and ergodicity).
    """
    X = np.atleast_2d(np.asarray(x_k, dtype=float))
    m, n = X.shape
    if n < 1:
        raise ValueError("x_k must be non-empty.")
    if T is not None and int(T) != n:
        raise ValueError(f"T = {T} does not match the {n} samples.")
    if float(dt) <= 0:
        raise ValueError("dt must be positive.")
    means = X.mean(axis=1)
    return RichResult(payload={"time_mean": float(means[0]) if m == 1 else means,
                               "spread_across_k": float(np.std(means)) if m > 1 else 0.0,
                               "T": int(n),
                               "method": "average ALONG the record; equals ensemble mean iff ergodic"})


def cheatsheet():
    return "rng019: time vs ensemble mean agreeing IS the ergodicity check"
