# morie.fn -- function file (rootcoder007/morie)
"""Running integral of a causal signal over [0, t]."""

from . import _array_core as np
from scipy import integrate

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_integral_causal"]


def rangayyan_ch3_integral_causal(x, dt=1.0):
    r"""Running integral :math:`y(t) = \int_0^t x(\tau)\, d\tau`.

    Cumulative trapezoidal integration of a causal signal (zero for
    t < 0), so ``y[0] = 0`` and each later value accumulates only past
    input -- integration is itself a causal LTI operation.

    Parameters
    ----------
    x : array-like, shape (m,)
        Samples on a uniform grid starting at t = 0.
    dt : float, default 1.0
        Sampling interval.

    Returns
    -------
    RichResult
        keys: ``y`` (m,), ``t`` (m,), ``total`` (y at the last
        sample), ``dt``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("need at least 2 samples.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    y = integrate.cumulative_trapezoid(x, dx=dt, initial=0.0)
    return RichResult(
        payload={
            "y": y,
            "t": np.arange(x.size) * dt,
            "total": float(y[-1]),
            "dt": dt,
            "method": "Running integral y(t) = int_0^t x(tau) dtau (cumulative trapezoid)",
        }
    )


def cheatsheet():
    return "rng103: cumulative trapezoid integral of a causal signal"
