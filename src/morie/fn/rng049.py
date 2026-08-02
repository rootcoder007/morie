# morie.fn -- function file (rootcoder007/morie)
"""Laplace transform of a causal finite-duration h(t) over [0, T]."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_laplace_transform_causal_finite"]


def rangayyan_ch3_laplace_transform_causal_finite(h, s, dt=1.0):
    r"""Numeric Laplace transform
    :math:`H(s) = \int_0^T h(t) e^{-st}\, dt`.

    For a causal impulse response of finite duration T the transform
    integral has finite limits, so it converges for every s and can be
    evaluated by quadrature. Complex s are accepted, so setting
    :math:`s = j\omega` recovers the Fourier transform of h.

    Parameters
    ----------
    h : array-like, shape (m,)
        Impulse response sampled on [0, T] with step ``dt``.
    s : complex or array-like of complex
        Transform variable(s).
    dt : float, default 1.0
        Sampling interval; T = (m - 1) * dt.

    Returns
    -------
    RichResult
        keys: ``H`` (complex scalar or array matching ``s``), ``s``,
        ``T``, ``dt``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3 (Laplace transform of causal systems).
    """
    h = np.asarray(h, dtype=float).ravel()
    if h.size < 2:
        raise ValueError("need at least 2 samples.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    t = np.arange(h.size) * dt
    sv = np.atleast_1d(np.asarray(s, dtype=complex))
    H = np.array([np.trapezoid(h * np.exp(-sk * t), dx=dt) for sk in sv])
    scalar = np.ndim(s) == 0

    return RichResult(
        payload={
            "H": complex(H[0]) if scalar else H,
            "s": complex(sv[0]) if scalar else sv,
            "T": float(t[-1]),
            "dt": dt,
            "method": "Laplace transform of a finite-duration causal h: int_0^T h e^{-st} dt",
        }
    )


def cheatsheet():
    return "rng049: H(s) = int_0^T h(t) e^{-st} dt by quadrature; s = jw gives the FT"
