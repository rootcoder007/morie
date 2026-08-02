# morie.fn -- function file (rootcoder007/morie)
"""Z-transform of a causal FIR system of length N (transfer function)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_z_transform_fir"]


def rangayyan_ch3_z_transform_fir(h, z):
    r"""FIR transfer function
    :math:`H(z) = \sum_{n=0}^{N-1} h(n) z^{-n}`.

    A finite causal impulse response gives a polynomial in
    :math:`z^{-1}`: it has no poles other than the origin, hence an
    FIR filter is always stable. Evaluating on the unit circle,
    :math:`z = e^{j\omega}`, gives the frequency response.

    Parameters
    ----------
    h : array-like, shape (N,)
        Impulse response taps h(0) .. h(N-1).
    z : complex or array-like of complex
        Evaluation point(s); must be non-zero.

    Returns
    -------
    RichResult
        keys: ``H`` (complex scalar or array matching ``z``), ``z``,
        ``N``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3 (z-transform, FIR transfer functions).
    """
    h = np.asarray(h, dtype=float).ravel()
    if h.size == 0:
        raise ValueError("h must be non-empty.")
    zv = np.atleast_1d(np.asarray(z, dtype=complex))
    if np.any(zv == 0):
        raise ValueError("z = 0 is a pole of a causal FIR transfer function.")
    n = np.arange(h.size)
    H = np.array([np.sum(h * zk ** (-n)) for zk in zv])
    scalar = np.ndim(z) == 0

    return RichResult(
        payload={
            "H": complex(H[0]) if scalar else H,
            "z": complex(zv[0]) if scalar else zv,
            "N": int(h.size),
            "method": "FIR transfer function H(z) = sum_n h(n) z^-n",
        }
    )


def cheatsheet():
    return "rng053: H(z) = sum_{n=0}^{N-1} h(n) z^-n; z = e^{jw} gives the frequency response"
