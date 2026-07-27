# morie.fn -- function file (rootcoder007/morie)
"""Z-transform of a causal discrete-time sequence."""

from ._richresult import RichResult
from .rng053 import rangayyan_ch3_z_transform_fir

__all__ = ["rangayyan_z_transform"]


def rangayyan_z_transform(x_coeffs, z=None):
    r"""One-sided z-transform :math:`X(z) = \sum_{n \ge 0} x(n) z^{-n}`.

    For a finite coefficient sequence this is the same polynomial in
    :math:`z^{-1}` as an FIR transfer function, so the evaluation
    delegates to :func:`morie.fn.rng053.rangayyan_ch3_z_transform_fir`.
    With ``z=None`` only the coefficient vector and the implied
    polynomial degree are returned.

    Parameters
    ----------
    x_coeffs : array-like
        The sequence x(0), x(1), ...
    z : complex or array-like, optional
        Where to evaluate.

    Returns
    -------
    RichResult
        keys: ``coefficients``, ``degree``, ``H`` (None when ``z`` is
        None), ``z``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    import numpy as np

    x = np.asarray(x_coeffs, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x_coeffs must be non-empty.")
    H = zz = None
    if z is not None:
        out = rangayyan_ch3_z_transform_fir(x, z)
        H, zz = out["H"], out["z"]
    return RichResult(
        payload={
            "coefficients": x,
            "degree": int(x.size - 1),
            "H": H,
            "z": zz,
            "method": "One-sided z-transform X(z) = sum_{n>=0} x(n) z^-n",
        }
    )


def cheatsheet():
    return "rgztf: X(z) = sum_{n>=0} x(n) z^-n (finite sequence, delegates to rng053)"
