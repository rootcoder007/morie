# morie.fn -- function file (rootcoder007/morie)
"""Inverse Fisher z transformation back to a correlation (Fisher 1921)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_fishers_z_inverse"]


def ma_fishers_z_inverse(z):
    r"""Map a Fisher :math:`z` back to the correlation scale.

    .. math::

        r = \frac{e^{2z}-1}{e^{2z}+1} = \tanh(z)

    Parameters
    ----------
    z : float or array-like
        Fisher-transformed correlation(s). Any finite value is admissible --
        :math:`z` is unbounded, unlike :math:`r`.

    Returns
    -------
    RichResult
        keys: ``r``, ``z``, ``method``.

    Raises
    ------
    ValueError
        If ``z`` is not finite.

    References
    ----------
    Fisher, R. A. (1921). On the "probable error" of a coefficient of
        correlation deduced from a small sample. *Metron*, 1, 3-32.

    Notes
    -----
    Computed as :func:`numpy.tanh`, which is the same function as the printed
    :math:`(e^{2z}-1)/(e^{2z}+1)` but does not overflow: at :math:`z = 400`
    the literal expression evaluates ``inf/inf`` and returns ``nan``, whereas
    ``tanh`` saturates correctly to ``1.0``. The two agree to the last bit
    everywhere the literal form is finite, and the tests check that.

    This is the step that makes a pooled meta-analytic estimate interpretable:
    the pooling happens on the :math:`z` scale, where the variance is
    stabilised, and only the final summary is mapped back. Because
    :math:`\tanh` is concave for :math:`z > 0`, the back-transformed mean is
    not the mean of the back-transformed values -- averaging on the wrong
    scale is exactly the bias :mod:`morie.fn.mafshz` exists to avoid.
    """
    zz = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(zz)):
        raise ValueError(f"z must be finite; got {z!r}")
    r = np.tanh(zz)
    scalar = r.ndim == 0
    return RichResult(
        payload={
            "r": float(r) if scalar else r,
            "z": float(zz) if scalar else zz,
            "method": "inverse Fisher z, r = tanh(z) (Fisher 1921)",
        }
    )


def cheatsheet():
    return "mafshi: r = (e^2z - 1)/(e^2z + 1) = tanh(z) (Fisher 1921)."
