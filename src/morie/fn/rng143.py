# morie.fn -- function file (rootcoder007/morie)
"""Wiener autocorrelation matrix Phi (Rangayyan Eq 3.163/3.164/3.165)."""

from __future__ import annotations

from . import _array_core as np
from ._sci_core import toeplitz

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_autocorrelation_matrix"]


def rangayyan_ch3_autocorrelation_matrix(x, M):
    r"""Autocorrelation matrix of the tap-input vector used in Wiener filtering.

    .. math::

        \Phi = E[\mathbf{x}(n)\,\mathbf{x}^T(n)]

    which in full :math:`M \times M` form (Eq. 3.164) is the symmetric
    Toeplitz matrix with element

    .. math::

        \phi(i-k) = E[x(n-k)\,x(n-i)], \qquad \phi(i-k) = \phi(k-i).

    Parameters
    ----------
    x : array-like
        Input signal :math:`x(n)`.
    M : int
        Filter length (number of taps), :math:`M \ge 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Phi`, shape ``(M, M)``), ``phi`` (the
        ``M`` autocorrelation lags that generate it), ``M``, ``n``,
        ``method``.

    Raises
    ------
    ValueError
        If ``M < 1`` or ``M`` exceeds the signal length.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.163), full matrix form Eq. (3.164), element Eq. (3.165),
        pp. 174-175. "With the assumption of wide-sense stationarity, the
        :math:`M \times M` matrix :math:`\Phi` is completely specified by
        :math:`M` values of the autocorrelation :math:`\phi(0), \ldots,
        \phi(M-1)`."

    Notes
    -----
    Built from the ``M`` lags rather than by forming and averaging outer
    products, because the book's own remark above says the matrix has only
    ``M`` degrees of freedom. Building it the direct way produces a matrix
    that is Toeplitz only up to sample noise, which then makes
    ``Phi @ w == Theta`` fail to reproduce the Wiener-Hopf solution exactly.
    """
    xs = np.asarray(x, dtype=float).ravel()
    M = int(M)
    if M < 1:
        raise ValueError(f"M (filter length) must be >= 1; got {M}")
    if M > xs.size:
        raise ValueError(f"M={M} exceeds the signal length {xs.size}")
    N = xs.size
    phi = np.empty(M, dtype=float)
    for k in range(M):
        # phi(k) = E[x(n)x(n-k)], averaged over the n where the tap vector exists.
        phi[k] = np.mean(xs[M - 1 :] * xs[M - 1 - k : N - k])
    Phi = toeplitz(phi)
    return RichResult(
        payload={
            "array": Phi,
            "phi": phi,
            "M": M,
            "n": int(N),
            "method": "Wiener autocorrelation matrix Phi (Rangayyan Eq 3.163/3.164/3.165)",
        }
    )


def cheatsheet():
    return "rng143: Phi = E[x(n) x^T(n)], symmetric Toeplitz (Rangayyan Eq 3.163/3.164)."
