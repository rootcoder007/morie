# morie.fn -- function file (rootcoder007/morie)
"""Wiener cross-correlation vector Theta (Rangayyan Eq 3.160/3.161)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_cross_correlation_vector"]


def rangayyan_ch3_cross_correlation_vector(x, d, M):
    r"""Cross-correlation vector between the tap-input vector and the desired response.

    .. math::

        \Theta = E[\mathbf{x}(n)\,d(n)]
        = [\theta(0), \theta(-1), \ldots, \theta(1-M)]^T

    with (Eq. 3.161)

    .. math::

        \theta(-k) = E[x(n-k)\,d(n)], \quad k = 0, 1, \ldots, M-1.

    Parameters
    ----------
    x : array-like
        Input signal :math:`x(n)`.
    d : array-like
        Desired response :math:`d(n)`, same length as ``x``. This is a
        *signal*, not a scalar lag.
    M : int
        Filter length (number of taps), :math:`M \ge 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Theta`, length ``M``), ``M``, ``n``,
        ``method``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length, if ``M < 1``, or if ``M``
        exceeds the signal length.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.160) and its expansion Eq. (3.161), p. 174, Section 3.9
        ("The Wiener filter").

    Notes
    -----
    The expectation is estimated by the sample average over the ``N - M + 1``
    time indices for which the full tap-input vector :math:`\mathbf{x}(n)` is
    available. Indices before the filter has filled are excluded rather than
    zero-padded: the book defines :math:`\Theta` as an expectation over the
    stationary process, and padding would bias every lag toward zero by a
    known factor of ``(N-k)/N``.

    The third argument was named ``n`` before this was implemented, which read
    as a time index; it is the filter length :math:`M`. The function has never
    returned a value -- the previous body referenced an undefined ``y`` and
    raised ``UnboundLocalError`` on every call -- so there is no caller to
    break.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ds = np.asarray(d, dtype=float).ravel()
    if xs.size != ds.size:
        raise ValueError(
            f"x and d must have the same length; got {xs.size} and {ds.size}. "
            "d is the desired-response SIGNAL d(n), not a scalar."
        )
    M = int(M)
    if M < 1:
        raise ValueError(f"M (filter length) must be >= 1; got {M}")
    if M > xs.size:
        raise ValueError(f"M={M} exceeds the signal length {xs.size}")
    N = xs.size
    theta = np.empty(M, dtype=float)
    for k in range(M):
        # theta(-k) = E[x(n-k) d(n)], averaged over the n for which the whole
        # tap vector exists: n = M-1 .. N-1.
        theta[k] = np.mean(xs[M - 1 - k : N - k] * ds[M - 1 :])
    return RichResult(
        payload={
            "array": theta,
            "M": M,
            "n": int(N),
            "method": "Wiener cross-correlation vector Theta (Rangayyan Eq 3.160/3.161)",
        }
    )


def cheatsheet():
    return "rng142: Theta = E[x(n) d(n)], theta(-k)=E[x(n-k)d(n)] (Rangayyan Eq 3.160/3.161)."
