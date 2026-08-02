# morie.fn -- function file (rootcoder007/morie)
"""RLS recursion for P(n) via the gain vector (Rangayyan Eq 3.218)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_p_recursion"]


def rangayyan_ch3_rls_p_recursion(P, k, r, lam):
    r"""Update :math:`P(n) = \Phi^{-1}(n)` using the precomputed gain vector.

    .. math::

        P(n) = \lambda^{-1}P(n-1) - \lambda^{-1}\mathbf{k}(n)\mathbf{r}^T(n)P(n-1)

    where (Eq. 3.217)

    .. math::

        \mathbf{k}(n) = \frac{\lambda^{-1}P(n-1)\mathbf{r}(n)}
                             {1 + \lambda^{-1}\mathbf{r}^T(n)P(n-1)\mathbf{r}(n)}.

    Parameters
    ----------
    P : array-like, shape (M, M)
        Previous :math:`P(n-1)`, initialised as :math:`\delta^{-1}I`.
    k : array-like, shape (M,)
        Gain vector :math:`\mathbf{k}(n)` from Eq. (3.217).
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    lam : float
        Forgetting factor, :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`P(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch or ``lam`` outside :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.218), p. 188, the simplified form of Eq. (3.215) under the
        notation :math:`P(n) = \Phi^{-1}(n)` of Eq. (3.216) and the gain
        vector Eq. (3.217). The book notes :math:`\mathbf{k}(n)` "is analogous
        to the Kalman gain vector".

    Notes
    -----
    ``k`` is taken as an argument rather than recomputed, because that is how
    the book states Eq. (3.218) and because the caller already needs
    :math:`\mathbf{k}(n)` for the tap-weight update Eq. (3.224). Passing a
    :math:`\mathbf{k}` inconsistent with ``P``, ``r`` and ``lam`` yields a
    :math:`P(n)` that is not :math:`\Phi^{-1}(n)`, and nothing here can detect
    that -- Eq. (3.221) :math:`\mathbf{k}(n) = P(n)\mathbf{r}(n)` is the
    identity to check it with, and the tests do.
    """
    Pm = np.asarray(P, dtype=float)
    kv = np.asarray(k, dtype=float).ravel()
    rv = np.asarray(r, dtype=float).ravel()
    lam = float(lam)
    if Pm.ndim != 2 or Pm.shape[0] != Pm.shape[1]:
        raise ValueError(f"P must be a square matrix; got shape {Pm.shape}")
    M = Pm.shape[0]
    if kv.size != M:
        raise ValueError(f"k must have length {M}; got {kv.size}")
    if rv.size != M:
        raise ValueError(f"r must have length {M}; got {rv.size}")
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must satisfy 0 < lam <= 1; got {lam!r} (Rangayyan p. 186)")
    inv_lam = 1.0 / lam
    P_n = inv_lam * Pm - inv_lam * np.outer(kv, rv @ Pm)
    return RichResult(
        payload={
            "array": P_n,
            "M": int(M),
            "lam": lam,
            "method": "RLS P(n) recursion via the gain vector (Rangayyan Eq 3.218)",
        }
    )


def cheatsheet():
    return "rng172: P(n) = lam^-1 P(n-1) - lam^-1 k(n) r'(n) P(n-1) (Rangayyan Eq 3.218)."
