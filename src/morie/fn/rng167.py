# morie.fn -- function file (rootcoder007/morie)
"""RLS recursion for the autocorrelation matrix (Rangayyan Eq 3.211)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_phi_recursion"]


def rangayyan_ch3_rls_phi_recursion(Phi, r, lam):
    r"""One RLS update of the time-averaged autocorrelation matrix.

    .. math::

        \Phi(n) = \lambda\,\Phi(n-1) + \mathbf{r}(n)\,\mathbf{r}^T(n)

    Parameters
    ----------
    Phi : array-like, shape (M, M)
        Previous matrix :math:`\Phi(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    lam : float
        Forgetting factor :math:`\lambda`, with :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Phi(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch or if ``lam`` is outside :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.211), p. 187, obtained by isolating the :math:`i = n` term of
        the definition Eq. (3.208)
        :math:`\Phi(n) = \sum_{i=1}^{n}\lambda^{n-i}\mathbf{r}(i)\mathbf{r}^T(i)`.

    Notes
    -----
    The book bounds the forgetting factor as :math:`0 < \lambda \le 1`, noting
    that :math:`1/(1-\lambda)` "is a measure of the memory of the algorithm".
    :math:`\lambda > 1` would grow past history without bound and is rejected;
    the vestigial ``n`` argument the previous signature carried is not in the
    equation and has been dropped.
    """
    P = np.asarray(Phi, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    lam = float(lam)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"Phi must be a square matrix; got shape {P.shape}")
    if rv.size != P.shape[0]:
        raise ValueError(f"r must have length {P.shape[0]}; got {rv.size}")
    if not (0.0 < lam <= 1.0):
        raise ValueError(
            f"lam (forgetting factor) must satisfy 0 < lam <= 1; got {lam!r} "
            "(Rangayyan p. 186)"
        )
    Phi_n = lam * P + np.outer(rv, rv)
    return RichResult(
        payload={
            "array": Phi_n,
            "M": int(P.shape[0]),
            "lam": lam,
            "method": "RLS autocorrelation-matrix recursion (Rangayyan Eq 3.211)",
        }
    )


def cheatsheet():
    return "rng167: Phi(n) = lam*Phi(n-1) + r r' (Rangayyan Eq 3.211)."
