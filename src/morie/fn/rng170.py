# morie.fn -- function file (rootcoder007/morie)
"""Riccati recursion for the inverse autocorrelation matrix (Rangayyan Eq 3.215)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_inverse_recursion"]


def rangayyan_ch3_rls_inverse_recursion(Phi_inv, r, lam):
    r"""Update :math:`\Phi^{-1}` directly, without inverting anything.

    .. math::

        \Phi^{-1}(n) = \lambda^{-1}\Phi^{-1}(n-1)
          - \frac{\lambda^{-2}\Phi^{-1}(n-1)\mathbf{r}(n)\mathbf{r}^T(n)\Phi^{-1}(n-1)}
                 {1 + \lambda^{-1}\mathbf{r}^T(n)\Phi^{-1}(n-1)\mathbf{r}(n)}

    Parameters
    ----------
    Phi_inv : array-like, shape (M, M)
        Previous inverse :math:`\Phi^{-1}(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    lam : float
        Forgetting factor, :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Phi^{-1}(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch or ``lam`` outside :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.215), p. 188, from applying the matrix-inversion ("ABCD")
        lemma Eq. (3.213) to the recursion Eq. (3.211) with
        :math:`A = \lambda\Phi(n-1)`, :math:`B = \mathbf{r}(n)`,
        :math:`C = 1`, :math:`D = \mathbf{r}^T(n)`.

    Notes
    -----
    The point of the lemma is that the bracketed quantity in Eq. (3.214) is a
    **scalar**, so the :math:`M \times M` inverse never has to be recomputed.
    The denominator is therefore a plain division here, not a solve.

    This is the same quantity :mod:`morie.fn.rng172` produces via the gain
    vector :math:`\mathbf{k}(n)`; the two agree to rounding, which is what the
    tests check.
    """
    Pi = np.asarray(Phi_inv, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    lam = float(lam)
    if Pi.ndim != 2 or Pi.shape[0] != Pi.shape[1]:
        raise ValueError(f"Phi_inv must be a square matrix; got shape {Pi.shape}")
    if rv.size != Pi.shape[0]:
        raise ValueError(f"r must have length {Pi.shape[0]}; got {rv.size}")
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must satisfy 0 < lam <= 1; got {lam!r} (Rangayyan p. 186)")
    inv_lam = 1.0 / lam
    Pr = Pi @ rv
    denom = 1.0 + inv_lam * float(rv @ Pr)
    Pi_n = inv_lam * Pi - (inv_lam**2) * np.outer(Pr, Pr) / denom
    return RichResult(
        payload={
            "array": Pi_n,
            "M": int(Pi.shape[0]),
            "lam": lam,
            "method": "RLS inverse-autocorrelation recursion (Rangayyan Eq 3.215)",
        }
    )


def cheatsheet():
    return "rng170: Riccati recursion for Phi^-1(n) (Rangayyan Eq 3.215)."
