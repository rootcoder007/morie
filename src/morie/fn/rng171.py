# morie.fn -- function file (rootcoder007/morie)
"""Kalman-like gain vector in RLS (Rangayyan Eq 3.217)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_kalman_gain"]


def rangayyan_ch3_rls_kalman_gain(P, r, lam):
    r"""Gain vector :math:`\mathbf{k}(n)` of the RLS algorithm.

    .. math::

        \mathbf{k}(n) = \frac{\lambda^{-1}P(n-1)\mathbf{r}(n)}
                             {1 + \lambda^{-1}\mathbf{r}^T(n)P(n-1)\mathbf{r}(n)}

    Parameters
    ----------
    P : array-like, shape (M, M)
        Previous inverse correlation matrix :math:`P(n-1) = \Phi^{-1}(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    lam : float
        Forgetting factor, :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\mathbf{k}(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch or ``lam`` outside :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.217), p. 188. The book notes :math:`\mathbf{k}(n)` "is
        analogous to the Kalman gain vector in Kalman filter theory".

    Notes
    -----
    The denominator is a scalar, which is the whole point of the
    matrix-inversion lemma this comes from -- no inverse is formed here.

    The previous body was the shared mean-and-standard-error stub,
    ``float(np.mean(P))`` returned under the key ``estimate``. It did not
    raise, so it was **green** in the suite while returning the mean of an
    inverse correlation matrix in place of a gain vector; the equation above
    was printed correctly in its own docstring the whole time. The vestigial
    trailing ``n`` argument, which appears nowhere in Eq. (3.217), is dropped.
    """
    Pm = np.asarray(P, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    lam = float(lam)
    if Pm.ndim != 2 or Pm.shape[0] != Pm.shape[1]:
        raise ValueError(f"P must be a square matrix; got shape {Pm.shape}")
    if rv.size != Pm.shape[0]:
        raise ValueError(f"r must have length {Pm.shape[0]}; got {rv.size}")
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must satisfy 0 < lam <= 1; got {lam!r} (Rangayyan p. 186)")
    inv_lam = 1.0 / lam
    num = inv_lam * (Pm @ rv)
    den = 1.0 + inv_lam * float(rv @ Pm @ rv)
    return RichResult(
        payload={
            "array": num / den,
            "M": int(Pm.shape[0]),
            "lam": lam,
            "method": "RLS Kalman-like gain vector (Rangayyan Eq 3.217)",
        }
    )


def cheatsheet():
    return "rng171: k(n) = lam^-1 P(n-1) r / (1 + lam^-1 r' P(n-1) r) (Rangayyan Eq 3.217)."
