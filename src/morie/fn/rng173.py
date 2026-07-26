# morie.fn -- function file (rootcoder007/morie)
"""RLS gain identity k(n) = P(n) r(n) (Rangayyan Eq 3.221)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_gain_identity"]


def rangayyan_ch3_rls_gain_identity(P, r):
    r"""Gain vector expressed through the *updated* inverse correlation matrix.

    .. math::

        \mathbf{k}(n) = P(n)\,\mathbf{r}(n)

    Parameters
    ----------
    P : array-like, shape (M, M)
        The **updated** matrix :math:`P(n)`, not :math:`P(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\mathbf{k}(n)`), ``M``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.221), p. 188, obtained by comparing the bracketed expression in
        Eq. (3.220) with the :math:`P(n)` recursion Eq. (3.218).

    Notes
    -----
    Same vector as :mod:`morie.fn.rng171`, reached from the other side of the
    update: rng171 uses :math:`P(n-1)` and needs the scalar denominator,
    this uses :math:`P(n)` and needs nothing. Feeding :math:`P(n-1)` here is
    the obvious mistake and is undetectable from shapes -- both are
    :math:`M \times M` -- so the tests pin the two against each other.

    The previous body returned ``float(np.mean(P))`` under the key
    ``estimate`` and was green in the suite.
    """
    Pm = np.asarray(P, dtype=float)
    rv = np.asarray(r, dtype=float).ravel()
    if Pm.ndim != 2 or Pm.shape[0] != Pm.shape[1]:
        raise ValueError(f"P must be a square matrix; got shape {Pm.shape}")
    if rv.size != Pm.shape[0]:
        raise ValueError(f"r must have length {Pm.shape[0]}; got {rv.size}")
    return RichResult(
        payload={
            "array": Pm @ rv,
            "M": int(Pm.shape[0]),
            "method": "RLS gain identity k(n) = P(n) r(n) (Rangayyan Eq 3.221)",
        }
    )


def cheatsheet():
    return "rng173: k(n) = P(n) r(n) (Rangayyan Eq 3.221)."
