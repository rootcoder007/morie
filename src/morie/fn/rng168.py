# morie.fn -- function file (rootcoder007/morie)
"""RLS recursion for the cross-correlation vector (Rangayyan Eq 3.212)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_theta_recursion"]


def rangayyan_ch3_rls_theta_recursion(Theta, r, x, lam):
    r"""One RLS update of the time-averaged cross-correlation vector.

    .. math::

        \Theta(n) = \lambda\,\Theta(n-1) + \mathbf{r}(n)\,x(n)

    Parameters
    ----------
    Theta : array-like, shape (M,)
        Previous vector :math:`\Theta(n-1)`.
    r : array-like, shape (M,)
        Reference input vector :math:`\mathbf{r}(n)`.
    x : float
        Primary input sample :math:`x(n)` -- a **scalar**, not a signal.
    lam : float
        Forgetting factor :math:`\lambda`, with :math:`0 < \lambda \le 1`.

    Returns
    -------
    RichResult
        keys: ``array`` (:math:`\Theta(n)`), ``M``, ``lam``, ``method``.

    Raises
    ------
    ValueError
        On shape mismatch, non-scalar ``x``, or ``lam`` outside
        :math:`(0, 1]`.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.212), p. 187, the recursive form of the definition
        Eq. (3.209) :math:`\Theta(n) = \sum_{i=1}^{n}\lambda^{n-i}\mathbf{r}(i)x(i)`.

    Notes
    -----
    In the adaptive-noise-cancelling arrangement of Figure 3.94, :math:`x(n)`
    is the *primary* input and :math:`\mathbf{r}(n)` the *reference*; getting
    the two the wrong way round is the usual error here and is not detectable
    from shapes, since only ``r`` is a vector.
    """
    th = np.asarray(Theta, dtype=float).ravel()
    rv = np.asarray(r, dtype=float).ravel()
    xs = np.asarray(x, dtype=float)
    lam = float(lam)
    if rv.size != th.size:
        raise ValueError(f"r must have the same length as Theta ({th.size}); got {rv.size}")
    if xs.ndim != 0:
        raise ValueError(
            f"x must be a scalar sample x(n); got shape {xs.shape}. Eq. (3.212) "
            "updates one time step."
        )
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must satisfy 0 < lam <= 1; got {lam!r} (Rangayyan p. 186)")
    Theta_n = lam * th + rv * float(xs)
    return RichResult(
        payload={
            "array": Theta_n,
            "M": int(th.size),
            "lam": lam,
            "method": "RLS cross-correlation-vector recursion (Rangayyan Eq 3.212)",
        }
    )


def cheatsheet():
    return "rng168: Theta(n) = lam*Theta(n-1) + r(n) x(n) (Rangayyan Eq 3.212)."
