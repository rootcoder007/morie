# morie.fn -- function file (rootcoder007/morie)
"""Dual norm -- Boyd & Vandenberghe Sec. A.1.6."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_dual_norm"]


def boyd_dual_norm(norm, z):
    r"""The dual norm :math:`\lVert z\rVert_* = \sup\{z^\top x :
    \lVert x\rVert \le 1\}`.

    For the :math:`\ell_p` family the dual is :math:`\ell_q` with
    :math:`1/p + 1/q = 1`: the dual of :math:`\ell_1` is
    :math:`\ell_\infty`, the dual of :math:`\ell_\infty` is
    :math:`\ell_1`, and :math:`\ell_2` is self-dual.

    The pairing matters in practice because it is what a norm constraint
    costs in the dual. An :math:`\ell_1` penalty -- the LASSO -- has an
    :math:`\ell_\infty` dual, which is why its optimality condition reads
    "every correlation is at most lambda in ABSOLUTE VALUE" rather than
    anything about a sum.

    Parameters
    ----------
    norm : {1, 2, "inf", "fro"} or float
        The primal norm.
    z : array-like
        Vector (or matrix, for ``"fro"``).

    Returns
    -------
    RichResult
        ``value``, ``dual_of``, ``conjugate_exponent``, ``maximizer`` (an
        x attaining the supremum).

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    The dual of the 1-norm is the sup-norm.

    >>> r = boyd_dual_norm(1, [3.0, -5.0, 2.0])
    >>> r["value"], r["dual_of"]
    (5.0, 'inf')

    The dual of the sup-norm is the 1-norm.

    >>> boyd_dual_norm("inf", [3.0, -5.0, 2.0])["value"]
    10.0

    The 2-norm is self-dual.

    >>> round(boyd_dual_norm(2, [3.0, 4.0])["value"], 10)
    5.0

    The maximizer attains the supremum: it lies in the primal unit ball
    and its inner product with z equals the dual norm.

    >>> import numpy as np
    >>> z = np.array([3.0, -5.0, 2.0])
    >>> x = boyd_dual_norm(1, z)["maximizer"]
    >>> bool(np.max(np.abs(x)) <= 1 + 1e-12 and abs(z @ x - 5.0) < 1e-12)
    True
    """
    zv = np.asarray(z, dtype=float)
    flat = zv.ravel()
    if flat.size == 0:
        raise ValueError("z must be non-empty")
    if norm in ("fro", "frobenius"):
        val = float(np.sqrt(np.sum(zv ** 2)))
        x = zv / val if val > 0 else np.zeros_like(zv)
        return RichResult(
            title="Dual norm (Frobenius)",
            summary_lines=[("value", val), ("dual of", "fro")],
            payload={"value": val, "dual_of": "fro",
                     "conjugate_exponent": 2.0, "maximizer": x,
                     "method": "boyd_dual_norm"})
    if norm in ("inf", np.inf, float("inf")):
        val = float(np.sum(np.abs(flat)))
        x = np.sign(flat)
        dual_of, q = "1", 1.0
    else:
        p = float(norm)
        if p < 1:
            raise ValueError("p must be at least 1 for a norm")
        if p == 1:
            val = float(np.max(np.abs(flat)))
            x = np.zeros_like(flat)
            # The supremum of z'x over the l1 ball is attained at a VERTEX,
            # a signed unit basis vector, not in the interior.
            x[int(np.argmax(np.abs(flat)))] = float(np.sign(
                flat[int(np.argmax(np.abs(flat)))]) or 1.0)
            dual_of, q = "inf", float("inf")
        else:
            q = p / (p - 1.0)
            val = float(np.sum(np.abs(flat) ** q) ** (1.0 / q))
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.sign(flat) * (np.abs(flat) ** (q - 1))
            nx = float(np.sum(np.abs(x) ** p) ** (1.0 / p))
            x = x / nx if nx > 0 else x
            dual_of = f"{q:g}"
    return RichResult(
        title=f"Dual norm (of l{norm})",
        summary_lines=[("value", val), ("dual of", dual_of)],
        payload={"value": val, "dual_of": dual_of,
                 "conjugate_exponent": q,
                 "maximizer": x.reshape(zv.shape) if zv.ndim > 1 else x,
                 "method": "boyd_dual_norm"},
    )


def cheatsheet():
    return "cvxdle: 1/p + 1/q = 1; l1 <-> l_inf, l2 self-dual. The LASSO's l_inf condition comes from here"
