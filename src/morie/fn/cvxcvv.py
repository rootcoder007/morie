# morie.fn -- function file (rootcoder007/morie)
"""Schur complement -- Boyd & Vandenberghe Sec. A.5.5."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_cvxlin_complement"]


def boyd_cvxlin_complement(A, B, C, tol=1e-09):
    r"""The Schur complement condition

    .. math::
        \begin{bmatrix} A & B \\ B^\top & C\end{bmatrix} \succeq 0
        \iff A \succeq 0,\; C - B^\top A^{+} B \succeq 0,\;
        (I - AA^{+})B = 0.

    This is the workhorse that turns quadratic conditions into LINEAR
    matrix inequalities -- it is how a quadratic constraint becomes an
    SDP, and therefore how a large class of apparently nonconvex problems
    turn out to be convex after all.

    The third condition is the one usually dropped. It is vacuous when A
    is invertible, so textbook statements often omit it, but with a
    SINGULAR A it is exactly what rules out the cases where the
    pseudo-inverse formula silently lies. This function checks all three
    and says which failed.

    Parameters
    ----------
    A, B, C : array-like
        Blocks; A and C symmetric.
    tol : float
        Eigenvalue tolerance.

    Returns
    -------
    RichResult
        ``psd``, ``A_psd``, ``schur_psd``, ``range_condition``,
        ``schur_complement``, ``min_eigenvalue``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A positive definite block matrix passes all three conditions.

    >>> import numpy as np
    >>> r = boyd_cvxlin_complement([[2.0, 0.0], [0.0, 2.0]],
    ...                            [[1.0], [0.0]], [[1.0]])
    >>> bool(r["psd"])
    True

    The Schur complement is what carries the information: here
    C - B'A^-1 B = 1 - 0.5 = 0.5.

    >>> round(float(r["schur_complement"][0, 0]), 6)
    0.5

    Shrinking C past that value breaks positive semi-definiteness, even
    though A and C are both individually positive.

    >>> bool(boyd_cvxlin_complement([[2.0, 0.0], [0.0, 2.0]],
    ...                             [[1.0], [0.0]], [[0.4]])["psd"])
    False

    With a SINGULAR A the range condition does the work the
    pseudo-inverse formula cannot: B must lie in the range of A.

    >>> s = boyd_cvxlin_complement([[1.0, 0.0], [0.0, 0.0]],
    ...                            [[0.0], [1.0]], [[1.0]])
    >>> bool(s["range_condition"]), bool(s["psd"])
    (False, False)
    """
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    Bm = np.atleast_2d(np.asarray(B, dtype=float))
    Cm = np.atleast_2d(np.asarray(C, dtype=float))
    n = Am.shape[0]
    m = Cm.shape[0]
    if Am.shape != (n, n) or Cm.shape != (m, m):
        raise ValueError("A and C must be square")
    if Bm.shape != (n, m):
        raise ValueError(f"B must be ({n}, {m}) to match A and C")
    Am = 0.5 * (Am + Am.T)
    Cm = 0.5 * (Cm + Cm.T)
    wA = np.linalg.eigvalsh(Am)
    A_psd = bool(wA.min() >= -tol)
    Ap = np.linalg.pinv(Am)
    schur = Cm - Bm.T @ Ap @ Bm
    schur = 0.5 * (schur + schur.T)
    wS = np.linalg.eigvalsh(schur)
    S_psd = bool(wS.min() >= -tol)
    # (I - A A^+) B = 0: B must lie in the range of A. Vacuous when A is
    # invertible, decisive when it is not.
    rng_ok = bool(np.max(np.abs((np.eye(n) - Am @ Ap) @ Bm)) <= 1e-08)
    full = np.block([[Am, Bm], [Bm.T, Cm]])
    wF = np.linalg.eigvalsh(0.5 * (full + full.T))
    return RichResult(
        title="Schur complement",
        summary_lines=[("A psd", A_psd), ("Schur psd", S_psd),
                       ("range condition", rng_ok),
                       ("min eigenvalue", float(wF.min()))],
        payload={
            "psd": bool(A_psd and S_psd and rng_ok), "A_psd": A_psd,
            "schur_psd": S_psd, "range_condition": rng_ok,
            "schur_complement": schur, "min_eigenvalue": float(wF.min()),
            "eigenvalues": wF, "method": "boyd_cvxlin_complement",
        },
    )


def cheatsheet():
    return "cvxcvv: turns quadratic conditions into LMIs; the range condition is only vacuous when A is invertible"
