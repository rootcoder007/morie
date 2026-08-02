# morie.fn -- function file (rootcoder007/morie)
"""Least squares via QR decomposition."""

from __future__ import annotations

from . import _array_core as np

from ._containers import DescriptiveResult


def lstsq_qr(
    A: np.ndarray,
    b: np.ndarray,
) -> DescriptiveResult:
    """Least squares solution via QR decomposition.

    Solves min ||Ax - b||_2 by factoring A = QR and solving Rx = Q^T b.

    Parameters
    ----------
    A : ndarray
        Design matrix (m x n), m >= n.
    b : ndarray
        Observation vector of length m.

    Returns
    -------
    DescriptiveResult
        ``value`` is the residual norm; ``extra`` has x (coefficients).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    Q, R = np.linalg.qr(A, mode="reduced")
    # A rank-deficient A leaves a diagonal entry of R at rounding scale
    # rather than exactly zero, so back-substitution "succeeds" and
    # blows up.  Test the rank of R and fall back to the minimum-norm
    # least-squares solution (Golub & Van Loan 2013, Matrix
    # Computations 4th ed., sec. 5.5).
    Rd = R.tolist()
    diag = [abs(Rd[i][i]) for i in range(min(len(Rd), len(Rd[0])))]
    dmax = max(diag) if diag else 0.0
    if diag and min(diag) <= dmax * max(m, n) * 2.220446049250313e-16:
        x = np.linalg.lstsq(A, b)[0]
    else:
        x = np.linalg.solve(R, Q.T @ b)
    residual = float(np.linalg.norm(A @ x - b))
    return DescriptiveResult(
        name="Least Squares (QR)",
        value=residual,
        extra={"x": x, "Q": Q, "R": R},
    )


lstqr = lstsq_qr
