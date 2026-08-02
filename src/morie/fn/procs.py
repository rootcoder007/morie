# morie.fn -- function file (rootcoder007/morie)
"""Orthogonal Procrustes rotation to align two configurations."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["procrustes_rotation"]


def procrustes_rotation(A, Z):
    r"""Schoenemann's closed-form orthogonal Procrustes solution.

    .. math:: \min_T \|A - Z T\|_F^2 \quad \text{s.t.}\quad T'T = I,
              \qquad T = U V' \text{ from } \mathrm{SVD}(Z'A) = U S V'.

    MDS solutions are only identified up to rotation and reflection,
    so comparing two configurations requires aligning one to the
    other first; the residual after the optimal rotation is the real
    disagreement.

    Parameters
    ----------
    A : array-like, shape (n, k)
        Target configuration.
    Z : array-like, shape (n, k)
        Configuration to rotate.

    Returns
    -------
    RichResult
        keys: ``rotation`` (k, k orthogonal), ``rotated`` (Z T),
        ``residual`` (||A - ZT||_F), ``residual_before``
        (||A - Z||_F), ``reflection`` (det T < 0), ``n``, ``method``.

    References
    ----------
    Schoenemann, P. H. (1966). A generalized solution of the
    orthogonal Procrustes problem. *Psychometrika*, 31(1), 1-10.
    """
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if A.shape != Z.shape or A.ndim != 2:
        raise ValueError("A and Z must be 2-D arrays of the same shape.")

    U, _, Vt = np.linalg.svd(Z.T @ A)
    T = U @ Vt
    ZT = Z @ T

    return RichResult(
        payload={
            "rotation": T,
            "rotated": ZT,
            "residual": float(np.linalg.norm(A - ZT)),
            "residual_before": float(np.linalg.norm(A - Z)),
            "reflection": bool(np.linalg.det(T) < 0),
            "n": int(A.shape[0]),
            "method": "Orthogonal Procrustes rotation (Schoenemann 1966)",
        }
    )


def cheatsheet():
    return "procs: T = UV' from SVD(Z'A); ZT is Z aligned to A"
