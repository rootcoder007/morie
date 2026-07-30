# morie.fn -- function file (rootcoder007/morie)
"""Nuclear norm -- Boyd & Vandenberghe Sec. A.1.6 / Ex. 4.28."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_nuclear_norm"]


def boyd_nuclear_norm(X, tol=None):
    r"""Nuclear norm :math:`\lVert X\rVert_* = \sum_i \sigma_i(X)`.

    The nuclear norm is to rank what the :math:`\ell_1` norm is to
    cardinality: it is the convex envelope of :math:`\operatorname{rank}`
    on the spectral-norm unit ball, which is why minimising it recovers
    low-rank matrices the way minimising :math:`\ell_1` recovers sparse
    vectors. Rank COUNTS nonzero singular values; the nuclear norm SUMS
    them, and that relaxation is tractable where the count is not.

    It is dual to the SPECTRAL norm, not to itself:

    .. math::

        \lVert X\rVert_* = \max\{\operatorname{tr}(Y^{\top}X) :
        \lVert Y\rVert_2 \le 1\},

    attained at :math:`Y = UV^{\top}` from the SVD. The three usual
    matrix norms then sit in a fixed order for every X -- spectral (the
    largest singular value) at the bottom, Frobenius (their
    root-sum-square) in the middle, nuclear (their sum) on top.

    Parameters
    ----------
    X : array-like
        Matrix, any shape.
    tol : float, optional
        Threshold below which a singular value counts as zero when
        reporting the rank. Defaults to ``max(shape) * eps * sigma_max``.

    Returns
    -------
    RichResult
        ``nuclear``, ``spectral``, ``frobenius``, ``singular_values``,
        ``rank``, ``dual_certificate`` (the maximising Y),
        ``dual_value``, ``norm_order_holds``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.
    Fazel, M. (2002). *Matrix Rank Minimization with Applications*.
        PhD thesis, Stanford University.

    Examples
    --------
    A diagonal matrix wears its singular values on its face.

    >>> import numpy as np
    >>> X = np.array([[3.0, 0.0], [0.0, 4.0]])
    >>> r = boyd_nuclear_norm(X)
    >>> [round(float(v), 6) for v in (r["nuclear"], r["frobenius"],
    ...                               r["spectral"])]
    [7.0, 5.0, 4.0]

    The ordering spectral <= Frobenius <= nuclear is not an accident of
    this example -- a sum of nonnegative numbers dominates their
    root-sum-square, which dominates the largest one.

    >>> bool(r["norm_order_holds"])
    True

    The dual characterisation is tight: tr(Y'X) with Y = UV' reproduces
    the nuclear norm exactly, and that Y is a legitimate certificate
    because its spectral norm is 1.

    >>> round(float(r["dual_value"]), 9)
    7.0
    >>> round(float(np.linalg.norm(r["dual_certificate"], 2)), 9)
    1.0

    A rank-one matrix has one term in the sum, so nuclear and Frobenius
    COINCIDE -- the gap between them is precisely what measures how
    spread out the spectrum is.

    >>> low = np.outer([1.0, 2.0], [3.0, 6.0])
    >>> lr = boyd_nuclear_norm(low)
    >>> int(lr["rank"]), bool(abs(lr["nuclear"] - lr["frobenius"]) < 1e-09)
    (1, True)
    """
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    if Xm.ndim != 2:
        raise ValueError("X must be a matrix")
    if not np.all(np.isfinite(Xm)):
        raise ValueError("X contains non-finite entries")
    U, s, Vt = np.linalg.svd(Xm, full_matrices=False)
    nuc = float(s.sum())
    spec = float(s[0]) if s.size else 0.0
    fro = float(np.sqrt(np.sum(s**2)))
    if tol is None:
        tol = max(Xm.shape) * np.finfo(float).eps * (spec if spec else 1.0)
    rank = int(np.sum(s > tol))
    # Y = U V' is the maximiser: tr(Y'X) = tr(V U' U diag(s) V') = sum s,
    # and all its singular values are 1, so it sits on the boundary of
    # the spectral-norm unit ball exactly as the dual requires.
    Y = U @ Vt
    return RichResult(
        title="Nuclear norm",
        summary_lines=[("shape", f"{Xm.shape[0]}x{Xm.shape[1]}"),
                       ("nuclear", nuc), ("spectral", spec),
                       ("frobenius", fro), ("rank", rank)],
        payload={
            "nuclear": nuc, "spectral": spec, "frobenius": fro,
            "singular_values": s, "rank": rank,
            "dual_certificate": Y,
            "dual_value": float(np.trace(Y.T @ Xm)),
            "norm_order_holds": bool(spec <= fro + 1e-09
                                     and fro <= nuc + 1e-09),
            "tol": float(tol), "method": "boyd_nuclear_norm",
        },
    )


def cheatsheet():
    return "cvxnch: convex envelope of RANK, as l1 is of cardinality; dual to the SPECTRAL norm, not itself"
