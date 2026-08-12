# morie.fn -- function file (rootcoder007/morie)
"""Pearl front-door adjustment from tabulated distributions."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_frontdoor_adjustment"]


def causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X):
    r"""Front-door formula evaluated on probability tables.

    .. math:: P(y \mid do(x)) = \sum_z P(z \mid x)
              \sum_{x'} P(y \mid x', z)\, P(x')

    (Pearl, Theorem 3.3.4). Unlike
    :func:`morie.fn.fdadj.frontdoor_adjustment`, which estimates the
    tables from data, this takes them directly -- for worked textbook
    examples and for propagating externally supplied estimates.

    Parameters
    ----------
    P_Z_X : array-like, shape (nx, nz)
        ``P_Z_X[i, k] = P(Z = k | X = i)``; rows sum to 1.
    P_Y_XZ : array-like, shape (nx, nz, ny)
        ``P_Y_XZ[i, k, j] = P(Y = j | X = i, Z = k)``; last axis sums
        to 1.
    P_X : array-like, shape (nx,)
        Marginal of X; sums to 1.

    Returns
    -------
    RichResult
        keys: ``p_y_do_x`` (nx, ny), ``expected`` (nx,) -- E[Y|do(x)]
        treating the Y levels as 0..ny-1, ``ate`` (last minus first x
        level), ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Theorem 3.3.4 (front-door adjustment formula).
    """
    Pzx = np.asarray(P_Z_X, dtype=float)
    Pyxz = np.asarray(P_Y_XZ, dtype=float)
    Px = np.asarray(P_X, dtype=float).ravel()
    if Pzx.ndim != 2 or Pyxz.ndim != 3:
        raise ValueError("P_Z_X must be 2-D (nx, nz) and P_Y_XZ 3-D (nx, nz, ny).")
    nx, nz = Pzx.shape
    if Pyxz.shape[:2] != (nx, nz) or Px.size != nx:
        raise ValueError("table shapes disagree on nx or nz.")
    for arr, axis, name in ((Pzx, 1, "P_Z_X"), (Pyxz, 2, "P_Y_XZ")):
        if not np.allclose(arr.sum(axis=axis), 1.0, atol=1e-8):
            raise ValueError(f"{name} must be a conditional distribution (rows sum to 1).")
    if not np.isclose(Px.sum(), 1.0, atol=1e-8):
        raise ValueError("P_X must sum to 1.")

    # The two sums of Theorem 3.3.4, written out. They used to be
    # np.tensordot(Px, Pyxz, axes=(0, 0)) and a matrix product, but the
    # native array core has no tensordot, so this function raised
    # AttributeError for every input it was ever given.
    ny = len(Pyxz[0][0])
    #   inner[z][y] = sum_x' P(y | x', z) P(x')
    inner = [[sum(float(Px[xp]) * float(Pyxz[xp][z][y])
                  for xp in range(nx))
              for y in range(ny)] for z in range(nz)]
    #   P(y | do(x)) = sum_z P(z | x) inner[z][y]
    p_do = [[sum(float(Pzx[x][z]) * inner[z][y] for z in range(nz))
             for y in range(ny)] for x in range(nx)]
    exp = [sum(y * p_do[x][y] for y in range(ny)) for x in range(nx)]
    return RichResult(
        payload={
            "p_y_do_x": p_do,
            "expected": exp,
            "ate": float(exp[-1] - exp[0]),
            "method": "Front-door adjustment from tables (Pearl Thm 3.3.4)",
        }
    )


def cheatsheet():
    return "causftbl: P(y|do(x)) = sum_z P(z|x) sum_x' P(y|x',z) P(x')"
