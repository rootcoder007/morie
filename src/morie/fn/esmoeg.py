# morie.fn -- function file (rootcoder007/morie)
"""Exploratory SEM with target rotation."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["esem_target_rotation"]


def esem_target_rotation(loadings, target, iters=200, tol=1e-13):
    """
    Exploratory SEM with target rotation

    Formula: rotate the exploratory loading matrix toward a hypothesised
    pattern.  For an ORTHOGONAL rotation the problem

        min_T || Lambda T - H ||_F   subject to  T'T = I

    is the orthogonal Procrustes problem, solved in closed form by the
    polar factor of M = Lambda' H: with M'M = V S^2 V',

        T = M V S^{-1} V'

    which is exactly U V' for the singular value decomposition
    M = U S V'.  No iteration is needed for a fully specified target.

    A PARTIALLY specified target -- the usual ESEM case, where only the
    zeros of the hypothesised pattern are stated and the salient
    loadings are left free -- is handled by the standard alternating
    scheme: free positions of H are refilled from the current rotated
    solution, then the Procrustes step is repeated, until T stops moving.
    Free positions are marked with NaN (Python) or NA (R).

    Parameters
    ----------
    loadings : array-like
        p x m exploratory loading matrix.
    target : array-like
        p x m hypothesised pattern; NaN/NA marks a free element.
    iters : int
        Maximum alternations for a partially specified target.
    tol : float
        Convergence tolerance on the change in T.

    Returns
    -------
    result : dict
        Keys: estimate (RMS discrepancy on the specified elements),
        rotated (row-major), rotation (row-major), rms, n_specified,
        iters_used, n_items, n_factors, n, method.

    References
    ----------
    Asparouhov & Muthen (2009), Structural Equation Modeling
    16(3):397-438, doi:10.1080/10705510903008204.
    """
    L = core.mat(loadings)
    H0 = core.mat(target)
    p = len(L)
    if p == 0:
        raise ValueError("empty input: loadings has no rows")
    m = len(L[0])
    if len(H0) != p or any(len(r) != m for r in H0) or any(len(r) != m for r in L):
        raise ValueError("loadings and target must have the same shape")
    if m == 0:
        raise ValueError("loadings has no columns")
    spec = [[H0[i][j] == H0[i][j] for j in range(m)] for i in range(p)]
    nspec = sum(1 for i in range(p) for j in range(m) if spec[i][j])
    if nspec == 0:
        raise ValueError("target specifies no elements")

    def _procrustes(H):
        M = [[sum(L[k][a] * H[k][b] for k in range(p)) for b in range(m)]
             for a in range(m)]
        MtM = [[sum(M[k][a] * M[k][b] for k in range(m)) for b in range(m)]
               for a in range(m)]
        vals, vecs = core.jacobi(MtM)
        for v in vals:
            if v <= 1e-24:
                raise ValueError("target rotation is degenerate "
                                 "(Lambda' H is rank deficient)")
        # T = M V S^-1 V'
        MV = [[sum(M[a][k] * vecs[k][b] for k in range(m)) for b in range(m)]
              for a in range(m)]
        for a in range(m):
            for b in range(m):
                MV[a][b] /= math.sqrt(vals[b])
        return [[sum(MV[a][k] * vecs[b][k] for k in range(m)) for b in range(m)]
                for a in range(m)]

    T = [[1.0 if a == b else 0.0 for b in range(m)] for a in range(m)]
    used = 0
    for used in range(1, int(iters) + 1):
        Rot = [[sum(L[i][k] * T[k][j] for k in range(m)) for j in range(m)]
               for i in range(p)]
        H = [[H0[i][j] if spec[i][j] else Rot[i][j] for j in range(m)]
             for i in range(p)]
        Tn = _procrustes(H)
        d = max(abs(Tn[a][b] - T[a][b]) for a in range(m) for b in range(m))
        T = Tn
        if d < float(tol):
            break
    Rot = [[sum(L[i][k] * T[k][j] for k in range(m)) for j in range(m)]
           for i in range(p)]
    ss = sum((Rot[i][j] - H0[i][j]) ** 2
             for i in range(p) for j in range(m) if spec[i][j])
    rms = math.sqrt(ss / nspec)
    return RichResult(payload={
        "estimate": rms,
        "rotated": [v for r in Rot for v in r],
        "rotation": [v for r in T for v in r],
        "rms": rms,
        "n_specified": nspec,
        "iters_used": used,
        "n_items": p,
        "n_factors": m,
        "n": p,
        "method": "Exploratory SEM with target rotation",
    })


def cheatsheet():
    return "esmoeg: Exploratory SEM with target rotation"


# compact alias per ledger/NAMING.md
esemtargetrotation = esem_target_rotation
