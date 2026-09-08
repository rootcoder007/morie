# morie.fn -- function file (rootcoder007/morie)
"""Standardized Mantel statistic under the Gaussian assumption."""

from math import fsum, sqrt

from ._richresult import RichResult
from ._spx import eye, matmul, mean, sqmat, trace, twosidep, vec

__all__ = [
    "schabenberger_mantel_standard",
    "mantelz",
]


def schabenberger_mantel_standard(coords, x, w, u=None):
    """z_M for Mantel's M2, Schabenberger & Gotway Sec. 1.3.1.

    Section 1.3.1 lists four ways to test M2 and names this one the
    "Asymptotic Z-Test With Gaussian Assumption":

        Zobs = (M2(obs) - Eg[M2]) / sqrt(Varg[M2]),

    compared against G(0,1), with the spatial weights W held fixed. The
    book states the route but does NOT print Eg[M2] or Varg[M2], so those
    two moments are DERIVED here rather than quoted, and the derivation is
    stated in full so it can be checked:

    Take the attribute proximity of eq (1.10), ``U_ij = (Z_i - Zbar)(Z_j -
    Zbar)``. Then M2 = d'Wd with d = MZ, M = I - 11'/n. Only the symmetric
    part of W contributes, so put A = (W + W')/2. Under H0 the Z(s_i) are
    iid G(mu, sigma^2), hence d ~ G(0, sigma^2 M) and, for a quadratic form
    in a Gaussian vector with symmetric A,

        E[d'Ad]   = sigma^2 tr(A M)
        Var[d'Ad] = 2 sigma^4 tr(A M A M).

    sigma^2 is estimated by S^2 = sum_i (Z_i - Zbar)^2 / (n-1).

    The moments are consistent with the book elsewhere: substituting
    A = W and the residual projector for M reproduces exactly the mean the
    book prints for Moran's I on OLS residuals in Sec. 1.3.2,
    ``Eg[Ires] = n tr[MW] / {(n-k) w..}``, which is the cross-check that
    the derivation is the right one.

    A supplied `u` is used verbatim; then M2 = sum_ij W_ij U_ij is still
    returned but ``z`` and ``p_value`` are None, because the Gaussian
    moments above hold only for the eq (1.10) form of U.

    Parameters
    ----------
    coords : (n, d) array-like
        Unused when `w` is supplied; kept for signature stability.
    x : (n,) array-like
        Attribute values.
    w : (n, n) array-like
        Spatial proximity weights, zero diagonal.
    u : (n, n) array-like, optional
        Explicit attribute proximity matrix; disables the Z-test.

    Returns
    -------
    RichResult
        ``m2``, ``expectation``, ``variance``, ``z``, ``p_value``,
        ``sigma2``, ``n``, ``method``.
    """
    z = vec(x, "x")
    n = len(z)
    if n < 3:
        raise ValueError("at least 3 sites are needed")
    ww = sqmat(w, n, "w")
    for i in range(n):
        if ww[i][i] != 0.0:
            raise ValueError("`w` must have a zero diagonal (W_ii = 0)")
    m = mean(z)
    d = [t - m for t in z]
    ss = fsum([t * t for t in d])
    if ss <= 0:
        raise ValueError("`x` is constant; the Mantel statistic is degenerate")

    if u is not None:
        uu = sqmat(u, n, "u")
        m2 = fsum([ww[i][j] * uu[i][j] for i in range(n) for j in range(n)])
        return RichResult(payload={
            "m2": m2, "expectation": None, "variance": None,
            "z": None, "p_value": None, "sigma2": ss / (n - 1.0), "n": n,
            "gaussian_moments_apply": False,
            "method": ("Mantel M2 with a user-supplied U; the Gaussian "
                       "Z-test of Schabenberger & Gotway Sec. 1.3.1 needs "
                       "U of eq (1.10) and is not reported"),
        })

    m2 = fsum([ww[i][j] * d[i] * d[j] for i in range(n) for j in range(n)])
    a = [[0.5 * (ww[i][j] + ww[j][i]) for j in range(n)] for i in range(n)]
    proj = eye(n)
    for i in range(n):
        for j in range(n):
            proj[i][j] = proj[i][j] - 1.0 / n
    am = matmul(a, proj)
    s2 = ss / (n - 1.0)
    ex = s2 * trace(am)
    var = 2.0 * s2 * s2 * trace(matmul(am, am))
    if var <= 0:
        raise ValueError("the null variance of M2 is not positive; "
                         "the weight matrix carries no information")
    zz = (m2 - ex) / sqrt(var)

    return RichResult(payload={
        "m2": m2,
        "expectation": ex,
        "variance": var,
        "z": zz,
        "p_value": twosidep(zz),
        "sigma2": s2,
        "n": n,
        "gaussian_moments_apply": True,
        "method": ("Standardized Mantel z_M, Gaussian Z-test of "
                   "Schabenberger & Gotway (2005) Sec. 1.3.1 with U of "
                   "eq (1.10); the moments are derived, the book states "
                   "only the approach"),
    })


def cheatsheet():
    return "spmani: standardized Mantel z_M under Gaussianity"


# compact alias per ledger/NAMING.md
mantelz = schabenberger_mantel_standard
