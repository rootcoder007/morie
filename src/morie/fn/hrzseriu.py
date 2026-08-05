# morie.fn -- function file (rootcoder007/morie)
"""Series truncation for NPIV when T is unknown."""

from . import _array_core as np
from . import _hrz3 as H

from ._richresult import RichResult

__all__ = ["horowitz_series_unknown_T"]


def horowitz_series_unknown_T(x, y, w, K=4, basis="poly"):
    r"""Nonparametric IV by series truncation when T is unknown.

    Horowitz (2009), Section 5.4.2, pages 178-181, following Blundell,
    Chen and Kristensen (2007).  Section 5.4.1 estimates the operator
    and then inverts it; this section avoids estimating eigenfunctions
    at all, because "consistent estimation of eigenfunctions requires
    assumptions about the spacing of eigenvalues that may be
    undesirable in applications" (p. 178).  The basis
    :math:`\{\psi_j\}` is KNOWN, and only coefficients are estimated.

    With :math:`\rho(w,h) = E[Y - h(X)|W = w]`, the model implies
    :math:`\rho(w,g) = 0`, so :math:`g` minimises
    :math:`E[\rho(W,h)]^2` (5.80).  Writing the conditional-mean
    operator as :math:`m = Ag` (5.84) and expanding both sides in the
    basis gives the finite linear system

    .. math:: m_k = \sum_{j=1}^{J} b_j q_{jk},\qquad k = 1,\dots,J
                                                              \quad (5.85)

    where :math:`\hat m = (\Psi'\Psi)^{-}\Psi'Y` and
    :math:`\hat q_{jk}` is the :math:`k`-th coefficient from the series
    regression of :math:`\psi_j(X)` on :math:`W`.  The estimator is
    :math:`\hat g = \sum_j \hat\beta_j\psi_j` (5.83).

    The Moore-Penrose generalised inverse is used exactly as the text
    specifies, so a rank-deficient :math:`\Psi'\Psi` degrades rather
    than raising.

    Printing error corrected here: (5.85) as printed on p. 181 carries
    the summation index ``k = 1`` beneath a summand ``b_j q_jk``, which
    would make the left side independent of :math:`k`.  It must be
    ``j = 1``, matching the display immediately above it.

    Parameters
    ----------
    x : array-like, shape (n,)
        Endogenous regressor.
    y : array-like, shape (n,)
        Response.
    w : array-like, shape (n,)
        Instrument.
    K : int, default 4
        Series length :math:`J`.  The text uses the same basis and
        length for :math:`g` and for :math:`\rho`, and so does this.
    basis : {"poly", "cos"}, default "poly"
        Basis family on the mid-rank [0, 1] scale.

    Returns
    -------
    RichResult
        keys: ``g_hat`` (at the observed X, in input order), ``beta``,
        ``m_hat``, ``Q``, ``J``, ``basis``, ``n``, ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 5.4.2, eqs. (5.80)-(5.85),
    pp. 178-181.
    Blundell, R., Chen, X. & Kristensen, D. (2007). Semi-nonparametric
    IV estimation of shape-invariant Engel curves.
    *Econometrica* 75(6), 1613-1669.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = int(x.size)
    if y.size != n or w.size != n:
        raise ValueError(
            f"x, y, w must have the same length; got {n}, {y.size}, {w.size}.")
    J = int(K)
    if J < 1:
        raise ValueError(f"K must be at least 1, got {J}.")
    if J > n:
        raise ValueError(f"K must not exceed n; got K={J}, n={n}.")

    u = H.u01(x)
    v = H.u01(w)
    Psi = H.sieve(v, J, basis)
    Phi = H.sieve(u, J, basis)

    PtP = np.asarray(Psi, dtype=float).T @ np.asarray(Psi, dtype=float)
    Pinv = np.linalg.pinv(PtP)

    m_hat = Pinv @ (np.asarray(Psi, dtype=float).T
                    @ np.asarray(y, dtype=float))
    # C[k][j] = k-th coefficient of the regression of psi_j(X) on W,
    # i.e. C[k][j] = q_hat_{jk}.
    C = Pinv @ (np.asarray(Psi, dtype=float).T @ np.asarray(Phi, dtype=float))

    # (5.85): m_hat = C beta.
    beta = np.linalg.lstsq(np.asarray(C, dtype=float),
                           np.asarray(m_hat, dtype=float), rcond=None)[0]
    g_hat = np.asarray(Phi, dtype=float) @ np.asarray(beta, dtype=float)

    return RichResult(payload={
        "g_hat": [float(t) for t in g_hat],
        "beta": [float(t) for t in beta],
        "m_hat": [float(t) for t in m_hat],
        "Q": [[float(C[k][j]) for j in range(J)] for k in range(J)],
        "J": J,
        "basis": str(basis),
        "n": n,
        "method": "Horowitz (2009) eqs. (5.83)-(5.85), series truncation",
    })


def cheatsheet():
    return "hrzseriu: known basis, estimated coefficients -- no eigenfunctions needed"
