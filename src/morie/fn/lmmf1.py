# morie.fn -- function file (rootcoder007/morie)
"""Linear mixed model in the form Y = X beta + Z u + e, with its two means and marginal variance."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['lmmform', 'lmm_form_eq2_1', 'lmmformeq21']


def lmmform(X, beta, Z, u, Sigma, R=None):
    """Linear mixed model in the form Y = X beta + Z u + e, with its two means and marginal variance.

    Formula: Y = X beta + Z u + e,  u ~ N(0, Sigma), e ~ N(0, R);  E(Y) = X beta, E(Y|u) = X beta + Z u, Var(Y) = Z Sigma Z' + R

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix of fixed effects.
    beta : array-like
        Fixed-effect coefficients, length p.
    Z : array-like, shape (n, q)
        Design matrix of random effects.
    u : array-like
        Realized random effects, length q.
    Sigma : array-like, shape (q, q)
        Variance-covariance matrix of the random effects.
    R : array-like or None
        Residual variance-covariance matrix; None uses the identity.

    Returns
    -------
    RichResult
        ``mean_marginal``, ``mean_conditional``, ``signal``, ``V``, ``n``, ``p``, ``q``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 2, Eq. (2.1) p. 37 and the paragraph beneath it: Y is n x 1, X is n x p, Z is n x q, u ~ N(0, Sigma) with Sigma of order q x q and e ~ N(0, R) with R of order n x n; the unconditional mean is E(Y) = X beta and the conditional mean given the random effects is E(Y|u) = X beta + Z u.  The marginal variance Z Sigma Z' + R is the V that Sect. 2.2 uses for the BLUE and BLUP.  Read from the chapter PDF, not recalled.
    """
    Xm = C.mat(X); Zm = C.mat(Z)
    b = C.vec(beta); uu = C.vec(u)
    n, p, q = len(Xm), len(Xm[0]), len(Zm[0])
    if len(Zm) != n:
        raise ValueError("X and Z must have the same number of rows")
    if len(b) != p or len(uu) != q:
        raise ValueError("beta and u must match the columns of X and Z")
    S = C.mat(Sigma)
    if len(S) != q or len(S[0]) != q:
        raise ValueError("Sigma must be q by q")
    Rm = C.eye(n) if R is None else C.mat(R)
    if len(Rm) != n or len(Rm[0]) != n:
        raise ValueError("R must be n by n")
    xb = C.matvec(Xm, b)
    zu = C.matvec(Zm, uu)
    ZS = C.matmul(Zm, S)
    V = C.matmul(ZS, C.transpose(Zm))
    V = [[V[i][j] + Rm[i][j] for j in range(n)] for i in range(n)]
    return RichResult(payload={
        "mean_marginal": xb, "mean_conditional": [a + c for a, c in zip(xb, zu)],
        "signal": zu, "V": V, "n": n, "p": p, "q": q,
        "method": "Linear mixed model, MVSML Eq. (2.1)"})


lmm_form_eq2_1 = lmmform
lmmformeq21 = lmmform


def cheatsheet():
    return 'lmmf1: Linear mixed model in the form Y = X beta + Z u + e, with its two means and marginal variance.'
