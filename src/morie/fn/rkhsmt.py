# morie.fn -- slice s04 (rootcoder007/morie)
"""Multi-trait Bayesian kernel regression with shared kernel matrix.

Book sections read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, all as rendered page images.

Volume [Pages 251-336], Chapter 8, Section 8.9 "Multi-trait Bayesian
Kernel", p. 288, is where the index sends you, and it contains no
equation at all.  Its whole specification is the sentence "In BGLR, it
is possible to fit multi-trait Bayesian kernel BLUP methods, and the
fitting process is exactly the same as fitting multi-trait Bayesian
GBLUP methods (see Chap. 6).  The only difference is that instead of
using a linear kernel, any kernel can be used."  The model is therefore
taken from where that sentence points.

Volume [Pages 171-208], Chapter 6, Section 6.8.1, equation (6.9), p. 191:

    Y = 1_J mu' + X B + Z_1 b_1 + E                             (6.9)

with E ~ MN_{J x n_T}(0, I_J, R) and b_1 ~ MN_{J x n_T}(0, G, Sigma_T),
that is vec(b_1) ~ N(0, Sigma_T (x) G).  Section 8.9 replaces G by the
kernel K.  The marginal that the function's own specification states,
vec(Y) ~ MVN(0, Sigma_g (x) K + Sigma_e (x) I), is that model's marginal;
it is NOT printed anywhere in the book, and no equation number should
ever be attached to it.

The Gibbs sampler is the six numbered steps on p. 193:

  1. beta ~ N(beta~_0, Sigma~_beta),
     Sigma~_beta = [Sigma_beta^-1 + (R^-1 (x) X'X)]^-1,
     beta~_0 = Sigma~_beta[Sigma_beta^-1 beta_0
                           + (R^-1 (x) X') vec(Y - 1_J mu' - Z_1 b_1)]
  2. mu ~ N_{n_T}(mu~, Sigma~_mu), Sigma~_mu = J^-1 R,
     mu~ = Sigma~_mu (R^-1 (x) 1_J) vec(Y - X B - Z_1 b_1)
  3. g = vec(b_1) ~ N(g~, G~),
     G~ = [(Sigma_T^-1 (x) G^-1) + (R^-1 (x) Z_1'Z_1)]^-1,
     g~ = G~ (R^-1 (x) Z_1') vec(Y - 1_J mu' - X B)
  4. Sigma_T ~ IW(v_T + J, b_1' G^-1 b_1 + S_T)
  5. R ~ IW(v_R + J, S_R + (Y - 1_J mu' - X B - Z_1 b_1)'
                          (Y - 1_J mu' - X B - Z_1 b_1))
  6. return to step 1.

BOOK ERRATUM, recorded.  Step 5 as printed writes the residual scale as
"S_T + ..." where it must be S_R; S_T is the trait genetic scale of step
4 and cannot also be the residual scale.  The same typo appears in the
Section 6.9 sampler on p. 196.  S_R is used here.

DETERMINISM.  Nothing is sampled.  Every step is taken at the exact mean
of its own full conditional -- the normal steps at their means, which are
already written above, and the two inverse-Wishart steps at
E[IW(v, S)] = S/(v - n_T - 1), the mean the book's own exponent
|Sigma|^{-(v+n_T+1)/2} implies.  Iterating those conditional means is the
EM fixed point of the same sampler, so both arms land on identical
numbers rather than on the same posterior.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["rkhs_multitrait"]


def _inv(A, ridge=1e-12):
    """Inverse of a symmetric positive definite matrix, by columns."""
    n = len(A)
    out = [[0.0] * n for _ in range(n)]
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        col = core.ridgesolve(A, e, ridge)
        for i in range(n):
            out[i][j] = col[i]
    # symmetrise, so that the two arms cannot drift on round-off
    for i in range(n):
        for j in range(i + 1, n):
            v = 0.5 * (out[i][j] + out[j][i])
            out[i][j] = v
            out[j][i] = v
    return out


def _kron(A, B):
    ra, ca, rb, cb = len(A), len(A[0]), len(B), len(B[0])
    out = [[0.0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    out[i * rb + k][j * cb + l] = A[i][j] * B[k][l]
    return out


def _vec(M):
    """Column-major vectorisation, the vec() the book uses."""
    return [M[i][j] for j in range(len(M[0])) for i in range(len(M))]


def _unvec(v, nr, nc):
    return [[v[j * nr + i] for j in range(nc)] for i in range(nr)]


def rkhs_multitrait(Y, K, n_iter=200, X=None, Z1=None, v_T=None, S_T=None,
                    v_R=None, S_R=None, tol=1e-12):
    """Multi-trait kernel BLUP of eq. (6.9) with G replaced by the kernel K.

    Parameters
    ----------
    Y : array-like
        J-by-n_T matrix of phenotypes; row j is line j, column t trait t.
    K : array-like
        J-by-J kernel (or genomic relationship) matrix, symmetric.
    n_iter : int
        Maximum number of conditional-mean sweeps.
    X : array-like, optional
        J-by-p fixed-effect design.  Absent means no fixed effects beyond
        the trait intercepts mu.
    Z1 : array-like, optional
        J-by-J incidence matrix of lines; the identity when absent, which
        is the one-record-per-line case the chapter writes.
    v_T, S_T, v_R, S_R : optional
        Inverse-Wishart hyperparameters of steps 4 and 5.  Default
        v = n_T + 2, S = I_{n_T}, the smallest degrees of freedom at which
        the inverse-Wishart mean exists.
    tol : float
        Fixed-point tolerance on the genetic effects.

    Returns
    -------
    estimate : the mean of the fitted genomic values
    gebv     : J-by-n_T matrix of genomic estimated breeding values Z_1 b_1
    b1       : J-by-n_T matrix of genetic effects
    Sigma_T  : the n_T-by-n_T trait genetic covariance
    R        : the n_T-by-n_T residual covariance
    mu       : the n_T trait intercepts
    beta     : the fixed-effect coefficients, empty when X is absent
    """
    YY = core.mat(Y)
    J = len(YY)
    if J < 2:
        raise ValueError("rkhs_multitrait: need at least two lines")
    nT = len(YY[0])
    if nT < 1:
        raise ValueError("rkhs_multitrait: need at least one trait")
    for r in YY:
        if len(r) != nT:
            raise ValueError("rkhs_multitrait: Y rows have unequal lengths")
    KK = core.mat(K)
    if len(KK) != J or len(KK[0]) != J:
        raise ValueError("rkhs_multitrait: K must be a square matrix of order J")
    for i in range(J):
        for j in range(i + 1, J):
            if abs(KK[i][j] - KK[j][i]) > 1e-12:
                raise ValueError("rkhs_multitrait: K must be symmetric")
    if X is None:
        XX = None
        p = 0
    else:
        XX = core.mat(X)
        if len(XX) != J:
            raise ValueError("rkhs_multitrait: X has a different number of rows than Y")
        p = len(XX[0])
    if Z1 is None:
        ZZ = [[1.0 if i == j else 0.0 for j in range(J)] for i in range(J)]
    else:
        ZZ = core.mat(Z1)
        if len(ZZ) != J or len(ZZ[0]) != J:
            raise ValueError("rkhs_multitrait: Z1 must be a J-by-J incidence matrix")
    vT = float(nT + 2) if v_T is None else float(v_T)
    vR = float(nT + 2) if v_R is None else float(v_R)
    if vT <= nT + 1 or vR <= nT + 1:
        raise ValueError("rkhs_multitrait: the degrees of freedom must exceed n_T + 1")
    ST = [[1.0 if i == j else 0.0 for j in range(nT)] for i in range(nT)] \
        if S_T is None else core.mat(S_T)
    SR = [[1.0 if i == j else 0.0 for j in range(nT)] for i in range(nT)] \
        if S_R is None else core.mat(S_R)
    it = int(n_iter)
    if it < 1:
        raise ValueError("rkhs_multitrait: n_iter must be at least 1")

    Kinv = _inv(KK)
    ZtZ = core.matmul([[ZZ[i][j] for j in range(J)] for i in range(J)], ZZ)
    ZtZ = [[sum(ZZ[k][i] * ZZ[k][j] for k in range(J)) for j in range(J)] for i in range(J)]
    b1 = [[0.0] * nT for _ in range(J)]
    beta = [[0.0] * nT for _ in range(p)]
    mu = [0.0] * nT
    SigT = [[1.0 if i == j else 0.0 for j in range(nT)] for i in range(nT)]
    Rm = [[1.0 if i == j else 0.0 for j in range(nT)] for i in range(nT)]

    def _resid(drop_mu=False, drop_beta=False, drop_b=False):
        out = [[YY[i][t] for t in range(nT)] for i in range(J)]
        if not drop_mu:
            for i in range(J):
                for t in range(nT):
                    out[i][t] -= mu[t]
        if not drop_beta and p:
            for i in range(J):
                for t in range(nT):
                    out[i][t] -= sum(XX[i][a] * beta[a][t] for a in range(p))
        if not drop_b:
            for i in range(J):
                for t in range(nT):
                    out[i][t] -= sum(ZZ[i][j] * b1[j][t] for j in range(J))
        return out

    for _ in range(it):
        prev = [row[:] for row in b1]
        # step 1: with the flat prior the R^-1 factors cancel and the mean of
        # beta is the ordinary least squares fit of the current residual
        if p:
            Rres = _resid(drop_beta=True)
            A = [[sum(XX[i][a] * XX[i][c] for i in range(J)) for c in range(p)] for a in range(p)]
            for t in range(nT):
                rhs = [sum(XX[i][a] * Rres[i][t] for i in range(J)) for a in range(p)]
                sol = core.ridgesolve(A, rhs, 1e-12)
                for a in range(p):
                    beta[a][t] = sol[a]
        # step 2: Sigma~_mu (R^-1 (x) 1_J') vec(.) collapses to the column mean
        Rres = _resid(drop_mu=True)
        for t in range(nT):
            mu[t] = sum(Rres[i][t] for i in range(J)) / J
        # step 3: the genetic effects, at the mean of their full conditional
        Rres = _resid(drop_b=True)
        Rinv = _inv(Rm)
        STinv = _inv(SigT)
        M = _kron(STinv, Kinv)
        Q = _kron(Rinv, ZtZ)
        for a in range(J * nT):
            for c in range(J * nT):
                M[a][c] += Q[a][c]
        # (R^-1 (x) Z_1') vec(Rres): Z_1' Rres R^-1, then vec
        ZtR = [[sum(ZZ[k][i] * Rres[k][t] for k in range(J)) for t in range(nT)] for i in range(J)]
        RHS = [[sum(ZtR[i][s] * Rinv[s][t] for s in range(nT)) for t in range(nT)] for i in range(J)]
        g = core.ridgesolve(M, _vec(RHS), 1e-12)
        b1 = _unvec(g, J, nT)
        # step 4: Sigma_T at E[IW] = S/(v - n_T - 1)
        Kb = [[sum(Kinv[i][k] * b1[k][t] for k in range(J)) for t in range(nT)] for i in range(J)]
        SS = [[sum(b1[i][s] * Kb[i][t] for i in range(J)) + ST[s][t] for t in range(nT)]
              for s in range(nT)]
        den = vT + J - nT - 1.0
        SigT = [[SS[s][t] / den for t in range(nT)] for s in range(nT)]
        # step 5: R at the same mean, with S_R and not the book's misprinted S_T
        E = _resid()
        SS = [[sum(E[i][s] * E[i][t] for i in range(J)) + SR[s][t] for t in range(nT)]
              for s in range(nT)]
        den = vR + J - nT - 1.0
        Rm = [[SS[s][t] / den for t in range(nT)] for s in range(nT)]
        d = 0.0
        for i in range(J):
            for t in range(nT):
                d = max(d, abs(b1[i][t] - prev[i][t]))
        if d < tol:
            break
    gebv = [[sum(ZZ[i][j] * b1[j][t] for j in range(J)) for t in range(nT)] for i in range(J)]
    tot = 0.0
    for i in range(J):
        for t in range(nT):
            tot += gebv[i][t]
    return RichResult(
        title="Multi-trait Bayesian kernel regression",
        summary_lines=[("lines", J), ("traits", nT)],
        payload={
            "estimate": tot / (J * nT),
            "gebv": gebv,
            "b1": b1,
            "Sigma_T": SigT,
            "R": Rm,
            "mu": mu,
            "beta": beta,
            "n": J,
            "method": "Chapter 6 eq. (6.9) with G replaced by the kernel K per Sect. 8.9, "
                      "every Gibbs step taken at its conditional mean",
        },
    )


def cheatsheet():
    return "rkhsmt: Multi-trait Bayesian kernel regression with shared kernel matrix"


# compact alias per ledger/NAMING.md
rkhsmultitrait = rkhs_multitrait
