# morie.fn -- slice s04 (rootcoder007/morie)
"""Bayesian multi-trait multi-environment model (BMTME).

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 171-208], Chapter 6, Section 6.9
"Bayesian Genomic Multi-trait and Multi-environment Model (BMTME)",
pp. 195-197, read as rendered page images.  The chapter attributes the
model to Montesinos-Lopez et al. (2016), *G3* 6, "A genomic Bayesian
multi-trait and multi-environment model".

Equation (6.11), p. 195:

    Y = 1_IJ mu' + X B + Z_1 b_1 + Z_2 b_2 + E                  (6.11)

with, on the same page,

    b_2 | Sigma_T, Sigma_E ~ MN_{IJ x n_T}(0, Sigma_E (x) G, Sigma_T)
    b_1 | Sigma_T          ~ MN_{J x n_T}(0, G, Sigma_T)
    E                      ~ MN_{IJ x n_T}(0, I_IJ, R)

I environments, J lines, n_T traits; Z_1 the incidence matrix of lines,
Z_2 the incidence matrix of the environment-by-line interaction, G the
genomic relationship matrix, Sigma_T the trait genetic covariance,
Sigma_E the environment covariance, R the residual covariance.

The Gibbs sampler is the eight numbered steps on p. 196.  Steps 3 and 4
are the ones that carry the structure:

    G~   = [(Sigma_T^-1 (x) G^-1) + (R^-1 (x) Z_1'Z_1)]^-1
    g~   = G~ (R^-1 (x) Z_1') vec(Y - 1_IJ mu' - X B - Z_2 b_2)
    G~_2 = [(Sigma_T^-1 (x) Sigma_E^-1 (x) G^-1) + (R^-1 (x) Z_2'Z_2)]^-1
    g~_2 = G~_2 (R^-1 (x) Z_2') vec(Y - 1_IJ mu' - X B - Z_1 b_1)

    Sigma_T ~ IW(v_T + J + IJ,
                 b_1'G^-1 b_1 + b_2'(Sigma_E^-1 (x) G^-1) b_2 + S_T)
    Sigma_E ~ IW(v_E + J L, b_2*'(G^-1 (x) Sigma_T^-1) b_2* + S_E)
    R       ~ IW(v_R + IJ, S_R + (Y - ...)'(Y - ...))

where b_2* is the J n_T by I matrix with vec(b_2') = vec(b_2*).  This
requires the IJ rows of the data to run environment-major, all J lines of
environment 1 and then all J lines of environment 2; the function checks
nothing about the row order because it cannot, and the docstring states
it instead.

BOOK ERRATA, all four recorded.  (1) The symbol L in the Sigma_E step is
never defined anywhere in Section 6.9.  It must be n_T: b_2* is J n_T by
I, so the exponent |Sigma_E|^{-JL/2} forces L = n_T, and step 6's
v_E + J L is then v_E + J n_T.  (2) Step 7 writes the residual scale as
"S_T + ..." where it must be S_R; the same typo appears in the Section
6.8 sampler on p. 193.  (3) Steps 3 and 4 both label the dimension N_J;
the correct dimensions are J n_T and I J n_T.  (4) Step 7 and the p. 193
analogue write X B and X beta inside the same quadratic form; they mean
X B.  The corrected readings are used here.

DETERMINISM.  Nothing is sampled.  Every step is taken at the exact mean
of its own full conditional, the three inverse-Wishart steps at
E[IW(v, S)] = S/(v - d - 1) with d the order of the matrix, the mean the
book's own exponents imply.  Iterating those conditional means is the EM
fixed point of the same sampler, so both arms land on identical numbers
rather than on the same posterior.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult
from .rkhsmt import _inv, _kron, _unvec, _vec

__all__ = ["bmtme_model"]


def bmtme_model(Y, G, n_env, n_iter=200, X=None, v_T=None, S_T=None, v_E=None,
                S_E=None, v_R=None, S_R=None, tol=1e-12):
    """BMTME of eq. (6.11), every Gibbs step at its conditional mean.

    Parameters
    ----------
    Y : array-like
        (I*J)-by-n_T matrix of phenotypes, rows running ENVIRONMENT-MAJOR:
        all J lines of environment 1, then all J lines of environment 2,
        and so on.  That row order is what makes b_2* of the Sigma_E step
        well defined.
    G : array-like
        J-by-J genomic relationship matrix, symmetric.
    n_env : int
        I, the number of environments.
    n_iter : int
        Maximum number of conditional-mean sweeps.
    X : array-like, optional
        (I*J)-by-p fixed-effect design, the environment design in the
        chapter's own R call.
    v_T, S_T, v_E, S_E, v_R, S_R : optional
        Inverse-Wishart hyperparameters of steps 5, 6 and 7.
    tol : float
        Fixed-point tolerance.

    Returns
    -------
    estimate : the mean of the fitted genomic values
    gebv     : (I*J)-by-n_T matrix, Z_1 b_1 + Z_2 b_2
    b1, b2   : the line and the environment-by-line effects
    sigma_g  : Sigma_T, the trait genetic covariance
    Sigma_T, Sigma_E, R : the three estimated covariance matrices
    mu       : the n_T trait intercepts
    """
    YY = core.mat(Y)
    N = len(YY)
    if N < 2:
        raise ValueError("bmtme_model: need at least two observations")
    nT = len(YY[0])
    if nT < 1:
        raise ValueError("bmtme_model: need at least one trait")
    for r in YY:
        if len(r) != nT:
            raise ValueError("bmtme_model: Y rows have unequal lengths")
    GG = core.mat(G)
    J = len(GG)
    if len(GG[0]) != J:
        raise ValueError("bmtme_model: G must be square")
    for i in range(J):
        for j in range(i + 1, J):
            if abs(GG[i][j] - GG[j][i]) > 1e-12:
                raise ValueError("bmtme_model: G must be symmetric")
    I = int(n_env)
    if I < 1:
        raise ValueError("bmtme_model: n_env must be at least 1")
    if I * J != N:
        raise ValueError("bmtme_model: Y must have n_env * nrow(G) rows")
    if X is None:
        XX = None
        p = 0
    else:
        XX = core.mat(X)
        if len(XX) != N:
            raise ValueError("bmtme_model: X has a different number of rows than Y")
        p = len(XX[0])
    vT = float(nT + 2) if v_T is None else float(v_T)
    vE = float(I + 2) if v_E is None else float(v_E)
    vR = float(nT + 2) if v_R is None else float(v_R)
    if vT <= nT + 1 or vR <= nT + 1:
        raise ValueError("bmtme_model: v_T and v_R must exceed n_T + 1")
    if vE <= I + 1:
        raise ValueError("bmtme_model: v_E must exceed n_env + 1")
    eye = lambda k: [[1.0 if a == b else 0.0 for b in range(k)] for a in range(k)]
    ST = eye(nT) if S_T is None else core.mat(S_T)
    SE = eye(I) if S_E is None else core.mat(S_E)
    SR = eye(nT) if S_R is None else core.mat(S_R)
    it = int(n_iter)
    if it < 1:
        raise ValueError("bmtme_model: n_iter must be at least 1")

    # Z_1 maps the J line effects onto the I*J environment-major rows;
    # Z_2 is the identity on the I*J interaction effects.
    Z1 = [[1.0 if (i % J) == j else 0.0 for j in range(J)] for i in range(N)]
    Ginv = _inv(GG)
    Z1tZ1 = [[sum(Z1[k][i] * Z1[k][j] for k in range(N)) for j in range(J)] for i in range(J)]
    Z2tZ2 = eye(N)
    b1 = [[0.0] * nT for _ in range(J)]
    b2 = [[0.0] * nT for _ in range(N)]
    beta = [[0.0] * nT for _ in range(p)]
    mu = [0.0] * nT
    SigT = eye(nT)
    SigE = eye(I)
    Rm = eye(nT)

    def resid(drop_mu=False, drop_beta=False, drop_b1=False, drop_b2=False):
        out = [[YY[i][t] for t in range(nT)] for i in range(N)]
        if not drop_mu:
            for i in range(N):
                for t in range(nT):
                    out[i][t] -= mu[t]
        if not drop_beta and p:
            for i in range(N):
                for t in range(nT):
                    out[i][t] -= sum(XX[i][a] * beta[a][t] for a in range(p))
        if not drop_b1:
            for i in range(N):
                for t in range(nT):
                    out[i][t] -= b1[i % J][t]
        if not drop_b2:
            for i in range(N):
                for t in range(nT):
                    out[i][t] -= b2[i][t]
        return out

    for _ in range(it):
        prev = [row[:] for row in b2]
        if p:
            Rr = resid(drop_beta=True)
            A = [[sum(XX[i][a] * XX[i][c] for i in range(N)) for c in range(p)] for a in range(p)]
            for t in range(nT):
                rhs = [sum(XX[i][a] * Rr[i][t] for i in range(N)) for a in range(p)]
                sol = core.ridgesolve(A, rhs, 1e-12)
                for a in range(p):
                    beta[a][t] = sol[a]
        Rr = resid(drop_mu=True)
        for t in range(nT):
            mu[t] = sum(Rr[i][t] for i in range(N)) / N
        Rinv = _inv(Rm)
        STinv = _inv(SigT)
        SEinv = _inv(SigE)
        # step 3: the line effects b_1
        Rr = resid(drop_b1=True)
        M = _kron(STinv, Ginv)
        Q = _kron(Rinv, Z1tZ1)
        for a in range(J * nT):
            for c in range(J * nT):
                M[a][c] += Q[a][c]
        ZtR = [[sum(Z1[k][i] * Rr[k][t] for k in range(N)) for t in range(nT)] for i in range(J)]
        RHS = [[sum(ZtR[i][s] * Rinv[s][t] for s in range(nT)) for t in range(nT)] for i in range(J)]
        b1 = _unvec(core.ridgesolve(M, _vec(RHS), 1e-12), J, nT)
        # step 4: the environment-by-line effects b_2
        Rr = resid(drop_b2=True)
        M = _kron(STinv, _kron(SEinv, Ginv))
        Q = _kron(Rinv, Z2tZ2)
        for a in range(N * nT):
            for c in range(N * nT):
                M[a][c] += Q[a][c]
        RHS = [[sum(Rr[i][s] * Rinv[s][t] for s in range(nT)) for t in range(nT)] for i in range(N)]
        b2 = _unvec(core.ridgesolve(M, _vec(RHS), 1e-12), N, nT)
        # step 5: Sigma_T
        Gb = [[sum(Ginv[i][k] * b1[k][t] for k in range(J)) for t in range(nT)] for i in range(J)]
        SEG = _kron(SEinv, Ginv)
        Wb = [[sum(SEG[i][k] * b2[k][t] for k in range(N)) for t in range(nT)] for i in range(N)]
        SS = [[sum(b1[i][s] * Gb[i][t] for i in range(J))
               + sum(b2[i][s] * Wb[i][t] for i in range(N)) + ST[s][t]
               for t in range(nT)] for s in range(nT)]
        den = vT + J + N - nT - 1.0
        SigT = [[SS[s][t] / den for t in range(nT)] for s in range(nT)]
        # step 6: Sigma_E, through the b_2* reshaping
        STinv = _inv(SigT)
        b2s = [[b2[e * J + j][t] for e in range(I)] for j in range(J) for t in range(nT)]
        GS = _kron(Ginv, STinv)
        Ab = [[sum(GS[r][k] * b2s[k][e] for k in range(J * nT)) for e in range(I)]
              for r in range(J * nT)]
        SS = [[sum(b2s[r][a] * Ab[r][b] for r in range(J * nT)) + SE[a][b] for b in range(I)]
              for a in range(I)]
        den = vE + J * nT - I - 1.0
        SigE = [[SS[a][b] / den for b in range(I)] for a in range(I)]
        # step 7: R, with S_R and not the book's misprinted S_T
        E = resid()
        SS = [[sum(E[i][s] * E[i][t] for i in range(N)) + SR[s][t] for t in range(nT)]
              for s in range(nT)]
        den = vR + N - nT - 1.0
        Rm = [[SS[s][t] / den for t in range(nT)] for s in range(nT)]
        d = 0.0
        for i in range(N):
            for t in range(nT):
                d = max(d, abs(b2[i][t] - prev[i][t]))
        if d < tol:
            break
    gebv = [[b1[i % J][t] + b2[i][t] for t in range(nT)] for i in range(N)]
    tot = 0.0
    for i in range(N):
        for t in range(nT):
            tot += gebv[i][t]
    return RichResult(
        title="Bayesian multi-trait multi-environment model",
        summary_lines=[("environments", I), ("lines", J), ("traits", nT)],
        payload={
            "estimate": tot / (N * nT),
            "gebv": gebv,
            "b1": b1,
            "b2": b2,
            "sigma_g": SigT,
            "Sigma_T": SigT,
            "Sigma_E": SigE,
            "R": Rm,
            "mu": mu,
            "beta": beta,
            "n": N,
            "method": "Chapter 6 eq. (6.11) BMTME, the eight-step p.196 sampler taken at "
                      "its conditional means",
        },
    )


def cheatsheet():
    return "bmtme: Bayesian multi-trait multi-environment model (BMTME)"


# compact alias per ledger/NAMING.md
bmtmemodel = bmtme_model
