# morie.fn -- function file (rootcoder007/morie)
"""High-throughput phenotyping functional predictor combining genomic + phenomic info.

SOURCE, AND A CORRECTION TO THE STUB THIS REPLACES.  The stub docstring
cited "Montesinos Lopez Ch 14" for the whole model, including the genomic
term ``g_i``.  Chapter 14 does not contain a genomic term at all.  It was
read in full -- volume [Pages 579-631] of Montesinos Lopez, Montesinos
Lopez and Crossa (2022), *Multivariate Statistical Machine Learning
Methods for Genomic Prediction*, Springer, doi:10.1007/978-3-030-89010-0
-- and Section 14.1 pp.579-583 gives ONLY the functional part: a scalar
response, one functional covariate, and no random line effect anywhere.
The genomic term therefore comes from a different chapter and is cited as
such below.  It is NOT in Chapter 14.

FUNCTIONAL PART -- Chapter 14, Section 14.1, volume [Pages 579-631],
pp.579-583, all equations read from rendered page images:

    (14.1)  Y = mu + int_0^T x(t) beta(t) dt + E
    (14.2)  beta(t) = sum_{l=1}^{L1} beta_l phi_l(t)
    (14.3)  Y = mu + sum_l beta_l int x(t) phi_l(t) dt = Xstar beta + E
    (14.4)  betahat = (Xstar' Xstar)^-1 Xstar' y
    (14.5)  sigma2hat = (1/n) (y - Xstar betahat)' (y - Xstar betahat)
    (14.6)  x_i(t) = sum_{o=1}^{L2} c_io psi_o(t)
    (14.7)  chat_i = (Psi' Psi)^-1 Psi' x_i(t)          <- p.581 and p.583
    (14.8)  Psi is m-by-L2 with Psi[j, o] = psi_o(t_j)
    (14.9)  Xstar = [1n  X],  X = Xtilde Psi (Psi'Psi)^-1 Q',
            Q[l, o] = int_0^T phi_l(t) psi_o(t) dt
    p.582   BIC = -2 loglik(betahat, sigma2hat; y) + (L1 + 1) log(n)
    p.583   CV1(L2) = sum_j (x(t_j) - xhat_j(t_j))^2, xhat_j the
            leave-point-j-out representation with L2 bases

    Equation (14.9) is implemented as X = Chat Q-transpose, which is the
    same matrix -- the book derives x_i = Q chat_i on p.581 and only then
    substitutes (14.7) into it to reach the printed form.  Going through
    Chat avoids forming and re-solving (Psi-transpose Psi) a second time.

GENOMIC PART -- Chapter 5, Section 5.3, equation (5.3), volume
[Pages 141-170], p.148, read from the same book:

    (5.3)   Y = 1n mu + Z_L b + e,  b ~ N_J(0, sigma_g^2 G),  R = sigma^2 In

with G the genomic relationship matrix of VanRaden, P. M. (2008),
Efficient methods to compute genomic predictions, *Journal of Dairy
Science* 91(11):4414-4423, doi:10.3168/jds.2008-0980, which is the source
Section 5.3 itself names.  G is built by this package's own
``grm_vanraden`` (method 1) rather than re-derived here.

THE COMBINED MODEL, which is this function's own composition of the two
and is not printed as a single equation anywhere in the book:

    y = 1n mu + X beta + Z_L g + e, g ~ N(0, sigma_g^2 G), e ~ N(0, sigma^2 I)

with X the functional design of (14.9).  It is solved by Henderson mixed
model equations at a fixed variance ratio lam = sigma^2 / sigma_g^2, with
one observation per line so that Z_L = I_n.

ERRATUM, confirmed by rendered page image, p.584.  The Fourier basis of
Section 14.2.1 is printed as

    phi_1 = 1/sqrt(P), phi_2 = sqrt(2/P) sin(wt), phi_3 = sqrt(2/P) sin(wt),
    phi_4 = sqrt(2/P) cos(2wt), phi_5 = sqrt(2/P) cos(2wt),
    phi_6 = sqrt(2/P) cos(3wt), phi_7 = sqrt(2/P) cos(3wt), ...

so phi_2 = phi_3, phi_4 = phi_5 and phi_6 = phi_7.  A repeated set is not
a basis: (Psi-transpose Psi) would be singular by construction.  Figure
14.1 on p.585 plots the first five elements for P = 4 on (0, 8) and shows
FIVE DISTINCT curves, with phi_1 flat at 0.5 = 1/sqrt(4), phi_2 zero at
t = 0 (a sine) and phi_3 at 0.7071 = sqrt(2/4) at t = 0 (a cosine).  The
alternating reading is the correct one and is what is implemented:

    phi_1 = 1/sqrt(P),
    phi_{2k}   = sqrt(2/P) sin(k w t),
    phi_{2k+1} = sqrt(2/P) cos(k w t),     w = 2 pi / P.

Determinism: nothing here is stochastic.  There is no sampling, no fold
assignment and no initialisation; the LOOCV of p.583 is exhaustive over
the m grid points.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult
from .gmatv import grm_vanraden

__all__ = ["htp_functional_predictor"]


def _fourier_row(s, L, P):
    """phi_1..phi_L evaluated at s: the alternating basis of p.584."""
    c = math.sqrt(2.0 / P)
    w = 2.0 * math.pi / P
    out = [1.0 / math.sqrt(P)]
    k = 1
    while len(out) < L:
        out.append(c * math.sin(k * w * s))
        if len(out) < L:
            out.append(c * math.cos(k * w * s))
        k += 1
    return out[:L]


def _inv_spd(A, ridge):
    n = len(A)
    B = [[A[i][j] + (ridge if i == j else 0.0) for j in range(n)] for i in range(n)]
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(core.cholsolve(B, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def htp_functional_predictor(y, markers, W_functional, n_basis=5, lam=1.0,
                             a=0.0, b=1.0, period=None, ridge=1e-8):
    """Genomic plus phenomic functional predictor.

    Parameters
    ----------
    y : array-like
        Length-n vector of scalar responses, one per line.
    markers : array-like
        n-by-p genotype matrix coded {0, 1, 2}; G is VanRaden method 1 of it.
    W_functional : array-like
        n-by-m matrix of high-throughput phenotyping curves; row i is
        w_i(t) sampled on the common equally spaced grid t_1 < ... < t_m.
    n_basis : int
        L1 = L2, the number of Fourier basis functions, 1 <= L <= m.
    lam : float
        The variance ratio sigma^2 / sigma_g^2 of the mixed model equations.
    a, b : float
        End points of the observation grid.
    period : float or None
        P of the Fourier basis; the book takes it as the range of observed
        t, which is the default, b - a.
    ridge : float
        Added to the diagonal of G before inversion; G from a finite marker
        panel is singular whenever two lines share a haplotype.

    Returns
    -------
    beta_func : beta(t) of (14.2) on the observation grid
    g_hat     : the BLUP of the genomic effect g of (5.3)
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("htp_functional_predictor: y is empty")
    Wm = core.mat(W_functional)
    if len(Wm) != n:
        raise ValueError("htp_functional_predictor: y and W_functional disagree on the number of lines")
    m = len(Wm[0])
    for r in Wm:
        if len(r) != m:
            raise ValueError("htp_functional_predictor: curves are sampled on grids of different length")
    if m < 2:
        raise ValueError("htp_functional_predictor: need at least two grid points")
    Mk = core.mat(markers)
    if len(Mk) != n:
        raise ValueError("htp_functional_predictor: y and markers disagree on the number of lines")
    L = int(n_basis)
    if L < 1 or L > m:
        raise ValueError("htp_functional_predictor: n_basis must lie between 1 and the number of grid points")
    lam = float(lam)
    if not lam > 0.0:
        raise ValueError("htp_functional_predictor: lam must be positive")
    a = float(a)
    b = float(b)
    if not b > a:
        raise ValueError("htp_functional_predictor: the grid must have positive width")
    P = (b - a) if period is None else float(period)
    if not P > 0.0:
        raise ValueError("htp_functional_predictor: period must be positive")

    h = (b - a) / (m - 1)
    tg = [a + j * h for j in range(m)]
    wq = [h * (0.5 if (j == 0 or j == m - 1) else 1.0) for j in range(m)]
    Psi = [_fourier_row(t - a, L, P) for t in tg]
    PtP = core.crossprod(Psi)
    Chat = []
    for i in range(n):
        rr = [sum(Psi[j][l] * Wm[i][j] for j in range(m)) for l in range(L)]
        Chat.append(core.ridgesolve(PtP, rr, 1e-12))
    Q = [[sum(wq[j] * Psi[j][l] * Psi[j][o] for j in range(m)) for o in range(L)] for l in range(L)]
    Xd = [[sum(Q[l][o] * Chat[i][o] for o in range(L)) for l in range(L)] for i in range(n)]
    Xs = [[1.0] + Xd[i] for i in range(n)]
    K = L + 1

    Gm = grm_vanraden(Mk, 1)["estimate"]
    G = [[float(Gm[i][j]) for j in range(n)] for i in range(n)]
    Ginv = _inv_spd(G, float(ridge))

    XtX = core.crossprod(Xs)
    C = [[0.0] * (K + n) for _ in range(K + n)]
    for i in range(K):
        for j in range(K):
            C[i][j] = XtX[i][j]
    for i in range(K):
        for j in range(n):
            C[i][K + j] = Xs[j][i]
            C[K + j][i] = Xs[j][i]
    for i in range(n):
        for j in range(n):
            C[K + i][K + j] = lam * Ginv[i][j] + (1.0 if i == j else 0.0)
    rhs = [sum(Xs[i][k] * yv[i] for i in range(n)) for k in range(K)] + list(yv)
    try:
        sol = core.cholsolve(C, rhs)
    except ValueError as exc:
        raise ValueError(
            "htp_functional_predictor: the mixed model equations are not positive "
            "definite (%s). The genomic block is regularised by lam G^-1 but the "
            "fixed-effect block is not, so this means the functional design Xstar "
            "is rank deficient -- typically n_basis >= n, or curves that are "
            "identical across lines." % (exc,)
        )
    # The Cholesky solve returns a ZERO VECTOR, silently and without error,
    # when the coefficient matrix is not positive definite. Check that the
    # solution actually solves the system rather than trusting it converged.
    scale = 1.0
    for v in rhs:
        scale = max(scale, abs(v))
    worst = 0.0
    for i in range(K + n):
        acc = 0.0
        for j in range(K + n):
            acc += C[i][j] * sol[j]
        worst = max(worst, abs(acc - rhs[i]))
    if not worst <= 1e-6 * scale:
        raise ValueError(
            "htp_functional_predictor: the mixed model equations did not solve "
            "(residual %.3g); the coefficient matrix is not positive definite" % worst
        )

    beta = list(sol[:K])
    ghat = list(sol[K:])
    mu = beta[0]
    beta_func = [sum(beta[l + 1] * Psi[j][l] for l in range(L)) for j in range(m)]
    fitted = [sum(Xs[i][k] * beta[k] for k in range(K)) + ghat[i] for i in range(n)]
    sse = 0.0
    for i in range(n):
        d = yv[i] - fitted[i]
        sse += d * d
    sigma2 = sse / n
    bic = (n * math.log(2.0 * math.pi * sigma2) + n + (L + 1) * math.log(n)) if sigma2 > 0.0 else float("-inf")

    cv1 = 0.0
    for i in range(n):
        for j in range(m):
            rows = [Psi[q] for q in range(m) if q != j]
            xs_ = [Wm[i][q] for q in range(m) if q != j]
            A = core.crossprod(rows)
            rr = [sum(rows[q][l] * xs_[q] for q in range(m - 1)) for l in range(L)]
            cj = core.ridgesolve(A, rr, 1e-12)
            pred = sum(cj[l] * Psi[j][l] for l in range(L))
            d = Wm[i][j] - pred
            cv1 += d * d

    return RichResult(
        title="HTP functional predictor",
        summary_lines=[("lines", n), ("grid", m), ("bases", L), ("lam", lam)],
        payload={
            "estimate": mu,
            "mu": mu,
            "beta_func": beta_func,
            "beta": beta,
            "g_hat": ghat,
            "coefs": Chat,
            "X": Xd,
            "Q": Q,
            "fitted": fitted,
            "sigma2": sigma2,
            "bic": bic,
            "cv1": cv1,
            "n": n,
            "method": (
                "y = 1n mu + X beta + Z_L g + e; X from eq. (14.9) with the Fourier "
                "basis of p.584 (alternating sin/cos; the printed set repeats), "
                "g ~ N(0, sigma_g^2 G) from eq. (5.3); Montesinos Lopez et al. (2022) "
                "Ch 14 and Ch 5, G by VanRaden (2008)"
            ),
        },
    )


def cheatsheet():
    return "htpfn: Ch 14 eq. (14.9) functional design with the Ch 5 eq. (5.3) genomic random effect"

