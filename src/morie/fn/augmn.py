# morie.fn -- slice s04 (rootcoder007/morie)
"""Albert-Chib data augmentation for binary ordinal Gibbs sampler.

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 209-249], Chapter 7, Section 7.2
"Bayesian Ordinal Regression Model", equation (7.2) on p. 214 and the
numbered Gibbs samplers on pp. 212 and 214, all read as rendered page
images.  The chapter attributes the scheme to Albert, J. H. and Chib, S.
(1993), Bayesian analysis of binary and polychotomous response data,
*Journal of the American Statistical Association* 88(422), 669-679,
which its reference list carries.

Equation (7.2) is p_ic = P(Y_i = c) = Phi(gamma_c + b_i) -
Phi(gamma_{c-1} + b_i), c = 1, ..., C, "where now L_i = - b_i + epsilon_i
is the latent variable".  Step 2 of the sampler is: "For each
i = 1, ..., n, simulate l_i from the normal distribution
N(-x_i^T beta, 1) truncated in (gamma_{y_i - 1}, gamma_{y_i})", and step
3 draws b from N_n(b~, Sigma~_b) with
Sigma~_b = (sigma_g^-2 G^-1 + I_n)^-1 and b~ = - Sigma~_b l.

SIGN CONVENTION, stated because the book's minus signs are load-bearing
and the PDF text layer drops them.  The book's latent has mean MINUS the
linear predictor, and its b~ and its
beta~_j = - sigma~^2_{beta_j} (x_j^T e_j) on p. 212 carry the matching
minus.  This implementation works with lstar = -l, the latent on the
positive orientation, so lstar has mean +eta and step 3 becomes
b~ = Sigma~_b lstar.  The two are algebraically the same model and the
returned beta and b are on the book's own scale; only z_samples and
estimate, which are lstar, are the negatives of the book's l.

A third erratum in this book, recorded here.  Page 214 writes "L_i = -
b_i + epsilon_i is the latent variable" and then, in the very next
sentence, "In matrix form the model for the latent variable can be
specified as L = b + epsilon".  Those two contradict each other; the
elementwise form is the one consistent with equation (7.2), with
b~ = - Sigma~_b l, and with the beta step on p. 212, so the matrix line
is the misprint.

DETERMINISM.  Steps 2 and 3 are not simulated.  The latent is set to the
exact mean of its truncated normal -- for the binary thresholds
(-inf, 0, +inf) that is eta_i + phi(eta_i)/Phi(eta_i) when y_i = 1 and
eta_i - phi(eta_i)/(1 - Phi(eta_i)) when y_i = 0 -- and step 3 is
replaced by its conditional mean.  Iterating those two exact conditional
means is the EM fixed point of the same augmentation, so both arms land
on identical numbers rather than on the same posterior.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["albert_chib_augmentation"]


def albert_chib_augmentation(y_bin, X, Z=None, sigma_g2=1.0, max_iter=200, tol=1e-13):
    """Albert-Chib augmentation, run to its exact conditional-mean fixed point.

    Parameters
    ----------
    y_bin : array-like
        Binary response, entries 0 or 1.
    X : array-like
        n-by-p fixed-effects design matrix.
    Z : array-like, optional
        n-by-q design for the shrunken effects; the identity when absent,
        which is the chapter's b = (b_1, ..., b_n) parameterisation.
    sigma_g2 : float
        The chapter's sigma_g^2, the prior variance of those effects.
    max_iter, tol : int, float
        Fixed-point controls.

    Returns
    -------
    estimate      : the mean of the augmented latent vector
    z_samples     : the augmented latents l
    beta_samples  : the fixed-effect coefficients
    b             : the shrunken effects
    """
    yy = k.vec(y_bin)
    n = len(yy)
    if n == 0:
        raise ValueError("albert_chib_augmentation: y_bin is empty")
    for v in yy:
        if v != 0.0 and v != 1.0:
            raise ValueError("albert_chib_augmentation: y_bin must be 0 or 1")
    XX = k.mat(X)
    if len(XX) != n:
        raise ValueError("albert_chib_augmentation: X has a different number of rows than y_bin")
    p = len(XX[0])
    if Z is None:
        ZZ = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    else:
        ZZ = k.mat(Z)
        if len(ZZ) != n:
            raise ValueError("albert_chib_augmentation: Z has a different number of rows than y_bin")
    q = len(ZZ[0])
    s2 = float(sigma_g2)
    if s2 <= 0.0:
        raise ValueError("albert_chib_augmentation: sigma_g2 must be positive")
    beta = [0.0] * p
    b = [0.0] * q
    lat = [0.0] * n
    inv2pi = 1.0 / math.sqrt(2.0 * math.pi)
    for _ in range(int(max_iter)):
        # step 2: the exact mean of the truncated normal on each side of 0
        prev = list(lat)
        for i in range(n):
            eta = 0.0
            for j in range(p):
                eta += XX[i][j] * beta[j]
            for j in range(q):
                eta += ZZ[i][j] * b[j]
            d = inv2pi * math.exp(-0.5 * eta * eta)
            P = k.pnorm(eta)
            if yy[i] == 1.0:
                lat[i] = eta + (d / P if P > 1e-300 else -eta)
            else:
                Q = 1.0 - P
                lat[i] = eta - (d / Q if Q > 1e-300 else eta)
        # steps 1 and 3: conditional means of beta and b, jointly, with the
        # chapter's sigma_g^-2 ridge on b alone
        M = [[0.0] * (p + q) for _ in range(p + q)]
        r = [0.0] * (p + q)
        for i in range(n):
            row = list(XX[i]) + list(ZZ[i])
            for a in range(p + q):
                r[a] += row[a] * lat[i]
                for c in range(p + q):
                    M[a][c] += row[a] * row[c]
        for a in range(p, p + q):
            M[a][a] += 1.0 / s2
        sol = k.ridgesolve(M, r, 1e-12)
        beta = sol[:p]
        b = sol[p:]
        d = 0.0
        for i in range(n):
            d = max(d, abs(lat[i] - prev[i]))
        if d < tol:
            break
    mu = 0.0
    for v in lat:
        mu += v
    mu = mu / n
    return RichResult(
        title="Albert-Chib augmentation",
        summary_lines=[("n", n), ("p", p), ("q", q)],
        payload={
            "estimate": mu,
            "z_samples": lat,
            "beta_samples": beta,
            "b": b,
            "n": n,
            "method": "Chapter 7 eq. (7.2) augmentation, steps 2-3 taken at their exact conditional means",
        },
    )


def cheatsheet():
    return "augmn: Albert-Chib data augmentation for binary ordinal Gibbs sampler"
