# morie.fn -- function file (rootcoder007/morie)
"""Mean-field ADVI evidence lower bound."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["advielbo", "advi"]


def advielbo(mu, omega, eta, logjoint):
    """Monte-Carlo ELBO of a mean-field Gaussian variational family.

    ADVI posits a factorised Gaussian in the unconstrained space,
    q(zeta; phi) = N(zeta; mu, diag(exp(omega)^2)) with omega = log sigma,
    and evaluates the evidence lower bound after elliptical
    standardisation zeta = mu + exp(omega) * eta with eta ~ N(0, I):

        L(mu, omega) = E_{N(eta;0,I)}[ log p(x, zeta) ] + H[q]
        H[q]         = sum_k omega_k + (K/2)(1 + log 2 pi)

    The expectation is replaced by the average over the supplied draws,
    so the caller owns the randomness and the value is reproducible.

    Parameters
    ----------
    mu : array-like
        Variational means, length K.
    omega : array-like
        Log standard deviations, length K.
    eta : array-like, shape (S, K)
        Standard-normal draws supplied by the caller.
    logjoint : callable
        Maps a length-K vector zeta to log p(x, zeta), already including
        any log-determinant Jacobian of the constraining transform.

    Returns
    -------
    RichResult
        ``elbo``, ``entropy``, ``meanlogjoint``, ``logjoints``, ``K``,
        ``S``.

    References
    ----------
    Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A. and Blei, D. M.
    (2017), "Automatic differentiation variational inference", Journal of
    Machine Learning Research 18(14), 1-45; arXiv:1603.00788.  The
    mean-field family q(zeta;phi)=N(zeta; mu, diag(exp(omega)^2)) with
    omega = log sigma and the ELBO of Equation (5) with an additive
    Gaussian entropy were read from the ar5iv rendering of the arXiv
    source, Sects. 2.3-2.5.  The entropy of a K-variate Gaussian with
    diagonal covariance, sum_k omega_k + (K/2)(1 + log 2 pi), is the
    standard closed form the paper refers to as analytic.
    """
    mu = C.vec(mu)
    omega = C.vec(omega)
    E = C.mat(eta)
    K = len(mu)
    if len(omega) != K:
        raise ValueError("mu and omega must have the same length")
    if len(E[0]) != K:
        raise ValueError("eta must have K columns")
    S = len(E)
    lj = []
    for s in range(S):
        zeta = [mu[k] + math.exp(omega[k]) * E[s][k] for k in range(K)]
        lj.append(float(logjoint(zeta)))
    ent = sum(omega) + 0.5 * K * (1.0 + math.log(2.0 * math.pi))
    mlj = sum(lj) / S
    return RichResult(payload={
        "elbo": mlj + ent, "entropy": ent, "meanlogjoint": mlj,
        "logjoints": lj, "K": K, "S": S,
        "method": "Mean-field ADVI ELBO (Kucukelbir et al. 2017 eq. 5)"})


advi = advielbo


def cheatsheet():
    return "advinf: Mean-field ADVI evidence lower bound."
