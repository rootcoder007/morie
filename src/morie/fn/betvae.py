# morie.fn -- function file (rootcoder007/morie)
"""beta-VAE objective with an optional capacity target."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["betavae", "beta_vae_disentangle"]


def betavae(x, xhat, mu, logvar, beta=4.0, capacity=None, gamma=None,
            noisevar=1.0):
    """Reconstruction term, KL term and the disentangling objective.

    The variational objective is the evidence lower bound with the
    Kullback-Leibler term scaled up, which pressures the approximate
    posterior towards the factorised prior and, at beta > 1, towards
    axis-aligned latent factors:

        L = E_q[ log p(x|z) ] - beta D_KL( q(z|x) || p(z) ).

    For a diagonal Gaussian posterior against a standard normal prior the
    divergence is closed form,

        D_KL = (1/2) sum_j ( mu_j^2 + sigma_j^2 - 1 - log sigma_j^2 ),

    and with a Gaussian decoder of fixed variance the reconstruction term
    is -||x - xhat||^2 / (2 noisevar) up to a constant.  The capacity
    variant replaces the plain penalty with gamma |D_KL - C|, which lets
    an increasing budget C be released over training instead of clamping
    the divergence at zero.

    Parameters
    ----------
    x : array-like
        Observation.
    xhat : array-like
        Decoder mean, same length as x.
    mu : array-like
        Posterior means, length J.
    logvar : array-like
        Posterior log variances, length J.
    beta : float
        Weight on the divergence.
    capacity : float or None
        Target C; ``None`` uses the plain beta penalty.
    gamma : float or None
        Weight on |D_KL - C|; ``None`` reuses ``beta``.
    noisevar : float
        Decoder variance, strictly positive.

    Returns
    -------
    RichResult
        ``objective``, ``recon``, ``kl``, ``klper``, ``penalty``,
        ``beta``, ``J``, ``d``.

    References
    ----------
    Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X.,
    Botvinick, M., Mohamed, S. and Lerchner, A. (2017), "beta-VAE:
    learning basic visual concepts with a constrained variational
    framework", International Conference on Learning Representations,
    which introduces the beta-weighted divergence; the capacity form
    gamma |D_KL - C| is Burgess, C. P. et al. (2018), "Understanding
    disentangling in beta-VAE", arXiv:1804.03599.  The closed-form
    Gaussian divergence is Kingma and Welling (2014), Appendix B.
    Standard published form; the ICLR paper was not in the local corpus
    and was not read for this implementation.
    """
    x = C.vec(x)
    xh = C.vec(xhat)
    m = C.vec(mu)
    lv = C.vec(logvar)
    if len(x) != len(xh):
        raise ValueError("x and xhat must have the same length")
    if len(m) != len(lv):
        raise ValueError("mu and logvar must have the same length")
    nv = float(noisevar)
    if nv <= 0.0:
        raise ValueError("noisevar must be strictly positive")
    d = len(x)
    J = len(m)
    rec = -sum((x[i] - xh[i]) ** 2 for i in range(d)) / (2.0 * nv) \
        - 0.5 * d * math.log(2.0 * math.pi * nv)
    per = [0.5 * (m[j] * m[j] + math.exp(lv[j]) - 1.0 - lv[j])
           for j in range(J)]
    kl = sum(per)
    b = float(beta)
    if capacity is None:
        pen = b * kl
    else:
        g = b if gamma is None else float(gamma)
        pen = g * abs(kl - float(capacity))
    return RichResult(payload={
        "objective": rec - pen, "recon": rec, "kl": kl, "klper": per,
        "penalty": pen, "beta": b, "J": J, "d": d,
        "method": "beta-VAE objective (Higgins et al. 2017)"})


beta_vae_disentangle = betavae


def cheatsheet():
    return "betvae: beta-VAE objective with an optional capacity target."
