# morie.fn -- function file (rootcoder007/morie)
"""Variational autoencoder evidence lower bound (SGVB estimator).

SOURCE.  Kingma, D.P. and Welling, M. (2014), "Auto-Encoding Variational
Bayes", ICLR 2014; arXiv:1312.6114.

The bound is the paper's Eq. (3),

    L(theta, phi; x) = -D_KL( q_phi(z|x) || p_theta(z) )
                       + E_{q_phi(z|x)}[ log p_theta(x|z) ],

estimated by the SGVB estimator of Eq. (7) with the reparameterisation
z = mu + sigma * eps, eps ~ N(0, I).  With a Gaussian encoder and a
standard normal prior the KL term is available in closed form, which is
the paper's Appendix B / Eq. (10):

    -D_KL = (1/2) sum_j ( 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 ).

DETERMINISM.  eps is not pseudo-random here: it comes from the shared
deterministic normal stream (base-2 van der Corput through AS 241), so
the Python and R arms hold the SAME draws and a 1e-9 parity comparison
means something.  That is this implementation's choice; the paper draws
eps at random.

DECODER.  p_theta(x|z) = N(x; W_dec z + b_dec, s^2 I), the Gaussian
decoder of the paper's Appendix C.2.  For this decoder the
reconstruction term also has a closed form,

    E_q[log p(x|z)] = -(1/2) sum_k [ log(2 pi s^2)
                        + ( (x_k - (W_dec mu + b)_k)^2
                            + sum_j sigma_j^2 W_dec[j,k]^2 ) / s^2 ],

which is returned as ``recon_analytic``.  The Monte Carlo estimate must
approach it as ``n_samples`` grows; that comparison is an anchor which
does not run through the other language arm.

Untrained weights come from the deterministic stream, as elsewhere in
this package: a reproducible reference implementation cannot ship
trained parameters.
"""

from __future__ import annotations

import math

from . import _s03core as core
from . import _vitcore as vc

from ._richresult import RichResult

__all__ = ["vae_elbo"]


def vae_elbo(x, encoder=None, decoder=None, latent_dim=2, n_samples=64,
             decoder_scale=1.0, skip=0):
    """SGVB estimate of the VAE evidence lower bound.

    Parameters
    ----------
    x : array-like
        n-by-d data matrix (a flat sequence is one row).
    encoder : mapping or None
        ``{"mu": n-by-m, "logvar": n-by-m}``.  ``None`` builds a
        deterministic linear encoder from the shared stream.
    decoder : mapping or None
        ``{"W": m-by-d, "b": length d}``.  ``None`` builds a
        deterministic linear decoder.
    latent_dim : int
        m, used only when ``encoder`` is ``None``.
    n_samples : int
        Number of reparameterised draws per data point.
    decoder_scale : float
        s, the decoder's standard deviation; > 0.
    skip : int
        Offset into the shared deterministic stream.

    Returns
    -------
    RichResult
        ``elbo``, ``kl``, ``recon``, ``recon_analytic``,
        ``elbo_analytic``, ``mc_error``, ``mu``, ``logvar``,
        ``elbo_per_point``, ``kl_per_point``, ``n``, ``d``,
        ``latent_dim``, ``n_samples``.

    Raises
    ------
    ValueError
        Empty ``x``, ragged rows, a non-positive latent dimension,
        sample count or decoder scale, or an encoder/decoder of the
        wrong shape.

    References
    ----------
    Kingma, D.P. and Welling, M. (2014).  Auto-Encoding Variational
    Bayes.  ICLR 2014; arXiv:1312.6114.
    """
    X = core.mat(x)
    n = len(X)
    if n == 0:
        raise ValueError("vae_elbo: x is empty")
    d = len(X[0])
    for r in X:
        if len(r) != d:
            raise ValueError("vae_elbo: rows of x have unequal length")
    m = int(latent_dim)
    L = int(n_samples)
    s = float(decoder_scale)
    if m < 1:
        raise ValueError("vae_elbo: latent_dim must be positive")
    if L < 1:
        raise ValueError("vae_elbo: n_samples must be positive")
    if not (s > 0.0):
        raise ValueError("vae_elbo: decoder_scale must be positive")
    skip = int(skip)
    if skip < 0:
        raise ValueError("vae_elbo: skip must be non-negative")
    if encoder is None:
        Wm = vc.draw(d, m, skip, 1.0 / math.sqrt(d))
        Wl = vc.draw(d, m, skip + d * m, 0.1 / math.sqrt(d))
        mu = core.matmul(X, Wm)
        lv = core.matmul(X, Wl)
    else:
        mu = core.mat(encoder["mu"])
        lv = core.mat(encoder["logvar"])
        if len(mu) != n or len(lv) != n:
            raise ValueError("vae_elbo: encoder mu/logvar must have one row per observation")
        m = len(mu[0])
        if any(len(r) != m for r in mu) or any(len(r) != m for r in lv):
            raise ValueError("vae_elbo: encoder mu/logvar must be n-by-m")
    if decoder is None:
        Wd = vc.draw(m, d, skip + 2 * d * m, 1.0 / math.sqrt(m))
        bd = [0.0] * d
    else:
        Wd = core.mat(decoder["W"])
        bd = core.vec(decoder["b"])
        if len(Wd) != m or len(Wd[0]) != d:
            raise ValueError("vae_elbo: decoder W must be m-by-d")
        if len(bd) != d:
            raise ValueError("vae_elbo: decoder b must have length d")
    eps = vc.draw(L, m, skip + 2 * d * m + m * d, 1.0)
    sig = [[math.exp(0.5 * lv[i][j]) for j in range(m)] for i in range(n)]
    klp = [0.0] * n
    for i in range(n):
        t = 0.0
        for j in range(m):
            v = sig[i][j] * sig[i][j]
            t += mu[i][j] * mu[i][j] + v - 1.0 - lv[i][j]
        klp[i] = 0.5 * t
    c = math.log(2.0 * math.pi * s * s)
    recp = [0.0] * n
    anap = [0.0] * n
    for i in range(n):
        acc = 0.0
        for l in range(L):
            z = [mu[i][j] + sig[i][j] * eps[l][j] for j in range(m)]
            t = 0.0
            for k in range(d):
                r = bd[k]
                for j in range(m):
                    r += z[j] * Wd[j][k]
                t += c + (X[i][k] - r) * (X[i][k] - r) / (s * s)
            acc += -0.5 * t
        recp[i] = acc / L
        t = 0.0
        for k in range(d):
            r = bd[k]
            for j in range(m):
                r += mu[i][j] * Wd[j][k]
            q = 0.0
            for j in range(m):
                q += sig[i][j] * sig[i][j] * Wd[j][k] * Wd[j][k]
            t += c + ((X[i][k] - r) * (X[i][k] - r) + q) / (s * s)
        anap[i] = -0.5 * t
    kl = sum(klp) / n
    rec = sum(recp) / n
    ana = sum(anap) / n
    per = [recp[i] - klp[i] for i in range(n)]
    return RichResult(
        title="VAE evidence lower bound (SGVB)",
        summary_lines=[("obs", n), ("latent dim", m), ("ELBO", rec - kl)],
        payload={
            "estimate": rec - kl,
            "elbo": rec - kl,
            "kl": kl,
            "recon": rec,
            "recon_analytic": ana,
            "elbo_analytic": ana - kl,
            "mc_error": abs(rec - ana),
            "mu": mu,
            "logvar": lv,
            "elbo_per_point": per,
            "kl_per_point": klp,
            "recon_per_point": recp,
            "n": n,
            "d": d,
            "latent_dim": m,
            "n_samples": L,
            "method": "SGVB ELBO with the closed-form Gaussian KL (Kingma and Welling 2014 Eqs. 3, 7, 10)",
        },
    )


def cheatsheet():
    return "vaeber: VAE evidence lower bound, SGVB estimator (Kingma & Welling 2014)"

# public names resolved by fn/_lazy_map.json
vaeelbo = vae_elbo
