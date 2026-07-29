# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Variational autoencoder ELBO: reconstruction term minus KL to the prior."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_vae_elbo"]

_METHOD = "VAE evidence lower bound"


def geron_vae_elbo(x, mu, logvar, recon, likelihood="gaussian", beta=1.0):
    r"""Evidence lower bound of a Gaussian-posterior VAE.

    .. math::
        \mathrm{ELBO} = \mathbb{E}_q[\log p(x|z)]
                        - \mathrm{KL}\bigl(q(z|x)\,\|\,p(z)\bigr)

    With :math:`q = \mathcal{N}(\mu, \sigma^2)` and
    :math:`p = \mathcal{N}(0, I)` the KL has the closed form

    .. math::
        \mathrm{KL} = -\tfrac{1}{2}\sum_j
            \bigl(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\bigr),

    so no sampling is needed for that half -- only the reconstruction
    term is a Monte Carlo estimate.  ELBO is a *lower* bound on
    :math:`\log p(x)`, and maximising it is the same as minimising the
    negative, which is what a training loop actually reports; both signs
    are returned so nobody has to guess which one is on screen.
    ``logvar``, not ``sigma``: the network predicts the log so the
    variance stays positive without a constraint.

    Parameters
    ----------
    x : array-like, shape (m, n)
    mu, logvar : array-like, shape (m, k)
    recon : array-like, shape (m, n)
        Decoder output: means for ``"gaussian"``, probabilities in
        ``[0, 1]`` for ``"bernoulli"``.
    likelihood : {"gaussian", "bernoulli"}, optional
    beta : float, optional
        KL weight (beta-VAE); 1.0 is the plain ELBO.

    Returns
    -------
    RichResult
        Payload keys ``elbo``, ``loss`` (negative ELBO),
        ``reconstruction_term``, ``kl``, ``kl_per_dim``,
        ``estimate`` (elbo), ``n``, ``method``.

    References
    ----------
    Géron Ch 18, Variational Autoencoder section.

    Examples
    --------
    A posterior equal to the prior (``mu = 0``, ``logvar = 0``) has zero
    KL, so the ELBO is the reconstruction term alone -- here a perfect
    reconstruction, hence 0:

    >>> r = geron_vae_elbo([[1.0]], [[0.0]], [[0.0]], [[1.0]])
    >>> r["kl"], abs(r["reconstruction_term"]), abs(r["elbo"])
    (0.0, 0.0, 0.0)

    Moving the posterior mean to 1 costs ``0.5`` of KL:

    >>> k = geron_vae_elbo([[1.0]], [[1.0]], [[0.0]], [[1.0]])
    >>> k["kl"], k["elbo"]
    (0.5, -0.5)

    Bernoulli likelihood on a certain, correct pixel also costs nothing:

    >>> b = geron_vae_elbo([[1.0]], [[0.0]], [[0.0]], [[1.0]], likelihood="bernoulli")
    >>> round(b["reconstruction_term"], 10)
    -0.0
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    M = np.atleast_2d(np.asarray(mu, dtype=float))
    LV = np.atleast_2d(np.asarray(logvar, dtype=float))
    R = np.atleast_2d(np.asarray(recon, dtype=float))
    if X.size == 0:
        raise ValueError("x is empty.")
    if R.shape != X.shape:
        raise ValueError(f"recon has shape {R.shape} but x has {X.shape}.")
    if M.shape != LV.shape:
        raise ValueError(f"mu has shape {M.shape} but logvar has {LV.shape}.")
    if M.shape[0] != X.shape[0]:
        raise ValueError(f"mu has {M.shape[0]} rows but x has {X.shape[0]}.")
    for name, A in (("x", X), ("mu", M), ("logvar", LV), ("recon", R)):
        if not np.all(np.isfinite(A)):
            raise ValueError(f"{name} contains non-finite values.")
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError(f"beta must be finite and non-negative, got {beta}.")

    if likelihood == "gaussian":
        recon_term = float(-0.5 * np.sum((X - R) ** 2) / X.shape[0])
    elif likelihood == "bernoulli":
        if np.any(R < 0) or np.any(R > 1):
            raise ValueError("bernoulli recon must lie in [0, 1]; these are not probabilities.")
        if np.any((X < 0) | (X > 1)):
            raise ValueError("bernoulli x must lie in [0, 1].")
        Rc = np.clip(R, 1e-12, 1 - 1e-12)
        recon_term = float(np.sum(X * np.log(Rc) + (1 - X) * np.log(1 - Rc)) / X.shape[0])
    else:
        raise ValueError(f"likelihood must be 'gaussian' or 'bernoulli', got {likelihood!r}.")

    kl_dim = -0.5 * (1.0 + LV - M**2 - np.exp(LV))
    kl = float(kl_dim.sum() / X.shape[0])
    elbo = recon_term - beta * kl

    return RichResult(
        title="VAE ELBO",
        summary_lines=[("ELBO", elbo), ("Reconstruction", recon_term), ("KL", kl)],
        payload={
            "elbo": elbo,
            "loss": -elbo,
            "reconstruction_term": recon_term,
            "kl": kl,
            "kl_per_dim": kl_dim.mean(axis=0).tolist(),
            "beta": beta,
            "estimate": elbo,
            "n": int(X.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grvae: ELBO = E_q[log p(x|z)] - KL; KL = -0.5 sum(1 + logvar - mu^2 - exp(logvar)) in closed form"
