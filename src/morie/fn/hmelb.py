# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evidence lower bound (ELBO) loss for VAE."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_elbo"]


def geron_elbo(x, mu, log_sigma, x_recon=None, likelihood="gaussian", sigma_x=1.0):
    """
    Evidence lower bound (ELBO) loss for VAE.

    Formula: ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

    With a diagonal Gaussian posterior and a standard normal prior the KL
    term is closed form,

        KL = -0.5 * sum_j (1 + 2 log_sigma_j - mu_j^2 - sigma_j^2),

    and is computed exactly here -- no sampling. The reconstruction term
    is a Gaussian log-likelihood with fixed variance ``sigma_x^2`` (or a
    Bernoulli log-likelihood with ``likelihood="bernoulli"``). If
    ``x_recon`` is omitted the decoder is treated as perfect, so the
    reconstruction term is at its maximum and ``elbo`` reduces to ``-KL``.

    ``loss`` is the quantity you minimise, ``-elbo``.

    Parameters
    ----------
    x : array-like, shape (m, d) or (d,)
        Inputs.
    mu, log_sigma : array-like, shape (m, k) or (k,)
        Posterior mean and log standard deviation.
    x_recon : array-like, optional
        Decoder output, same shape as ``x``. Default: ``x`` itself.
    likelihood : {"gaussian", "bernoulli"}, default "gaussian"
        With "bernoulli", ``x`` and ``x_recon`` must lie in [0, 1].
    sigma_x : float, default 1.0
        Fixed decoder standard deviation for the Gaussian likelihood.

    Returns
    -------
    result : RichResult
        Keys: elbo, loss, kl, reconstruction_log_lik, per_sample_kl,
        per_sample_elbo, estimate, n, method.

    Examples
    --------
    A posterior equal to the prior costs nothing, and a perfect decoder
    leaves only the Gaussian normaliser ``-0.5 log(2 pi)`` per dimension:

    >>> import math
    >>> r = geron_elbo([[0.0]], mu=[[0.0]], log_sigma=[[0.0]])
    >>> round(r["kl"], 12)
    0.0
    >>> round(r["reconstruction_log_lik"], 9) == round(-0.5 * math.log(2 * math.pi), 9)
    True

    Shifting the posterior mean to 1 costs exactly 0.5 nats of KL:

    >>> round(geron_elbo([[0.0]], [[1.0]], [[0.0]])["kl"], 12)
    0.5

    A reconstruction error of 1 costs 0.5 more nats:

    >>> a = geron_elbo([[0.0]], [[0.0]], [[0.0]])["elbo"]
    >>> b = geron_elbo([[0.0]], [[0.0]], [[0.0]], x_recon=[[1.0]])["elbo"]
    >>> round(a - b, 12)
    0.5

    References
    ----------
    Géron Ch 18
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    M = np.atleast_2d(np.asarray(mu, dtype=float))
    LS = np.atleast_2d(np.asarray(log_sigma, dtype=float))
    if M.shape != LS.shape:
        raise ValueError(f"geron_elbo: mu has shape {M.shape} but log_sigma has shape {LS.shape}")
    if X.shape[0] != M.shape[0]:
        raise ValueError(f"geron_elbo: x has {X.shape[0]} rows but mu has {M.shape[0]}")
    if X.size == 0 or M.size == 0:
        raise ValueError("geron_elbo: x and mu must be non-empty")
    for name, arr in (("x", X), ("mu", M), ("log_sigma", LS)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_elbo: {name} contains non-finite values")
    if likelihood not in ("gaussian", "bernoulli"):
        raise ValueError(f"geron_elbo: likelihood must be 'gaussian' or 'bernoulli', got {likelihood!r}")
    sx = float(sigma_x)
    if not np.isfinite(sx) or sx <= 0:
        raise ValueError(f"geron_elbo: sigma_x must be positive and finite, got {sigma_x!r}")

    R = X.copy() if x_recon is None else np.atleast_2d(np.asarray(x_recon, dtype=float))
    if R.shape != X.shape:
        raise ValueError(f"geron_elbo: x_recon must have shape {X.shape}, got {R.shape}")
    if not np.all(np.isfinite(R)):
        raise ValueError("geron_elbo: x_recon contains non-finite values")

    var = np.exp(2.0 * LS)
    kl_i = -0.5 * np.sum(1.0 + 2.0 * LS - M**2 - var, axis=1)

    if likelihood == "gaussian":
        rec_i = -0.5 * np.sum((X - R) ** 2, axis=1) / (sx * sx) - X.shape[1] * (math.log(2 * math.pi) / 2 + math.log(sx))
    else:
        if np.any(X < 0) or np.any(X > 1) or np.any(R < 0) or np.any(R > 1):
            raise ValueError("geron_elbo: Bernoulli likelihood requires x and x_recon in [0, 1]")
        eps = 1e-12
        rec_i = np.sum(X * np.log(R + eps) + (1 - X) * np.log(1 - R + eps), axis=1)

    elbo_i = rec_i - kl_i
    elbo = float(np.mean(elbo_i))
    kl = float(np.mean(kl_i))
    rec = float(np.mean(rec_i))

    return RichResult(
        title="VAE ELBO",
        summary_lines=[("ELBO", elbo), ("KL", kl), ("Reconstruction", rec)],
        interpretation="Maximise the ELBO, i.e. minimise `loss` = -ELBO; KL = 0 means the posterior is the prior.",
        payload={
            "elbo": elbo,
            "loss": -elbo,
            "kl": kl,
            "reconstruction_log_lik": rec,
            "per_sample_kl": kl_i.tolist(),
            "per_sample_elbo": elbo_i.tolist(),
            "latent_dim": int(M.shape[1]),
            "likelihood": likelihood,
            "estimate": elbo,
            "n": int(X.shape[0]),
            "method": "ELBO = E_q[log p(x|z)] - KL(q||p) with closed-form Gaussian KL",
        },
    )


def cheatsheet():
    return "hmelb: Evidence lower bound (ELBO) loss for VAE"
