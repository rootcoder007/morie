# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reparameterization trick: z = mu + sigma * eps."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_reparameterization_trick"]

_METHOD = "Reparameterization trick"


def _standard_normals(count, seed):
    """Box-Muller normals from the reproducible LCG in the house style."""
    s = int(seed) % 2**32
    out = np.empty(count + (count % 2))
    for i in range(0, out.size, 2):
        s = (1664525 * s + 1013904223) % 2**32
        u1 = max((s + 0.5) / 2**32, 1e-12)
        s = (1664525 * s + 1013904223) % 2**32
        u2 = (s + 0.5) / 2**32
        r = math.sqrt(-2.0 * math.log(u1))
        out[i] = r * math.cos(2 * math.pi * u2)
        out[i + 1] = r * math.sin(2 * math.pi * u2)
    return out[:count]


def geron_reparameterization_trick(mu, logvar, eps=None, seed=42):
    r"""Sample the latent in a way the gradient can pass through.

    .. math::
        z = \mu + \exp(\tfrac{1}{2}\log\sigma^2)\,\epsilon,
        \qquad \epsilon \sim \mathcal{N}(0, I)

    Sampling :math:`z \sim \mathcal{N}(\mu, \sigma^2)` directly puts a
    stochastic node between the loss and the encoder, and you cannot
    backpropagate through a random draw.  Rewriting the same distribution
    as a deterministic function of :math:`(\mu, \sigma)` and an
    *external* noise variable moves the randomness off the gradient path:
    :math:`\partial z/\partial\mu = 1` and
    :math:`\partial z/\partial\log\sigma^2 = \tfrac{1}{2}\sigma\epsilon`,
    both returned here.

    ``logvar`` is halved and exponentiated -- ``exp(0.5 * logvar)``, not
    ``exp(logvar)``; that factor of two is the most common transcription
    error in VAE code, and it silently squares the noise scale.  Noise
    comes from the reproducible LCG unless ``eps`` is supplied, and the
    achieved sample statistics are reported.

    Parameters
    ----------
    mu, logvar : array-like, same shape
    eps : array-like, optional
        Noise; drawn deterministically when omitted.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``z``, ``sigma``, ``eps``, ``dz_dmu``,
        ``dz_dlogvar``, ``sample_mean``, ``sample_variance``,
        ``estimate``, ``n``, ``method``.

    NOTE this module clamps the Box-Muller u1 at 1e-12 where the
    sibling initialisers do not; the clamp only matters on the
    measure-zero event u1 == 0 from the LCG, which the shared stream
    never actually produces, but the difference is stated so the
    conventions cannot be assumed identical.

    References
    ----------
    Géron Ch 18, Reparameterization Trick (VAE).

    Examples
    --------
    ``logvar = 0`` means ``sigma = 1``, so ``z = mu + eps``:

    >>> r = geron_reparameterization_trick([1.0, 2.0], [0.0, 0.0], eps=[0.5, -0.5])
    >>> r["z"]
    [1.5, 1.5]
    >>> r["sigma"]
    [1.0, 1.0]

    ``logvar = 2 log 3`` gives ``sigma = 3``, via the half:

    >>> import math
    >>> s = geron_reparameterization_trick([0.0], [2 * math.log(3)], eps=[1.0])
    >>> round(s["sigma"][0], 10)
    3.0
    >>> round(s["dz_dlogvar"][0], 10)
    1.5
    """
    M = np.asarray(mu, dtype=float)
    LV = np.asarray(logvar, dtype=float)
    if M.size == 0:
        raise ValueError("mu is empty.")
    if M.shape != LV.shape:
        raise ValueError(f"mu has shape {M.shape} but logvar has {LV.shape}.")
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(LV)):
        raise ValueError("mu and logvar must be finite.")

    sigma = np.exp(0.5 * LV)
    if eps is None:
        E = _standard_normals(M.size, seed).reshape(M.shape)
    else:
        E = np.asarray(eps, dtype=float)
        if E.shape != M.shape:
            raise ValueError(f"eps has shape {E.shape} but mu has {M.shape}.")
        if not np.all(np.isfinite(E)):
            raise ValueError("eps contains non-finite values.")
    z = M + sigma * E

    return RichResult(
        title="Reparameterization trick",
        summary_lines=[("Latent dims", int(M.size)),
                       ("Sample eps variance", float(np.var(E)))],
        payload={
            "z": z.tolist(),
            "sigma": sigma.tolist(),
            "eps": E.tolist(),
            "dz_dmu": np.ones_like(M).tolist(),
            "dz_dlogvar": (0.5 * sigma * E).tolist(),
            "sample_mean": float(np.mean(E)),
            "sample_variance": float(np.var(E)),
            "estimate": z.tolist(),
            "n": int(M.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrep: z = mu + exp(0.5*logvar)*eps; the 0.5 is load-bearing; noise off the gradient path"
