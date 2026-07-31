"""Shared kriging primitives (Schabenberger & Gotway 2005, Ch. 5)."""

import numpy as np

from ._schab_vario import correlogram

__all__ = []


def cov_from_model(h, cov_model=None):
    """C(h) from a nugget / sill / range / model spec.

    C(h) = sill * R(h) for h > 0, and C(0) = nugget + sill: the nugget is
    a variance component present only at zero lag (Sec. 4.3.6).
    """
    cm = dict(cov_model or {})
    nugget = float(cm.get("nugget", 0.0))
    sill = float(cm.get("sill", 1.0))
    rng = float(cm.get("range", cm.get("range_", 1.0)))
    model = cm.get("model", "exponential")
    h = np.asarray(h, dtype=float)
    c = sill * correlogram(h.ravel(), rng, model).reshape(h.shape)
    return np.where(h == 0, nugget + sill, c)


def _dist(a, b):
    return np.linalg.norm(np.atleast_2d(a)[:, None, :]
                          - np.atleast_2d(b)[None, :, :], axis=-1)


def simple_kriging(coords, z, target, cov_model=None, mu=None):
    """Simple kriging, Sec. 5.2.1.

    lambda = Sigma^-1 sigma, and with the mean known,

        p_sk(Z; s0) = mu(s0) + sigma' Sigma^-1 (Z(s) - mu(s))     (5.10)
        sigma^2_sk(s0) = sigma^2 - sigma' Sigma^-1 sigma          (5.11)

    Returns (prediction, variance, weights).
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    target = np.atleast_2d(np.asarray(target, dtype=float))
    if coords.shape[0] != z.size:
        raise ValueError("`coords` and `z` must have the same number of rows")
    mu = float(np.mean(z)) if mu is None else float(mu)

    Sigma = cov_from_model(_dist(coords, coords), cov_model)
    sig = cov_from_model(_dist(coords, target), cov_model)      # (n, m)
    sigma2 = float(cov_from_model(np.zeros(1), cov_model)[0])

    lam = np.linalg.solve(Sigma, sig)                            # (n, m)
    pred = mu + lam.T @ (z - mu)
    var = sigma2 - np.einsum("ij,ij->j", sig, lam)
    return pred, np.maximum(var, 0.0), lam
