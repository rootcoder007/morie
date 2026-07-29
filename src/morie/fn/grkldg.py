# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-form KL divergence of a diagonal Gaussian from N(0, I)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_kl_divergence_gaussian"]

_METHOD = "KL(N(mu, sigma^2) || N(0, I)), closed form"


def geron_kl_divergence_gaussian(mu, logvar):
    r"""The VAE's regularisation term.

    .. math::
        \mathrm{KL} = -\tfrac12 \sum_i \bigl(
        1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2 \bigr)

    Parameterising by ``logvar`` rather than ``sigma`` is what keeps
    this stable: the network can output any real number and the
    variance is positive by construction, with no clamping and no
    ``log 0``.

    The term is 0 exactly when :math:`\mu = 0` and
    :math:`\sigma^2 = 1`, and positive everywhere else -- it is a
    divergence, so a negative result would be a bug and is rejected
    here.  Its job in the VAE is to keep the latent codes overlapping
    enough that the space between them decodes to something.

    Parameters
    ----------
    mu : array-like, shape (d,) or (m, d)
        Encoder means.
    logvar : array-like, same shape as ``mu``
        Encoder log-variances.

    Returns
    -------
    RichResult
        Payload keys ``kl``, ``per_dimension``, ``per_sample``,
        ``variance``, ``n_active_dims`` (dimensions carrying more than
        0.01 nats -- the rest have collapsed to the prior),
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 18, VAE KL term section (Kingma and Welling 2014).

    Examples
    --------
    At the prior the divergence is exactly zero:

    >>> geron_kl_divergence_gaussian([0.0, 0.0], [0.0, 0.0])["kl"]
    0.0

    A unit-variance Gaussian displaced by 1 costs ``mu^2 / 2``:

    >>> r = geron_kl_divergence_gaussian([1.0], [0.0])
    >>> r["kl"]
    0.5

    Halving the variance (``logvar = -log 2``) costs the log term:
    ``-0.5 * (1 - log 2 - 0.5)``:

    >>> import math
    >>> r2 = geron_kl_divergence_gaussian([0.0], [-math.log(2.0)])
    >>> round(r2["kl"], 10)
    0.0965735903
    >>> round(r2["variance"][0], 10)
    0.5
    """
    m_arr = np.asarray(mu, dtype=float)
    lv = np.asarray(logvar, dtype=float)
    if m_arr.shape != lv.shape:
        raise ValueError(f"mu {m_arr.shape} and logvar {lv.shape} must have the same shape.")
    if m_arr.size == 0:
        raise ValueError("mu is empty.")
    if not np.all(np.isfinite(m_arr)) or not np.all(np.isfinite(lv)):
        raise ValueError("mu and logvar must be finite.")
    if np.any(lv > 80.0):
        raise ValueError(
            f"logvar up to {lv.max()} would overflow exp(); the encoder has diverged."
        )

    var = np.exp(lv)
    per_dim = -0.5 * (1.0 + lv - m_arr**2 - var)
    A = np.atleast_2d(per_dim)
    per_sample = A.sum(axis=1)
    total = float(per_sample.sum())
    if total < -1e-9:
        raise ValueError(f"KL came out negative ({total}); a divergence cannot be, so this is a bug.")
    total = float(max(total, 0.0))

    return RichResult(
        title="Gaussian KL to N(0, I)",
        summary_lines=[("KL", total), ("Dims", int(A.shape[1]))],
        payload={
            "kl": total,
            "per_dimension": per_dim.tolist(),
            "per_sample": per_sample.tolist(),
            "variance": var.tolist(),
            "n_active_dims": int(np.sum(A.mean(axis=0) > 0.01)),
            "estimate": total,
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grkldg: KL = -0.5 sum(1 + logvar - mu^2 - exp(logvar)); zero exactly at the prior"
