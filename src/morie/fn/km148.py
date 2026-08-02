# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.20: the conditional latent-diffusion generation loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_ldm_loss"]


def kamath_ch9_ldm_loss(epsilon, z_t, H_X, eps_net=None, t=None):
    r"""L_X-gen = E_{eps,t} || eps - eps_X(z_t, t, H_X) ||_2^2.

    ``eps_net`` is the caller's denoising U-Net, a callable
    ``eps_net(z_t, t, H_X) -> predicted noise``; alternatively pass the
    prediction itself as ``eps_net`` (an array of the same shape as
    ``epsilon``). The expectation is the mean over the batch of the
    SQUARED L2 norm per sample, as the ||.||_2^2 in the equation says.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.20, printed
    p. 398.

    Examples
    --------
    >>> out = kamath_ch9_ldm_loss([[1.0, 0.0]], [[0.0, 0.0]], [[0.0]],
    ...                           eps_net=lambda z, tt, h: [[0.0, 0.0]])
    >>> out["estimate"]
    1.0
    """
    if eps_net is None:
        raise ValueError("eps_net= is required: the denoising network "
                         "eps_X(z_t, t, H_X), or its prediction.")
    pred = np.asarray(eps_net(z_t, t, H_X) if callable(eps_net)
                      else eps_net, dtype=float)
    eps = np.asarray(epsilon, dtype=float)
    if eps.shape != pred.shape:
        raise ValueError(
            f"the noise is {eps.shape} but the prediction is "
            f"{pred.shape}.")
    if eps.size == 0:
        raise ValueError("the noise array is empty.")
    if not np.all(np.isfinite(pred)):
        raise ValueError("the denoising network returned non-finite "
                         "values.")
    d = (eps - pred).reshape(eps.shape[0], -1) if eps.ndim > 1 \
        else (eps - pred).reshape(1, -1)
    per = (d ** 2).sum(axis=1)
    return RichResult(payload={
        "estimate": float(per.mean()),
        "per_sample": [float(v) for v in per], "n": int(per.size),
        "method": "conditional LDM noise-prediction loss "
                  "(Kamath Eq 9.20)"})


def cheatsheet():
    return "km148: mean squared L2 error between true and predicted noise"
