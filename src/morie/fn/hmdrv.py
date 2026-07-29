# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diffusion reverse process denoises from x_T back to x_0."""

import numpy as np

from ._richresult import RichResult
from .hmdfw import beta_schedule_values, lcg_normal

__all__ = ["geron_diffusion_reverse"]


def geron_diffusion_reverse(x_T, model, T, beta_schedule="linear", seed=0, clip_x0=None):
    """
    Diffusion reverse process denoises from x_T back to x_0.

    Formula: p_theta(x_{t-1} | x_t) = N(mu_theta(x_t,t), Sigma_theta)

    Ancestral (DDPM) sampling. At each step the model's noise prediction
    is turned into the posterior mean

        mu = (x_t - beta_t/sqrt(1 - abar_t) * eps) / sqrt(alpha_t)

    and a draw is taken with variance ``beta_t``, except at ``t = 1``
    where the chain is deterministic -- adding noise to the last step is
    the classic off-by-one that leaves visible grain in the output.

    ``model`` must be a callable ``model(x_t, t) -> eps`` returning an
    array shaped like ``x_t``; the contract is enforced on every call, so
    a model that returns a scalar or the wrong shape raises rather than
    broadcasting silently.

    The schedule comes from :mod:`morie.fn.hmdfw` so the forward and
    reverse processes cannot drift apart.

    Parameters
    ----------
    x_T : array-like
        Starting noise.
    model : callable
        ``model(x_t, t) -> eps`` with ``t`` a 1-based step index.
    T : int
        Number of steps, >= 1.
    beta_schedule : {"linear", "cosine"} or array-like, default "linear"
    seed : int, default 0
    clip_x0 : tuple, optional
        ``(lo, hi)`` clamp applied to the implied ``x_0`` each step.

    Returns
    -------
    result : RichResult
        Keys: x_0, trajectory, means, betas, alpha_bar, n_steps,
        model_calls, estimate, n, method.

    Examples
    --------
    A model that predicts zero noise just rescales the sample by
    ``1/sqrt(alpha_t)`` at every step; with beta = 0.75 (alpha = 0.25) one
    step doubles it:

    >>> zero = lambda x, t: np.zeros_like(x)
    >>> r = geron_diffusion_reverse([1.0], zero, T=1, beta_schedule=[0.75])
    >>> [round(float(v), 12) for v in r["x_0"]]
    [2.0]
    >>> r["model_calls"]
    1

    A model that predicts exactly the noise present recovers the clean
    signal: with abar = 0.25, x_1 = 0.5*x0 + sqrt(0.75)*eps, so feeding
    back eps returns x0.

    >>> import math
    >>> eps = 1.0
    >>> x1 = 0.5 * 3.0 + math.sqrt(0.75) * eps
    >>> r2 = geron_diffusion_reverse([x1], lambda x, t: np.full_like(x, eps),
    ...                              T=1, beta_schedule=[0.75])
    >>> round(float(r2["x_0"][0]), 9)
    3.0

    References
    ----------
    Géron Ch 18
    """
    x = np.atleast_1d(np.asarray(x_T, dtype=float))
    if x.size == 0:
        raise ValueError("geron_diffusion_reverse: x_T is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("geron_diffusion_reverse: x_T contains non-finite values")
    if not callable(model):
        raise ValueError("geron_diffusion_reverse: model must be a callable model(x_t, t) -> eps")
    Ti = int(T)
    if Ti < 1:
        raise ValueError(f"geron_diffusion_reverse: T must be >= 1, got {T!r}")
    betas = beta_schedule_values(Ti, beta_schedule)
    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("geron_diffusion_reverse: every beta must lie strictly in (0, 1)")
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)

    cur = x.copy()
    traj = [cur.copy()]
    means = []
    calls = 0
    for t in range(Ti, 0, -1):
        eps = np.asarray(model(cur, t), dtype=float)
        calls += 1
        if eps.shape != cur.shape:
            raise ValueError(
                f"geron_diffusion_reverse: model returned shape {eps.shape} at t={t} but x_t has shape {cur.shape}"
            )
        if not np.all(np.isfinite(eps)):
            raise ValueError(f"geron_diffusion_reverse: model returned non-finite values at t={t}")
        b, a, ab = float(betas[t - 1]), float(alphas[t - 1]), float(abar[t - 1])
        if clip_x0 is not None:
            lo, hi = (float(v) for v in clip_x0)
            if lo >= hi:
                raise ValueError(f"geron_diffusion_reverse: clip_x0 must be (lo, hi) with lo < hi, got {clip_x0!r}")
            x0_hat = np.clip((cur - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab), lo, hi)
            ab_prev = float(abar[t - 2]) if t > 1 else 1.0
            coef0 = np.sqrt(ab_prev) * b / (1.0 - ab)
            coef_t = np.sqrt(a) * (1.0 - ab_prev) / (1.0 - ab)
            mu = coef0 * x0_hat + coef_t * cur
        else:
            mu = (cur - (b / np.sqrt(1.0 - ab)) * eps) / np.sqrt(a)
        means.append(mu.tolist())
        if t > 1:
            z = lcg_normal(cur.shape, seed + t)
            cur = mu + np.sqrt(b) * z
        else:
            cur = mu
        traj.append(cur.copy())

    return RichResult(
        title="Diffusion reverse process",
        summary_lines=[("Steps", Ti), ("Model calls", calls)],
        interpretation="The final step is noise-free by construction; adding noise there is the classic grainy-output bug.",
        payload={
            "x_0": cur,
            "trajectory": [v.tolist() for v in traj],
            "means": means,
            "betas": betas.tolist(),
            "alphas": alphas.tolist(),
            "alpha_bar": abar.tolist(),
            "n_steps": Ti,
            "model_calls": calls,
            "estimate": float(np.mean(cur)),
            "n": int(x.size),
            "method": "ancestral DDPM reverse sampling with an enforced model(x_t, t) -> eps contract",
        },
    )


def cheatsheet():
    return "hmdrv: Diffusion reverse process denoises from x_T back to x_0"
