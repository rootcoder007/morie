# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Denoising diffusion implicit models (DDIM) for faster sampling."""

from . import _array_core as np

from ._richresult import RichResult
from .grddim import geron_ddim_sampling_step
from .hmdfw import beta_schedule_values

__all__ = ["geron_ddim"]


def geron_ddim(x_T, model, T, n_steps, beta_schedule="linear", clip_x0=None):
    """
    Denoising diffusion implicit models (DDIM) for faster sampling.

    Formula: non-Markovian deterministic reverse:
    x_{t-1} = sqrt(a_{t-1})*x0_pred + sqrt(1 - a_{t-1})*eps

    Each step is DELEGATED to
    :func:`morie.fn.grddim.geron_ddim_sampling_step`, which implements the
    update above. What this module adds is the sub-sequence: DDIM's whole
    point is that the reverse chain need not visit every training
    timestep, so ``n_steps`` evenly spaced indices are taken out of ``T``
    and the model is called exactly ``n_steps`` times.

    The map is deterministic -- no noise is injected -- so the same
    ``x_T`` always gives the same sample, and the trajectory is a genuine
    ODE-like path rather than a stochastic chain.

    ``model`` must be a callable ``model(x_t, t) -> eps`` returning an
    array shaped like ``x_t``.

    Parameters
    ----------
    x_T : array-like
        Starting noise.
    model : callable
    T : int
        Length of the training schedule.
    n_steps : int
        Number of reverse steps actually taken, ``1 <= n_steps <= T``.
    beta_schedule : {"linear", "cosine"} or array-like, default "linear"
    clip_x0 : tuple, optional
        ``(lo, hi)`` clamp on the predicted ``x_0``.

    Returns
    -------
    result : RichResult
        Keys: x_0, trajectory, timesteps, x0_preds, model_calls,
        speedup, alpha_bar, estimate, n, method.

    Examples
    --------
    With a zero noise prediction the chain just rescales the sample from
    one signal level to the next, ending at ``x_T / sqrt(abar_T)``:

    >>> zero = lambda x, t: np.zeros_like(x)
    >>> r = geron_ddim([1.0], zero, T=4, n_steps=2, beta_schedule=[0.5, 0.5, 0.5, 0.5])
    >>> r["timesteps"]
    [4, 1]
    >>> r["model_calls"]
    2
    >>> round(float(r["x_0"][0]), 9)
    4.0

    Two steps instead of four is a 2x speedup for the same endpoint:

    >>> r2 = geron_ddim([1.0], zero, T=4, n_steps=4, beta_schedule=[0.5, 0.5, 0.5, 0.5])
    >>> round(float(r2["x_0"][0]), 9)
    4.0
    >>> r["speedup"], r2["speedup"]
    (2.0, 1.0)

    References
    ----------
    Géron Ch 18
    """
    x = np.atleast_1d(np.asarray(x_T, dtype=float))
    if x.size == 0:
        raise ValueError("geron_ddim: x_T is empty")
    if not callable(model):
        raise ValueError("geron_ddim: model must be a callable model(x_t, t) -> eps")
    Ti, K = int(T), int(n_steps)
    if Ti < 1:
        raise ValueError(f"geron_ddim: T must be >= 1, got {T!r}")
    if not (1 <= K <= Ti):
        raise ValueError(f"geron_ddim: n_steps must lie in 1..{Ti}, got {n_steps!r}")

    betas = beta_schedule_values(Ti, beta_schedule)
    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("geron_ddim: every beta must lie strictly in (0, 1)")
    # grddim indexes alpha_bar as abar[0] = 1 (the clean end) through abar[T].
    abar = np.concatenate([[1.0], np.cumprod(1.0 - betas)])

    steps = sorted({int(round(v)) for v in np.linspace(Ti, 1, K)}, reverse=True)
    seq = steps + [0]

    cur = x.copy()
    traj = [cur.copy()]
    x0s = []
    calls = 0
    for i in range(len(seq) - 1):
        t, t_prev = seq[i], seq[i + 1]
        eps = np.asarray(model(cur, t), dtype=float)
        calls += 1
        if eps.shape != cur.shape:
            raise ValueError(f"geron_ddim: model returned shape {eps.shape} at t={t} but x_t has shape {cur.shape}")
        if not np.all(np.isfinite(eps)):
            raise ValueError(f"geron_ddim: model returned non-finite values at t={t}")
        step = geron_ddim_sampling_step(cur, t=t, t_prev=t_prev, eps_pred=eps, alpha_bar=abar, clip_x0=clip_x0)
        x0s.append(step["x0_pred"])
        cur = np.atleast_1d(np.asarray(step["x_prev"], dtype=float))
        traj.append(cur.copy())

    return RichResult(
        title="DDIM sampling",
        summary_lines=[("Steps taken", len(steps)), ("Training steps", Ti), ("Speedup", float(Ti / len(steps)))],
        interpretation="DDIM is deterministic, so the same x_T always yields the same sample.",
        payload={
            "x_0": cur,
            "trajectory": [v.tolist() for v in traj],
            "timesteps": steps,
            "x0_preds": x0s,
            "model_calls": calls,
            "speedup": float(Ti / len(steps)),
            "alpha_bar": abar.tolist(),
            "T": Ti,
            "n_steps": int(len(steps)),
            "estimate": float(np.mean(cur)),
            "n": int(x.size),
            "method": "DDIM sub-sequence sampling; each step delegated to grddim",
        },
    )


def cheatsheet():
    return "hmddim: Denoising diffusion implicit models (DDIM) for faster sampling"
