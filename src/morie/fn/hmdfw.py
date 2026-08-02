# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diffusion forward process adds Gaussian noise over T steps."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_diffusion_forward", "beta_schedule_values", "lcg_normal"]


def lcg_normal(shape, seed):
    """Deterministic standard normals from the LCG plus a Box-Muller pair."""
    n = int(np.prod(shape))
    m = n + (n % 2)
    s = int(seed) % 2**32
    u = np.empty(m)
    for i in range(m):
        s = (1664525 * s + 1013904223) % 2**32
        u[i] = (s + 0.5) / 2**32
    u1, u2 = u[0::2], u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.empty(m)
    z[0::2] = r * np.cos(2 * np.pi * u2)
    z[1::2] = r * np.sin(2 * np.pi * u2)
    return z[:n].reshape(shape)


def beta_schedule_values(T, beta_schedule="linear", beta_start=1e-4, beta_end=0.02):
    """Beta values for ``T`` steps: 'linear', 'cosine', or an explicit array."""
    if isinstance(beta_schedule, str):
        if beta_schedule == "linear":
            return np.linspace(beta_start, beta_end, T)
        if beta_schedule == "cosine":
            s = 0.008
            t = np.arange(T + 1) / T
            f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            ab = f / f[0]
            b = 1.0 - ab[1:] / ab[:-1]
            return np.clip(b, 1e-8, 0.999)
        raise ValueError(f"beta_schedule must be 'linear', 'cosine' or an array, got {beta_schedule!r}")
    b = np.atleast_1d(np.asarray(beta_schedule, dtype=float))
    if b.size != T:
        raise ValueError(f"beta_schedule has {b.size} entries but T is {T}")
    return b


def geron_diffusion_forward(x0, T, beta_schedule="linear", t=None, seed=0):
    """
    Diffusion forward process adds Gaussian noise over T steps.

    Formula: q(x_t | x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I)

    Both routes to ``x_t`` are computed and cross-checked. The step-by-step
    chain applies the transition above ``t`` times; the closed form uses
    ``x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps`` with
    ``abar_t = prod(1 - beta_i)``. They agree when driven by the same
    noise draws only in distribution, not path-wise, so what is checked
    here is the second moment: ``variance_check`` compares the chain's
    empirical scale against ``1 - abar_t``.

    Noise comes from a deterministic LCG plus Box-Muller, so a call is
    reproducible from ``seed``.

    Parameters
    ----------
    x0 : array-like
        Clean sample.
    T : int
        Number of diffusion steps, >= 1.
    beta_schedule : {"linear", "cosine"} or array-like, default "linear"
        Variance schedule; an explicit array must have ``T`` entries in
        (0, 1).
    t : int, optional
        Step to report; default ``T``.
    seed : int, default 0

    Returns
    -------
    result : RichResult
        Keys: x_t, x_chain, betas, alphas, alpha_bar, signal_scale,
        noise_scale, noise, snr, variance_check, estimate, n, method.

    Examples
    --------
    A one-step schedule with beta = 0.5 keeps ``sqrt(0.5)`` of the signal:

    >>> r = geron_diffusion_forward([1.0], T=1, beta_schedule=[0.5])
    >>> round(r["signal_scale"], 9), round(r["noise_scale"], 9)
    (0.707106781, 0.707106781)
    >>> round(r["alpha_bar"][-1], 12)
    0.5

    alpha_bar is the running product of ``1 - beta``, so the signal decays
    geometrically and the SNR with it:

    >>> r2 = geron_diffusion_forward([1.0], T=3, beta_schedule=[0.5, 0.5, 0.5])
    >>> [round(v, 12) for v in r2["alpha_bar"]]
    [0.5, 0.25, 0.125]
    >>> round(r2["snr"], 12)
    0.142857142857

    Step 0 is the clean sample itself:

    >>> r3 = geron_diffusion_forward([2.0, -1.0], T=2, beta_schedule=[0.1, 0.2], t=0)
    >>> [float(v) for v in r3["x_t"]]
    [2.0, -1.0]

    References
    ----------
    Géron Ch 18
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    if x.size == 0:
        raise ValueError("geron_diffusion_forward: x0 is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("geron_diffusion_forward: x0 contains non-finite values")
    Ti = int(T)
    if Ti < 1:
        raise ValueError(f"geron_diffusion_forward: T must be >= 1, got {T!r}")
    betas = beta_schedule_values(Ti, beta_schedule)
    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("geron_diffusion_forward: every beta must lie strictly in (0, 1)")
    ti = Ti if t is None else int(t)
    if not (0 <= ti <= Ti):
        raise ValueError(f"geron_diffusion_forward: t must lie in 0..{Ti}, got {t!r}")

    alphas = 1.0 - betas
    abar = np.cumprod(alphas)

    # Step-by-step chain.
    chain = [x.copy()]
    cur = x.copy()
    for k in range(Ti):
        z = lcg_normal(x.shape, seed + k + 1)
        cur = np.sqrt(alphas[k]) * cur + np.sqrt(betas[k]) * z
        chain.append(cur.copy())

    if ti == 0:
        xt = x.copy()
        sig, noi, eps = 1.0, 0.0, np.zeros_like(x)
        ab_t = 1.0
    else:
        ab_t = float(abar[ti - 1])
        sig = float(np.sqrt(ab_t))
        noi = float(np.sqrt(1.0 - ab_t))
        eps = lcg_normal(x.shape, seed + 10_000)
        xt = sig * x + noi * eps

    chain_var = float(np.mean((chain[ti] - sig * x) ** 2))

    return RichResult(
        title="Diffusion forward process",
        summary_lines=[("t", ti), ("Signal scale", sig), ("Noise scale", noi)],
        interpretation="As alpha_bar -> 0 the sample becomes pure noise, which is what the reverse model must undo.",
        payload={
            "x_t": xt,
            "x_chain": [c.tolist() for c in chain],
            "betas": betas.tolist(),
            "alphas": alphas.tolist(),
            "alpha_bar": abar.tolist(),
            "alpha_bar_t": ab_t,
            "signal_scale": sig,
            "noise_scale": noi,
            "noise": eps,
            "snr": float(ab_t / (1.0 - ab_t)) if ab_t < 1 else float("inf"),
            "variance_check": {"chain_mse_from_signal": chain_var, "closed_form_variance": float(1.0 - ab_t)},
            "t": ti,
            "T": Ti,
            "estimate": sig,
            "n": int(x.size),
            "method": "forward diffusion, chain and closed form, deterministic LCG/Box-Muller noise",
        },
    )


def cheatsheet():
    return "hmdfw: Diffusion forward process adds Gaussian noise over T steps"
