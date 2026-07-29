# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DDPM reverse (denoising) step."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ddpm_reverse_step"]

_METHOD = "DDPM reverse denoising step"


def _lcg_normals(count, seed):
    """Standard normals from the reference LCG via Box-Muller."""
    s = int(seed) % 2**32
    n_pairs = (count + 1) // 2
    out = np.empty(2 * n_pairs, dtype=float)
    for i in range(n_pairs):
        s = (1664525 * s + 1013904223) % 2**32
        u1 = (s + 0.5) / 2**32
        s = (1664525 * s + 1013904223) % 2**32
        u2 = (s + 0.5) / 2**32
        rad = np.sqrt(-2.0 * np.log(u1))
        out[2 * i] = rad * np.cos(2.0 * np.pi * u2)
        out[2 * i + 1] = rad * np.sin(2.0 * np.pi * u2)
    return out[:count]


def geron_ddpm_reverse_step(x_t, t, eps_pred, alpha, alpha_bar, sigma, z=None, seed=0):
    r"""One step back down the chain.

    .. math::
        x_{t-1} = \frac{1}{\sqrt{\alpha_t}}
        \Bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,
        \varepsilon_\theta(x_t, t)\Bigr) + \sigma_t z

    The bracket subtracts the model's estimate of the noise and the
    division rescales what is left; the ``sigma * z`` term puts fresh
    noise *back*.  That last part is what makes DDPM a sampler rather
    than a denoiser -- without it every run from the same ``x_T`` would
    give the same image.  Setting ``sigma = 0`` is exactly the
    deterministic DDIM step (:mod:`morie.fn.grddim`).

    At the final step ``sigma`` should be 0: there is no more noise to
    add to a finished sample.

    Parameters
    ----------
    x_t : array-like
        Current noisy sample.
    t : int
        Timestep, indexing ``alpha`` and ``alpha_bar``.
    eps_pred : array-like, same shape as ``x_t``
    alpha : array-like
        Per-step :math:`\alpha_t \in (0, 1]`.
    alpha_bar : array-like
        Cumulative products.
    sigma : float
        Non-negative noise scale for this step.
    z : array-like, optional
        Supply the fresh noise; otherwise drawn from the reference LCG.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``x_prev``, ``mean`` (before the noise term),
        ``noise_term``, ``eps_coef``, ``x0_estimate`` (the implied clean
        sample), ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 18, DDPM reverse process section (Ho et al. 2020).

    Examples
    --------
    With a zero noise prediction and ``sigma = 0`` the step is a pure
    rescale by :math:`1/\sqrt{\alpha_t}`:

    >>> r = geron_ddpm_reverse_step([1.0], 0, [0.0], [0.9], [0.9], sigma=0.0)
    >>> round(r["x_prev"][0], 10)
    1.0540925534
    >>> r["noise_term"]
    [0.0]

    A non-zero prediction is subtracted first, scaled by
    :math:`(1-\alpha)/\sqrt{1-\bar\alpha}`:

    >>> r2 = geron_ddpm_reverse_step([1.0], 0, [1.0], [0.9], [0.9], sigma=0.0)
    >>> round(r2["eps_coef"], 10)
    0.316227766
    >>> round(r2["x_prev"][0], 10)
    0.7207592201
    """
    X = np.asarray(x_t, dtype=float)
    E = np.asarray(eps_pred, dtype=float)
    if E.shape != X.shape:
        raise ValueError(f"eps_pred shape {E.shape} != x_t shape {X.shape}.")
    if X.size == 0:
        raise ValueError("x_t is empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(E)):
        raise ValueError("x_t and eps_pred must be finite.")
    a = np.asarray(alpha, dtype=float).ravel()
    ab = np.asarray(alpha_bar, dtype=float).ravel()
    t = int(t)
    if not (0 <= t < a.size) or not (0 <= t < ab.size):
        raise ValueError(f"t = {t} must index both alpha ({a.size}) and alpha_bar ({ab.size}).")
    at, abt = float(a[t]), float(ab[t])
    if not (0.0 < at <= 1.0):
        raise ValueError(f"alpha[{t}] must lie in (0, 1], got {at}.")
    if not (0.0 <= abt < 1.0):
        raise ValueError(
            f"alpha_bar[{t}] must lie in [0, 1); at exactly 1 the noise coefficient "
            f"divides by zero. Got {abt}."
        )
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be non-negative and finite, got {sigma}.")

    coef = (1.0 - at) / np.sqrt(1.0 - abt)
    mean = (X - coef * E) / np.sqrt(at)
    if sigma == 0:
        zz = np.zeros_like(X)
    elif z is None:
        zz = _lcg_normals(X.size, seed).reshape(X.shape)
    else:
        zz = np.asarray(z, dtype=float)
        if zz.shape != X.shape:
            raise ValueError(f"z shape {zz.shape} != x_t shape {X.shape}.")
        if not np.all(np.isfinite(zz)):
            raise ValueError("z must be finite.")
    noise_term = sigma * zz
    x_prev = mean + noise_term
    x0 = (X - np.sqrt(1.0 - abt) * E) / np.sqrt(abt) if abt > 0 else np.full_like(X, np.nan)

    return RichResult(
        title="DDPM reverse step",
        summary_lines=[("t", t), ("alpha_t", at), ("sigma", sigma)],
        payload={
            "x_prev": x_prev.tolist(),
            "mean": mean.tolist(),
            "noise_term": noise_term.tolist(),
            "eps_coef": float(coef),
            "x0_estimate": x0.tolist(),
            "alpha_t": at,
            "alpha_bar_t": abt,
            "sigma": sigma,
            "estimate": x_prev.tolist(),
            "n": int(X.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdpmr: x_{t-1} = (x_t - (1-a)/sqrt(1-ab) eps)/sqrt(a) + sigma z; sigma=0 is DDIM"
