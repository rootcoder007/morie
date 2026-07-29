# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DDIM deterministic sampling step (subset schedule, eta=0)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ddim_sampling_step"]

_METHOD = "DDIM deterministic sampling step"


def geron_ddim_sampling_step(x_t, t, t_prev, eps_pred, alpha_bar, clip_x0=None):
    r"""One deterministic DDIM step from timestep ``t`` to ``t_prev``.

    The predicted clean sample is read straight out of the noise
    prediction,

    .. math::
        \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,
        \varepsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}},

    and is then re-noised to the earlier level,

    .. math::
        x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0
                + \sqrt{1-\bar\alpha_{t-1}}\,\varepsilon_\theta(x_t, t).

    With :math:`\eta = 0` no fresh noise enters, so the map from latent
    to image is deterministic and ``t_prev`` may skip arbitrarily far --
    which is exactly how DDIM samples in 50 steps what DDPM needs 1000
    for.

    Parameters
    ----------
    x_t : array-like
        Current noisy sample.
    t, t_prev : int
        Indices into ``alpha_bar``; ``t_prev < t`` (denoising runs
        backwards).
    eps_pred : array-like
        Network's noise prediction at ``t``, same shape as ``x_t``.
    alpha_bar : array-like
        Cumulative product schedule :math:`\bar\alpha_0 \dots
        \bar\alpha_T`, non-increasing, all in ``(0, 1]``.
    clip_x0 : tuple, optional
        ``(lo, hi)`` to clamp :math:`\hat x_0` before re-noising.

    Returns
    -------
    RichResult
        Payload keys ``x_prev``, ``x0_pred``, ``alpha_bar_t``,
        ``alpha_bar_prev``, ``signal_scale``, ``noise_scale``,
        ``estimate`` (mean of ``x_prev``), ``n``, ``method``.

    References
    ----------
    Géron Ch 18, DDIM sampling section.

    Examples
    --------
    With a zero noise prediction the step just rescales the sample from
    one signal level to the next:

    >>> ab = [1.0, 0.8, 0.5]
    >>> r = geron_ddim_sampling_step([1.0], t=2, t_prev=1, eps_pred=[0.0],
    ...                             alpha_bar=ab)
    >>> round(r["x0_pred"][0], 6)
    1.414214
    >>> round(r["x_prev"][0], 6)
    1.264911

    Stepping to ``t_prev = 0`` returns the clean prediction itself:

    >>> r2 = geron_ddim_sampling_step([1.0], t=2, t_prev=0, eps_pred=[0.0],
    ...                              alpha_bar=ab)
    >>> round(r2["x_prev"][0], 6) == round(r2["x0_pred"][0], 6)
    True
    """
    x_t = np.asarray(x_t, dtype=float)
    eps = np.asarray(eps_pred, dtype=float)
    ab = np.asarray(alpha_bar, dtype=float).ravel()
    if x_t.shape != eps.shape:
        raise ValueError(f"eps_pred shape {eps.shape} must match x_t shape {x_t.shape}.")
    if x_t.size == 0:
        raise ValueError("x_t is empty.")
    if not np.all(np.isfinite(x_t)) or not np.all(np.isfinite(eps)):
        raise ValueError("x_t and eps_pred must be finite.")
    if ab.size < 2:
        raise ValueError(f"alpha_bar needs at least 2 entries, got {ab.size}.")
    if np.any(ab <= 0) or np.any(ab > 1):
        raise ValueError("alpha_bar entries must lie in (0, 1].")
    if np.any(np.diff(ab) > 1e-12):
        raise ValueError(
            "alpha_bar must be non-increasing in t (it is a cumulative product "
            "of alphas < 1)."
        )
    t = int(t)
    t_prev = int(t_prev)
    for name, v in (("t", t), ("t_prev", t_prev)):
        if not (0 <= v < ab.size):
            raise ValueError(f"{name}={v} out of range for alpha_bar of length {ab.size}.")
    if t_prev >= t:
        raise ValueError(
            f"denoising runs backwards, so t_prev ({t_prev}) must be strictly "
            f"below t ({t})."
        )

    ab_t = float(ab[t])
    ab_p = float(ab[t_prev])
    x0 = (x_t - np.sqrt(1.0 - ab_t) * eps) / np.sqrt(ab_t)
    if clip_x0 is not None:
        lo, hi = (float(v) for v in clip_x0)
        if lo >= hi:
            raise ValueError(f"clip_x0 must be (lo, hi) with lo < hi, got {clip_x0!r}.")
        x0 = np.clip(x0, lo, hi)
    sig = float(np.sqrt(ab_p))
    noi = float(np.sqrt(1.0 - ab_p))
    x_prev = sig * x0 + noi * eps

    return RichResult(
        title="DDIM sampling step",
        summary_lines=[("t -> t_prev", f"{t} -> {t_prev}"),
                       ("Signal scale", sig), ("Noise scale", noi)],
        payload={
            "x_prev": x_prev.tolist(),
            "x0_pred": x0.tolist(),
            "alpha_bar_t": ab_t,
            "alpha_bar_prev": ab_p,
            "signal_scale": sig,
            "noise_scale": noi,
            "t": t,
            "t_prev": t_prev,
            "estimate": float(np.mean(x_prev)),
            "n": int(x_t.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grddim: DDIM step -- x0_pred from eps, then re-noise to t_prev; deterministic (eta=0)"
