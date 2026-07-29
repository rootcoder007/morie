# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DDPM forward (noising) process."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_ddpm_forward_process"]

_METHOD = "DDPM forward diffusion q(x_t | x_0)"


def _lcg_normals(count, seed):
    """Standard normals from the reference LCG via Box-Muller.

    ``s = (1664525 s + 1013904223) mod 2**32``, ``u = (s + 0.5)/2**32``.
    """
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


def geron_ddpm_forward_process(x0, t, alpha_bar, noise=None, seed=0):
    r"""Jump straight to step ``t`` of the noising chain.

    .. math::
        x_t = \sqrt{\bar\alpha_t}\, x_0
        + \sqrt{1 - \bar\alpha_t}\, \varepsilon,
        \qquad \varepsilon \sim \mathcal N(0, I)

    The closed form is what makes DDPM trainable.  The forward process
    is a thousand tiny Gaussian steps, but their composition is itself
    Gaussian, so any ``t`` can be sampled in one shot -- no chain to
    run, and training can pick ``t`` uniformly at random.

    The coefficients are chosen so the variance is preserved: with
    :math:`\mathrm{var}(x_0) = 1`, :math:`\bar\alpha + (1-\bar\alpha) =
    1` keeps :math:`\mathrm{var}(x_t) = 1` at every step.

    Parameters
    ----------
    x0 : array-like
        Clean data.
    t : int
        Timestep index into ``alpha_bar``.
    alpha_bar : array-like
        Cumulative products :math:`\bar\alpha`, each in ``[0, 1]`` and
        non-increasing.
    noise : array-like, optional
        Supply :math:`\varepsilon` to make the draw exact; otherwise it
        comes from the deterministic LCG above.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``x_t``, ``noise``, ``signal_coef``,
        ``noise_coef``, ``snr`` (:math:`\bar\alpha/(1-\bar\alpha)`),
        ``alpha_bar_t``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 18, DDPM forward diffusion section (Ho et al. 2020).

    Examples
    --------
    At :math:`\bar\alpha = 1` no noise is added at all -- the start of
    the chain:

    >>> r = geron_ddpm_forward_process([2.0, -1.0], 0, [1.0, 0.25])
    >>> r["x_t"]
    [2.0, -1.0]
    >>> r["noise_coef"]
    0.0

    At :math:`\bar\alpha = 0.25` the signal is halved and
    :math:`\sqrt{0.75}` of noise is mixed in:

    >>> r2 = geron_ddpm_forward_process([2.0], 1, [1.0, 0.25], noise=[1.0])
    >>> round(r2["x_t"][0], 10)
    1.8660254038
    >>> round(r2["snr"], 10)
    0.3333333333
    """
    X = np.asarray(x0, dtype=float)
    if X.size == 0:
        raise ValueError("x0 is empty.")
    if not np.all(np.isfinite(X)):
        raise ValueError("x0 must be finite.")
    ab = np.asarray(alpha_bar, dtype=float).ravel()
    if ab.size == 0:
        raise ValueError("alpha_bar is empty.")
    if np.any(ab < 0) or np.any(ab > 1):
        raise ValueError("alpha_bar entries are cumulative products and must lie in [0, 1].")
    if np.any(np.diff(ab) > 1e-12):
        raise ValueError("alpha_bar must be non-increasing; it is a cumulative product of alphas <= 1.")
    t = int(t)
    if not (0 <= t < ab.size):
        raise ValueError(f"t must index alpha_bar, i.e. lie in [0, {ab.size - 1}], got {t}.")

    a = float(ab[t])
    if noise is None:
        eps = _lcg_normals(X.size, seed).reshape(X.shape)
    else:
        eps = np.asarray(noise, dtype=float)
        if eps.shape != X.shape:
            raise ValueError(f"noise shape {eps.shape} != x0 shape {X.shape}.")
        if not np.all(np.isfinite(eps)):
            raise ValueError("noise must be finite.")

    sc = float(np.sqrt(a))
    nc = float(np.sqrt(1.0 - a))
    xt = sc * X + nc * eps
    snr = float("inf") if a == 1.0 else a / (1.0 - a)

    return RichResult(
        title="DDPM forward process",
        summary_lines=[("t", t), ("alpha_bar_t", a), ("SNR", snr)],
        payload={
            "x_t": xt.tolist(),
            "noise": eps.tolist(),
            "signal_coef": sc,
            "noise_coef": nc,
            "snr": snr,
            "alpha_bar_t": a,
            "t": t,
            "estimate": xt.tolist(),
            "n": int(X.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdpmf: x_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps -- any t in one shot, variance preserved"
