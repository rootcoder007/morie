# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""YaRN context-window scaling: NTK-by-parts interpolation + ramp +
attention temperature.

Peng, B., Quesnelle, J., Fan, H. and Shippole, E. (2023), "YaRN:
Efficient Context Window Extension of Large Language Models",
arXiv:2309.00071 (ICLR 2024). Implemented equations:

* Eq 17: the rotation count of RoPE dimension d,
  r(d) = L / lambda_d = L / (2 pi b^(2d/|D|)), where b is the RoPE
  base, |D| the head width and L the ORIGINAL context length.
* Eq 18: the ramp
  gamma(r) = 0 if r < alpha; 1 if r > beta; (r - alpha)/(beta - alpha)
  otherwise (alpha = 1, beta = 32 recommended for LLaMA).
* Eq 20 (NTK-by-parts): the blended frequency
  h(theta_d) = (1 - gamma(r(d))) theta_d / s + gamma(r(d)) theta_d,
  i.e. dimensions that complete many rotations inside L (high
  frequency, r > beta) are left alone (extrapolation) while dimensions
  with fewer than alpha rotations are fully position-interpolated by
  the scale factor s, with a linear ramp between.
* Eq 22 (attention temperature): sqrt(1/t) = 0.1 ln(s) + 1; the
  pre-softmax logits are multiplied by 1/t, implemented here by
  reporting t and the equivalent per-dimension logit scaling.

The related module kmyarn implements only the uniform "NTK-aware"
rescaling theta * s^(-2i/|D|) with a generic index ramp -- that is a
DIFFERENT interpolation (Section 3.1/3.2 background of the same
paper), so this is not aliased to it.

Note on the stub: its argument list (q, m, theta, s, beta_fast,
beta_slow) matched the reference implementation's naming, where
beta_fast = 32 and beta_slow = 1 are the rotation bounds; in the
paper's Eq 18 notation alpha = beta_slow and beta = beta_fast.

Source: fetched-wave3/peng-etal-2023-yarn-arxiv2309.00071.pdf
(Sections 3.2-3.4, Eqs 17, 18, 20, 22).
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["yarn", "yarn_context_scaling"]


def _gamma(r, alpha, beta):
    # Eq 18 ramp on the rotation count.
    if r < alpha:
        return 0.0
    if r > beta:
        return 1.0
    return (r - alpha) / (beta - alpha)


def yarn(base, s, d, L, beta_fast=32.0, beta_slow=1.0):
    """YaRN frequency blend + temperature (Peng et al. 2023).

    Parameters
    ----------
    base : float
        RoPE base b (e.g. 10000), or the d/2 frequencies directly.
    s : float
        Context extension scale factor (>= 1 extends).
    d : int
        Head embedding width |D| (even).
    L : float
        Original (pre-extension) context length in positions.
    beta_fast, beta_slow : float
        Rotation-count bounds: dimensions with more than `beta_fast`
        rotations inside L are extrapolated untouched, fewer than
        `beta_slow` fully interpolated (paper Eq 18 with
        alpha = beta_slow, beta = beta_fast; defaults 32 and 1).

    Returns
    -------
    result : RichResult
        Keys: theta (original frequencies), theta_new (Eq 20 blend),
        rotations (Eq 17 per dimension), gamma (Eq 18 per dimension),
        temperature (Eq 22 t), logit_scale (1/t), estimate, n, method.
    """
    d = int(d)
    if d < 2 or d % 2 != 0:
        raise ValueError(f"yarn: d must be a positive even width, got {d}")
    s = float(s)
    if s <= 0:
        raise ValueError(f"yarn: scale factor must be positive, got {s}")
    L = float(L)
    if L <= 0:
        raise ValueError(f"yarn: original context length must be positive, got {L}")
    alpha = float(beta_slow)
    beta = float(beta_fast)
    if not alpha < beta:
        raise ValueError(
            f"yarn: need beta_slow < beta_fast, got {beta_slow} and {beta_fast}")
    half = d // 2
    t = None if isinstance(base, (int, float)) else np.asarray(base, dtype=float)
    if t is None:
        b = float(base)
        if b <= 1.0:
            raise ValueError(f"yarn: a scalar base must exceed 1, got {b}")
        freqs = [b ** (-2.0 * i / d) for i in range(half)]
    else:
        freqs = [float(v) for v in t.ravel()]
        if len(freqs) != half:
            raise ValueError(
                f"yarn: need the d/2 = {half} frequencies, got {len(freqs)}")
        if any(v <= 0 for v in freqs):
            raise ValueError("yarn: frequencies must be positive")
    # Eq 17: lambda_d = 2 pi / theta_d, r(d) = L / lambda_d
    rot = [L * th / (2.0 * math.pi) for th in freqs]
    gam = [_gamma(r, alpha, beta) for r in rot]
    # Eq 20
    new = [(1.0 - g) * th / s + g * th for g, th in zip(gam, freqs)]
    # Eq 22
    sqrt_inv_t = 0.1 * math.log(s) + 1.0
    inv_t = sqrt_inv_t * sqrt_inv_t
    temperature = 1.0 / inv_t
    return RichResult(payload={
        "theta": freqs,
        "theta_new": new,
        "rotations": rot,
        "gamma": gam,
        "temperature": temperature,
        "logit_scale": inv_t,
        "scale": s,
        "estimate": float(new[-1]),
        "n": half,
        "method": "YaRN NTK-by-parts + ramp + temperature (Peng et al. 2023, Eqs 17/18/20/22)",
    })


def yarn_context_scaling(y=None, q=None, m=None, theta=None, s=None,
                         beta_fast=32.0, beta_slow=1.0, d=None, L=None):
    """Back-compatible wrapper over :func:`yarn` (old stub name).

    The stub's argument list carried unused placeholders (y, q, m);
    the meaningful ones map to yarn(theta, s, d, L, beta_fast,
    beta_slow). d defaults to twice the frequency count, L to 2048.
    """
    if theta is None or s is None:
        raise ValueError("yarn_context_scaling: theta and s are required")
    th = np.asarray(theta, dtype=float)
    if d is None:
        d = 2 * (th.ravel().shape[0] if th.ndim else 32)
    if L is None:
        L = 2048.0
    return yarn(theta, s, d, L, beta_fast=beta_fast, beta_slow=beta_slow)


def cheatsheet():
    return "yarn: YaRN NTK-by-parts + ramp + temperature (Peng et al. 2023, arXiv:2309.00071, Eqs 17/18/20/22)"
