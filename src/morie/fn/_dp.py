# morie.fn -- shared helpers (rootcoder007/morie)
"""Shared plumbing for the differential-privacy mechanisms.

Every mechanism here is calibrated to a *sensitivity*: how much the output can
move when one record changes. Getting that number wrong is the only way to
silently lose the privacy guarantee, so each mechanism states the sensitivity
it assumes and validates the budget it is handed.

Neighbouring-dataset convention throughout is **bounded DP** (one record
replaced, dataset size fixed) unless a docstring says otherwise; for counts
that makes the L1 sensitivity 1, not 2.
"""

from __future__ import annotations

from . import _array_core as np

__all__ = ["check_budget", "laplace_noise", "gaussian_sigma", "clip_to_range"]


def check_budget(epsilon, delta=None):
    """Validate a privacy budget, returning it as floats."""
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if delta is None:
        return epsilon, None
    delta = float(delta)
    if not 0 <= delta < 1:
        raise ValueError("delta must be in [0, 1)")
    return epsilon, delta


def laplace_noise(scale, size, rng):
    """Laplace(0, scale) draws."""
    return rng.laplace(0.0, scale, size)


def gaussian_sigma(sensitivity, epsilon, delta):
    r"""Classical Gaussian-mechanism sigma.

    :math:`\sigma = \Delta_2 \sqrt{2\ln(1.25/\delta)}/\varepsilon`, valid only
    for :math:`\varepsilon \le 1`. Outside that range the bound does not hold
    and the caller is told rather than silently under-noised.
    """
    if delta <= 0:
        raise ValueError(
            "the Gaussian mechanism needs delta > 0; use the Laplace mechanism "
            "for pure epsilon-DP"
        )
    return float(sensitivity) * np.sqrt(2.0 * np.log(1.25 / delta)) / float(epsilon)


def clip_to_range(x, a, b):
    """Clip to ``[a, b]``, which is what makes the sensitivity finite."""
    a, b = float(a), float(b)
    if a >= b:
        raise ValueError(f"need a < b, got a={a}, b={b}")
    return np.clip(np.asarray(x, dtype=float), a, b), a, b
