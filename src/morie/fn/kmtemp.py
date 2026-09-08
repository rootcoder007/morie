# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Temperature sampling: softmax over scaled logits."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_temperature_sampling"]


def kamath_temperature_sampling(logits, T):
    """p_i = exp(z_i / T) / sum_j exp(z_j / T).

    T = 1 is the model's own distribution, T > 1 flattens it and
    T < 1 sharpens it. T = 0 is refused: the limit is greedy decoding,
    a different function with a different (degenerate) distribution,
    and dividing by zero to reach it produces inf/inf = nan.

    Computed with the max-shift, so logits of a few hundred -- routine
    after a long generation -- do not overflow exp. The entropy is
    reported because it, not T, is what actually changed.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, temperature
    sampling.

    Examples
    --------
    >>> out = kamath_temperature_sampling([1.0, 1.0], 1.0)
    >>> out["probabilities"]
    [0.5, 0.5]
    >>> import math
    >>> hot = kamath_temperature_sampling([0.0, math.log(3)], 1.0)
    >>> [round(v, 12) for v in hot["probabilities"]]
    [0.25, 0.75]
    >>> cold = kamath_temperature_sampling([0.0, math.log(3)], 0.5)
    >>> [round(v, 12) for v in cold["probabilities"]]
    [0.1, 0.9]
    >>> big = kamath_temperature_sampling([1000.0, 1000.0], 1.0)
    >>> big["probabilities"]
    [0.5, 0.5]
    """
    z = np.atleast_1d(np.asarray(logits, dtype=float)).ravel()
    T = float(T)
    if z.size == 0:
        raise ValueError("no logits supplied.")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite.")
    if T <= 0:
        raise ValueError(
            f"the temperature must be positive; got {T}. T -> 0 is "
            "greedy decoding (argmax), not a temperature.")
    s = z / T
    s = s - s.max()
    e = np.exp(s)
    p = e / e.sum()
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)))
    return RichResult(payload={
        "probabilities": [float(v) for v in p],
        "estimate": float(p.max()),
        "entropy": entropy,
        "max_entropy": float(np.log(z.size)),
        "argmax": int(np.argmax(p)),
        "temperature": T, "n": int(z.size),
        "method": "Temperature-scaled softmax"})


def cheatsheet():
    return "kmtemp: softmax(z/T), max-shifted; T <= 0 refused"
