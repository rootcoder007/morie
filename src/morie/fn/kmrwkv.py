# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RWKV time-mixing: an attention-free recurrent weighted sum with
exponential decay."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_rwkv_time_mix"]


def kamath_rwkv_time_mix(k, v, w, u=0.0):
    """wkv_t = sum_{i<=t} exp(-(t - i) w + k_i) v_i
              / sum_{i<=t} exp(-(t - i) w + k_i),
    with the current token's term additionally carrying the bonus u.

    A normalised weighted average of the values, with weight decaying
    exponentially in distance -- no pairwise interaction, so the whole
    thing is one O(T) scan. ``u`` is the "first-token bonus" the
    RWKV signature carries; ``u = 0`` reduces exactly to the ratio
    above, and the spec line is recovered.

    Computed with the running-maximum trick, so a large k does not
    overflow exp -- the naive form silently returns nan for k around
    750, which is well inside the range trained models produce.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 10,
    RWKV; that section is not in the 2024 PDF, so the operator is
    implemented exactly as the spec line states (Peng et al. 2023).

    Examples
    --------
    >>> out = kamath_rwkv_time_mix([0.0], [5.0], 1.0)
    >>> out["wkv"]
    [5.0]
    >>> flat = kamath_rwkv_time_mix([0.0, 0.0], [1.0, 3.0], 0.0)
    >>> flat["wkv"]
    [1.0, 2.0]
    >>> import math
    >>> dec = kamath_rwkv_time_mix([0.0, 0.0], [1.0, 3.0], 1.0)
    >>> e = math.exp(-1.0)
    >>> abs(dec["wkv"][1] - (e * 1 + 3) / (e + 1)) < 1e-12
    True
    >>> huge = kamath_rwkv_time_mix([1000.0, 1000.0], [1.0, 3.0], 0.0)
    >>> huge["wkv"][1]
    2.0
    """
    kv = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
    vv = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    w = float(w)
    u = float(u)
    if kv.size != vv.size:
        raise ValueError(
            f"got {kv.size} keys and {vv.size} values; time-mixing pairs "
            "them per position.")
    if kv.size == 0:
        raise ValueError("the sequence is empty.")
    if not (np.all(np.isfinite(kv)) and np.all(np.isfinite(vv))):
        raise ValueError("k and v must be finite.")
    if w < 0:
        raise ValueError(
            f"the decay w must be non-negative; got {w}. A negative w "
            "makes distant tokens matter MORE, which is not a decay.")
    T = kv.size
    out = np.empty(T)
    # Running (num, den) carried in a shifted exponent so nothing
    # overflows: a = num * exp(-p), b = den * exp(-p).
    a = b = 0.0
    p = -np.inf
    for t in range(T):
        # Current token, with the bonus.
        q = max(p, u + kv[t])
        e1 = np.exp(p - q) if np.isfinite(p) else 0.0
        e2 = np.exp(u + kv[t] - q)
        out[t] = (a * e1 + e2 * vv[t]) / (b * e1 + e2)
        # Fold the current token into the running state (no bonus) and
        # decay by w for the next step.
        q2 = max(p, kv[t])
        f1 = np.exp(p - q2) if np.isfinite(p) else 0.0
        f2 = np.exp(kv[t] - q2)
        a = a * f1 + f2 * vv[t]
        b = b * f1 + f2
        p = q2 - w
    return RichResult(payload={
        "wkv": [float(x) for x in out],
        "estimate": float(out[-1]),
        "w": w, "u": u, "n": T,
        "method": "RWKV time-mixing (log-space stable scan)"})


def cheatsheet():
    return "kmrwkv: decayed normalised value average; u bonuses the current token"
