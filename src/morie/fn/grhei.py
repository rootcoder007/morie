# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""He initialization for ReLU-family activations."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_he_init"]

_METHOD = "He (Kaiming) normal initialization"


def _lcg_normals(count, seed):
    """``count`` standard normals from the reference LCG via Box-Muller.

    ``s = (1664525 s + 1013904223) mod 2**32``, ``u = (s + 0.5)/2**32``;
    pairs of uniforms become pairs of normals, so the draw is fully
    reproducible and the achieved variance can be checked by hand.
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


def geron_he_init(fan_in, fan_out=None, seed=0):
    r"""Draw weights with variance :math:`2/\text{fan\_in}`.

    .. math::
        \mathrm{var}(W) = \frac{2}{\text{fan\_in}},\qquad
        W \sim \mathcal N(0, \mathrm{var})

    The 2 is what separates He from Glorot.  ReLU zeroes half its
    inputs, so it halves the variance of what passes through; doubling
    the initial variance cancels that exactly and keeps the signal
    magnitude stable as the layers stack.  Initialise a deep ReLU net
    with Glorot's :math:`1/\text{fan\_in}` instead and activations decay
    by :math:`2^{-L/2}`.

    Draws come from the deterministic LCG above, and the *achieved*
    sample variance of the draw is reported next to the target so the
    rule can be verified rather than assumed.

    Parameters
    ----------
    fan_in : int
        Number of inputs to the layer, at least 1.
    fan_out : int, optional
        Number of outputs; defaults to ``fan_in``. The returned matrix
        has shape ``(fan_out, fan_in)`` -- the ``(out, in)`` layout of
        :mod:`morie.fn.grlinf`.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``W``, ``target_variance`` (``2/fan_in``),
        ``achieved_variance``, ``std``, ``achieved_mean``,
        ``relative_error``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 11, He Initialization section (He et al. 2015).

    Examples
    --------
    The target variance is fixed by the fan-in alone:

    >>> r = geron_he_init(100, 100, seed=0)
    >>> r["target_variance"]
    0.02
    >>> round(r["std"], 10)
    0.1414213562

    Over 10 000 draws the sample variance lands within a few percent of
    the target -- the rule is doing what it claims:

    >>> r["relative_error"] < 0.05
    True

    Doubling the fan-in halves the variance:

    >>> geron_he_init(200)["target_variance"]
    0.01
    """
    fan_in = int(fan_in)
    if fan_in < 1:
        raise ValueError(f"fan_in must be a positive integer, got {fan_in}.")
    fan_out = fan_in if fan_out is None else int(fan_out)
    if fan_out < 1:
        raise ValueError(f"fan_out must be a positive integer, got {fan_out}.")

    var = 2.0 / fan_in
    std = np.sqrt(var)
    W = _lcg_normals(fan_in * fan_out, seed).reshape(fan_out, fan_in) * std
    achieved = float(np.var(W, ddof=1)) if W.size > 1 else 0.0
    rel = abs(achieved - var) / var

    return RichResult(
        title="He initialization",
        summary_lines=[("Shape", (fan_out, fan_in)), ("Target var", var),
                       ("Achieved var", achieved)],
        payload={
            "W": W.tolist(),
            "target_variance": var,
            "achieved_variance": achieved,
            "achieved_mean": float(W.mean()),
            "std": float(std),
            "relative_error": float(rel),
            "fan_in": fan_in,
            "fan_out": fan_out,
            "seed": int(seed),
            "estimate": W.tolist(),
            "n": int(W.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grhei: W ~ N(0, 2/fan_in); the 2 compensates for ReLU killing half the signal"


# compact alias per ledger/NAMING.md
geronheinit = geron_he_init
