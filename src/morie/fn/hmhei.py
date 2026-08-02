# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""He initialization for ReLU networks."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_he_init"]

_METHOD = "He (Kaiming) initialization"


def geron_he_init(fan_in, seed=0, fan_out=None, distribution="normal"):
    """
    He initialization for ReLU networks.

    Formula: Var(W) = 2 / fan_in

    ReLU zeroes half the pre-activations, so a Glorot-style ``1/fan_in``
    variance halves the signal at every layer and a deep stack fades to
    nothing.  He initialization puts the factor 2 back:
    ``Var(W) = 2/fan_in``, i.e. ``sigma = sqrt(2/fan_in)`` for the normal
    variant and ``limit = sqrt(6/fan_in)`` for the uniform variant
    (a uniform on ``[-a, a]`` has variance ``a^2/3``, so ``a =
    sqrt(3 * 2/fan_in) = sqrt(6/fan_in)``).

    Parameters
    ----------
    fan_in : int
        Number of inputs to the layer (positive).
    seed : int
        Seed for the draw.
    fan_out : int, optional
        Number of units; defaults to ``fan_in`` so the result is square.
    distribution : {"normal", "uniform"}
        Which He variant to draw from.

    Returns
    -------
    result : RichResult
        Keys: W, std_target, var_target, limit, empirical_std,
        estimate, n, method.

    Examples
    --------
    The target is set by the formula, not by the draw:

    >>> r = geron_he_init(8, seed=0, fan_out=4)
    >>> r["var_target"]
    0.25
    >>> round(r["std_target"], 10)
    0.5
    >>> r["W"].shape
    (8, 4)

    The uniform variant uses ``sqrt(6/fan_in)``, and its bound really
    does hold:

    >>> u = geron_he_init(6, seed=1, fan_out=3, distribution="uniform")
    >>> round(u["limit"], 10)
    1.0
    >>> import numpy as np
    >>> bool(np.all(np.abs(u["W"]) <= u["limit"]))
    True

    References
    ----------
    Géron Ch 11
    """
    n_in = int(fan_in)
    if n_in < 1:
        raise ValueError(f"geron_he_init: fan_in must be a positive integer, got {fan_in!r}")
    n_out = n_in if fan_out is None else int(fan_out)
    if n_out < 1:
        raise ValueError(f"geron_he_init: fan_out must be a positive integer, got {fan_out!r}")
    if distribution not in ("normal", "uniform"):
        raise ValueError(f"geron_he_init: distribution must be 'normal' or 'uniform', got {distribution!r}")

    var = 2.0 / n_in
    std = float(np.sqrt(var))
    limit = float(np.sqrt(6.0 / n_in))
    rng = np.random.default_rng(int(seed))
    if distribution == "normal":
        W = rng.normal(0.0, std, size=(n_in, n_out))
    else:
        W = rng.uniform(-limit, limit, size=(n_in, n_out))

    return RichResult(
        title="He initialization",
        summary_lines=[
            ("fan_in", n_in),
            ("Target Var(W)", var),
            ("Target sd(W)", std),
            ("Empirical sd", float(np.std(W))),
        ],
        interpretation=(
            "The factor 2 compensates for ReLU discarding half the pre-activations; "
            "use Glorot (1/fan_avg) for tanh or logistic units instead."
        ),
        payload={
            "W": W,
            "std_target": std,
            "var_target": float(var),
            "limit": limit,
            "empirical_std": float(np.std(W)),
            "fan_in": n_in,
            "fan_out": n_out,
            "distribution": distribution,
            "estimate": std,
            "n": int(W.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmhei: He init Var(W) = 2/fan_in (normal sd sqrt(2/fan_in), uniform limit sqrt(6/fan_in))"
