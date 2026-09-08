# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.39: the mixture-of-experts combination."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_moe_output"]


def kamath_ch2_moe_output(x, G, E_i):
    """y = sum_i G(x)_i E_i(x). ``G`` is the gate -- a weight vector or
    a callable x -> weights; ``E_i`` a list of expert callables (or
    precomputed expert outputs). Zero-weight experts are SKIPPED, the
    sparsity that makes MoE cheap, and the payload reports how many
    actually ran.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.39, printed
    p. 74.

    Examples
    --------
    >>> out = kamath_ch2_moe_output([1.0], [0.5, 0.5, 0.0],
    ...     [lambda x: x[0] * 2, lambda x: x[0] * 4, lambda x: 1e9])
    >>> out["estimate"]
    3.0
    >>> out["experts_evaluated"]
    2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    g = np.atleast_1d(np.asarray(G(x) if callable(G) else G,
                                 dtype=float))
    if len(g) != len(E_i):
        raise ValueError(
            f"the gate produced {len(g)} weights for {len(E_i)} "
            "experts.")
    if np.any(g < 0):
        raise ValueError("gate weights must be non-negative.")
    total = None
    ran = 0
    for gi, Ei in zip(g, E_i):
        if gi == 0.0:
            continue
        out_i = np.atleast_1d(np.asarray(
            Ei(x) if callable(Ei) else Ei, dtype=float))
        ran += 1
        total = gi * out_i if total is None else total + gi * out_i
    if total is None:
        raise ValueError("every gate weight is 0; the mixture selects "
                         "no expert.")
    return RichResult(payload={
        "output": [float(v) for v in np.atleast_1d(total)],
        "gate_weights": [float(v) for v in g],
        "experts_evaluated": ran,
        "estimate": float(np.atleast_1d(total)[0]), "n": len(E_i),
        "method": "Mixture-of-experts combination (Kamath Eq 2.39)"})


def cheatsheet():
    return "km039: gated expert sum, zero-weight experts never run"
