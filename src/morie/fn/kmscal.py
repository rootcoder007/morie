# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kaplan-style scaling law: loss against parameters, data or
compute."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_scaling_laws"]


def kamath_scaling_laws(N, N_c, alpha_N, L_inf=0.0):
    """L(N) = (N_c / N)^alpha_N + L_inf.

    ``L_inf`` is the irreducible loss -- the entropy of the data that
    no model size removes. It defaults to 0 (the original Kaplan form,
    a pure power law); with it, the law is the Chinchilla-style form
    where the reducible part decays and the floor stays. Both are
    reported, so how much of a predicted loss is floor rather than
    model quality is visible instead of buried.

    ``N`` may be an array, and the same functional form serves the D
    (data) and C (compute) laws with their own constants.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 1, scaling laws
    (Kaplan et al. 2020).

    Examples
    --------
    >>> out = kamath_scaling_laws(1e9, 1e9, 0.076, 1.5)
    >>> out["estimate"]
    2.5
    >>> ten = kamath_scaling_laws([1e9, 1e10], 1e9, 0.5, 0.0)
    >>> abs(ten["loss"][1] - 10 ** -0.5) < 1e-12
    True
    >>> ten["reducible"][0]
    1.0
    """
    n = np.atleast_1d(np.asarray(N, dtype=float)).ravel()
    N_c = float(N_c)
    alpha = float(alpha_N)
    L_inf = float(L_inf)
    if n.size == 0:
        raise ValueError("no scale supplied.")
    if np.any(n <= 0):
        raise ValueError(
            "N must be positive; a model with zero parameters has no "
            "loss curve.")
    if N_c <= 0:
        raise ValueError(f"N_c must be positive; got {N_c}.")
    if alpha <= 0:
        raise ValueError(
            f"alpha_N must be positive; got {alpha}. A non-positive "
            "exponent makes the loss grow with scale.")
    if L_inf < 0:
        raise ValueError(
            f"the irreducible loss must be non-negative; got {L_inf}.")
    reducible = (N_c / n) ** alpha
    loss = reducible + L_inf
    scalar = n.size == 1
    return RichResult(payload={
        "estimate": float(loss[0]) if scalar else [float(v) for v in loss],
        "loss": float(loss[0]) if scalar else [float(v) for v in loss],
        "reducible": float(reducible[0]) if scalar
        else [float(v) for v in reducible],
        "irreducible": L_inf,
        "N_c": N_c, "alpha_N": alpha, "n": int(n.size),
        "method": "Power-law scaling L(N) = (N_c/N)^alpha + L_inf"})


def cheatsheet():
    return "kmscal: (N_c/N)^alpha_N + L_inf, reducible part reported"
