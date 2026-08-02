# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Curse of dimensionality: sample sparsity grows exponentially with d."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_curse_dimensionality"]


def geron_curse_dimensionality(d, n):
    """
    Curse of dimensionality: sample sparsity grows exponentially with d.

    Formula: expected distance to nearest neighbor ~ d^{1/d}

    Three exact quantities are computed for ``n`` points uniform in the
    unit ``d``-cube, because the hand-wave in the formula line is a
    scaling statement, not a number:

    * ``nn_distance``: the radius at which one point is expected inside
      the ball, ``r = (Gamma(d/2+1) / (n pi^{d/2}))^{1/d}`` -- it grows
      towards 0.5 and beyond as ``d`` rises, meaning the "nearest"
      neighbour is no nearer than the box itself.
    * ``mean_pairwise_distance``: ``sqrt(d/6)``, the exact expectation for
      two uniform points, which grows without bound.
    * ``border_fraction``: ``1 - (1-2t)^d``, the share of the cube within
      ``t`` of a face -- at ``d = 10000`` essentially every point is on the
      boundary.

    ``n_for_density`` inverts the first: how many points you would need in
    ``d`` dimensions to match the nearest-neighbour distance you get from
    ``n`` points in one dimension.

    Parameters
    ----------
    d : int
        Dimensionality, >= 1.
    n : int
        Sample size, >= 1.

    Returns
    -------
    result : RichResult
        Keys: nn_distance, mean_pairwise_distance, border_fraction,
        n_for_density, sparsity_factor, estimate, n, method.

    Examples
    --------
    In one dimension with 100 points the nearest neighbour sits about
    1/200 away, and two random points are on average 1/sqrt(6) apart:

    >>> r = geron_curse_dimensionality(1, 100)
    >>> round(r["nn_distance"], 12)
    0.005
    >>> round(r["mean_pairwise_distance"], 9)
    0.40824829

    By 100 dimensions the same 100 points leave the nearest neighbour
    more than four times a box-width away, and 18% of the volume sits
    within 0.001 of a face:

    >>> r2 = geron_curse_dimensionality(100, 100)
    >>> round(r2["nn_distance"], 6)
    2.378241
    >>> round(r2["mean_pairwise_distance"], 6)
    4.082483
    >>> round(r2["border_fraction"], 6)
    0.181433

    Matching 1-D density in 10 dimensions is hopeless:

    >>> geron_curse_dimensionality(10, 100)["n_for_density"] > 1e17
    True

    References
    ----------
    Géron Ch 7
    """
    dd = int(d)
    nn = int(n)
    if dd < 1:
        raise ValueError(f"geron_curse_dimensionality: d must be >= 1, got {d!r}")
    if nn < 1:
        raise ValueError(f"geron_curse_dimensionality: n must be >= 1, got {n!r}")

    log_r = (math.lgamma(dd / 2.0 + 1.0) - math.log(nn) - (dd / 2.0) * math.log(math.pi)) / dd
    r_nn = math.exp(log_r)
    mean_pair = math.sqrt(dd / 6.0)
    t = 0.001
    border = 1.0 - (1.0 - 2.0 * t) ** dd
    r_1d = 0.5 / nn
    n_for_density = math.exp(math.lgamma(dd / 2.0 + 1.0) - (dd / 2.0) * math.log(math.pi) - dd * math.log(r_1d))

    return RichResult(
        title="Curse of dimensionality",
        summary_lines=[("Dimensions", dd), ("Sample size", nn), ("NN distance", r_nn)],
        interpretation=(
            "The nearest neighbour distance grows with d at fixed n: in high dimensions every point is "
            "equally far from every other, which is what breaks distance-based methods."
        ),
        payload={
            "nn_distance": float(r_nn),
            "mean_pairwise_distance": float(mean_pair),
            "border_fraction": float(border),
            "border_tolerance": t,
            "n_for_density": float(n_for_density),
            "sparsity_factor": float(r_nn / r_1d),
            "d": dd,
            "estimate": float(r_nn),
            "n": nn,
            "method": "exact uniform-cube nearest-neighbour radius, mean pairwise distance and border share",
        },
    )


def cheatsheet():
    return "hmcod: Curse of dimensionality: sample sparsity grows exponentially with d"
