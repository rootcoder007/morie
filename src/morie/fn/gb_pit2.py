# morie.fn -- function file (rootcoder007/morie)
"""Inverse probability integral transform sampling."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_pit_rng"]


def gibbons_pit_rng(U, F_inv):
    r"""Example 2.5.2: the converse of the PIT. If U ~ Uniform(0, 1)
    then :math:`X = F^{-1}(U)` has CDF F -- inverse transform
    sampling, the oldest general random-variate generator.

    Parameters
    ----------
    U : array-like in [0, 1]
        Uniform draws.
    F_inv : callable
        Quantile function of the target distribution.

    Returns
    -------
    RichResult
        keys: ``X`` (transformed draws), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Example 2.5.2.
    """
    U = np.asarray(U, dtype=float).ravel()
    if U.size < 1:
        raise ValueError("U must be non-empty.")
    if np.any((U < 0) | (U > 1)):
        raise ValueError("U values must lie in [0, 1].")
    X = np.asarray([F_inv(u) for u in U], dtype=float)
    return RichResult(
        payload={
            "X": X, "n": int(U.size),
            "method": "X = F^{-1}(U), inverse transform sampling (Example 2.5.2)",
        }
    )


def cheatsheet():
    return "gb_pit2: X = F^{-1}(U); converse of the PIT"
