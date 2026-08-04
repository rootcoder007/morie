# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Boosting shrinkage schedule (ESL Ch 10.12.1)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_shrinkage"]


def esl_shrinkage(nu, M=None, target_capacity=None):
    """
    Shrinkage in boosting: f_m = f_{m-1} + nu * h_m, 0 < nu <= 1.

    ESL Ch 10.12.1's empirical finding is that small nu with large M
    beats large nu with small M for the same fitting effort, because
    each stage makes a smaller, more correctable commitment. The
    practical consequence is that nu and M are NOT independent knobs:
    total capacity scales roughly as nu * M, so halving nu without
    doubling M underfits.

    This routine makes that trade explicit. Give it nu and M and it
    reports the effective capacity nu * M; give it nu and a target
    capacity and it reports the M you would need to reach it.

    Parameters
    ----------
    nu : float
        Learning rate in (0, 1].
    M : int, optional
        Number of stages, >= 1.
    target_capacity : float, optional
        Desired nu * M; returns the M required.

    Returns
    -------
    result : dict
        Keys: estimate (capacity nu*M, or the required M when
        target_capacity is given), nu, M, capacity, required_M,
        regime, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 10.12.1.

    Examples
    --------
    >>> esl_shrinkage(0.1, M=100)["capacity"]
    10.0
    >>> esl_shrinkage(0.1, M=100)["regime"]
    'shrunk (nu <= 0.1): prefer many stages'

    Halving the rate doubles the stages needed for the same capacity:

    >>> esl_shrinkage(0.05, target_capacity=10.0)["required_M"]
    200
    >>> esl_shrinkage(0.1, target_capacity=10.0)["required_M"]
    100
    >>> esl_shrinkage(1.5)
    Traceback (most recent call last):
        ...
    ValueError: the learning rate must lie in (0, 1]; got 1.5.
    """
    nu = float(nu)
    if not 0 < nu <= 1:
        raise ValueError(f"the learning rate must lie in (0, 1]; got {nu}.")
    cap = None if M is None else nu * int(M)
    req = None
    if target_capacity is not None:
        tc = float(target_capacity)
        if tc <= 0:
            raise ValueError(f"the target capacity must be positive; got {tc}.")
        req = int(np.ceil(tc / nu))
    if nu <= 0.1:
        regime = "shrunk (nu <= 0.1): prefer many stages"
    elif nu < 1.0:
        regime = "moderate shrinkage"
    else:
        regime = "no shrinkage (nu = 1): each stage fully committed"
    est = req if req is not None else cap
    return RichResult(payload={
        "estimate": est, "nu": nu, "M": None if M is None else int(M),
        "capacity": cap, "required_M": req, "regime": regime,
        "method": "shrinkage f_m = f_{m-1} + nu h_m; capacity ~ nu * M"})


def cheatsheet():
    return "eslshk: nu and M trade off; capacity ~ nu*M, halve nu => double M"


# compact alias per ledger/NAMING.md
eslshrinkage = esl_shrinkage
