# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mutual information I(X;Y)."""

from . import _array_core as np

from ._richresult import RichResult
from .wsmkbk import wasserman_kullback_leibler

__all__ = ["wasserman_mutual_info"]


def wasserman_mutual_info(x, y):
    """
    Mutual information of two discrete samples, in nats.

    Formula: I(X;Y) = D_KL(p(x,y) || p(x) p(y)), with the joint and
    marginals estimated by empirical frequencies over the observed
    label pairs. Delegates the divergence to wsmkbk (single source
    of truth). I = 0 iff the empirical joint factorises; I is
    symmetric in its arguments.

    Parameters
    ----------
    x, y : array-like
        Paired discrete observations (labels of any hashable kind),
        equal length >= 1.

    Returns
    -------
    result : dict
        Keys: estimate (nats), bits, levels_x, levels_y, n, method.

    References
    ----------
    Wasserman (2004), Ch 23; Cover & Thomas Ch 2.

    Examples
    --------
    Perfectly dependent binary pair: I = log 2. Independent: I = 0.

    >>> import math
    >>> out = wasserman_mutual_info([0, 0, 1, 1], [0, 0, 1, 1])
    >>> abs(out["estimate"] - math.log(2)) < 1e-15
    True
    >>> out["bits"]
    1.0
    >>> wasserman_mutual_info([0, 0, 1, 1], [0, 1, 0, 1])["estimate"]
    0.0
    >>> wasserman_mutual_info([1], [1, 2])
    Traceback (most recent call last):
        ...
    ValueError: paired samples must have equal length; got 1 and 2.
    """
    x = list(np.atleast_1d(np.asarray(x)).tolist())
    y = list(np.atleast_1d(np.asarray(y)).tolist())
    if len(x) != len(y):
        raise ValueError(f"paired samples must have equal length; got {len(x)} and {len(y)}.")
    n = len(x)
    if n == 0:
        raise ValueError("mutual information of an empty sample is undefined.")
    lx = sorted(set(x), key=repr)
    ly = sorted(set(y), key=repr)
    joint = np.zeros((len(lx), len(ly)))
    ix = {v: i for i, v in enumerate(lx)}
    iy = {v: i for i, v in enumerate(ly)}
    for a, b in zip(x, y):
        joint[ix[a], iy[b]] += 1.0
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    prod = np.outer(px, py)
    kl = wasserman_kullback_leibler(joint.ravel(), prod.ravel())
    return RichResult(payload={
        "estimate": kl["estimate"], "bits": kl["bits"],
        "levels_x": len(lx), "levels_y": len(ly), "n": int(n),
        "method": "I(X;Y) = KL(joint || product of marginals), empirical"})


def cheatsheet():
    return "wsmmtl: I = KL(p(x,y)||p(x)p(y)); delegates to wsmkbk"
