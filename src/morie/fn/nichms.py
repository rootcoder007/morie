# morie.fn -- function file (rootcoder007/morie)
"""Meyer-Miller nicheness of the parties in one election."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._qpcore import ssum


def _rival_mean(x, w):
    tx, tw = ssum(a * b for a, b in zip(x, w)), ssum(w)
    return [(tx - a * b) / (tw - b) for a, b in zip(x, w)]


def party_nicheness(
    emphases, weights=None, *, normalize: bool = True, only_non_zero: bool = True, transform=None
) -> DescriptiveResult:
    """Meyer and Miller (2015) nicheness of the parties in one election.

    ``emphases`` holds each party's attention to each policy dimension. For
    every dimension, a party's squared deviation from the (weighted) mean
    of its rivals is taken; the party's nicheness is the root mean square
    of these over the dimensions (dimensions nobody mentions are dropped
    when ``only_non_zero``), and, with ``normalize``, minus the weighted
    rival mean of that quantity (party-system normalisation). ``transform
    = "bischof"`` applies ``log(x + 1)`` to the emphases first. This is
    ``manifestoR``'s single-election Meyer-Miller computation.

    :param emphases: Parties x dimensions attention shares.
    :param weights: Party weights (e.g. vote shares); default 1.
    :param normalize: Subtract the weighted rival mean.
    :param only_non_zero: Drop dimensions with zero total attention.
    :param transform: ``None`` or ``"bischof"``.
    :return: DescriptiveResult; ``value`` is the nicheness of each party.

    References
    ----------
    Meyer, T. M. and Miller, B. (2015). The niche party concept and its
    measurement. Party Politics 21, 259-271.

    Bischof, D. (2017). Towards a renewal of the niche party concept.
    Party Politics 23, 220-235.

    Examples
    --------
    >>> [round(v, 12) for v in party_nicheness([[10, 0], [4, 6], [2, 8]], normalize=False).value]
    [7.0, 2.0, 5.0]
    """
    E = [[float(v) for v in row] for row in emphases]
    if transform == "bischof":
        E = [[math.log(v + 1) for v in row] for row in E]
    p, k = len(E), len(E[0])
    w = [1.0] * p if weights is None else [float(v) for v in weights]
    cols = [c for c in range(k) if not only_non_zero or ssum(E[r][c] for r in range(p)) > 0]
    dev = [[0.0] * len(cols) for _ in range(p)]
    for a, c in enumerate(cols):
        rm = _rival_mean([E[r][c] for r in range(p)], w)
        for r in range(p):
            dev[r][a] = (E[r][c] - rm[r]) ** 2
    nic = [math.sqrt(ssum(row) / len(cols)) for row in dev]
    if normalize:
        rm = _rival_mean(nic, w)
        nic = [a - b for a, b in zip(nic, rm)]
    return DescriptiveResult(name="party_nicheness", value=nic, extra={"dimensions": cols})


nichms = party_nicheness


def cheatsheet() -> str:
    return "party_nicheness(emphases, weights) -> Meyer-Miller party nicheness"
