# morie.fn -- function file (rootcoder007/morie)
"""Weighted voting game value"""

from __future__ import annotations

from ._containers import DescriptiveResult
from .svpowr import power_indices


def weighted_vote(weights, *, quota=None, index: str = "shapley_shubik"):
    r"""Power index of each player in the weighted voting game ``[q; w_1, ..., w_n]``.

    ``index`` is ``shapley_shubik`` (Shapley-Shubik value), ``banzhaf``
    (normalised), ``banzhaf_absolute``, ``deegan_packel``, ``johnston`` or
    ``holler``; the default quota is a simple majority of total weight.
    Delegates to :func:`morie.fn.svpowr.power_indices`; ``extra`` holds all
    indices.

    References
    ----------
    Shapley, L. S. and Shubik, M. (1954). A method for evaluating the
    distribution of power in a committee system. *American Political Science
    Review*, 48(3), 787-792.

    Examples
    --------
    >>> weighted_vote([3, 2, 2], quota=4).value
    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    """
    w = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    r = power_indices(w, quota)
    return DescriptiveResult(name="svwvt", value=list(r[index]), extra=dict(r))


weig = weighted_vote


def cheatsheet() -> str:
    return "weighted_vote(weights, quota, index) -> weighted voting game power indices"


# compact alias per ledger/NAMING.md
weightedvote = weighted_vote
