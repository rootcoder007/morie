# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Information gain from a split using entropy."""

from . import _array_core as np

from ._richresult import RichResult
from .grcart import geron_cart_split_cost

__all__ = ["geron_information_gain"]

_METHOD = "Information gain (entropy impurity decrease)"


def geron_information_gain(y, split):
    """
    Information gain from a split using entropy.

    Formula: IG = H(parent) - sum_c (m_c/m) H(child_c)

    The weighted child entropy is exactly the CART split cost with the
    entropy criterion, so the arithmetic is delegated to
    :func:`morie.fn.grcart.geron_cart_split_cost` and the gain is its
    ``impurity_decrease``.  Entropy is in bits (log base 2).

    Information gain is never negative -- conditioning cannot increase
    expected entropy -- and it is zero exactly when the split leaves the
    class proportions unchanged, or sends everything one way.  Both
    cases are reported rather than left to be inferred from a 0.

    The known bias of this criterion is toward *many-valued* splits: a
    split into singletons has zero child entropy and therefore maximal
    gain while generalising not at all.  The split's intrinsic
    information and the resulting gain *ratio* are returned so that bias
    is visible.

    Parameters
    ----------
    y : array-like
        Class labels at the node.
    split : array-like
        Group assignment per instance -- boolean, or any labels defining
        the child nodes.  Must have the same length as ``y``.

    Returns
    -------
    result : RichResult
        Keys: information_gain, parent_entropy, child_entropy,
        weighted_child_entropy, intrinsic_information, gain_ratio,
        estimate, n, method.

    Examples
    --------
    A perfect binary split: parent entropy is 1 bit, children are pure,
    so the gain is the whole bit.

    >>> r = geron_information_gain([0, 0, 1, 1], [False, False, True, True])
    >>> float(r["parent_entropy"]), float(r["information_gain"])
    (1.0, 1.0)

    A split that separates nothing gains nothing:

    >>> z = geron_information_gain([0, 0, 1, 1], [True, False, True, False])
    >>> float(z["information_gain"])
    0.0

    A three-way split with one impure child: parent 1 bit, the mixed
    child of two contributes ``(2/4) * 1 = 0.5``:

    >>> t = geron_information_gain([0, 0, 1, 1], ["a", "b", "b", "c"])
    >>> round(float(t["weighted_child_entropy"]), 6)
    0.5
    >>> round(float(t["information_gain"]), 6)
    0.5

    Splitting into singletons maximises the gain and the intrinsic
    information exposes why that is worthless:

    >>> s = geron_information_gain([0, 0, 1, 1], [0, 1, 2, 3])
    >>> float(s["information_gain"]), float(s["intrinsic_information"])
    (1.0, 2.0)
    >>> float(s["gain_ratio"])
    0.5

    References
    ----------
    Géron Ch 5
    """
    yy = np.asarray(y).ravel()
    ss = np.asarray(split).ravel()
    if yy.size == 0:
        raise ValueError("geron_information_gain: y is empty")
    if yy.size != ss.size:
        raise ValueError(f"geron_information_gain: y has {yy.size} entries but split has {ss.size}")

    groups, inverse, counts = np.unique(ss, return_inverse=True, return_counts=True)
    n_groups = groups.size
    m = yy.size

    def _entropy(labels):
        if labels.size == 0:
            return 0.0
        _, c = np.unique(labels, return_counts=True)
        p = c / labels.size
        return float(-np.sum(p * np.log2(p)))

    if n_groups == 2:
        # Delegate the binary case to the CART cost, which is the same
        # weighted-entropy arithmetic.
        indicator = inverse.astype(float).reshape(-1, 1)
        cart = geron_cart_split_cost(indicator, yy, feature=0, threshold=0.5, criterion="entropy")
        weighted = float(cart["cost"])
        parent = float(cart["impurity_parent"])
        child = [float(cart["impurity_left"]), float(cart["impurity_right"])]
    else:
        parent = _entropy(yy)
        child = [_entropy(yy[inverse == g]) for g in range(n_groups)]
        weighted = float(np.sum((counts / m) * np.asarray(child)))

    gain = parent - weighted
    p_groups = counts / m
    intrinsic = float(-np.sum(p_groups * np.log2(p_groups))) if n_groups > 1 else 0.0
    ratio = float(gain / intrinsic) if intrinsic > 0 else 0.0

    warns = []
    if n_groups == 1:
        warns.append("the split sends every instance to one child, so it gains nothing by construction.")
    if n_groups == m and m > 2:
        warns.append(
            f"the split makes {m} singleton children: the gain is maximal by construction and "
            f"generalises not at all -- read gain_ratio instead."
        )

    return RichResult(
        title="Information gain",
        summary_lines=[
            ("Parent entropy (bits)", parent),
            ("Weighted child entropy", weighted),
            ("Information gain", gain),
            ("Gain ratio", ratio),
        ],
        warnings=warns,
        interpretation=(
            "Gain is never negative; it is zero when the split leaves the class proportions alone, "
            "and inflated when the split has many branches."
        ),
        payload={
            "information_gain": float(gain),
            "parent_entropy": parent,
            "child_entropy": child,
            "weighted_child_entropy": weighted,
            "group_sizes": counts,
            "n_groups": int(n_groups),
            "intrinsic_information": intrinsic,
            "gain_ratio": ratio,
            "estimate": float(gain),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmigr: information gain H(parent) - weighted child entropy (binary case delegated to grcart)"
