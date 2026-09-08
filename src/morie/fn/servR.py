# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Serendipity of a recommendation list."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["serendipity"]


def _ids(v, name):
    x = [int(t) for t in np.atleast_1d(np.asarray(v, dtype=float)).tolist()]
    seen = []
    for t in x:
        if t not in seen:
            seen.append(t)
    return seen


def serendipity(pred, baseline, relevant):
    r"""Serendipity of a recommendation list, the share of its items that
    are both unexpected and relevant.

    Let RS be the recommended set, PM the set a primitive prediction model
    (the baseline the user would have expected anyway) would have produced,
    and REL the set the user actually found useful.  The unexpected set is
    the part of the recommendation the baseline did not anticipate,

    .. math::  \mathrm{UNEXP} = \mathrm{RS} \setminus \mathrm{PM},

    and serendipity is the fraction of the recommendation that lands in the
    intersection with the useful items,

    .. math::

        \mathrm{SRDP}
        = \frac{|\mathrm{UNEXP} \cap \mathrm{REL}|}{|\mathrm{RS}|} .

    Adamopoulos & Tuzhilin's unexpectedness of the list relative to the
    expected set is reported alongside as
    :math:`|\mathrm{RS} \setminus \mathrm{PM}| / |\mathrm{RS}|`; serendipity
    is that quantity restricted to items the user valued, so
    :math:`\mathrm{SRDP} \le \mathrm{unexpectedness}` always, and
    :math:`\mathrm{SRDP} \le \mathrm{precision}` always.

    Sets are compared by identity, not by position: repeated ids collapse.

    Parameters
    ----------
    pred : array-like
        Item ids in the recommendation list RS.
    baseline : array-like
        Item ids the primitive/baseline model would have recommended, PM.
    relevant : array-like
        Item ids the user found useful, REL.

    Returns
    -------
    RichResult
        ``estimate`` is the serendipity SRDP.  ``tp``/``fp``/``fn``/``tn``
        are the confusion matrix of the recommendation against REL over the
        union of all three id sets.

    References
    ----------
    Ge, M., Delgado-Battenfeld, C. & Jannach, D. (2010). Beyond accuracy:
    evaluating recommender systems by coverage and serendipity. Proceedings
    of the Fourth ACM Conference on Recommender Systems, 257-260.
    doi:10.1145/1864708.1864761
    Adamopoulos, P. & Tuzhilin, A. (2014). On unexpectedness in recommender
    systems: or how to better expect the unexpected. ACM Transactions on
    Intelligent Systems and Technology 5(4), 54. doi:10.1145/2559952
    """
    rs = _ids(pred, "pred")
    pm = _ids(baseline, "baseline")
    rel = _ids(relevant, "relevant")
    if len(rs) == 0:
        raise ValueError("serendipity: pred must be a non-empty set of item ids")

    unexp = [t for t in rs if t not in pm]
    hit = [t for t in unexp if t in rel]
    inter_rel = [t for t in rs if t in rel]

    universe = []
    for t in rs + pm + rel:
        if t not in universe:
            universe.append(t)
    tp = float(len([t for t in universe if t in rs and t in rel]))
    fp = float(len([t for t in universe if t in rs and t not in rel]))
    fn = float(len([t for t in universe if t not in rs and t in rel]))
    tn = float(len([t for t in universe if t not in rs and t not in rel]))

    nrs = float(len(rs))
    srdp = len(hit) / nrs
    unexpectedness = len(unexp) / nrs
    precision = len(inter_rel) / nrs
    recall = (len(inter_rel) / len(rel)) if len(rel) > 0 else float("nan")

    return RichResult(
        payload={
            "estimate": srdp,
            "serendipity": srdp,
            "unexpectedness": unexpectedness,
            "precision": precision,
            "recall": recall,
            "n_unexpected": float(len(unexp)),
            "n_serendipitous": float(len(hit)),
            "n_recommended": nrs,
            "n_baseline": float(len(pm)),
            "n_relevant": float(len(rel)),
            "n_universe": float(len(universe)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "method": "Serendipity of a recommendation list (Ge et al. 2010; Adamopoulos & Tuzhilin 2014)",
        }
    )


def cheatsheet():
    return "servR: serendipity of a recommendation list, |UNEXP and REL| / |RS|"


# compact alias per ledger/NAMING.md
serendipityscore = serendipity
