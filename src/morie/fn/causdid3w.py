# morie.fn -- function file (rootcoder007/morie)
"""Triple-difference (DDD) estimator."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_did_three_way"]


def causal_did_three_way(y, treated, post, group):
    r"""Difference-in-difference-in-differences.

    .. math::
        \hat\tau = \big[(\bar y_{11} - \bar y_{10}) - (\bar y_{01} - \bar y_{00})\big]_{g=1}
                 - \big[(\bar y_{11} - \bar y_{10}) - (\bar y_{01} - \bar y_{00})\big]_{g=0},

    the difference between two DiDs -- one in the group that could be affected
    and one in a group that could not.

    The point of the third difference is to relax parallel trends. Plain DiD
    requires treated and control to have moved identically absent treatment;
    DDD only requires that any *violation* of that be common to both groups.
    If a policy region was already diverging from its comparison region for
    unrelated reasons, DDD differences that away provided the divergence hits
    the ineligible group too.

    That is a weaker assumption but not a free one, and it is the one people
    forget to state: DDD assumes the **differential trend is the same across
    groups**. Both component DiDs are returned so the assumption can be
    inspected rather than asserted -- if the placebo group's DiD is large, the
    third difference is doing heavy lifting and deserves scrutiny.

    Parameters
    ----------
    y : array-like
        Outcome.
    treated : array-like
        Treated-unit indicator, 0/1.
    post : array-like
        Post-period indicator, 0/1.
    group : array-like
        Eligible-group indicator, 0/1. The 0 group is the placebo.

    Returns
    -------
    RichResult
        ``ddd``, ``did_eligible``, ``did_placebo``, ``cell_means``,
        ``se``, ``p_value``.

    References
    ----------
    Gruber, J. (1994). The incidence of mandated maternity benefits.
        *American Economic Review*, 84(3), 622-641.

    Examples
    --------
    A true effect present only in the eligible group is recovered, and the
    placebo DiD is near zero.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 8000
    >>> tr = rng.integers(0, 2, n).astype(float)
    >>> po = rng.integers(0, 2, n).astype(float)
    >>> gr = rng.integers(0, 2, n).astype(float)
    >>> y = 1 + 0.5 * tr + 0.3 * po + 0.2 * gr + 2.0 * tr * po * gr + rng.normal(0, 0.5, n)
    >>> r = causal_did_three_way(y, tr, po, gr)
    >>> bool(abs(r["ddd"] - 2.0) < 0.15)
    True
    >>> bool(abs(r["did_placebo"]) < 0.15)
    True

    A confound common to both groups is differenced away, which is the whole
    reason for the third difference.

    >>> y2 = y + 5.0 * tr * po                       # affects BOTH groups
    >>> bool(abs(causal_did_three_way(y2, tr, po, gr)["ddd"] - 2.0) < 0.15)
    True

    Plain DiD on the eligible group alone is contaminated by that confound.

    >>> bool(causal_did_three_way(y2, tr, po, gr)["did_eligible"] > 5.0)
    True

    >>> causal_did_three_way([1.0, 2.0], [0, 1], [0, 1], [0, 2])
    Traceback (most recent call last):
        ...
    ValueError: group must be 0/1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    tr = np.atleast_1d(np.asarray(treated, dtype=float)).ravel()
    po = np.atleast_1d(np.asarray(post, dtype=float)).ravel()
    gr = np.atleast_1d(np.asarray(group, dtype=float)).ravel()
    if not (y.size == tr.size == po.size == gr.size):
        raise ValueError("y, treated, post and group must all have the same length")
    for nm, v in (("treated", tr), ("post", po), ("group", gr)):
        if not np.all((v == 0) | (v == 1)):
            raise ValueError(f"{nm} must be 0/1")

    cells = {}
    for g in (0, 1):
        for t in (0, 1):
            for p in (0, 1):
                m = (gr == g) & (tr == t) & (po == p)
                if not m.any():
                    raise ValueError(f"cell group={g} treated={t} post={p} is empty")
                cells[(g, t, p)] = (float(y[m].mean()),
                                    float(y[m].var(ddof=1) / m.sum()))
    def did(g):
        return ((cells[(g, 1, 1)][0] - cells[(g, 1, 0)][0])
                - (cells[(g, 0, 1)][0] - cells[(g, 0, 0)][0]))
    d1, d0 = did(1), did(0)
    ddd = d1 - d0
    var = sum(v for (_, v) in cells.values())
    se = float(np.sqrt(var))
    from scipy.stats import norm

    z = ddd / se if se > 0 else np.nan
    return RichResult(
        title="Triple difference (DDD)",
        summary_lines=[("DDD", float(ddd)), ("DiD eligible", float(d1)),
                       ("DiD placebo", float(d0)), ("se", se)],
        warnings=["DDD assumes the differential trend is the SAME across "
                  "groups; inspect did_placebo, since a large value means the "
                  "third difference is doing heavy lifting"],
        payload={
            "ddd": float(ddd), "did_eligible": float(d1),
            "did_placebo": float(d0),
            "cell_means": {k: v[0] for k, v in cells.items()},
            "se": se, "z": float(z),
            "p_value": float(2 * norm.sf(abs(z))) if se > 0 else float("nan"),
            "n": int(y.size), "method": "causal_did_three_way",
        },
    )


def cheatsheet():
    return "causdid3w: relaxes parallel trends to 'the VIOLATION is common'; check did_placebo"
