# morie.fn -- function file (rootcoder007/morie)
"""Propensity score matching."""

from . import _array_core as np

from ._richresult import RichResult
from ._did import add_intercept, logit_fit, logit_predict

__all__ = ["propensity_score_matching"]


def propensity_score_matching(y, d, X=None, propensity=None, n_neighbors=1,
                              caliper=None, replace=True, estimand="att"):
    r"""Match on the propensity score and compare matched outcomes.

    Rosenbaum and Rubin's result is that if treatment is ignorable
    given :math:`X`, it is ignorable given :math:`e(X)` alone -- a
    scalar. That is what makes matching on a single number legitimate.

    What the result does NOT say is that matched groups will look alike
    on :math:`X`. The propensity score balances :math:`X` IN
    EXPECTATION, so any particular matched sample can remain
    imbalanced, and checking is not optional. ``balance_before`` and
    ``balance_after`` give the standardised mean difference per
    covariate; the usual threshold is 0.1, and ``balanced`` applies it.

    A caliper refuses matches worse than a given distance. It improves
    balance and changes the estimand: dropped treated units are no
    longer represented, so the answer is the ATT among the matchable,
    not the ATT. ``n_unmatched`` records how many were lost.

    Matching WITH replacement lowers bias -- every unit gets its best
    available match -- at the cost of variance, since a few controls
    can be reused many times. ``max_control_reuse`` shows whether that
    happened.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
    X : array-like, optional
    propensity : array-like, optional
    n_neighbors : int
    caliper : float, optional
        In propensity units.
    replace : bool
    estimand : {'att', 'ate'}

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``matched_pairs``, ``n_unmatched``,
        ``balance_before``, ``balance_after``, ``balanced``,
        ``max_control_reuse``.

    References
    ----------
    Molak (2023), *Causal Inference and Discovery in Python*, chapter 9.
    Rosenbaum and Rubin (1983), *Biometrika* 70:41-55.
    Austin (2011), *Multivariate Behavioral Research* 46:399-424, for
    the 0.1 standardised-difference threshold.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> d = np.r_[np.ones(50), np.zeros(50)]
    >>> e = np.r_[rng.uniform(0.4, 0.6, 50), rng.uniform(0.4, 0.6, 50)]
    >>> y = 2.0 * d + rng.normal(size=100)
    >>> out = propensity_score_matching(y, d, propensity=e)
    >>> bool(abs(out["estimate"] - 2.0) < 0.8)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    n = yv.size
    if dv.size != n:
        raise ValueError("y and d must agree in length.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if estimand not in ("att", "ate"):
        raise ValueError("estimand must be 'att' or 'ate', got %r." % estimand)
    Xa = None if X is None else np.atleast_2d(np.asarray(X, dtype=float))
    if Xa is not None and Xa.shape[0] != n:
        Xa = Xa.T
    if propensity is None:
        if Xa is None:
            raise ValueError("supply X or propensity.")
        beta, _ = logit_fit(add_intercept(Xa), dv)
        e = logit_predict(add_intercept(Xa), beta)
    else:
        e = np.asarray(propensity, dtype=float).ravel()
        if e.size != n:
            raise ValueError("propensity has %d entries for %d rows."
                             % (e.size, n))
    ti = np.nonzero(dv == 1)[0]
    ci = np.nonzero(dv == 0)[0]
    if ti.size == 0 or ci.size == 0:
        raise ValueError("need both treated and control units.")
    k = max(int(n_neighbors), 1)

    pairs = []
    unmatched = 0
    pool = list(ci)
    for i in ti:
        avail = np.asarray(pool if not replace else ci)
        if avail.size == 0:
            unmatched += 1
            continue
        dist = np.abs(e[avail] - e[i])
        order = np.argsort(dist)[:k]
        sel = avail[order]
        if caliper is not None:
            keep = np.abs(e[sel] - e[i]) <= float(caliper)
            sel = sel[keep]
        if sel.size == 0:
            unmatched += 1
            continue
        pairs.append((int(i), [int(v) for v in sel]))
        if not replace:
            for v in sel:
                if v in pool:
                    pool.remove(int(v))

    if not pairs:
        raise ValueError(
            "no treated unit found a match; the caliper may be too tight."
        )
    diffs = np.array([yv[i] - float(np.mean(yv[m])) for i, m in pairs])
    est = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(diffs.size)) \
        if diffs.size > 1 else np.nan

    used = [v for _, m in pairs for v in m]
    reuse = int(np.max(np.bincount(used))) if used else 0

    bb = ba = None
    balanced = None
    if Xa is not None:
        def smd(a, b):
            sp = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2)
            sp = np.where(sp > 0, sp, 1.0)
            return (a.mean(axis=0) - b.mean(axis=0)) / sp
        bb = smd(Xa[ti], Xa[ci])
        mt = np.array([i for i, _ in pairs])
        mc = np.array(used)
        ba = smd(Xa[mt], Xa[mc])
        balanced = bool(np.all(np.abs(ba) < 0.1))
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "ci": (est - 1.959963984540054 * se,
                   est + 1.959963984540054 * se) if se == se else None,
            "matched_pairs": pairs,
            "n_matched": int(len(pairs)),
            "n_unmatched": int(unmatched),
            "unmatched_note": (
                None if unmatched == 0 else
                "%d treated unit(s) found no match, so the estimand is the "
                "ATT among the MATCHABLE rather than the ATT" % unmatched
            ),
            "balance_before": bb,
            "balance_after": ba,
            "balanced": balanced,
            "balance_note": (
                "the propensity score balances X only IN EXPECTATION; any "
                "particular matched sample can stay imbalanced, which is why "
                "the standardised differences have to be looked at"
            ),
            "max_control_reuse": reuse,
            "reuse_note": (
                "matching with replacement lowers bias and raises variance; "
                "a control reused many times is carrying the estimate"
            ),
            "replace": bool(replace),
            "caliper": None if caliper is None else float(caliper),
            "n_neighbors": k,
            "estimand": estimand,
            "n_treated": int(ti.size),
            "n_control": int(ci.size),
            "n": int(n),
            "method": "Propensity score matching (%s)" % estimand.upper(),
        }
    )


def cheatsheet():
    return (
        "pscsm: nearest-neighbour propensity matching with before/after "
        "balance, caliper losses and control reuse"
    )
