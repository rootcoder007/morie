# morie.fn -- function file (rootcoder007/morie)
"""Caliper matching on the propensity score."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_caliper_matching"]


def causal_caliper_matching(ps, treat, caliper=None, k=1, replace=True,
                            on_logit=True):
    r"""Nearest-neighbour matching with a maximum acceptable distance.

    Matches on the propensity score, refusing any pair further apart than
    ``caliper``. The default is Austin's recommendation of 0.2 standard
    deviations **of the logit** of the propensity score, which is why matching
    is done on the logit scale by default: the propensity score is compressed
    near 0 and 1, so a fixed caliper on the raw scale is far stricter in the
    middle than at the extremes, while on the logit scale it is uniform.

    The caliper is what makes matching honest. Without it, nearest-neighbour
    matching always returns a match -- the nearest control to a treated unit
    with no comparable control is still *some* control, and the estimator
    quietly extrapolates. Unmatched units are dropped, which changes the
    estimand from the ATT to the ATT among matchable units, and
    ``n_unmatched`` is what tells you how far that has gone.

    Parameters
    ----------
    ps : array-like
        Propensity scores in (0, 1).
    treat : array-like
        Treatment indicator, 0/1.
    caliper : float, optional
        Maximum distance. Defaults to 0.2 sd of the logit.
    k : int
        Controls per treated unit.
    replace : bool
        Allow control reuse.
    on_logit : bool
        Match on the logit scale.

    Returns
    -------
    RichResult
        ``matches``, ``distances``, ``n_unmatched``, ``caliper_used``,
        ``match_rate``, ``estimand``.

    References
    ----------
    Austin, P. C. (2011). Optimal caliper widths for propensity-score matching.
        *Pharmaceutical Statistics*, 10(2), 150-161.
    Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics,
        Social, and Biomedical Sciences*. Cambridge University Press.

    Examples
    --------
    With good overlap almost every treated unit finds a match.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ps = rng.beta(2, 2, 1000)
    >>> tr = (rng.random(1000) < ps).astype(float)
    >>> r = causal_caliper_matching(ps, tr)
    >>> bool(r["match_rate"] > 0.9)
    True

    A tight caliper refuses poor matches instead of extrapolating, and says
    so by dropping units.

    >>> tight = causal_caliper_matching(ps, tr, caliper=0.001)
    >>> bool(tight["n_unmatched"] > r["n_unmatched"])
    True

    Dropping units changes the estimand, and that is reported rather than
    left implicit.

    >>> str(tight["estimand"])
    'ATT among matchable units'

    The logit scale makes the caliper uniform: on the raw scale the same width
    is far stricter in the middle of the distribution than at the extremes.

    >>> a = causal_caliper_matching(ps, tr, on_logit=True)["caliper_used"]
    >>> b = causal_caliper_matching(ps, tr, on_logit=False)["caliper_used"]
    >>> bool(a != b)
    True
    """
    e = np.atleast_1d(np.asarray(ps, dtype=float)).ravel()
    tr = np.atleast_1d(np.asarray(treat, dtype=float)).ravel()
    if e.size != tr.size:
        raise ValueError(f"ps has {e.size} entries but treat has {tr.size}")
    if np.any(e <= 0) or np.any(e >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1)")
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    scale = np.log(e / (1.0 - e)) if on_logit else e
    if caliper is None:
        caliper = float(0.2 * np.std(scale, ddof=1))

    ti = np.flatnonzero(tr == 1)
    ci = np.flatnonzero(tr == 0)
    if ti.size == 0 or ci.size == 0:
        raise ValueError("both treatment groups must be non-empty")
    matches = np.full((ti.size, k), -1, dtype=int)
    dists = np.full((ti.size, k), np.nan)
    used = np.zeros(ci.size, dtype=int)
    for a, i in enumerate(ti):
        d = np.abs(scale[ci] - scale[i])
        picked = 0
        for j in np.argsort(d):
            if not replace and used[j] > 0:
                continue
            if d[j] > caliper:
                break
            matches[a, picked] = int(ci[j])
            dists[a, picked] = d[j]
            used[j] += 1
            picked += 1
            if picked == k:
                break
    ok = matches[:, 0] >= 0
    n_un = int((~ok).sum())
    rate = float(ok.mean())
    warn = []
    if n_un:
        warn.append(
            f"{n_un} of {ti.size} treated units found no match within the "
            "caliper and are dropped; the estimand is now the ATT among "
            "matchable units, not the ATT"
        )
    return RichResult(
        title="Caliper matching",
        summary_lines=[("treated", int(ti.size)), ("matched", int(ok.sum())),
                       ("caliper", float(caliper)), ("match rate", rate)],
        warnings=warn,
        payload={
            "matches": matches, "distances": dists, "n_unmatched": n_un,
            "caliper_used": float(caliper), "match_rate": rate,
            "estimand": "ATT among matchable units" if n_un else "ATT",
            "matched_treated": ti[ok], "on_logit": bool(on_logit),
            "reuse_max": int(used.max()) if used.size else 0,
            "method": "causal_caliper_matching",
        },
    )


def cheatsheet():
    return "causmtchcm: caliper on the LOGIT so it is uniform; without one, matching always extrapolates"
