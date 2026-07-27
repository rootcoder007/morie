# morie.fn -- function file (rootcoder007/morie)
"""Propensity score caliper matching (restrict to within-caliper pairs)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["caliper_psm"]


def caliper_psm(e_score, T, caliper=None, y=None):
    r"""Greedy 1:1 caliper matching on the logit of the propensity score.

    Each treated unit is matched, in descending order of propensity, to
    the nearest unused control on the logit scale, subject to

    .. math:: |\,\mathrm{logit}\,e(X_i) - \mathrm{logit}\,e(X_j)\,| < c.

    The default caliper is Austin's recommendation, :math:`c = 0.2`
    pooled standard deviations of the logit of the propensity score,
    which in Monte Carlo work minimised the MSE of the matched
    estimate and removed at least 98% of the crude bias.

    Parameters
    ----------
    e_score : array-like, shape (n,)
        Estimated propensity scores in (0, 1).
    T : array-like of {0, 1}, shape (n,)
        Treatment indicator.
    caliper : float, optional
        Caliper width on the logit scale. Default
        ``0.2 * std(logit(e_score))``.
    y : array-like, shape (n,), optional
        Outcome; if given, the matched-pair ATT is reported.

    Returns
    -------
    RichResult
        keys: ``matched_idx`` (m, 2) array of (treated, control) row
        indices, ``n_matched``, ``n_treated``, ``caliper``,
        ``balance`` dict with the standardised mean difference of the
        propensity score before and after matching, ``att`` (or None),
        ``method``.

    References
    ----------
    Austin, P. C. (2011). Optimal caliper widths for propensity-score
    matching when estimating differences in means and differences in
    proportions in observational studies. *Pharmaceutical Statistics*,
    10(2), 150-161. doi:10.1002/pst.433.

    Rosenbaum, P. R. & Rubin, D. B. (1985). Constructing a control
    group using multivariate matched sampling methods that incorporate
    the propensity score. *The American Statistician*, 39(1), 33-38.
    """
    e = np.asarray(e_score, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    if e.size != T.size:
        raise ValueError(f"e_score and T must have equal length, got {e.size} and {T.size}.")
    if not np.all(np.isin(T, (0.0, 1.0))):
        raise ValueError("T must be binary 0/1.")
    if np.any((e <= 0) | (e >= 1)):
        raise ValueError("propensity scores must lie strictly in (0, 1).")

    lg = np.log(e / (1.0 - e))
    if caliper is None:
        caliper = 0.2 * float(lg.std(ddof=1))
    caliper = float(caliper)
    if caliper <= 0:
        raise ValueError(f"caliper must be positive, got {caliper}.")

    tr = np.flatnonzero(T == 1)
    co = np.flatnonzero(T == 0)
    if tr.size == 0 or co.size == 0:
        raise ValueError("need at least one treated and one control unit.")

    def _smd(idx_t, idx_c):
        mt, mc = e[idx_t].mean(), e[idx_c].mean()
        s = np.sqrt((e[idx_t].var(ddof=1) + e[idx_c].var(ddof=1)) / 2.0)
        return float((mt - mc) / s) if s > 0 else 0.0

    used = np.zeros(co.size, dtype=bool)
    pairs = []
    for i in tr[np.argsort(-e[tr])]:  # high propensity first: hardest to match
        d = np.abs(lg[co] - lg[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] < caliper:
            used[j] = True
            pairs.append((int(i), int(co[j])))

    matched = np.array(pairs, dtype=int).reshape(-1, 2)
    att = None
    if y is not None and matched.shape[0] > 0:
        y = np.asarray(y, dtype=float).ravel()
        att = float(np.mean(y[matched[:, 0]] - y[matched[:, 1]]))

    balance = {"smd_before": _smd(tr, co)}
    if matched.shape[0] > 1:
        balance["smd_after"] = _smd(matched[:, 0], matched[:, 1])

    return RichResult(
        payload={
            "matched_idx": matched,
            "n_matched": int(matched.shape[0]),
            "n_treated": int(tr.size),
            "caliper": caliper,
            "balance": balance,
            "att": att,
            "method": "Propensity score caliper matching (logit scale, greedy 1:1)",
        }
    )


def cheatsheet():
    return "cipsc: caliper PSM on logit(e), default 0.2*sd (Austin 2011)"
