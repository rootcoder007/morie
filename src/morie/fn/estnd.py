# morie.fn -- function file (rootcoder007/morie)
"""Causal estimand definition and selection."""

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_estimand"]

ESTIMANDS = ("ate", "att", "atc", "late", "cate")


def causal_estimand(d, weights=None, propensity=None, estimand="ate",
                    complier=None):
    r"""Define which population an effect is averaged over.

    The four aggregate estimands differ only in the weight each unit
    receives, and the weights are what should be inspected:

    ==========  ===========================  ==========================
    estimand    population                   weight
    ==========  ===========================  ==========================
    ATE         everyone                     1
    ATT         the treated                  :math:`e(X)/\bar e`
    ATC         the untreated                :math:`(1-e(X))/(1-\bar e)`
    LATE        compliers                    unidentified individually
    ==========  ===========================  ==========================

    Choosing among them is a question about the DECISION, not about the
    data. The ATT answers "should we have treated those we did"; the
    ATC answers "should we extend treatment to those we did not"; the
    ATE answers a question about a population that may not correspond
    to any available intervention. Under effect heterogeneity these are
    genuinely different numbers, and ``spread`` reports how different
    the implied weighting is.

    The LATE is the odd one out: its population is defined by response
    to an instrument, so no unit can be identified as belonging to it.
    ``identifiable_population`` records that.

    ``overlap`` is the practical constraint. When treated and control
    propensity distributions barely overlap, the ATE requires
    extrapolating each group into a region where the other has no data,
    while the ATT may still be estimable. The estimand should be chosen
    after looking at this, not before.

    Parameters
    ----------
    d : array-like of {0, 1}, shape (n,)
    weights : array-like, optional
        Sampling weights.
    propensity : array-like, optional
        Needed for ATT and ATC weights and for the overlap diagnostic.
    estimand : {'ate', 'att', 'atc', 'late', 'cate'}
    complier : array-like of bool, optional
        Known complier indicator, if a design supplies one.

    Returns
    -------
    RichResult
        ``weights``, ``target_share``, ``effective_sample_size``,
        ``overlap``, ``identifiable_population``, ``spread``,
        ``question``.

    References
    ----------
    Molak (2023), *Causal Inference and Discovery in Python*, Packt,
    chapter 7, on defining the estimand before estimating.
    Imbens (2004), *Review of Economics and Statistics* 86:4-29.

    Examples
    --------
    >>> out = causal_estimand([1, 1, 0, 0], estimand="att")
    >>> float(out["target_share"])
    0.5
    """
    dv = np.asarray(d, dtype=float).ravel()
    n = dv.size
    if n < 1:
        raise ValueError("need at least one observation.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if estimand not in ESTIMANDS:
        raise ValueError(
            "estimand must be one of %s, got %r." % (ESTIMANDS, estimand)
        )
    sw = np.ones(n) if weights is None else np.asarray(
        weights, dtype=float
    ).ravel()
    if sw.size != n:
        raise ValueError("weights has %d entries for %d rows." % (sw.size, n))

    e = None
    if propensity is not None:
        e = np.asarray(propensity, dtype=float).ravel()
        if e.size != n:
            raise ValueError("propensity has %d entries for %d rows." % (e.size, n))
        e = np.clip(e, 1e-6, 1 - 1e-6)

    if estimand == "ate":
        w = sw.copy()
        share = 1.0
        question = ("the effect of treating everyone, which may not "
                    "correspond to any available intervention")
    elif estimand == "att":
        if e is None:
            w = sw * dv
        else:
            w = sw * e / float(np.mean(e))
        share = float(np.mean(dv))
        question = "whether treating those who were treated was right"
    elif estimand == "atc":
        if e is None:
            w = sw * (1 - dv)
        else:
            w = sw * (1 - e) / float(np.mean(1 - e))
        share = float(np.mean(1 - dv))
        question = "whether to extend treatment to those not treated"
    elif estimand == "late":
        w = sw * (np.asarray(complier, dtype=float).ravel()
                  if complier is not None else np.ones(n))
        share = (float(np.mean(complier)) if complier is not None else np.nan)
        question = ("the effect among compliers, a group defined by "
                    "response to the instrument")
    else:
        w = sw.copy()
        share = np.nan
        question = "the effect as a function of covariates, not a scalar"

    s = w.sum()
    ess = float(s ** 2 / np.sum(w ** 2)) if s > 0 else np.nan
    overlap = None
    if e is not None:
        t, c = e[dv == 1], e[dv == 0]
        if t.size and c.size:
            lo = max(t.min(), c.min())
            hi = min(t.max(), c.max())
            overlap = float(np.mean((e >= lo) & (e <= hi)))
    # how much the ATT and ATC weightings differ, as a heterogeneity cue
    spread = np.nan
    if e is not None:
        wa = e / float(np.mean(e))
        wc = (1 - e) / float(np.mean(1 - e))
        spread = float(np.mean(np.abs(wa - wc)))
    return RichResult(
        payload={
            "estimate": w / s if s > 0 else w,
            "weights": w / s if s > 0 else w,
            "estimand": estimand,
            "target_share": share,
            "question": question,
            "question_note": (
                "the choice among ATE, ATT and ATC is a question about the "
                "DECISION, not about the data; under heterogeneity they are "
                "genuinely different numbers"
            ),
            "effective_sample_size": ess,
            "ess_fraction": float(ess / n) if ess == ess else np.nan,
            "overlap": overlap,
            "overlap_note": (
                "where the propensity distributions barely overlap the ATE "
                "requires extrapolating each arm into a region the other "
                "never visits, while the ATT may still be estimable"
            ),
            "spread": spread,
            "identifiable_population": estimand != "late",
            "late_note": (
                "the complier population is defined by response to the "
                "instrument, so no unit can be identified as belonging to it"
                if estimand == "late" else None
            ),
            "n_treated": int(dv.sum()),
            "n": int(n),
            "method": "Causal estimand definition (%s)" % estimand.upper(),
        }
    )


def cheatsheet():
    return (
        "estnd: the weights each estimand implies, with overlap and the "
        "decision question each one answers"
    )
