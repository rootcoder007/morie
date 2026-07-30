# morie.fn -- function file (rootcoder007/morie)
"""IPTW and overlap (ATO) weights."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_iptw_atoweights"]


def causal_iptw_atoweights(treat, ps, estimand="ato", trim=None, stabilize=True):
    r"""Propensity weights for the ATE, ATT or the overlap-weighted estimand.

    ==========  =======================  ====================================
    estimand    treated weight           control weight
    ==========  =======================  ====================================
    ``"ate"``   :math:`1/e`              :math:`1/(1-e)`
    ``"att"``   :math:`1`                :math:`e/(1-e)`
    ``"ato"``   :math:`1-e`              :math:`e`
    ==========  =======================  ====================================

    IPTW weights blow up as the propensity approaches 0 or 1: a control with
    :math:`e = 0.99` receives a weight of 100 and can dominate the estimate
    single-handed. Trimming is the usual patch, and it silently changes the
    estimand -- you are no longer estimating the ATE but the ATE on whoever
    survived the trim.

    **Overlap weights** avoid the problem by construction. The per-unit
    weights -- :math:`1-e` for the treated and :math:`e` for the controls --
    are bounded by 1, against IPTW's unbounded :math:`1/e`, and go to zero
    exactly where IPTW explodes. The implied tilting function is
    :math:`e(1-e)`, bounded by 1/4 and maximised at :math:`e = 1/2`. They target the ATO -- the effect among units
    whose treatment was genuinely uncertain -- which is a different estimand
    but one that is actually identified, and they minimise the asymptotic
    variance among all balancing weights. Preferring a well-estimated ATO to
    a badly-estimated ATE is usually the right trade.

    ``max_weight_share`` reports the largest single unit's share of the total
    weight, which is the number that reveals a dominated estimate.

    Parameters
    ----------
    treat : array-like
        Treatment indicator, 0/1.
    ps : array-like
        Estimated propensity scores in (0, 1).
    estimand : {"ato", "ate", "att"}
        Target estimand.
    trim : float, optional
        Drop units with ``ps`` outside ``[trim, 1-trim]``. Changes the
        estimand.
    stabilize : bool
        Multiply by the marginal treatment probability (ATE/ATT only).

    Returns
    -------
    RichResult
        ``weights``, ``estimand``, ``ess``, ``max_weight_share``,
        ``n_trimmed``.

    References
    ----------
    Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing covariates via
        propensity score weighting. *JASA*, 113(521), 390-400.
    Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics,
        Social, and Biomedical Sciences*. Cambridge University Press.

    Examples
    --------
    Overlap weights are bounded by 1, and the tilting function e(1-e) they
    imply is bounded by 1/4 -- so no single unit can dominate.

    >>> import numpy as np
    >>> ps = np.array([0.01, 0.5, 0.99])
    >>> tr = np.array([1, 1, 0])
    >>> w = causal_iptw_atoweights(tr, ps, estimand="ato")["weights"]
    >>> bool(w.max() <= 1.0 + 1e-12)
    True
    >>> bool((ps * (1 - ps)).max() <= 0.25 + 1e-12)
    True

    IPTW explodes exactly where overlap weights vanish -- an extreme
    propensity gives a weight of 100.

    >>> wi = causal_iptw_atoweights(tr, ps, estimand="ate", stabilize=False)["weights"]
    >>> bool(wi.max() > 90)
    True

    The dominance is visible in the weight share, which is the diagnostic.

    >>> a = causal_iptw_atoweights(tr, ps, estimand="ate", stabilize=False)
    >>> b = causal_iptw_atoweights(tr, ps, estimand="ato")
    >>> bool(a["max_weight_share"] > b["max_weight_share"])
    True

    Trimming changes the estimand, and that is stated rather than assumed.

    >>> t = causal_iptw_atoweights(tr, ps, estimand="ate", trim=0.05)
    >>> int(t["n_trimmed"])
    2
    >>> bool(t.warnings)
    True

    >>> causal_iptw_atoweights([1, 0], [0.0, 0.5])
    Traceback (most recent call last):
        ...
    ValueError: propensity scores must lie strictly inside (0, 1)
    """
    tr = np.atleast_1d(np.asarray(treat, dtype=float)).ravel()
    e = np.atleast_1d(np.asarray(ps, dtype=float)).ravel()
    if tr.size != e.size:
        raise ValueError(f"treat has {tr.size} entries but ps has {e.size}")
    if not np.all((tr == 0) | (tr == 1)):
        raise ValueError("treat must be 0/1")
    if np.any(e <= 0) or np.any(e >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1)")
    if estimand not in ("ato", "ate", "att"):
        raise ValueError('estimand must be "ato", "ate" or "att"')

    keep = np.ones(tr.size, dtype=bool)
    warn = []
    if trim is not None:
        keep = (e >= trim) & (e <= 1 - trim)
        warn.append(
            f"trimming dropped {int((~keep).sum())} units, which changes the "
            "estimand: this is no longer the ATE but the ATE among units that "
            "survived the trim"
        )

    if estimand == "ato":
        w = np.where(tr == 1, 1.0 - e, e)
    elif estimand == "ate":
        w = np.where(tr == 1, 1.0 / e, 1.0 / (1.0 - e))
        if stabilize:
            p = float(tr.mean())
            w = w * np.where(tr == 1, p, 1.0 - p)
    else:
        w = np.where(tr == 1, 1.0, e / (1.0 - e))
        if stabilize:
            w = w / max(w.mean(), 1e-12)
    w = np.where(keep, w, 0.0)
    tot = float(w.sum())
    share = float(w.max() / tot) if tot > 0 else float("nan")
    ess = float(tot**2 / max(float(np.sum(w**2)), 1e-300))
    if share > 0.1:
        warn.append(f"one unit carries {share:.1%} of the total weight; the "
                    "estimate is dominated by a handful of observations")
    return RichResult(
        title=f"Propensity weights ({estimand.upper()})",
        summary_lines=[("n", int(tr.size)), ("estimand", estimand.upper()),
                       ("ESS", ess), ("max weight share", share)],
        warnings=warn,
        payload={
            "weights": w, "estimand": estimand, "ess": ess,
            "max_weight_share": share, "n_trimmed": int((~keep).sum()),
            "kept": keep, "n": int(tr.size),
            "method": "causal_iptw_atoweights",
        },
    )


def cheatsheet():
    return "causipsw0: overlap weights e(1-e) are bounded by 1/4 and vanish where IPTW explodes; trimming CHANGES the estimand"
