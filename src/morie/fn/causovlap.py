# morie.fn -- function file (rootcoder007/morie)
"""Propensity overlap (positivity) diagnostic."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_overlap_diagnostic"]


def causal_overlap_diagnostic(ps, treat, bins=20, eps=0.05):
    r"""Assess whether the positivity assumption is plausible.

    Every propensity-based estimator requires
    :math:`0 < P(A=1 \mid X) < 1` for all :math:`X` -- if some covariate
    pattern is never treated, no amount of weighting recovers its treated
    outcome, because it was never observed. The estimator will nonetheless
    return a number.

    Reported here: the propensity ranges in each arm and their intersection,
    the count of units outside the common support, and the mass in the
    extremes. **Positivity is an assumption about the population, and no
    sample can verify it** -- an empty region might be structurally impossible
    or merely unobserved, and the data cannot distinguish those. What a
    diagnostic can do is show that the sample provides no information where
    the estimator is nonetheless producing an answer.

    The distinction between *structural* and *random* non-positivity matters
    for what to do next: structural violations mean the estimand is not
    identified and should be redefined, while random ones may be fixable with
    more data or a restricted target population.

    Parameters
    ----------
    ps : array-like
        Estimated propensity scores.
    treat : array-like
        Treatment indicator, 0/1.
    bins : int
        Histogram bins for the overlap profile.
    eps : float
        Threshold defining the extremes.

    Returns
    -------
    RichResult
        ``common_support``, ``n_outside``, ``prop_extreme``,
        ``min_treated_ps``, ``max_control_ps``, ``overlap_coefficient``.

    References
    ----------
    Petersen, M. L., Porter, K. E., Gruber, S., Wang, Y., & van der Laan, M. J.
        (2012). Diagnosing and responding to violations in the positivity
        assumption. *Statistical Methods in Medical Research*, 21(1), 31-54.
    Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics,
        Social, and Biomedical Sciences*. Cambridge University Press.

    Examples
    --------
    Good overlap gives a wide common support and a high overlap coefficient.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ps = rng.beta(2, 2, 2000)
    >>> tr = (rng.random(2000) < ps).astype(float)
    >>> r = causal_overlap_diagnostic(ps, tr)
    >>> bool(r["overlap_coefficient"] > 0.5)
    True

    Near-deterministic treatment destroys overlap, and the diagnostic says so
    rather than letting the estimator proceed quietly.

    >>> ps_bad = np.r_[rng.uniform(0.9, 0.99, 500), rng.uniform(0.01, 0.1, 500)]
    >>> tr_bad = np.r_[np.ones(500), np.zeros(500)]
    >>> rb = causal_overlap_diagnostic(ps_bad, tr_bad)
    >>> float(rb["overlap_coefficient"])
    0.0

    The common support comes back *empty* -- its lower bound exceeds its
    upper one, meaning no propensity value is occupied by both arms. That is
    the sharpest possible statement that the estimand is not identified, and
    an estimator handed this data will still return a number.

    >>> lo, hi = rb["common_support"]
    >>> bool(lo > hi)
    True

    The contrast is the point: an order of magnitude separates usable overlap
    from unusable.

    >>> bool(r["overlap_coefficient"] > 10 * rb["overlap_coefficient"])
    True
    >>> bool(rb.warnings)
    True

    Common support is the intersection of the two arms' ranges.

    >>> lo, hi = r["common_support"]
    >>> bool(lo <= hi)
    True
    """
    e = np.atleast_1d(np.asarray(ps, dtype=float)).ravel()
    tr = np.atleast_1d(np.asarray(treat, dtype=float)).ravel()
    if e.size != tr.size:
        raise ValueError(f"ps has {e.size} entries but treat has {tr.size}")
    if not np.all((tr == 0) | (tr == 1)):
        raise ValueError("treat must be 0/1")
    t1, t0 = e[tr == 1], e[tr == 0]
    if t1.size == 0 or t0.size == 0:
        raise ValueError("both treatment groups must be non-empty")

    lo = max(float(t1.min()), float(t0.min()))
    hi = min(float(t1.max()), float(t0.max()))
    outside = int(np.sum((e < lo) | (e > hi)))
    extreme = float(np.mean((e < eps) | (e > 1 - eps)))

    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    h1, _ = np.histogram(t1, bins=edges, density=False)
    h0, _ = np.histogram(t0, bins=edges, density=False)
    p1 = h1 / max(h1.sum(), 1)
    p0 = h0 / max(h0.sum(), 1)
    ovl = float(np.sum(np.minimum(p1, p0)))

    warn = []
    if ovl < 0.5:
        warn.append(f"the overlap coefficient is {ovl:.2f}; the two arms barely "
                    "occupy the same propensity region and the estimand is "
                    "close to unidentified")
    if extreme > 0.1:
        warn.append(f"{extreme:.1%} of units have propensity beyond {eps}; "
                    "weights there are extreme and the estimate is fragile")
    warn.append("positivity is an assumption about the population and cannot "
                "be verified from a sample; an empty region may be structurally "
                "impossible or merely unobserved")
    return RichResult(
        title="Overlap / positivity diagnostic",
        summary_lines=[("n", int(e.size)), ("common support", f"[{lo:.3f}, {hi:.3f}]"),
                       ("overlap coefficient", ovl), ("outside support", outside)],
        warnings=warn,
        payload={
            "common_support": (lo, hi), "n_outside": outside,
            "prop_extreme": extreme, "min_treated_ps": float(t1.min()),
            "max_control_ps": float(t0.max()),
            "overlap_coefficient": ovl, "hist_treated": p1,
            "hist_control": p0, "n": int(e.size),
            "method": "causal_overlap_diagnostic",
        },
    )


def cheatsheet():
    return "causovlap: positivity is a POPULATION assumption no sample can verify; the estimator answers anyway"
