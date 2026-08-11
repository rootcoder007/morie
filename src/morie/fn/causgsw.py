# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generalizability diagnostics from sampling scores: SMD of logits."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causgsw", "causal_generalisability_smd"]


def causgsw(s_sample, s_target):
    """
    Generalizability diagnostics comparing a trial sample with its
    target population through estimated sampling (participation)
    scores.

    Let s(x) be the estimated probability of trial participation and
    l(x) = logit(s(x)). Tipton and Hartman give, as the diagnostic for
    similarity of sample and target population, the standardized mean
    difference of the logits of the sampling scores,

        SMD = (mean l in sample - mean l in target) / sd,

    where sd is the standard deviation of the logit distribution in
    the target population (Handbook of Matching and Weighting
    Adjustments for Causal Inference, Ch. 3, Eq. 3.7, crediting Stuart
    et al. 2011). Stuart et al. (2011) themselves propose the
    difference in mean sampling scores as the primary similarity
    metric, so the raw-scale difference is reported as well.

    Sign convention: positive values mean the sample sits higher on
    the participation-score scale than the target population. The
    printed Eq. 3.7 is written target-minus-sample; the absolute value
    is identical and is reported as ``smd_abs``.

    Parameters
    ----------
    s_sample : array-like
        Estimated sampling scores s(x) for the trial sample units,
        strictly in (0, 1).
    s_target : array-like
        Estimated sampling scores for the target-population units,
        strictly in (0, 1).

    Returns
    -------
    result : RichResult
        Keys: estimate (SMD of logits, sample minus target, divided by
        the target-population logit sd with ddof = 1), smd_abs,
        diff_means (difference in mean scores on the raw scale),
        mean_sample, mean_target, n_sample, n_target.

    References
    ----------
    Tipton, E. and Hartman, E. (2023), "Generalizability and
    transportability", Ch. 3 in Zubizarreta, Stuart, Small, Rosenbaum
    (eds), Handbook of Matching and Weighting Adjustments for Causal
    Inference, Chapman and Hall/CRC, doi:10.1201/9781003102670,
    Eq. 3.7. Local source: /run/media/rootcoder/WD_BLACK/library/pdf/
    (Zubizarreta et al. Handbook PDF), Ch. 3.
    Stuart, E. A., Cole, S. R., Bradshaw, C. P. and Leaf, P. J. (2011),
    "The use of propensity scores to assess the generalizability of
    results from randomized trials", Journal of the Royal Statistical
    Society Series A 174(2), 369-386,
    doi:10.1111/j.1467-985x.2010.00673.x.
    """
    ss = np.asarray(s_sample, dtype=float)
    st = np.asarray(s_target, dtype=float)
    if len(ss) < 1 or len(st) < 2:
        raise ValueError("need at least 1 sample and 2 target scores")
    for v in list(ss) + list(st):
        if not 0.0 < float(v) < 1.0:
            raise ValueError("sampling scores must lie strictly in (0, 1)")
    ls = np.asarray([math.log(v / (1.0 - v)) for v in ss])
    lt = np.asarray([math.log(v / (1.0 - v)) for v in st])
    sd = float(np.std(lt, ddof=1))
    if sd == 0.0:
        raise ValueError("target-population logits are constant")
    smd = (float(np.mean(ls)) - float(np.mean(lt))) / sd
    return RichResult(payload={
        "estimate": smd,
        "smd_abs": abs(smd),
        "diff_means": float(np.mean(ss)) - float(np.mean(st)),
        "mean_sample": float(np.mean(ss)),
        "mean_target": float(np.mean(st)),
        "n_sample": len(ss),
        "n_target": len(st),
        "method": "Tipton-Hartman Eq. 3.7 SMD of sampling-score logits",
    })


causal_generalisability_smd = causgsw


def cheatsheet():
    return "causgsw(s_sample, s_target) -> standardized mean difference of sampling-score logits."
