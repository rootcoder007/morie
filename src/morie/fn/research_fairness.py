# morie.fn -- function file (rootcoder007/morie)
"""Research P5: what can be certified about the fairness of a risk score.

Python twin of ``R/fairness_bounds.R``; machine-checked in
``research/lean/P5Fairness.lean`` and companions:

- ``Research.P5.Table.chouldechova``: fpr = p/(1-p) (1-ppv)/ppv (1-fnr)
- ``Research.P5.impossibility``: equal ppv and fnr with p_A != p_B force fpr_A != fpr_B
- ``Research.P5.true_base_rate_bounds``: (p_obs - a)/(1 - a) <= p <= p_obs/(1 - b), both ends attained
- ``Research.P5.compare_decided`` / ``compare_undecided``
- ``Research.P5.reduced_coefficient`` and companions: logit coefficient rescaling
- ``Research.P5.rank_reversal_exists`` / ``rank_stable_of_gap``: ranking resolution
- ``Research.P5.survivor_hazard_mono`` / ``hr2_gt_one_of_depletion``: built-in selection in hazards
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = [
    "fairness_rates",
    "fairness_implied_fpr",
    "fairness_base_rate_bounds",
    "fairness_true_rate",
    "fairness_compare_groups",
    "logit_rescale",
    "ranking_resolution",
    "hazard_selection",
]


def _num1(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def _vec(v):
    return [float(v)] if isinstance(v, (int, float)) else [float(a) for a in v]


def fairness_rates(tp, fp, fn, tn):
    """Base rate, PPV, FPR and FNR from a confusion table.

    Parameters
    ----------
    tp, fp, fn, tn : float
        Cell counts; positive.

    Returns
    -------
    RichResult
        ``p`` (base rate), ``ppv``, ``fpr``, ``fnr`` and ``n``.

    Examples
    --------
    >>> r = fairness_rates(tp=60, fp=40, fn=20, tn=180)
    >>> round(r.p, 12), r.ppv, round(r.fpr, 12), r.fnr, r.n
    (0.266666666667, 0.6, 0.181818181818, 0.25, 300)
    """
    for v in (tp, fp, fn, tn):
        if not _num1(v) or v <= 0:
            raise ValueError("tp, fp, fn and tn must be single positive numbers")
    n = tp + fp + fn + tn
    return RichResult(
        title="Confusion-table rates",
        payload={"p": (tp + fn) / n, "ppv": tp / (tp + fp), "fpr": fp / (fp + tn), "fnr": fn / (tp + fn), "n": n},
    )


def fairness_implied_fpr(p, ppv, fnr):
    """False positive rate implied by base rate, PPV and FNR (Chouldechova 2017).

    ``fpr = p/(1-p) * (1-ppv)/ppv * (1-fnr)``: two groups with different
    base rates cannot share PPV, FNR and FPR at once.

    Parameters
    ----------
    p : float
        Base rate in (0, 1).
    ppv : float
        Positive predictive value in (0, 1].
    fnr : float
        False negative rate in [0, 1).

    Returns
    -------
    float

    Examples
    --------
    >>> round(fairness_implied_fpr(80 / 300, 0.6, 0.25), 12)
    0.181818181818
    >>> round(fairness_implied_fpr(0.5, 0.6, 0.25), 12)
    0.5
    """
    if not _num1(p) or p <= 0 or p >= 1:
        raise ValueError("p must be a single number in (0, 1)")
    if not _num1(ppv) or ppv <= 0 or ppv > 1:
        raise ValueError("ppv must be a single number in (0, 1]")
    if not _num1(fnr) or fnr < 0 or fnr >= 1:
        raise ValueError("fnr must be a single number in [0, 1)")
    return p / (1 - p) * ((1 - ppv) / ppv) * (1 - fnr)


def fairness_base_rate_bounds(p_obs, alpha_max, beta_max):
    """Sharp bounds on a true base rate from a noisy recorded rate.

    With over-recording ``alpha in [0, a]`` and under-recording
    ``beta in [0, b]``, ``a + b < 1``, the true rate lies in
    ``[max(0, (p_obs - a)/(1 - a)), min(1, p_obs/(1 - b))]``, both ends
    attained.

    Parameters
    ----------
    p_obs : float or sequence of float
        Recorded rate(s) in [0, 1].
    alpha_max, beta_max : float
        Noise boxes, non-negative with sum below 1.

    Returns
    -------
    dict
        Columns ``p_obs``, ``lower``, ``upper``, ``width``.

    Examples
    --------
    >>> b = fairness_base_rate_bounds([0.1, 0.3], alpha_max=0.05, beta_max=0.2)
    >>> [round(v, 12) for v in b["lower"]], [round(v, 12) for v in b["upper"]]
    ([0.052631578947, 0.263157894737], [0.125, 0.375])
    """
    ps = _vec(p_obs)
    if any(math.isnan(v) or v < 0 or v > 1 for v in ps):
        raise ValueError("p_obs must be numeric in [0, 1]")
    for v in (alpha_max, beta_max):
        if not _num1(v) or v < 0:
            raise ValueError("alpha_max and beta_max must be single non-negative numbers")
    if alpha_max + beta_max >= 1:
        raise ValueError("alpha_max + beta_max must be below 1")
    lo = [max(0.0, (v - alpha_max) / (1 - alpha_max)) for v in ps]
    up = [min(1.0, v / (1 - beta_max)) for v in ps]
    return {"p_obs": ps, "lower": lo, "upper": up, "width": [u - v for u, v in zip(up, lo)]}


def fairness_true_rate(p_obs, alpha, beta):
    """True rate recovered from a recorded rate under known noise.

    ``p = (p_obs - alpha) / (1 - alpha - beta)``.

    Parameters
    ----------
    p_obs : float or sequence of float
        Recorded rate(s).
    alpha, beta : float
        Over- and under-recording rates, non-negative with sum below 1.

    Returns
    -------
    float or list of float

    Examples
    --------
    >>> round(fairness_true_rate(0.3, 0.05, 0.2), 12)
    0.333333333333
    """
    scalar = isinstance(p_obs, (int, float))
    ps = _vec(p_obs)
    if any(math.isnan(v) for v in ps):
        raise ValueError("p_obs must be numeric")
    if not (_num1(alpha) and _num1(beta)) or alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("alpha and beta must be non-negative with alpha + beta < 1")
    out = [(v - alpha) / (1 - alpha - beta) for v in ps]
    return out[0] if scalar else out


def fairness_compare_groups(p_obs_a, p_obs_b, alpha_max, beta_max):
    """Can two groups' true base rates be ordered under a common noise box?

    Disjoint bound intervals order the true rates for every admissible
    noise pair; overlapping intervals admit noise pairs in either order.
    The breakdown value is the largest under-recording box (at the given
    ``alpha_max``) for which the observed order stays certified.

    Parameters
    ----------
    p_obs_a, p_obs_b : float
        Recorded rates of the two groups.
    alpha_max, beta_max : float
        Noise boxes.

    Returns
    -------
    RichResult
        ``decided``, ``order`` (``"a < b"``, ``"b < a"`` or ``"undecided"``),
        ``interval_a``, ``interval_b``, ``breakdown_beta_max`` and ``theorem``.

    Examples
    --------
    >>> c = fairness_compare_groups(0.1, 0.3, alpha_max=0.02, beta_max=0.2)
    >>> c.order, round(c.breakdown_beta_max, 12)
    ('a < b', 0.65)
    >>> fairness_compare_groups(0.1, 0.12, 0.02, 0.2).order
    'undecided'
    """
    ba = fairness_base_rate_bounds(p_obs_a, alpha_max, beta_max)
    bb = fairness_base_rate_bounds(p_obs_b, alpha_max, beta_max)
    if ba["upper"][0] < bb["lower"][0]:
        order = "a < b"
    elif bb["upper"][0] < ba["lower"][0]:
        order = "b < a"
    else:
        order = "undecided"
    breakdown = math.nan
    lo_b = (p_obs_b - alpha_max) / (1 - alpha_max)
    lo_a = (p_obs_a - alpha_max) / (1 - alpha_max)
    if p_obs_a < p_obs_b and lo_b > 0:
        b = 1 - p_obs_a / lo_b
        if b > 0:
            breakdown = min(b, 1 - alpha_max - 1e-12)
    elif p_obs_b < p_obs_a and lo_a > 0:
        b = 1 - p_obs_b / lo_a
        if b > 0:
            breakdown = min(b, 1 - alpha_max - 1e-12)
    return RichResult(
        title="Certified ordering of two base rates",
        payload={
            "decided": order != "undecided",
            "order": order,
            "interval_a": {"lower": ba["lower"][0], "upper": ba["upper"][0]},
            "interval_b": {"lower": bb["lower"][0], "upper": bb["upper"][0]},
            "breakdown_beta_max": breakdown,
            "theorem": "Research.P5.compare_decided" if order != "undecided" else "Research.P5.compare_undecided",
        },
    )


def logit_rescale(beta, omitted_var, error_var=math.pi**2 / 3):
    """Rescaling of logit coefficients across nested models.

    Dropping an omitted covariate independent of the others that carries
    latent variance ``v`` shrinks the identified coefficient by
    ``c = sqrt(error_var / (error_var + v))`` with no confounding at all;
    the apparent odds-ratio change is ``exp((c - 1) beta)`` (Karlson, Holm
    & Breen 2012). Exact for a probit index (``error_var = 1``), the
    standard approximation for the logit.

    Parameters
    ----------
    beta : float or sequence of float
        Identified coefficient(s) in the full model.
    omitted_var : float or sequence of float
        Latent variance of the omitted covariate(s), non-negative.
    error_var : float
        Error variance of the latent index (pi^2/3 logit, 1 probit).

    Returns
    -------
    RichResult
        ``rescale``, ``beta_reduced``, ``odds_ratio_full``,
        ``odds_ratio_reduced``, ``apparent_change`` (scalars when both
        inputs are scalars, lists otherwise, R recycling) and ``theorems``.

    Examples
    --------
    >>> r = logit_rescale(beta=0.8, omitted_var=1)
    >>> round(r.rescale, 12), round(r.beta_reduced, 12), round(r.apparent_change, 12)
    (0.875724044222, 0.700579235378, 0.905361683702)
    """
    scalar = isinstance(beta, (int, float)) and isinstance(omitted_var, (int, float))
    bs = _vec(beta)
    ov = _vec(omitted_var)
    if any(v < 0 for v in ov) or error_var <= 0:
        raise ValueError("omitted_var must be non-negative and error_var positive")
    c = [math.sqrt(error_var / (error_var + v)) for v in ov]
    n = max(len(bs), len(c))
    bs_r = [bs[i % len(bs)] for i in range(n)]
    c_r = [c[i % len(c)] for i in range(n)]
    red = [b * k for b, k in zip(bs_r, c_r)]
    out = {
        "rescale": c,
        "beta_reduced": red,
        "odds_ratio_full": [math.exp(b) for b in bs],
        "odds_ratio_reduced": [math.exp(v) for v in red],
        "apparent_change": [math.exp((k - 1) * b) for b, k in zip(bs_r, c_r)],
    }
    if scalar:
        out = {k: v[0] for k, v in out.items()}
    out["theorems"] = [
        "Research.P5.rescale_lt_one",
        "Research.P5.rescale_eq_one_iff",
        "Research.P5.reduced_coefficient",
        "Research.P5.ratio_is_rescaling",
    ]
    return RichResult(title="Logit coefficient rescaling", payload=out)


def ranking_resolution(estimate, half_width):
    """Resolution of a ranking built from noisy risk scores.

    Two scores whose intervals of half-width ``delta_i`` overlap can be
    reversed by admissible noise; a gap above the sum of the half-widths
    cannot. The resolution of a ranking is twice the largest half-width.

    Parameters
    ----------
    estimate : sequence of float
        Estimated scores, at least two.
    half_width : float or sequence of float
        Half-width per estimate (recycled).

    Returns
    -------
    RichResult
        ``n_pairs``, ``identified_pairs`` (matrix of bools, ``None`` on the
        diagonal), ``share_unidentified``, ``resolution`` and ``theorems``.

    Examples
    --------
    >>> r = ranking_resolution([0.2, 0.35, 0.8], half_width=[0.1, 0.1, 0.05])
    >>> r.identified_pairs
    [[None, False, True], [False, None, True], [True, True, None]]
    >>> round(r.share_unidentified, 12), r.resolution
    (0.333333333333, 0.2)
    """
    est = _vec(estimate)
    n = len(est)
    if n < 2:
        raise ValueError("need at least two scores")
    hw = _vec(half_width)
    hw = [hw[i % len(hw)] for i in range(n)]
    if any(v < 0 for v in hw):
        raise ValueError("half_width must be non-negative")
    ident = [[None if i == j else abs(est[i] - est[j]) > hw[i] + hw[j] for j in range(n)] for i in range(n)]
    pairs = [ident[i][j] for j in range(n) for i in range(j)]
    return RichResult(
        title="Ranking resolution",
        payload={
            "n_pairs": len(pairs),
            "identified_pairs": ident,
            "share_unidentified": sum(1 for p in pairs if not p) / len(pairs),
            "resolution": 2 * max(hw),
            "theorems": [
                "Research.P5.rank_reversal_exists",
                "Research.P5.rank_stable_of_gap",
                "Research.P5.identified_scores_le",
            ],
        },
    )


def hazard_selection(s, h, l, survive_high, survive_low):  # noqa: E741
    """Built-in selection in period-by-period hazard ratios.

    Two risk types with per-period hazards ``h > l`` and initial high-risk
    share ``s``. The period-2 hazard among survivors is the average of ``h``
    and ``l`` weighted by the surviving high-risk share, so an arm that
    depletes the high-risk type less in period 1 shows a higher period-2
    hazard with no period-2 effect at all.

    Parameters
    ----------
    s : float
        Initial high-risk share in (0, 1).
    h, l : float
        Per-period hazards of the high and low types, ``0 <= l < h <= 1``.
    survive_high, survive_low : sequence of two floats
        Period-1 survival probabilities of each type as (control, treated).

    Returns
    -------
    RichResult
        ``surviving_high_share`` and ``period2_hazard`` (dicts with
        ``control``, ``treated``), ``period2_hazard_ratio`` and ``theorems``.

    Examples
    --------
    >>> r = hazard_selection(0.5, 0.5, 0.1, survive_high=(0.5, 0.8), survive_low=(0.9, 0.9))
    >>> {k: round(v, 12) for k, v in r.period2_hazard.items()}
    {'control': 0.242857142857, 'treated': 0.288235294118}
    >>> round(r.period2_hazard_ratio, 12)
    1.186851211073
    """
    if s <= 0 or s >= 1:
        raise ValueError("s must lie in (0, 1)")
    if not (l < h) or l < 0 or h > 1:
        raise ValueError("need 0 <= l < h <= 1")
    sh = _vec(survive_high)
    sl = _vec(survive_low)
    if len(sh) != 2 or len(sl) != 2 or any(v <= 0 for v in sh + sl):
        raise ValueError("survival probabilities must be length-2 positive vectors")
    w = [s * a / (s * a + (1 - s) * b) for a, b in zip(sh, sl)]
    hz = [x * h + (1 - x) * l for x in w]
    return RichResult(
        title="Selection in period-2 hazards",
        payload={
            "surviving_high_share": {"control": w[0], "treated": w[1]},
            "period2_hazard": {"control": hz[0], "treated": hz[1]},
            "period2_hazard_ratio": hz[1] / hz[0],
            "theorems": [
                "Research.P5.survivor_hazard_mono",
                "Research.P5.hr2_gt_one_of_depletion",
                "Research.P5.hr2_witness",
            ],
        },
    )


def cheatsheet() -> str:
    return (
        "fairness_rates(tp, fp, fn, tn) -> p, ppv, fpr, fnr (Research P5)\n"
        "fairness_implied_fpr(p, ppv, fnr) -> Chouldechova identity\n"
        "fairness_base_rate_bounds(p_obs, a, b) / fairness_true_rate(p_obs, alpha, beta)\n"
        "fairness_compare_groups(pa, pb, a, b) -> certified order and breakdown\n"
        "logit_rescale(beta, omitted_var) -> coefficient shrink sqrt(s2/(s2+v))\n"
        "ranking_resolution(estimate, half_width) -> identified pairs, resolution 2 delta\n"
        "hazard_selection(s, h, l, survive_high, survive_low) -> period-2 hazard ratio from selection"
    )
