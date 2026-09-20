# SPDX-License-Identifier: AGPL-3.0-or-later
"""Design and measurement tools for small or under-enumerated subpopulations.

Full parity with rmorie's R/subpop_design.R and R/subpop_psychometrics.R.
The arithmetic is implemented here rather than delegated: morie is not a
wrapper.

The functions are general survey methodology. The motivating application is
estimation for Indigenous populations in Canada, where three conditions
usually hold together: the group is a small share of any general frame, it is
reached through designs that are not simple random samples, and the frame
itself identifies it imperfectly.

Three results drive the planning arithmetic.

* A subgroup that is a share ``q`` of the frame needs roughly ``1/q`` times
  the general sample to reach a given subgroup size. At q = 0.05, a subgroup
  target of 385 needs about 7,700 general interviews.
* Clustering by community multiplies the requirement by ``1 + (m - 1) * rho``
  and unequal weights multiply it again; the two sources compose.
* If the frame records only a fraction of true members, that fraction divides
  the achievable subgroup size a second time, and every rate computed on the
  recorded counts is biased downward (Rogan and Gladen, 1978).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from morie.fn import _stats_core as stats

__all__ = [
    "morie_deff_cluster",
    "morie_neff_cluster",
    "morie_sample_size_proportion",
    "morie_sample_size_domain",
    "morie_oversample_factor",
    "morie_screen_design",
    "morie_alloc_optimal",
    "morie_rake",
    "morie_misclass_correct",
    "morie_misclass_count",
    "morie_dif_sample_size",
    "morie_dif_delta_mh",
    "morie_invariance_compare",
    "morie_irt_theta_se",
    "morie_irt_marginal_reliability",
]


def _prob(x, name):
    x = float(x)
    if math.isnan(x) or x < 0.0 or x > 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1]")
    return x


def _z(conf):
    conf = _prob(conf, "conf")
    if conf <= 0.0 or conf >= 1.0:
        raise ValueError("conf must be strictly between 0 and 1")
    return stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)


# ---------------------------------------------------------------------------
# Design effects
# ---------------------------------------------------------------------------

def morie_deff_cluster(m, icc):
    """Cluster design effect ``1 + (m - 1) * icc``."""
    m = float(m)
    icc = float(icc)
    if math.isnan(m) or m < 1.0:
        raise ValueError("m must be a number >= 1")
    if math.isnan(icc):
        raise ValueError("icc must be a number")
    deff = 1.0 + (m - 1.0) * icc
    if deff <= 0.0:
        raise ValueError("icc implies a non-positive design effect")
    return deff


def morie_neff_cluster(n, m, icc):
    """Effective sample size ``n / deff`` under clustering."""
    n = float(n)
    if math.isnan(n) or n <= 0.0:
        raise ValueError("n must be a positive number")
    return n / morie_deff_cluster(m, icc)


# ---------------------------------------------------------------------------
# Sample size
# ---------------------------------------------------------------------------

@dataclass
class SampleSizeProportion:
    """Sample size for a proportion under a complex design."""

    n_srs: float
    n_design: float
    n_invite: float
    p: float
    moe: float
    conf: float
    N: float
    deff: float
    response_rate: float


def morie_sample_size_proportion(p, moe, conf=0.95, N=float("inf"), deff=1.0,
                                 response_rate=1.0):
    """Completed sample size for a proportion, with FPC, deff and non-response.

    Precision first, then the population limit, then the design, then
    fieldwork loss.
    """
    p = _prob(p, "p")
    moe = float(moe)
    if math.isnan(moe) or moe <= 0.0 or moe >= 1.0:
        raise ValueError("moe must be in (0, 1)")
    deff = float(deff)
    if math.isnan(deff) or deff <= 0.0:
        raise ValueError("deff must be a positive number")
    response_rate = _prob(response_rate, "response_rate")
    if response_rate <= 0.0:
        raise ValueError("response_rate must be greater than 0")
    N = float(N)
    if math.isnan(N) or N <= 0.0:
        raise ValueError("N must be a positive number or infinity")
    z = _z(conf)
    n0 = z * z * p * (1.0 - p) / (moe * moe)
    n_fpc = n0 / (1.0 + (n0 - 1.0) / N) if math.isfinite(N) else n0
    n_design = n_fpc * deff
    return SampleSizeProportion(
        n_srs=n0, n_design=n_design, n_invite=n_design / response_rate,
        p=p, moe=moe, conf=float(conf), N=N, deff=deff,
        response_rate=response_rate,
    )


@dataclass
class SampleSizeDomain:
    """Sample size for a subpopulation within a larger frame."""

    n_domain: float
    n_domain_invite: float
    n_overall: float
    domain_prevalence: float
    coverage: float
    deff: float
    response_rate: float
    moe: float
    conf: float


def morie_sample_size_domain(p, moe, domain_prevalence, conf=0.95,
                             N_domain=float("inf"), deff=1.0,
                             response_rate=1.0, coverage=1.0):
    """How large the whole sample must be for the subgroup to be large enough."""
    dp = _prob(domain_prevalence, "domain_prevalence")
    if dp <= 0.0:
        raise ValueError("domain_prevalence must be greater than 0")
    cov = _prob(coverage, "coverage")
    if cov <= 0.0:
        raise ValueError("coverage must be greater than 0")
    base = morie_sample_size_proportion(p, moe, conf=conf, N=N_domain,
                                        deff=deff,
                                        response_rate=response_rate)
    return SampleSizeDomain(
        n_domain=base.n_design,
        n_domain_invite=base.n_invite,
        n_overall=base.n_invite / (dp * cov),
        domain_prevalence=dp, coverage=cov, deff=base.deff,
        response_rate=base.response_rate, moe=base.moe, conf=float(conf),
    )


@dataclass
class OversampleFactor:
    """Selection-probability multiplier and the weight variation it creates."""

    factor: float
    weight_ratio: float
    deff_weights: float


def morie_oversample_factor(domain_prevalence, target_share):
    """Multiplier on the subgroup's selection probability, and its weight cost."""
    dp = _prob(domain_prevalence, "domain_prevalence")
    ts = _prob(target_share, "target_share")
    if dp <= 0.0 or dp >= 1.0:
        raise ValueError("domain_prevalence must be in (0, 1)")
    if ts <= 0.0 or ts >= 1.0:
        raise ValueError("target_share must be in (0, 1)")
    fac = (ts / (1.0 - ts)) * ((1.0 - dp) / dp)
    wr = 1.0 / fac
    wbar = ts * wr + (1.0 - ts)
    w2bar = ts * wr * wr + (1.0 - ts)
    return OversampleFactor(factor=fac, weight_ratio=wr,
                            deff_weights=w2bar / (wbar * wbar))


@dataclass
class ScreenDesign:
    """Two-phase screening design for a rare subpopulation."""

    n_screen: float
    n_eligible: float
    cost_total: float
    cost_per_completed_interview: float
    screening_share_of_cost: float


def morie_screen_design(prevalence, target_n, cost_screen=1.0,
                        cost_interview=1.0, screen_response=1.0,
                        interview_response=1.0):
    """Expected screens, cost, and the share of budget spent screening."""
    pr = _prob(prevalence, "prevalence")
    if pr <= 0.0:
        raise ValueError("prevalence must be greater than 0")
    sr = _prob(screen_response, "screen_response")
    ir = _prob(interview_response, "interview_response")
    if sr <= 0.0 or ir <= 0.0:
        raise ValueError("response rates must be greater than 0")
    target_n = float(target_n)
    if math.isnan(target_n) or target_n <= 0.0:
        raise ValueError("target_n must be a positive number")
    cost_screen = float(cost_screen)
    cost_interview = float(cost_interview)
    if cost_screen < 0.0 or cost_interview < 0.0:
        raise ValueError("costs must be non-negative")
    n_eligible = target_n / ir
    n_screen = n_eligible / (pr * sr)
    cost_s = n_screen * cost_screen
    cost_i = target_n * cost_interview
    total = cost_s + cost_i
    return ScreenDesign(
        n_screen=n_screen, n_eligible=n_eligible, cost_total=total,
        cost_per_completed_interview=total / target_n,
        screening_share_of_cost=(cost_s / total) if total > 0 else float("nan"),
    )


@dataclass
class Allocation:
    """Optimal allocation across strata."""

    n_h: list
    n_h_int: list
    share: list


def morie_alloc_optimal(N_h, S_h, n, cost_h=None):
    """Cost-constrained optimal allocation, ``n_h propto N_h S_h / sqrt(c_h)``.

    With equal costs this is Neyman allocation.
    """
    N_h = [float(v) for v in N_h]
    S_h = [float(v) for v in S_h]
    if len(N_h) != len(S_h):
        raise ValueError("N_h and S_h must have the same length")
    if not N_h:
        raise ValueError("at least one stratum is required")
    if any(math.isnan(v) or v <= 0.0 for v in N_h):
        raise ValueError("N_h must be positive with no missing values")
    if any(math.isnan(v) or v < 0.0 for v in S_h):
        raise ValueError("S_h must be non-negative with no missing values")
    n = float(n)
    if math.isnan(n) or n <= 0.0:
        raise ValueError("n must be a positive number")
    if cost_h is None:
        cost_h = [1.0] * len(N_h)
    else:
        cost_h = [float(v) for v in cost_h]
        if len(cost_h) != len(N_h):
            raise ValueError("cost_h must match N_h in length")
        if any(math.isnan(v) or v <= 0.0 for v in cost_h):
            raise ValueError("cost_h must be positive")
    num = [N_h[i] * S_h[i] / math.sqrt(cost_h[i]) for i in range(len(N_h))]
    denom = sum(num)
    if denom <= 0.0:
        raise ValueError("allocation is undefined when every stratum has zero variance")
    share = [v / denom for v in num]
    nh = [n * s for s in share]
    base = [math.floor(v) for v in nh]
    rem = int(round(n - sum(base)))
    if rem > 0:
        order = sorted(range(len(nh)), key=lambda i: nh[i] - base[i], reverse=True)
        for i in order[:rem]:
            base[i] += 1
    return Allocation(n_h=nh, n_h_int=[int(v) for v in base], share=share)


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

@dataclass
class RakeResult:
    """Iterative proportional fitting of weights to known margins."""

    weights: list
    converged: bool
    iterations: int
    max_discrepancy: float


def morie_rake(data, margins, weights=None, max_iter=50, tol=1e-8):
    """Rake weights to known population margins by iterative proportional fitting.

    ``data`` is a mapping of column name to a sequence of level labels, one
    entry per row. ``margins`` maps a subset of those columns to a mapping of
    level to population total.
    """
    if not isinstance(data, dict) or not data:
        raise ValueError("data must be a non-empty mapping of column to values")
    if not isinstance(margins, dict) or not margins:
        raise ValueError("margins must be a non-empty mapping")
    missing = [v for v in margins if v not in data]
    if missing:
        raise ValueError("margin variables not found in data: " + ", ".join(missing))
    cols = {k: [str(x) for x in v] for k, v in data.items()}
    n = len(next(iter(cols.values())))
    if any(len(v) != n for v in cols.values()):
        raise ValueError("every column of data must have the same length")
    if n == 0:
        raise ValueError("data has no rows")
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    if len(w) != n:
        raise ValueError("weights must have one element per row")
    if any(math.isnan(v) or v <= 0.0 for v in w):
        raise ValueError("starting weights must be positive")
    totals = [sum(m.values()) for m in margins.values()]
    if any(abs(t - totals[0]) > 1e-6 * max(1.0, abs(totals[0])) for t in totals):
        raise ValueError("every margin must sum to the same population total")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    converged = False
    it = 0
    disc = float("nan")
    for iteration in range(1, max_iter + 1):
        it = iteration
        for v, tgt in margins.items():
            lev = cols[v]
            unknown = sorted(set(lev) - set(tgt))
            if unknown:
                raise ValueError(
                    f"variable {v} has levels absent from its margin: "
                    + ", ".join(unknown))
            cur = {k: 0.0 for k in tgt}
            for i, lab in enumerate(lev):
                cur[lab] += w[i]
            for level, target in tgt.items():
                if cur[level] > 0.0:
                    f = target / cur[level]
                    for i, lab in enumerate(lev):
                        if lab == level:
                            w[i] *= f
                elif target > 0.0:
                    raise ValueError(
                        f"level {level} of {v} has a positive target but no sample units")
        disc = 0.0
        for v, tgt in margins.items():
            cur = {k: 0.0 for k in tgt}
            for i, lab in enumerate(cols[v]):
                cur[lab] += w[i]
            for level, target in tgt.items():
                disc = max(disc, abs(cur[level] - target))
        if disc < tol:
            converged = True
            break
    return RakeResult(weights=w, converged=converged, iterations=it,
                      max_discrepancy=disc)


# ---------------------------------------------------------------------------
# Identification error
# ---------------------------------------------------------------------------

@dataclass
class MisclassCorrection:
    """Rogan-Gladen correction of a prevalence for identification error."""

    p_corrected: float
    p_obs: float
    youden: float
    ratio: float
    sensitivity: float
    specificity: float
    se: float = float("nan")
    ci: tuple = ()
    conf: float = float("nan")


def morie_misclass_correct(p_obs, sensitivity, specificity, n=None, conf=0.95):
    """``(p_obs + spec - 1) / (sens + spec - 1)``, after Rogan and Gladen (1978).

    When ``n`` is given, the interval propagates sampling error in ``p_obs``
    only: sensitivity and specificity are treated as known constants. If they
    come from a validation study of finite size, their own uncertainty widens
    the true interval and this one is optimistic.
    """
    p_obs = _prob(p_obs, "p_obs")
    se_ = _prob(sensitivity, "sensitivity")
    sp_ = _prob(specificity, "specificity")
    youden = se_ + sp_ - 1.0
    if abs(youden) < 1e-8:
        raise ValueError(
            "sensitivity + specificity must differ from 1; "
            "the classifier carries no information")
    p_true = (p_obs + sp_ - 1.0) / youden
    out = MisclassCorrection(
        p_corrected=p_true, p_obs=p_obs, youden=youden,
        ratio=(p_true / p_obs) if p_obs > 0 else float("nan"),
        sensitivity=se_, specificity=sp_,
    )
    if n is not None:
        n = float(n)
        if math.isnan(n) or n <= 0.0:
            raise ValueError("n must be a positive number")
        se_p = math.sqrt(p_obs * (1.0 - p_obs) / n) / abs(youden)
        z = _z(conf)
        out.se = se_p
        out.ci = (p_true - z * se_p, p_true + z * se_p)
        out.conf = float(conf)
    return out


@dataclass
class MisclassCount:
    """Count adjusted for identification error."""

    count_corrected: float
    count_observed: float
    undercount: float
    undercount_pct: float


def morie_misclass_count(observed_count, n, sensitivity, specificity):
    """Adjust an observed member count and report what the records miss."""
    observed_count = float(observed_count)
    n = float(n)
    if math.isnan(observed_count) or observed_count < 0.0:
        raise ValueError("observed_count must be a non-negative number")
    if math.isnan(n) or n <= 0.0 or n < observed_count:
        raise ValueError("n must be positive and at least as large as observed_count")
    fit = morie_misclass_correct(observed_count / n, sensitivity, specificity)
    corrected = fit.p_corrected * n
    return MisclassCount(
        count_corrected=corrected,
        count_observed=observed_count,
        undercount=corrected - observed_count,
        undercount_pct=(100.0 * (corrected - observed_count) / corrected)
        if corrected > 0 else float("nan"),
    )


# ---------------------------------------------------------------------------
# Measurement equivalence
# ---------------------------------------------------------------------------

@dataclass
class DifSampleSize:
    """Focal-group sample size for detecting differential item functioning."""

    n_focal: float
    n_reference: float
    n_total: float
    p_focal: float
    p_reference: float
    odds_ratio: float
    ratio: float
    power: float
    alpha: float


def morie_dif_sample_size(p_reference, odds_ratio=None, p_focal=None,
                          ratio=1.0, power=0.8, alpha=0.05):
    """Two-proportion sample size; power is governed by the smaller group."""
    p_r = _prob(p_reference, "p_reference")
    if p_r <= 0.0 or p_r >= 1.0:
        raise ValueError("p_reference must be in (0, 1)")
    if p_focal is None:
        if odds_ratio is None:
            raise ValueError("supply either odds_ratio or p_focal")
        orv = float(odds_ratio)
        if math.isnan(orv) or orv <= 0.0:
            raise ValueError("odds_ratio must be a positive number")
        odds_r = p_r / (1.0 - p_r)
        odds_f = orv * odds_r
        p_f = odds_f / (1.0 + odds_f)
    else:
        p_f = _prob(p_focal, "p_focal")
        orv = (p_f / (1.0 - p_f)) / (p_r / (1.0 - p_r))
    if abs(p_f - p_r) < 1e-12:
        raise ValueError(
            "the two groups have the same response probability; no effect to detect")
    ratio = float(ratio)
    if math.isnan(ratio) or ratio <= 0.0:
        raise ValueError("ratio must be a positive number")
    power = _prob(power, "power")
    alpha = _prob(alpha, "alpha")
    if power <= 0.0 or power >= 1.0:
        raise ValueError("power must be in (0, 1)")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_b = stats.norm.ppf(power)
    p_bar = (p_f + ratio * p_r) / (1.0 + ratio)
    term1 = z_a * math.sqrt((1.0 + 1.0 / ratio) * p_bar * (1.0 - p_bar))
    term2 = z_b * math.sqrt(p_f * (1.0 - p_f) + p_r * (1.0 - p_r) / ratio)
    n_focal = (term1 + term2) ** 2 / (p_r - p_f) ** 2
    return DifSampleSize(
        n_focal=n_focal, n_reference=ratio * n_focal,
        n_total=n_focal * (1.0 + ratio), p_focal=p_f, p_reference=p_r,
        odds_ratio=orv, ratio=ratio, power=power, alpha=alpha,
    )


def morie_dif_delta_mh(or_mh):
    """ETS delta scale ``-2.35 * log(or)`` with the A/B/C classification.

    Returns a list of dicts with ``or_mh``, ``delta_mh``, ``magnitude`` and
    ``favours``.
    """
    if isinstance(or_mh, (int, float)):
        or_mh = [or_mh]
    or_mh = [float(v) for v in or_mh]
    if not or_mh:
        raise ValueError("or_mh must have at least one element")
    if any(v <= 0.0 for v in or_mh if not math.isnan(v)):
        raise ValueError("or_mh must be positive")
    out = []
    for v in or_mh:
        delta = -2.35 * math.log(v)
        ad = abs(delta)
        mag = "A" if ad < 1.0 else ("B" if ad < 1.5 else "C")
        fav = "neither" if ad < 1e-12 else ("focal" if delta < 0 else "reference")
        out.append({"or_mh": v, "delta_mh": delta, "magnitude": mag,
                    "favours": fav})
    return out


def morie_invariance_compare(fits, cfi_cut=0.01, rmsea_cut=0.015):
    """Compare nested invariance models on chi-square, CFI and RMSEA.

    ``fits`` is a sequence of mappings with ``model``, ``chisq``, ``df``,
    ``cfi`` and ``rmsea``, ordered from least to most constrained.
    """
    rows = list(fits)
    need = ("model", "chisq", "df", "cfi", "rmsea")
    for r in rows:
        missing = [k for k in need if k not in r]
        if missing:
            raise ValueError("fits is missing keys: " + ", ".join(missing))
    if len(rows) < 2:
        raise ValueError("at least two models are required to compare")
    df = [float(r["df"]) for r in rows]
    if any(df[i + 1] - df[i] <= 0 for i in range(len(df) - 1)):
        raise ValueError(
            "models must be ordered from least to most constrained (df must increase)")
    chisq = [float(r["chisq"]) for r in rows]
    cfi = [float(r["cfi"]) for r in rows]
    rmsea = [float(r["rmsea"]) for r in rows]
    out = []
    for i in range(len(rows) - 1):
        d_chisq = chisq[i + 1] - chisq[i]
        d_df = df[i + 1] - df[i]
        pval = (stats.chi2.sf(d_chisq, df=d_df)
                if d_chisq > 0 else float("nan"))
        d_cfi = cfi[i + 1] - cfi[i]
        d_rmsea = rmsea[i + 1] - rmsea[i]
        out.append({
            "step": f"{rows[i]['model']} -> {rows[i + 1]['model']}",
            "delta_chisq": d_chisq,
            "delta_df": d_df,
            "p_value": pval,
            "delta_cfi": d_cfi,
            "delta_rmsea": d_rmsea,
            "supported": (-d_cfi <= cfi_cut) and (d_rmsea <= rmsea_cut),
        })
    return out


# ---------------------------------------------------------------------------
# Score precision
# ---------------------------------------------------------------------------

def morie_irt_theta_se(information):
    """``SE(theta) = 1 / sqrt(I(theta))``."""
    scalar = isinstance(information, (int, float))
    vals = [float(information)] if scalar else [float(v) for v in information]
    if not vals:
        raise ValueError("information must have at least one element")
    if any(v <= 0.0 for v in vals if not math.isnan(v)):
        raise ValueError("information must be positive")
    out = [1.0 / math.sqrt(v) for v in vals]
    return out[0] if scalar else out


def morie_irt_marginal_reliability(se, var_theta=1.0):
    """``1 - mean(se^2) / var_theta``, the marginal reliability."""
    if isinstance(se, (int, float)):
        se = [se]
    vals = [float(v) for v in se]
    if not vals:
        raise ValueError("se must have at least one element")
    if any(v < 0.0 for v in vals if not math.isnan(v)):
        raise ValueError("se must be non-negative")
    var_theta = float(var_theta)
    if math.isnan(var_theta) or var_theta <= 0.0:
        raise ValueError("var_theta must be a positive number")
    finite = [v for v in vals if not math.isnan(v)]
    mean_sq = sum(v * v for v in finite) / len(finite)
    return 1.0 - mean_sq / var_theta
