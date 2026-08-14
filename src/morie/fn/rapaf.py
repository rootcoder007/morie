# morie.fn -- function file (rootcoder007/morie)
r"""Adjusted population attributable risk from case-control data.

The population attributable risk is the fraction of disease that would
not have occurred had the exposure's effect been absent. The naive
version -- Levin's formula with a single factor -- is fine when there
is one factor and nothing confounds it. Neither is usually true.

**The formula that does the work.** Cross-classify the population into
:math:`J` strata by the :math:`K` risk factors, let stratum
:math:`j = 0` be the lowest-risk one, and let :math:`R_j` be the rate
ratio of stratum :math:`j` against it. Bruzzi et al.'s eq. (3) writes
the disease rate under the counterfactual in terms of the *cases*, and
their eq. (6) is

.. math:: AR = 1 - \frac{1}{x}\sum_{\text{cases}} \frac{1}{R_j}
              = 1 - \sum_{j} \frac{\rho_j}{R_j},

where :math:`\rho_j` is the proportion of **cases** in stratum
:math:`j`. Note what is absent: the distribution of exposure in the
population. Only the case distribution and the rate ratios appear.

**Why that matters more than it looks.** A case-control study samples
cases and gives relative risks, but its controls are not a
representative population sample -- especially when matched. The
formula above never asks them to be. That is the paper's central
point, made against the claim that multifactorial attributable risk
requires knowing the joint exposure distribution in the population. It
does not. ``population_attributable_risk`` therefore takes case counts
and rate ratios, and nothing else.

**Adjustment and interaction come free.** Because :math:`R_j` is the
rate ratio for a full cross-classified stratum, confounding between
factors is handled by the stratification itself, and interactions are
handled if the model that produced :math:`R_j` included them. The
paper's recommendation -- fit logistic regression rather than
tabulate, so that thin strata borrow strength, and include the
interactions that matter rather than all of them -- is available here
as ``rate_ratios_from_logit``.

**The single-factor case is a special case, and it is checked as one.**
With one dichotomous factor at case prevalence :math:`\rho_1` and rate
ratio :math:`R`,

.. math:: AR = 1 - \left(\rho_0 + \frac{\rho_1}{R}\right)
             = \rho_1\left(1 - \frac1R\right),

which is Levin's formula rewritten in case proportions. The anchor
verifies that this agrees with Levin's population-prevalence form
:math:`p(R-1)/[1 + p(R-1)]` exactly, which it must, since the two are
the same quantity written from different data.

**Partial attributable risk is not additive.** The AR for a set of
factors is generally **less** than the sum of the single-factor ARs,
because a case exposed to two factors is counted once. Anyone adding
them up is double-counting; ``partial_ar`` computes the correct
quantity for a subset by setting only that subset to baseline, and the
anchor demonstrates the shortfall rather than asserting it.

References
----------
Bruzzi, P., Green, S. B., Byar, D. P., Brinton, L. A. & Schairer, C.
(1985) "Estimating the Population Attributable Risk for Multiple Risk
Factors Using Case-Control Data", *American Journal of Epidemiology*
122(5), 904-914. The article prints no DOI. Sec. "Statistical
development": the stratification and notation, eq. (3) for the
counterfactual rate and eq. (6) for the case-based attributable risk
implemented here; the discussion of matched controls and of logistic
regression for thin strata.

Levin, M. L. (1953) "The occurrence of lung cancer in man", *Acta
Unio Internationalis Contra Cancrum* 9(3), 531-541. The single-factor
formula this generalises; cited by Bruzzi et al. as reference (1) and
used here only as the closed form the anchor checks against.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["population_attributable_risk", "levin_ar", "partial_ar",
           "rate_ratios_from_logit", "ar_confidence_interval"]

_EPS = 1e-12


def _norm(case_counts, rate_ratios):
    n = len(case_counts)
    if len(rate_ratios) != n:
        raise ValueError("rapaf: %d strata of cases but %d rate "
                         "ratios" % (n, len(rate_ratios)))
    if n < 2:
        raise ValueError("rapaf: need at least 2 strata, got %d" % n)
    cc = [float(v) for v in case_counts]
    rr = [float(v) for v in rate_ratios]
    if any(v < 0.0 for v in cc):
        raise ValueError("rapaf: case counts must be non-negative")
    if any(v <= 0.0 for v in rr):
        raise ValueError("rapaf: rate ratios must be positive, got %r"
                         % (min(rr),))
    tot = sum(cc)
    if tot <= _EPS:
        raise ValueError("rapaf: there are no cases")
    return cc, rr, tot


def population_attributable_risk(case_counts, rate_ratios):
    r"""Bruzzi eq. (6): :math:`AR = 1 - \sum_j \rho_j / R_j`.

    ``case_counts[j]`` is the number of **cases** in stratum *j* and
    ``rate_ratios[j]`` the adjusted rate ratio of that stratum against
    the baseline. The control distribution is not used and is not
    required -- which is what makes this usable with matched controls.
    """
    cc, rr, tot = _norm(case_counts, rate_ratios)
    rho = [v / tot for v in cc]
    s = sum(rho[j] / rr[j] for j in range(len(cc)))
    return RichResult(payload={
        "estimate": 1.0 - s, "ar": 1.0 - s,
        "case_proportions": rho, "rate_ratios": rr,
        "n_cases": tot, "n_strata": len(cc),
        "uses_control_distribution": False,
        "method": "Bruzzi, Green, Byar, Brinton & Schairer (1985) "
                  "eq. (6), case-based adjusted attributable risk",
    })


def levin_ar(prevalence, rate_ratio):
    r"""Levin's single-factor formula, :math:`p(R-1)/[1+p(R-1)]`.

    Written from the **population** prevalence rather than the case
    proportion. Provided so the general formula can be checked against
    the classical one on the case they share.
    """
    p, R = float(prevalence), float(rate_ratio)
    if not 0.0 <= p <= 1.0:
        raise ValueError("rapaf: prevalence must be in [0, 1], got %r"
                         % (prevalence,))
    if R <= 0.0:
        raise ValueError("rapaf: the rate ratio must be positive")
    d = 1.0 + p * (R - 1.0)
    if abs(d) <= _EPS:
        raise ValueError("rapaf: Levin's formula is undefined here "
                         "(1 + p(R-1) = 0)")
    return p * (R - 1.0) / d


def partial_ar(case_counts, rate_ratios, baseline_map):
    r"""AR for a subset of factors, by moving only that subset to
    baseline.

    ``baseline_map[j]`` names the stratum that *j* would become if the
    factors of interest were absent, leaving the others as they are.
    The result is the fraction of disease attributable to those factors
    alone; summing single-factor ARs instead double-counts cases
    exposed to more than one.
    """
    cc, rr, tot = _norm(case_counts, rate_ratios)
    n = len(cc)
    bm = [int(v) for v in baseline_map]
    if len(bm) != n:
        raise ValueError("rapaf: %d baseline targets for %d strata"
                         % (len(bm), n))
    if any(not 0 <= v < n for v in bm):
        raise ValueError("rapaf: a baseline target is out of range")
    s = 0.0
    for j in range(n):
        s += (cc[j] / tot) * (rr[bm[j]] / rr[j])
    return {"estimate": 1.0 - s, "ar": 1.0 - s,
            "baseline_map": bm,
            "note": "partial ARs are NOT additive across factor sets; "
                    "a case exposed to two factors is counted once"}


def rate_ratios_from_logit(case_counts, control_counts, design,
                           ridge=1e-8):
    r"""Adjusted rate ratios by logistic regression, as the paper
    recommends for thin strata.

    ``design[j]`` is the covariate row for stratum *j*; the odds ratio
    relative to stratum 0 is returned. For a rare disease the odds
    ratio approximates the rate ratio, which is the standard reading
    and is stated rather than hidden.
    """
    ca = [float(v) for v in case_counts]
    co = [float(v) for v in control_counts]
    D = k.mat(design)
    n = len(ca)
    if not (len(co) == len(D) == n):
        raise ValueError("rapaf: cases, controls and design must "
                         "agree in length (%d, %d, %d)"
                         % (n, len(co), len(D)))
    rows, y, w = [], [], []
    for j in range(n):
        if ca[j] > 0:
            rows.append(list(D[j]))
            y.append(1.0)
            w.append(ca[j])
        if co[j] > 0:
            rows.append(list(D[j]))
            y.append(0.0)
            w.append(co[j])
    if not rows:
        raise ValueError("rapaf: no stratum has any observations")
    X = k.design(rows, len(rows))
    beta = k.logit_irls(X, y, ridge=ridge, obs_weights=w)
    lin = [sum(k.design([list(D[j])], 1)[0][a] * beta[a]
               for a in range(len(beta))) for j in range(n)]
    return {"rate_ratios": [math.exp(lin[j] - lin[0])
                            for j in range(n)],
            "coef": beta,
            "note": "odds ratios; equal to rate ratios only under the "
                    "rare-disease approximation"}


def ar_confidence_interval(case_counts, rate_ratios, log_rr_se,
                           level=0.95, draws=2000, seed=0):
    r"""A Monte Carlo interval that respects :math:`AR \le 1`.

    The rate ratios are resampled on the log scale, the AR recomputed
    each time, and the percentiles reported. A delta-method interval on
    the AR scale can cross 1, which is impossible; this cannot.
    """
    cc, rr, tot = _norm(case_counts, rate_ratios)
    se = [float(v) for v in log_rr_se]
    if len(se) != len(rr):
        raise ValueError("rapaf: %d standard errors for %d rate "
                         "ratios" % (len(se), len(rr)))
    if any(v < 0.0 for v in se):
        raise ValueError("rapaf: standard errors must be non-negative")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("rapaf: level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    rho = [v / tot for v in cc]
    vals = []
    for _ in range(int(draws)):
        rs = [math.exp(math.log(rr[j]) + se[j] * float(rng.normal()))
              for j in range(len(rr))]
        vals.append(1.0 - sum(rho[j] / rs[j] for j in range(len(rr))))
    vals.sort()
    lo_q = (1.0 - float(level)) / 2.0
    return {"estimate": 1.0 - sum(rho[j] / rr[j]
                                  for j in range(len(rr))),
            "lower": vals[int(lo_q * len(vals))],
            "upper": vals[min(len(vals) - 1,
                              int((1.0 - lo_q) * len(vals)))],
            "level": float(level), "draws": int(draws)}


def cheatsheet():
    return ("rapaf: AR = 1 - sum_j rho_j / R_j with rho over CASES "
            "(Bruzzi et al. 1985 eq. 6). The population exposure "
            "distribution never enters, so MATCHED controls are fine. "
            "Stratifying on the full cross-classification handles "
            "confounding; interactions come from the model that made "
            "R_j. Single dichotomous factor reduces to Levin exactly. "
            "Partial ARs do NOT add up -- a case exposed twice is one "
            "case.")


# compact alias per ledger/NAMING.md
adjustedpaf = population_attributable_risk

# public names resolved by fn/_lazy_map.json
adjusted_paf = population_attributable_risk
