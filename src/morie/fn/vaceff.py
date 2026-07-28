# morie.fn -- function file (rootcoder007/morie)
"""Vaccine efficacy."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["vaccine_efficacy"]


def vaccine_efficacy(cases_vaccinated, n_vaccinated, cases_control,
                     n_control, person_time_vaccinated=None,
                     person_time_control=None, alpha=0.05):
    r"""Vaccine efficacy with an interval that behaves at the boundary.

    .. math::
       VE = 1 - RR = 1 - \frac{\text{risk}_v}{\text{risk}_c}

    The interval is built on :math:`\log RR`, not on :math:`VE`
    directly, and that choice is what makes it usable. :math:`VE` is
    bounded above by 1 and unbounded below, so its sampling
    distribution is badly skewed; :math:`\log RR` is close to normal,
    and transforming its interval back gives bounds that respect the
    boundary automatically. A Wald interval on :math:`VE` routinely
    produces an upper bound above 1, which is not a possible value.

    When zero cases occur in the vaccinated arm -- the situation a
    successful trial is designed to produce -- the log risk ratio is
    :math:`-\infty` and the standard interval fails entirely.
    ``zero_cases`` flags it and the interval falls back to the exact
    conditional binomial (Clopper-Pearson on the case split), which is
    what regulatory submissions use.

    ``person_time_*`` switches from risk to INCIDENCE RATE. The
    distinction matters whenever follow-up differs between arms, which
    it usually does: a risk ratio then confounds efficacy with
    differential follow-up.

    Parameters
    ----------
    cases_vaccinated, n_vaccinated : int
    cases_control, n_control : int
    person_time_vaccinated, person_time_control : float, optional
    alpha : float

    Returns
    -------
    RichResult
        ``efficacy``, ``ci``, ``risk_ratio``, ``basis``,
        ``zero_cases``, ``attack_rates``, ``prevented_fraction``.

    References
    ----------
    Halloran, Longini and Struchiner (2010), *Design and Analysis of
    Vaccine Studies*, Springer, chapters 6-7.
    Clopper and Pearson (1934) for the exact interval.

    Examples
    --------
    >>> out = vaccine_efficacy(8, 1000, 80, 1000)
    >>> round(float(out["efficacy"]), 3)
    0.9
    """
    av, nv = int(cases_vaccinated), int(n_vaccinated)
    ac, nc = int(cases_control), int(n_control)
    for name, v in (("cases_vaccinated", av), ("cases_control", ac)):
        if v < 0:
            raise ValueError("%s must be non-negative." % name)
    if nv <= 0 or nc <= 0:
        raise ValueError("arm sizes must be positive.")
    if av > nv or ac > nc:
        raise ValueError("cases cannot exceed the arm size.")

    rate_basis = (person_time_vaccinated is not None
                  and person_time_control is not None)
    if rate_basis:
        pv = float(person_time_vaccinated)
        pc = float(person_time_control)
        if pv <= 0 or pc <= 0:
            raise ValueError("person-time must be positive.")
        rv, rc = av / pv, ac / pc
        var_log = (1.0 / av if av else np.inf) + (1.0 / ac if ac else np.inf)
    else:
        rv, rc = av / nv, ac / nc
        var_log = ((1.0 / av - 1.0 / nv) if av else np.inf) + \
                  ((1.0 / ac - 1.0 / nc) if ac else np.inf)

    if rc <= 0:
        raise ValueError(
            "no cases in the control arm; efficacy is not estimable."
        )
    rr = rv / rc
    ve = 1.0 - rr
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z(1 - alpha / 2)
    zero = av == 0

    if not zero and np.isfinite(var_log) and var_log > 0:
        se = math.sqrt(var_log)
        lo_rr = rr * math.exp(-z * se)
        hi_rr = rr * math.exp(z * se)
        ci = (1.0 - hi_rr, 1.0 - lo_rr)
        method_ci = "log risk ratio"
    else:
        # exact conditional interval on the case split
        total = av + ac
        ratio = (nv / nc) if not rate_basis else (pv / pc)
        lo_p, hi_p = _clopper_pearson(av, total, alpha)
        def to_ve(p):
            if p >= 1.0:
                return -np.inf
            return 1.0 - (p / (1.0 - p)) / ratio
        ci = (to_ve(hi_p), to_ve(lo_p))
        method_ci = "exact conditional binomial"
    return RichResult(
        payload={
            "estimate": float(ve),
            "efficacy": float(ve),
            "ci": (float(ci[0]), float(ci[1])),
            "risk_ratio": float(rr),
            "basis": "incidence rate" if rate_basis else "risk",
            "basis_note": (
                "risk ratios confound efficacy with differential follow-up; "
                "supply person-time when the arms were not followed equally"
            ),
            "attack_rates": (float(rv), float(rc)),
            "zero_cases": bool(zero),
            "interval_method": method_ci,
            "interval_note": (
                "built on log RR, which is near-normal, then transformed "
                "back; a Wald interval on VE itself routinely returns an "
                "upper bound above 1, which is not a possible value"
            ),
            "zero_note": (
                None if not zero else
                "no cases in the vaccinated arm, so log RR is -inf and the "
                "standard interval fails; the exact conditional binomial is "
                "used instead, as in regulatory practice"
            ),
            "prevented_fraction": float(ve * ac / max(ac, 1)),
            "cases": (av, ac),
            "n": (nv, nc),
            "method": "Vaccine efficacy (1 - %s ratio)"
                      % ("rate" if rate_basis else "risk"),
        }
    )


def _clopper_pearson(k, n, alpha):
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else _beta_q(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_q(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def _beta_q(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _beta_cdf(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # regularised incomplete beta by its continued fraction
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-14:
            break
    val = front * (f - 1.0)
    if x < (a + 1) / (a + b + 2):
        return min(max(val, 0.0), 1.0)
    return min(max(1.0 - _beta_cdf(1 - x, b, a), 0.0), 1.0)


def _z(q):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return (
        "vaceff: vaccine efficacy on the log-RR scale, with an exact "
        "fallback when the vaccinated arm has zero cases"
    )
