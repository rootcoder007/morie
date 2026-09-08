"""Effective reproduction number by Wallinga-Teunis (2004)."""

from ._richresult import RichResult

__all__ = ["rtwall", "wallinga_teunis_rt"]


def rtwall(onset_times, gi_pmf):
    """
    Case reproduction numbers from symptom-onset times.

    Wallinga & Teunis (2004), likelihood-based pairs method: with
    w(tau) the generation-interval distribution, the relative
    likelihood that case i was infected by case j is

        p_ij = w(t_i - t_j) / sum_{k != i} w(t_i - t_k),

    and the effective reproduction number of case j is
    R_j = sum_i p_ij (their equations on p. 511; no assumption on
    the offspring distribution).  Each case i with at least one
    possible infector distributes exactly one unit of probability,
    so sum_j R_j equals the number of such cases -- an identity this
    implementation preserves exactly.  A day-aggregated R_t (mean
    R_j over cases with onset on day t) is also returned.

    Sources
    -------
    Wallinga, J. & Teunis, P. (2004). Different epidemic curves for
    severe acute respiratory syndrome reveal similar impacts of
    control measures. *American Journal of Epidemiology*, 160(6),
    509-516, Eqs. for p_ij and R_j and Appendix 1 (local copy
    fetched-wave3/wallinga-teunis-2004-rt.pdf).

    Parameters
    ----------
    onset_times : sequence of int
        Symptom-onset day per case (any integer scale).
    gi_pmf : sequence of float
        Generation-interval probability masses w(1), w(2), ...
        (w at lag 0 is taken as 0; lags beyond the vector are 0).

    Returns
    -------
    RichResult
        Keys: r_case (R_j per case), r_daily ({day: mean R_j}),
        n_cases, mass_check (sum R_j minus number of cases with an
        infector; 0 up to float error).
    """
    t = [int(v) for v in onset_times]
    w = [float(v) for v in gi_pmf]
    n = len(t)
    if n < 2:
        raise ValueError("need at least two cases")
    if not w or any(v < 0 for v in w):
        raise ValueError("gi_pmf must be non-negative and non-empty")

    def wfun(lag):
        return w[lag - 1] if 1 <= lag <= len(w) else 0.0

    r = [0.0] * n
    n_with_infector = 0
    for i in range(n):
        denom = 0.0
        for k in range(n):
            if k != i:
                denom += wfun(t[i] - t[k])
        if denom <= 0.0:
            continue
        n_with_infector += 1
        for j in range(n):
            if j != i:
                r[j] += wfun(t[i] - t[j]) / denom
    days = sorted(set(t))
    r_daily = {}
    for d in days:
        vals = [r[j] for j in range(n) if t[j] == d]
        r_daily[d] = sum(vals) / len(vals)
    return RichResult(payload={
        "r_case": r,
        "r_daily": r_daily,
        "n_cases": n,
        "mass_check": sum(r) - n_with_infector,
        "method": "Wallinga-Teunis (2004) case reproduction numbers",
    })


# long descriptive alias (stub-era name)
wallinga_teunis_rt = rtwall


def cheatsheet():
    return "rtwall: p_ij = w(ti-tj)/sum_k w(ti-tk); R_j = sum_i p_ij"

# public names resolved by fn/_lazy_map.json
rt_wallinga_teunis = rtwall
