# morie.fn -- function file (rootcoder007/morie)
r"""Self-controlled case series: cases only, each its own control.

A cohort study of a rare vaccine reaction needs the whole cohort. A
case-control study needs matched controls. The self-controlled case
series needs **neither** -- only the people who had the event, and for
each of them the dates of exposure and of the event. Everything else
cancels.

**Why it cancels.** Events arise in an age-dependent Poisson process
whose rate for individual :math:`i` is

.. math:: \lambda_i(t \mid v_i, x_i)
          = \lambda_{i0}(t)\,
            \exp\!\big(\gamma^{\top}x_i + \textstyle\sum_r
            \beta_r X_{ir}(t)\big),

with :math:`X_{ir}(t) = 1` when :math:`t` falls in the *r*-th risk
interval :math:`(v_i + a_r,\, v_i + b_r]` after vaccination at age
:math:`v_i`, and :math:`\lambda_{i0}` piecewise constant on age bands.
Writing the log baseline as :math:`\varphi_i + \alpha_j` splits it into
an **individual** effect and an **age** effect. Conditioning on the
number of events a person had -- and on their exposure history --
removes :math:`\varphi_i` and :math:`\gamma^{\top}x_i` **exactly**,
because both are constants that multiply every interval of that
person's follow-up alike. The conditional likelihood is multinomial:

.. math:: L = \prod_i \prod_{k=1}^{n_i}
          \frac{\exp(\alpha_{j(t_{ik})} + \beta_{r(t_{ik})})}
               {\sum_{j} e_{ij}\,
                \exp(\alpha_{j} + \beta_{r(j)})},

where :math:`e_{ij}` is the time individual *i* spent in interval
*j*. Any fixed characteristic -- genotype, frailty, socioeconomic
status, anything measured or not -- is inside :math:`\varphi_i` and
therefore cannot confound. That is the whole point, and the anchor
tests it as an exact invariance rather than as a claim: scaling one
person's underlying rate by any constant leaves every estimate
bit-for-bit unchanged.

**What does NOT cancel.** Age does, and only because it is modelled.
Anything that varies *within* a person over time is a live confounder:
if the age bands are too coarse, age leaks into
:math:`\hat\beta`. The anchor plants an age effect and shows the
estimate is biased when the bands are omitted and recovered when they
are not.

**The assumption in this module's name.** Farrington's derivation
requires that the event does not alter subsequent observation -- no
event-dependent censoring (the event must not be fatal or
observation-terminating) and no event-dependent exposure (having the
event must not change whether or when you get vaccinated). This
module implements the case where those hold. ``check_assumptions``
reports the two diagnostics that bear on them; it does not pretend to
test what is untestable from cases alone.

**A pre-exposure window is a diagnostic, not decoration.** If
vaccination is deferred because a child is unwell, event rates dip
just *before* exposure. Fitting an explicit pre-exposure interval
makes that visible: a pre-exposure relative incidence far from 1 is
evidence the exposure was event-dependent, which invalidates the
design rather than merely inflating a standard error.

References
----------
Farrington, C. P. (1995) "Relative Incidence Estimation from Case
Series for Vaccine Safety Evaluation", *Biometrics* 51(1), 228-235.
JSTOR stable URL https://www.jstor.org/stable/2533328. The article
prints no DOI. Secs. 2-3: the incidence parameterisation, the
cutpoint construction of Fig. 1, and the conditioning argument that
eliminates the individual effects.

Whitaker, H. J., Farrington, C. P., Spiessens, B. & Musonda, P. (2006)
"Tutorial in biostatistics: The self-controlled case series method",
*Statistics in Medicine* 25, 1768-1797, doi:10.1002/sim.2302
(published online 11 October 2005). The practical treatment of age
groups, multiple risk periods and pre-exposure windows followed here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["build_intervals", "sccs_loglik", "sccs_fit",
           "relative_incidence", "check_assumptions"]

_EPS = 1e-12


def _cuts(start, end, exposure, risk_periods, age_breaks):
    """Ordered distinct cutpoints for one individual (Fig. 1)."""
    pts = {float(start), float(end)}
    for b in age_breaks:
        if float(start) < float(b) < float(end):
            pts.add(float(b))
    if exposure is not None:
        for a, b in risk_periods:
            for p in (float(exposure) + float(a),
                      float(exposure) + float(b)):
                if float(start) < p < float(end):
                    pts.add(p)
    return sorted(pts)


def _band(t, age_breaks):
    """Index of the age band containing t."""
    j = 0
    for b in age_breaks:
        if t >= float(b):
            j += 1
        else:
            break
    return j


def _risk(t, exposure, risk_periods):
    """Index of the risk period containing t; 0 is the control period."""
    if exposure is None:
        return 0
    for r, (a, b) in enumerate(risk_periods, start=1):
        if float(exposure) + float(a) < t <= float(exposure) + float(b):
            return r
    return 0


def build_intervals(start, end, exposure, event_times, risk_periods,
                    age_breaks):
    r"""One individual's follow-up, cut into (age band, risk period)
    cells with their exposure times and event counts.

    Returns a list of ``(age_band, risk_period, e_ij, n_ij)``. The
    :math:`e_{ij}` are the observation times of Farrington's Sec. 3;
    the :math:`n_{ij}` are the events falling in each cell.
    """
    s, e = float(start), float(end)
    if not e > s:
        raise ValueError("sccsno: the observation period must have "
                         "positive length, got [%g, %g]" % (s, e))
    for a, b in risk_periods:
        if not float(b) > float(a):
            raise ValueError("sccsno: a risk period must satisfy "
                             "b > a, got (%g, %g]" % (a, b))
    if exposure is not None and not s <= float(exposure) <= e:
        raise ValueError("sccsno: the exposure at %g lies outside the "
                         "observation period [%g, %g]"
                         % (exposure, s, e))
    cuts = _cuts(s, e, exposure, risk_periods, age_breaks)
    cells = []
    for q in range(len(cuts) - 1):
        lo, hi = cuts[q], cuts[q + 1]
        mid = 0.5 * (lo + hi)
        cells.append([_band(mid, age_breaks),
                      _risk(mid, exposure, risk_periods),
                      hi - lo, 0])
    for t in event_times:
        tv = float(t)
        if not s <= tv <= e:
            raise ValueError("sccsno: an event at %g lies outside the "
                             "observation period [%g, %g]"
                             % (tv, s, e))
        placed = False
        for q in range(len(cuts) - 1):
            if cuts[q] < tv <= cuts[q + 1] or (q == 0 and tv == cuts[0]):
                cells[q][3] += 1
                placed = True
                break
        if not placed:
            cells[-1][3] += 1
    return [tuple(c) for c in cells]


def sccs_loglik(params, cells_by_person, n_risk, n_age):
    r"""The conditional log-likelihood of Sec. 3.

    ``params`` is :math:`(\beta_1..\beta_s, \alpha_1..\alpha_{m-1})`
    with :math:`\beta_0 = \alpha_0 = 0`. The individual effects
    :math:`\varphi_i` do not appear -- that is the point of
    conditioning.
    """
    beta = [0.0] + [float(v) for v in params[:n_risk]]
    alpha = [0.0] + [float(v) for v in params[n_risk:n_risk + n_age - 1]]
    ll = 0.0
    for cells in cells_by_person:
        tot = sum(n for _, _, _, n in cells)
        if tot == 0:
            continue
        den = 0.0
        for j, r, e, _ in cells:
            den += e * math.exp(alpha[j] + beta[r])
        if den <= _EPS:
            raise ValueError("sccsno: an individual has no observation "
                             "time")
        for j, r, e, n in cells:
            if n:
                ll += n * (alpha[j] + beta[r])
        ll -= tot * math.log(den)
    return ll


def _grad_hess(params, cells_by_person, n_risk, n_age):
    p = n_risk + n_age - 1
    g = [0.0] * p
    H = [[0.0] * p for _ in range(p)]
    beta = [0.0] + [float(v) for v in params[:n_risk]]
    alpha = [0.0] + [float(v) for v in params[n_risk:p]]

    def idx(j, r):
        """Design row: risk dummies then age dummies, both baseline 0."""
        row = [0.0] * p
        if r > 0:
            row[r - 1] = 1.0
        if j > 0:
            row[n_risk + j - 1] = 1.0
        return row

    for cells in cells_by_person:
        tot = sum(n for _, _, _, n in cells)
        if tot == 0:
            continue
        w, rows = [], []
        den = 0.0
        for j, r, e, _ in cells:
            v = e * math.exp(alpha[j] + beta[r])
            den += v
            w.append(v)
            rows.append(idx(j, r))
        pr = [v / den for v in w]
        for c, (j, r, e, n) in enumerate(cells):
            if n:
                for a in range(p):
                    g[a] += n * rows[c][a]
        mean = [sum(pr[c] * rows[c][a] for c in range(len(cells)))
                for a in range(p)]
        for a in range(p):
            g[a] -= tot * mean[a]
            for b in range(p):
                sec = sum(pr[c] * rows[c][a] * rows[c][b]
                          for c in range(len(cells)))
                H[a][b] -= tot * (sec - mean[a] * mean[b])
    return g, H


def sccs_fit(cases, risk_periods, age_breaks=(), iters=100, tol=1e-10,
             ridge=1e-10):
    r"""Maximise the conditional likelihood by Newton-Raphson.

    ``cases`` is a sequence of dicts with keys ``start``, ``end``,
    ``exposure`` (or ``None``) and ``events``. Only individuals with at
    least one event contribute -- the rest carry no information, which
    is why the design needs cases only.
    """
    rp = [(float(a), float(b)) for a, b in risk_periods]
    ab = [float(v) for v in age_breaks]
    if ab != sorted(ab):
        raise ValueError("sccsno: age_breaks must be increasing")
    n_risk, n_age = len(rp), len(ab) + 1
    if n_risk < 1:
        raise ValueError("sccsno: at least one risk period is needed")
    cells_by_person, used = [], 0
    for c in cases:
        ev = list(c.get("events", ()))
        if not ev:
            continue
        cells = build_intervals(c["start"], c["end"], c.get("exposure"),
                                ev, rp, ab)
        cells_by_person.append(cells)
        used += 1
    if used == 0:
        raise ValueError("sccsno: no case contributed an event")
    p = n_risk + n_age - 1
    par = [0.0] * p
    conv, it = False, 0
    for it in range(1, int(iters) + 1):
        g, H = _grad_hess(par, cells_by_person, n_risk, n_age)
        A = [[-H[a][b] + (ridge if a == b else 0.0) for b in range(p)]
             for a in range(p)]
        try:
            step = k.cholsolve(A, g)
        except Exception:
            raise ValueError("sccsno: the information matrix is "
                             "singular -- some interval carries no "
                             "events or no exposure time")
        mx = 0.0
        for a in range(p):
            par[a] += step[a]
            mx = max(mx, abs(step[a]))
        if mx < tol:
            conv = True
            break
    g, H = _grad_hess(par, cells_by_person, n_risk, n_age)
    A = [[-H[a][b] + (ridge if a == b else 0.0) for b in range(p)]
         for a in range(p)]
    cols = [k.cholsolve(A, [1.0 if q == a else 0.0 for q in range(p)])
            for a in range(p)]
    se = [math.sqrt(cols[a][a]) if cols[a][a] > 0 else float("nan")
          for a in range(p)]
    beta = par[:n_risk]
    return RichResult(payload={
        "estimate": [math.exp(v) for v in beta],
        "relative_incidence": [math.exp(v) for v in beta],
        "log_ri": beta, "se_log_ri": se[:n_risk],
        "age_effects": par[n_risk:], "se_age": se[n_risk:],
        "coef": par, "se": se,
        "loglik": sccs_loglik(par, cells_by_person, n_risk, n_age),
        "n_cases": used, "converged": conv, "iterations": it,
        "n_risk_periods": n_risk, "n_age_bands": n_age,
        "method": "self-controlled case series, conditional "
                  "likelihood of Farrington (1995) Sec. 3",
        "conditions_out": "individual frailty and every "
                          "time-invariant covariate",
    })


def relative_incidence(fit, level=0.95):
    """Point estimates and Wald intervals on the incidence scale."""
    z = k.qnorm(0.5 + float(level) / 2.0)
    out = []
    for b, s in zip(fit["log_ri"], fit["se_log_ri"]):
        out.append({"ri": math.exp(b),
                    "lower": math.exp(b - z * s),
                    "upper": math.exp(b + z * s),
                    "log_ri": b, "se": s})
    return {"intervals": out, "level": float(level)}


def check_assumptions(fit_with_pre, pre_index=0, tol=0.25):
    r"""Read the pre-exposure window as a design diagnostic.

    A relative incidence far from 1 in a window *before* exposure means
    the event influenced whether or when exposure happened. That breaks
    the derivation itself -- it is not a bias to be adjusted away.
    """
    ri = fit_with_pre["relative_incidence"][int(pre_index)]
    ok = abs(math.log(ri)) <= float(tol)
    return {"pre_exposure_ri": ri, "consistent_with_design": ok,
            "tolerance_log": float(tol),
            "interpretation":
                "a pre-exposure RI near 1 is consistent with "
                "event-independent exposure; far from 1 indicates the "
                "event affected exposure, which invalidates the "
                "design rather than biasing it"}


def cheatsheet():
    return ("sccsno: SCCS. Cases ONLY. Conditioning on each person's "
            "event count cancels phi_i exactly, so every "
            "time-INVARIANT confounder -- measured or not -- is gone "
            "by construction. What does NOT cancel is anything varying "
            "WITHIN a person: age must be modelled with bands or it "
            "leaks into beta. Requires no event-dependent censoring "
            "and no event-dependent exposure; a pre-exposure window "
            "with RI far from 1 says the latter failed.")


# compact alias per ledger/NAMING.md
sccsnoevent = sccs_fit

# public names resolved by fn/_lazy_map.json
sccs_no_replacement = sccs_fit
