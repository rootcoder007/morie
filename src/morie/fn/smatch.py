# morie.fn -- function file (rootcoder007/morie)
r"""Fitting and sizing a self-controlled case series.

:mod:`sccsno` derives the case-series estimator and maximises its
conditional likelihood directly. This module covers the two things the
tutorial adds on top: **how the model is actually fitted in general
software**, and **how many events a study needs**.

**The multinomial model is a Poisson model in disguise.** Sec. 4 of
the tutorial: put the event count in each interval on the left, the
log of the time spent in that interval in as an offset, and include a
factor for **each individual** alongside the age and exposure factors,

.. math:: n_{ijk} \sim \mathrm{Poisson}(\mu_{ijk} e_{ijk}), \qquad
          \log \mu_{ijk} = \varphi_i + \alpha_j + \beta_k .

The per-individual factors force the fitted totals to equal the
observed totals, which is exactly the conditioning that produced the
multinomial likelihood. So the two fits are not similar, they are the
**same fit** -- and the anchor checks that the coefficients agree to
solver precision rather than merely to a tolerance that would hide a
different model.

That equivalence is what lets the design be run in any package with a
Poisson regression, which is the tutorial's practical point. It also
explains the one thing that surprises people: the individual effects
are estimated, but they are nuisance parameters whose only job is to
reproduce the totals. Their number grows with the sample, so they are
never interpreted.

**Sizing the study.** Sec. 7.6 gives the number of *events* -- not
people -- needed to detect a log relative incidence :math:`\beta` when
age effects can be ignored. With :math:`r` the ratio of risk period to
observation period,

.. math:: \rho = \frac{r e^{\beta}}{r e^{\beta} + 1 - r},

and with :math:`p` the proportion of the **population** exposed during
the observation period (not the proportion of cases):

.. math:: A &= 2\{\rho\beta - \log(r e^{\beta} + 1 - r)\},\\
          B &= \frac{\beta^{2}\rho(1-\rho)}{A},\\
          C &= 1 + \frac{1-p}{p(r e^{\beta} + 1 - r)},\\
          n &= \frac{C}{A}\left(z_{\alpha/2}
              + z_{\gamma}\sqrt{B}\right)^{2}.

:math:`A` is the signed-root-likelihood-ratio information, and it is
worth seeing why :math:`B` is a *correction near one* rather than a
free quantity. Expanding :math:`A` about :math:`\beta = 0` gives
:math:`A \approx \rho(1-\rho)\beta^{2}`, so :math:`B \to 1` as the
effect shrinks and the whole expression collapses to the familiar
:math:`n = C(z_{\alpha/2}+z_{\gamma})^{2}/A`. The anchor measures that
limit, which is what distinguishes this :math:`B` from the plausible
misreadings of it.

**Two things the formula does not do.** It assumes age effects are
negligible; with real age variation the requirement is larger, and the
tutorial points to Musonda et al. for that case. And :math:`p` refers
to the population, so a study where nearly everyone is exposed has
:math:`C \to 1` while a rare exposure inflates it sharply -- which is
the term that usually decides feasibility.

References
----------
Whitaker, H. J., Farrington, C. P., Spiessens, B. & Musonda, P. (2006)
"Tutorial in biostatistics: The self-controlled case series method",
*Statistics in Medicine* 25, 1768-1797, doi:10.1002/sim.2302
(published online 11 October 2005). Sec. 4 (the associated Poisson
model with an individual factor and a log-time offset), Sec. 7.3-7.5
(risk-period choice, covariates, relative efficiency) and Sec. 7.6
(the sample size expression implemented here).

Farrington, C. P. (1995) "Relative Incidence Estimation from Case
Series for Vaccine Safety Evaluation", *Biometrics* 51(1), 228-235,
JSTOR https://www.jstor.org/stable/2533328. The conditional
likelihood the Poisson form reproduces.

Musonda, P., Farrington, C. P. & Whitaker, H. J. (2006) "Sample sizes
for self-controlled case series studies", *Statistics in Medicine*
25(15), 2618-2631. Cited by the tutorial as reference [41] for the
derivation and for the age-varying case; not implemented here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .sccsno import build_intervals, sccs_fit

__all__ = ["poisson_design", "sccs_poisson_fit", "sample_size",
           "power", "relative_efficiency"]

_EPS = 1e-12


def poisson_design(cases, risk_periods, age_breaks=()):
    r"""Build the Sec. 4 design: counts, offsets and factor columns.

    Returns the response ``n``, the offset ``log_e``, and a design
    matrix whose columns are the risk-period dummies, the age-band
    dummies and one dummy per individual.
    """
    rp = [(float(a), float(b)) for a, b in risk_periods]
    ab = [float(v) for v in age_breaks]
    n_risk, n_age = len(rp), len(ab) + 1
    people = []
    for c in cases:
        ev = list(c.get("events", ()))
        if not ev:
            continue
        people.append(build_intervals(c["start"], c["end"],
                                      c.get("exposure"), ev, rp, ab))
    if not people:
        raise ValueError("smatch: no case contributed an event")
    P = len(people)
    ncol = n_risk + (n_age - 1) + P
    y, off, X = [], [], []
    for i, cells in enumerate(people):
        for j, r, e, n in cells:
            if e <= _EPS:
                continue
            row = [0.0] * ncol
            if r > 0:
                row[r - 1] = 1.0
            if j > 0:
                row[n_risk + j - 1] = 1.0
            row[n_risk + n_age - 1 + i] = 1.0
            X.append(row)
            y.append(float(n))
            off.append(math.log(e))
    return {"y": y, "offset": off, "X": X, "n_risk": n_risk,
            "n_age": n_age, "n_people": P, "n_rows": len(y)}


def sccs_poisson_fit(cases, risk_periods, age_breaks=(), iters=200,
                     tol=1e-12, ridge=1e-9):
    r"""Fit the associated Poisson model of Sec. 4 by IRLS.

    Returns the same relative incidences the conditional multinomial
    fit returns, because it is the same model. The individual factors
    are estimated and reported but are nuisance parameters: there is
    one per case, so they do not accumulate information.
    """
    d = poisson_design(cases, risk_periods, age_breaks=age_breaks)
    y, off, X = d["y"], d["offset"], d["X"]
    p = len(X[0])
    beta = [0.0] * p
    conv, it = False, 0
    for it in range(1, int(iters) + 1):
        mu, W, z = [], [], []
        for i in range(len(y)):
            eta = off[i] + sum(X[i][a] * beta[a] for a in range(p))
            eta = max(-500.0, min(500.0, eta))
            m = math.exp(eta)
            mu.append(m)
            W.append(max(m, 1e-12))
            z.append(eta - off[i] + (y[i] - m) / max(m, 1e-12))
        XtWX = [[0.0] * p for _ in range(p)]
        XtWz = [0.0] * p
        for i in range(len(y)):
            for a in range(p):
                if X[i][a] == 0.0:
                    continue
                XtWz[a] += X[i][a] * W[i] * z[i]
                for b in range(p):
                    if X[i][b] != 0.0:
                        XtWX[a][b] += X[i][a] * W[i] * X[i][b]
        for a in range(p):
            XtWX[a][a] += ridge
        try:
            nb = k.cholsolve(XtWX, XtWz)
        except Exception:
            raise ValueError("smatch: the Poisson design is singular "
                             "-- an interval has no exposure time or "
                             "an individual has no variation")
        mx = max(abs(nb[a] - beta[a]) for a in range(p))
        beta = nb
        if mx < tol:
            conv = True
            break
    nr = d["n_risk"]
    return RichResult(payload={
        "estimate": [math.exp(v) for v in beta[:nr]],
        "relative_incidence": [math.exp(v) for v in beta[:nr]],
        "log_ri": beta[:nr],
        "age_effects": beta[nr:nr + d["n_age"] - 1],
        "individual_effects": beta[nr + d["n_age"] - 1:],
        "coef": beta, "converged": conv, "iterations": it,
        "n_rows": d["n_rows"], "n_people": d["n_people"],
        "method": "associated Poisson model with a per-individual "
                  "factor and log-time offset; Whitaker et al. (2006) "
                  "Sec. 4",
        "identical_to": "the conditional multinomial fit of sccsno",
    })


def sample_size(log_ri, r, p_exposed, alpha=0.05, power=0.8):
    r"""Events required, Sec. 7.6. Age effects assumed negligible.

    ``log_ri`` is :math:`\beta`, ``r`` the ratio of risk period to
    observation period, ``p_exposed`` the proportion of the
    **population** exposed during the observation period.
    """
    b = float(log_ri)
    rr = float(r)
    p = float(p_exposed)
    if b == 0.0:
        raise ValueError("smatch: the sample size is unbounded at a "
                         "log relative incidence of 0")
    if not 0.0 < rr < 1.0:
        raise ValueError("smatch: r must lie strictly in (0, 1), got "
                         "%r -- it is the risk period as a fraction "
                         "of the observation period" % (r,))
    if not 0.0 < p <= 1.0:
        raise ValueError("smatch: p_exposed must lie in (0, 1], got "
                         "%r" % (p_exposed,))
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("smatch: alpha must lie in (0, 1)")
    if not 0.0 < float(power) < 1.0:
        raise ValueError("smatch: power must lie in (0, 1)")
    eb = math.exp(b)
    den = rr * eb + 1.0 - rr
    rho = rr * eb / den
    A = 2.0 * (rho * b - math.log(den))
    if A <= _EPS:
        raise ValueError("smatch: the information A is non-positive "
                         "(%.3e) -- the design carries no signal here"
                         % A)
    B = b * b * rho * (1.0 - rho) / A
    C = 1.0 + (1.0 - p) / (p * den)
    za = k.qnorm(1.0 - float(alpha) / 2.0)
    zg = k.qnorm(float(power))
    n = (C / A) * (za + zg * math.sqrt(B)) ** 2
    return {"n_events": n, "n_events_ceiling": int(math.ceil(n)),
            "rho": rho, "A": A, "B": B, "C": C,
            "z_alpha_2": za, "z_power": zg,
            "log_ri": b, "r": rr, "p_exposed": p,
            "assumes": "age effects negligible; see Musonda, "
                       "Farrington & Whitaker (2006) otherwise",
            "method": "Whitaker et al. (2006) Sec. 7.6"}


def power(n_events, log_ri, r, p_exposed, alpha=0.05):
    r"""Invert Sec. 7.6 for the power at a given number of events."""
    s = sample_size(log_ri, r, p_exposed, alpha=alpha, power=0.5)
    A, B, C = s["A"], s["B"], s["C"]
    za = s["z_alpha_2"]
    root = math.sqrt(max(float(n_events) * A / C, 0.0))
    zg = (root - za) / math.sqrt(B) if B > _EPS else float("inf")
    return {"power": k.pnorm(zg), "z_power": zg,
            "n_events": float(n_events), "A": A, "B": B, "C": C}


def relative_efficiency(r, log_ri):
    r"""Asymptotic efficiency of the case series against the cohort
    design it is derived from (Sec. 7.5).

    Conditioning on each person's event total discards the marginal
    information, so efficiency is lost. It is negligible when the risk
    period is short relative to the observation period -- which the
    returned value makes concrete rather than assertable.
    """
    rr, b = float(r), float(log_ri)
    if not 0.0 < rr < 1.0:
        raise ValueError("smatch: r must lie strictly in (0, 1)")
    eb = math.exp(b)
    den = rr * eb + 1.0 - rr
    rho = rr * eb / den
    return {"rho": rho, "efficiency": 1.0 - rho,
            "r": rr, "log_ri": b,
            "interpretation": "the fraction of cases falling in the "
                              "risk period is rho; the marginal "
                              "information lost grows with it, so a "
                              "SHORT risk period keeps efficiency "
                              "high (Sec. 7.5)"}


def cheatsheet():
    return ("smatch: the case series fitted as a POISSON model -- "
            "counts n_ijk, offset log(e_ijk), factors for age, "
            "exposure AND one per individual. The individual factors "
            "force the fitted totals to match the observed ones, "
            "which IS the conditioning, so this is the same fit as "
            "the multinomial, not an approximation. Sample size "
            "(Sec. 7.6): rho = re^b/(re^b+1-r), A = 2{rho b - "
            "log(re^b+1-r)}, B = b^2 rho(1-rho)/A -> 1 as b -> 0, "
            "C = 1 + (1-p)/(p(re^b+1-r)), n = (C/A)(z_a2 + z_g sqrt "
            "B)^2. p is the POPULATION exposed fraction, not the "
            "cases.")


# compact alias per ledger/NAMING.md
selfcontrolledcaseseries = sccs_poisson_fit

# public names resolved by fn/_lazy_map.json
sccs_design = sccs_poisson_fit
sccsdesign = sccs_poisson_fit
