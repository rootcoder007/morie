# morie.fn -- function file (rootcoder007/morie)
r"""Cumulative vaccine sieve effects.

An HIV vaccine is built from a few antigens, so it may protect well
against strains resembling them and poorly against antigenically
distant ones. **Sieve analysis** asks how efficacy varies with the
virus's genetic characteristics -- the vaccine as a sieve, with "holes"
that particular strains pass through. A sieve effect at a genetic locus
is the difference in vaccine efficacy comparing viruses *matched* to
the vaccine at that locus with viruses *mismatched* there, and it
guides which antigens a future multivalent vaccine should carry.

**Statistically this is competing risks.** Each viral genotype is a
distinct endpoint; a participant infected by one type is no longer at
risk of a first infection by another. The cumulative parameter is
**cumulative incidence** -- the probability of being infected by time
:math:`t` *with a virus of that type*:

.. math:: F_j(t) = \int_0^t S(u^-)\, d\Lambda_j(u),

the type-:math:`j` hazard integrated against overall survival. Vaccine
efficacy at :math:`t` for type :math:`j` is
:math:`VE_j(t) = 1 - F_j^{vac}(t)/F_j^{pla}(t)`, and the sieve effect
contrasts :math:`VE` between matched and mismatched types. The
cumulative parameter, rather than the instantaneous hazard, is the one
with public health meaning when vaccine effects **wane**.

**Why not the standard estimator.** Aalen-Johansen is consistent under
uninformative censoring and, with no covariates, nonparametric
efficient. But informative censoring is routine in a longitudinal
trial and prognostic covariates -- sexual risk behaviour, say -- are
collected as a matter of course. Using them weakens the censoring
assumption *and* improves efficiency. The semiparametric alternatives
that do use covariates require a correctly specified finite-dimensional
regression, which is the assumption TMLE removes.

The anchor exploits the structure rather than a reference
implementation: with no censoring and no covariates the TMLE must
reproduce Aalen-Johansen exactly, the cumulative incidences of all
types plus survival must sum to one at every time, and under
informative censoring the covariate-using estimator must beat the
naive one.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 11
(Benkeser, Carone & Gilbert): sieve analysis as the study of how
vaccine efficacy varies with viral genetics; the sieve effect at a
locus as the difference in efficacy between matched and mismatched
viruses, and its use in selecting antigens for multivalent vaccines;
the competing risks framework with each genotype a separate endpoint;
cumulative incidence as the cumulative parameter, of greater public
health relevance when vaccine effects wane; the Aalen-Johansen
estimator's consistency under uninformative censoring and
nonparametric efficiency absent covariates; the concern of informative
censoring and the routine availability of prognostic covariates; and
the drawback that semiparametric hazard-based alternatives require a
correctly specified finite-dimensional regression model.

Aalen, O. O. & Johansen, S. (1978) "An Empirical Transition Matrix for
Non-Homogeneous Markov Chains Based on Censored Observations",
*Scandinavian Journal of Statistics* 5(3), 141-150. The estimator
being improved on.

Gilbert, P. B., Self, S. G. & Ashby, M. A. (1998) "Statistical methods
for assessing differential vaccine protection against human
immunodeficiency virus types", *Biometrics* 54(3), 799-814,
doi:10.2307/2533838. Sieve analysis.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["cause_specific_hazard", "cumulative_incidence",
           "aalen_johansen", "vaccine_efficacy", "sieve_effect"]

_EPS = 1e-12


def cause_specific_hazard(time, event_type, times, weights=None):
    r"""The type-specific hazard at each event time.

    ``event_type = 0`` marks censoring. Weights allow an
    inverse-probability-of-censoring or targeted correction.
    """
    t = [float(v) for v in k.vec(time)]
    e = [int(v) for v in k.vec(event_type)]
    n = len(t)
    if len(e) != n:
        raise ValueError("tlsieve: %d times but %d event types"
                         % (n, len(e)))
    w = [1.0] * n if weights is None else [float(v)
                                           for v in k.vec(weights)]
    types = sorted(set(v for v in e if v > 0))
    out = {}
    for j in types:
        h = []
        for u in times:
            at_risk = sum(w[i] for i in range(n) if t[i] >= u)
            ev = sum(w[i] for i in range(n)
                     if abs(t[i] - u) < _EPS and e[i] == j)
            h.append(ev / at_risk if at_risk > _EPS else 0.0)
        out[j] = h
    return {"hazards": out, "types": types, "times": list(times)}


def cumulative_incidence(hazards, times):
    r""":math:`F_j(t) = \sum_{u \le t} S(u^-)\,\lambda_j(u)`.

    Overall survival multiplies the type-specific hazard, which is why
    a competing type reduces a type's cumulative incidence even when
    its own hazard is unchanged.
    """
    types = sorted(hazards)
    if not types:
        raise ValueError("tlsieve: no event types given")
    m = len(times)
    S = 1.0
    F = {j: [] for j in types}
    surv = []
    for u in range(m):
        tot = sum(hazards[j][u] for j in types)
        for j in types:
            prev = F[j][-1] if F[j] else 0.0
            F[j].append(prev + S * hazards[j][u])
        S *= (1.0 - tot)
        surv.append(S)
    return {"F": F, "survival": surv, "times": list(times),
            "types": types,
            "closure": [sum(F[j][u] for j in types) + surv[u]
                        for u in range(m)],
            "note": "cumulative incidences plus survival sum to 1 at "
                    "every time"}


def aalen_johansen(time, event_type, times, weights=None):
    r"""The nonparametric estimator: hazards then integration."""
    h = cause_specific_hazard(time, event_type, times, weights)
    ci = cumulative_incidence(h["hazards"], times)
    return RichResult(payload={
        "estimate": ci["F"], "F": ci["F"],
        "survival": ci["survival"], "types": ci["types"],
        "times": list(times), "closure": ci["closure"],
        "method": "Aalen-Johansen cumulative incidence; van der Laan "
                  "& Rose (2018) Chap. 11",
        "caveat": "consistent under UNINFORMATIVE censoring and "
                  "efficient only absent covariates",
    })


def vaccine_efficacy(F_vaccine, F_placebo):
    r""":math:`VE_j(t) = 1 - F_j^{vac}(t)/F_j^{pla}(t)`."""
    a = [float(v) for v in k.vec(F_vaccine)]
    b = [float(v) for v in k.vec(F_placebo)]
    if len(a) != len(b):
        raise ValueError("tlsieve: the two arms differ in length")
    return [1.0 - a[i] / b[i] if b[i] > _EPS else float("nan")
            for i in range(len(a))]


def sieve_effect(F_vac_matched, F_pla_matched, F_vac_mismatched,
                 F_pla_mismatched):
    r"""The sieve effect: efficacy against matched minus mismatched.

    Zero means the vaccine sieves nothing -- efficacy does not depend
    on the locus.
    """
    ve_m = vaccine_efficacy(F_vac_matched, F_pla_matched)
    ve_x = vaccine_efficacy(F_vac_mismatched, F_pla_mismatched)
    if len(ve_m) != len(ve_x):
        raise ValueError("tlsieve: matched and mismatched series "
                         "differ in length")
    d = [ve_m[i] - ve_x[i] for i in range(len(ve_m))]
    return RichResult(payload={
        "estimate": d, "sieve_effect": d,
        "ve_matched": ve_m, "ve_mismatched": ve_x,
        "method": "cumulative sieve effect; van der Laan & Rose "
                  "(2018) Chap. 11",
        "note": "zero at every time means the vaccine does not sieve "
                "at this locus",
    })


def cheatsheet():
    return ("tlsieve: an HIV vaccine built from a few antigens "
            "protects unevenly across strains, so SIEVE ANALYSIS asks "
            "how efficacy varies with viral genetics -- the sieve "
            "effect at a locus is efficacy against MATCHED minus "
            "MISMATCHED viruses. Each genotype is a competing risk, "
            "and the parameter is CUMULATIVE INCIDENCE, "
            "F_j = integral of S(u-) dLambda_j -- the cumulative form "
            "matters when vaccine effects WANE. Aalen-Johansen is "
            "consistent under uninformative censoring and efficient "
            "without covariates; using covariates weakens the "
            "censoring assumption and gains efficiency, which is what "
            "TMLE does without a parametric hazard model.")


# compact alias per ledger/NAMING.md
vaccinesieve = sieve_effect
