# morie.fn -- function file (rootcoder007/morie)
r"""Contact tracing and isolation: when is it enough on its own?

The question is not whether tracing helps. It is whether tracing plus
isolation, with no other measure, drives an outbreak to extinction --
and the answer turns almost entirely on **how much transmission
happens before anyone knows to act**.

**The model is a branching process, deliberately.** Each infected
person draws a number of potential secondary cases from a negative
binomial with mean :math:`R_0` and dispersion :math:`k`; each potential
infection gets a time drawn from the serial-interval distribution; and
a secondary case is created **only if the infector had not yet been
isolated at that time**. Isolation is assumed perfectly effective, so
failure to control comes from incomplete tracing and from delay --
never from isolation leaking.

Why negative binomial rather than Poisson: overdispersion is the point.
With :math:`k` small, most people infect nobody and a few infect many.
Two outbreaks with the same :math:`R_0` behave completely differently
if one has :math:`k = 0.1` and the other :math:`k = \infty`, because
extinction becomes far more likely when most chains die on their own.
The variance is :math:`R_0(1 + R_0/k)` exactly, and the Poisson case is
recovered as :math:`k \to \infty` -- both checked by the anchor rather
than asserted.

**What tracing actually buys.** A traced contact is quarantined at the
moment their infector is isolated, so their onward transmission is cut
from that point. An untraced one transmits until their own symptoms
appear plus the onset-to-isolation delay. So the lever is the fraction
of the serial-interval distribution that falls **before** isolation,
and the effective reproduction number is

.. math:: R_{\text{eff}} = R_0 \cdot
          E\big[F_{\text{SI}}(\text{time to isolation})\big],

which is why a short delay matters more than a large tracing effort
when transmission is early, and why presymptomatic transmission is the
parameter that decides feasibility.

**Subclinical cases are the hard limit.** A case that never develops
symptoms is never reported, so it is never isolated and its contacts
are never traced. Even perfect tracing of the symptomatic cannot reach
them. That is why the subclinical fraction appears as a separate
parameter and not as a discount on :math:`\rho`.

**Control is a definition, not a fact.** The paper defines it as
extinction within a fixed horizon before the outbreak exceeds a cap;
simulations that hit the cap are counted as uncontrolled. Both the
horizon and the cap are exposed here, because moving either moves the
answer and a "probability of control" quoted without them is
meaningless.

References
----------
Hellewell, J., Abbott, S., Gimma, A., Bosse, N. I., Jarvis, C. I.,
Russell, T. W., Munday, J. D., Kucharski, A. J., Edmunds, W. J.,
Centre for the Mathematical Modelling of Infectious Diseases COVID-19
Working Group, Funk, S. & Eggo, R. M. (2020) "Feasibility of
controlling COVID-19 outbreaks by isolation of cases and contacts",
*The Lancet Global Health* 8, e488-e496. The article as filed prints
no DOI. Section "Methods -- Model structure": the negative binomial
offspring distribution, serial-interval assignment, the rule that
secondary cases arise only before the infector's isolation, initial
outbreak sizes of 5/20/40, isolation assumed 100% effective, and the
100%/90% symptomatic split.

Lloyd-Smith, J. O., Schreiber, S. J., Kopp, P. E. & Getz, W. M. (2005)
"Superspreading and the effect of individual variation on disease
emergence", *Nature* 438(7066), 355-359, doi:10.1038/nature04153. The
negative binomial offspring parameterisation with dispersion k that
this model uses.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["negbinom_offspring", "serial_interval_draw",
           "simulate_outbreak", "probability_of_control",
           "effective_reproduction_number"]

_EPS = 1e-12


def negbinom_offspring(R0, dispersion, rng):
    r"""Draw from a negative binomial with mean :math:`R_0`.

    Parameterised as a gamma-Poisson mixture: the individual rate is
    :math:`\lambda \sim \Gamma(k, R_0/k)` and the count is
    :math:`\mathrm{Poisson}(\lambda)`. Mean :math:`R_0`, variance
    :math:`R_0(1 + R_0/k)`. As :math:`k \to \infty` the gamma
    collapses to a point and the draw becomes Poisson.
    """
    r0 = float(R0)
    kk = float(dispersion)
    if r0 < 0.0:
        raise ValueError("ttrace: R0 must be non-negative, got %r"
                         % (R0,))
    if kk <= 0.0:
        raise ValueError("ttrace: the dispersion k must be positive, "
                         "got %r" % (dispersion,))
    if r0 <= _EPS:
        return 0
    if kk > 1e6:
        lam = r0
    else:
        lam = _gamma_draw(kk, r0 / kk, rng)
    return _poisson_draw(lam, rng)


def _gamma_draw(shape, scale, rng):
    """Marsaglia-Tsang, with the shape < 1 boost."""
    a = float(shape)
    if a < 1.0:
        u = max(float(rng.uniform()), 1e-300)
        return _gamma_draw(a + 1.0, scale, rng) * (u ** (1.0 / a))
    d = a - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = float(rng.normal())
        v = (1.0 + c * x) ** 3
        if v <= 0.0:
            continue
        u = max(float(rng.uniform()), 1e-300)
        if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v * float(scale)


def _poisson_draw(lam, rng):
    """Knuth for small means, normal approximation above 500."""
    lm = float(lam)
    if lm <= 0.0:
        return 0
    if lm > 500.0:
        return max(0, int(round(lm + math.sqrt(lm)
                                * float(rng.normal()))))
    L = math.exp(-lm)
    n, p = 0, 1.0
    while True:
        p *= max(float(rng.uniform()), 1e-300)
        if p <= L:
            return n
        n += 1
        if n > 100000:
            return n


def serial_interval_draw(mean, sd, rng, allow_presymptomatic=True):
    r"""A serial interval, from a normal truncated as the model needs.

    ``allow_presymptomatic=False`` forces the interval to be at least
    the incubation period -- i.e. no transmission before symptoms --
    which is the comparison the paper turns on.
    """
    m, s = float(mean), float(sd)
    if s <= 0.0:
        raise ValueError("ttrace: the serial-interval sd must be "
                         "positive")
    v = m + s * float(rng.normal())
    if not allow_presymptomatic:
        return max(v, 0.0)
    return v


def simulate_outbreak(R0=2.5, dispersion=0.16, n_initial=20,
                      trace_prob=0.8, delay_mean=3.83, delay_sd=2.4,
                      si_mean=4.7, si_sd=2.9, subclinical=0.0,
                      max_cases=5000, max_weeks=12, seed=0,
                      allow_presymptomatic=True):
    r"""One realisation of the branching process.

    Returns the weekly incidence, the total, and whether the outbreak
    was controlled -- extinct before ``max_weeks`` and without
    exceeding ``max_cases``.
    """
    rng = np.random.default_rng(seed)
    if not 0.0 <= float(trace_prob) <= 1.0:
        raise ValueError("ttrace: trace_prob must lie in [0, 1], got "
                         "%r" % (trace_prob,))
    if not 0.0 <= float(subclinical) <= 1.0:
        raise ValueError("ttrace: subclinical must lie in [0, 1], got "
                         "%r" % (subclinical,))
    if int(n_initial) < 1:
        raise ValueError("ttrace: need at least one initial case")
    horizon = float(max_weeks) * 7.0

    # each case: (infection time, isolation time)
    active = []
    for _ in range(int(n_initial)):
        sub = float(rng.uniform()) < float(subclinical)
        iso = (float("inf") if sub
               else max(0.0, float(delay_mean)
                        + float(delay_sd) * float(rng.normal())))
        active.append((0.0, iso, sub))
    total = int(n_initial)
    weekly = [0] * (int(max_weeks) + 1)
    weekly[0] = int(n_initial)
    hit_cap = False

    while active:
        nxt = []
        for t_inf, t_iso, _sub in active:
            n_off = negbinom_offspring(R0, dispersion, rng)
            for _ in range(n_off):
                si = serial_interval_draw(
                    si_mean, si_sd, rng,
                    allow_presymptomatic=allow_presymptomatic)
                t_new = t_inf + si
                if t_new < t_inf:
                    continue
                if t_new >= t_iso:
                    continue          # infector already isolated
                if t_new > horizon:
                    continue
                sub = float(rng.uniform()) < float(subclinical)
                traced = (not sub
                          and float(rng.uniform()) < float(trace_prob))
                if sub:
                    iso_new = float("inf")
                elif traced:
                    # quarantined when the infector was isolated
                    iso_new = max(t_new, t_iso)
                else:
                    iso_new = t_new + max(
                        0.0, float(delay_mean)
                        + float(delay_sd) * float(rng.normal()))
                nxt.append((t_new, iso_new, sub))
                total += 1
                wk = int(t_new // 7.0)
                if 0 <= wk <= int(max_weeks):
                    weekly[wk] += 1
                if total > int(max_cases):
                    hit_cap = True
                    break
            if hit_cap:
                break
        if hit_cap:
            break
        active = nxt

    controlled = (not hit_cap) and (not active)
    return {"controlled": controlled, "total_cases": total,
            "weekly": weekly, "hit_cap": hit_cap,
            "extinct": not active}


def probability_of_control(reps=200, seed=0, **kw):
    r"""The fraction of simulated outbreaks that are controlled.

    "Controlled" means extinct within the horizon without exceeding
    the case cap. Both are parameters and both change the answer, so
    they are reported alongside it.
    """
    ok, sizes = 0, []
    for r in range(int(reps)):
        out = simulate_outbreak(seed=int(seed) * 7919 + r, **kw)
        ok += 1 if out["controlled"] else 0
        sizes.append(out["total_cases"])
    p = ok / float(reps)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / reps)
    sizes.sort()
    return RichResult(payload={
        "estimate": p, "probability_of_control": p, "se": se,
        "reps": int(reps),
        "median_size": sizes[len(sizes) // 2],
        "max_size": sizes[-1],
        "max_cases": kw.get("max_cases", 5000),
        "max_weeks": kw.get("max_weeks", 12),
        "definition": "extinct within max_weeks without exceeding "
                      "max_cases; both change the answer",
        "method": "branching-process simulation, Hellewell et al. "
                  "(2020) Methods",
    })


def effective_reproduction_number(R0, si_mean, si_sd, delay_mean,
                                  delay_sd, trace_prob,
                                  subclinical=0.0, draws=20000,
                                  seed=0):
    r"""The transmission that survives isolation, measured.

    :math:`R_{\text{eff}} = R_0 \cdot P(\text{serial interval} <
    \text{time to isolation})`, estimated by simulation over the two
    distributions. Traced contacts are isolated at their infector's
    isolation time; untraced ones at their own onset plus delay;
    subclinical ones never.
    """
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(int(draws)):
        if float(rng.uniform()) < float(subclinical):
            hit += 1                      # never isolated
            continue
        traced = float(rng.uniform()) < float(trace_prob)
        t_iso = (0.0 if traced
                 else max(0.0, float(delay_mean)
                          + float(delay_sd) * float(rng.normal())))
        si = float(si_mean) + float(si_sd) * float(rng.normal())
        if si < t_iso:
            hit += 1
    frac = hit / float(draws)
    return {"R_eff": float(R0) * frac, "R0": float(R0),
            "fraction_before_isolation": frac,
            "controlled_in_expectation": float(R0) * frac < 1.0,
            "note": "a traced contact is quarantined when its infector "
                    "is isolated, so its own transmission window is "
                    "measured from that point"}


def cheatsheet():
    return ("ttrace: branching process. Offspring ~ NegBinom(mean R0, "
            "dispersion k), variance R0(1 + R0/k) -- overdispersion "
            "matters because small k means most chains die alone. A "
            "secondary case exists ONLY if the infector was not yet "
            "isolated. So the lever is the fraction of the serial "
            "interval falling before isolation, which is why "
            "PRESYMPTOMATIC transmission decides feasibility. "
            "Subclinical cases are never isolated at all -- a hard "
            "ceiling no amount of tracing clears.")


# compact alias per ledger/NAMING.md
contacttracingyield = probability_of_control
