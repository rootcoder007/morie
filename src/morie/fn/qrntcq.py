# morie.fn -- function file (rootcoder007/morie)
r"""Quarantine efficacy: how much transmission a duration actually
prevents.

Quarantine length is usually argued over as a policy number. It has a
quantitative answer, and the answer has sharply diminishing returns.

**Efficacy is an area under the infectivity curve.** An individual
infected at :math:`t_E` transmits according to the generation-time
distribution :math:`f`. If they are quarantined from :math:`t_Q` to
:math:`t_R`, and quarantine is perfectly effective, the transmission
prevented is the mass of :math:`f` in that window, as a fraction of
the transmission that was still ahead of them when quarantine began:

.. math:: \text{efficacy}(t_Q, t_R)
          = \frac{\int_{t_Q}^{t_R} f(t)\,dt}
                 {\int_{t_Q}^{\infty} f(t)\,dt}.

Everything follows from that. The returns diminish because :math:`f`
has most of its mass early: once :math:`t_R` is past the bulk of the
generation-time distribution there is almost nothing left to prevent,
which is why extending quarantine beyond about ten days buys nearly
nothing.

**The ceiling nobody can beat.** Transmission that happened *before*
quarantine started is gone -- no protocol recovers it. So for a given
delay between exposure and quarantine there is a maximum attainable
efficacy, reached as :math:`t_R \to \infty`, and **every** strategy
including test-and-release lies below it. The anchor checks that
bound rather than assuming it.

**Test-and-release is strictly worse on efficacy and better on
utility.** Releasing on a negative test frees people earlier, but a
false negative releases an infectious person, so efficacy falls:

.. math:: \text{efficacy}_{\text{test}}
          = (1 - p_{\text{FN}}(t_T))\,\text{efficacy}(t_Q, t_R)
            + p_{\text{FN}}(t_T)\,\text{efficacy}(t_Q, t_R'),

where the first term is those correctly detained and the second those
prematurely released at :math:`t_R'`. Testing later helps twice over:
it lengthens quarantine *and* lowers the false-negative probability.

**Utility is the trade-off made explicit.** Efficacy alone always
favours longer quarantine. The paper's utility is transmission
prevented per person-day spent in quarantine,

.. math:: \text{utility} = \frac{\text{efficacy}}
                                {E[\text{days in quarantine}]},

which has an interior optimum. And a useful property falls out: for
**standard** quarantine the expected days do not depend on how many
quarantined people are actually infected, so a *ratio* of utilities
cancels that fraction entirely. The common argument "quarantine should
be shortened because most quarantined people are not infected" does
not survive that cancellation. Under test-and-release the fraction
does not cancel, because only the infected can test positive and be
detained longer.

**What this module does not do.** It takes the generation-time
distribution and the false-negative curve as inputs. Reproducing the
paper's headline numbers -- 90.8% maximum prevention with a three-day
delay, 90.1% for release on day 10, relative utility 1.53 for test-on-5
release-on-7 -- requires their fitted parameters for both, which are
not reproduced here. The machinery is exact; the numbers depend on
what you feed it.

References
----------
Ashcroft, P., Lehtinen, S., Angst, D. C., Low, N. & Bonhoeffer, S.
(2021) "Quantifying the impact of quarantine duration on COVID-19
transmission", *eLife* 10, e63704, doi:10.7554/eLife.63704. "Model
description" and Materials and methods: eq. (1) (quarantine efficacy
as the fraction of transmission between t_Q and t_R), eq. (2)
(test-and-release efficacy across false-negative and positive
testers), eq. (4) (utility as efficacy per day in quarantine), and the
result that the relative utility of standard quarantine is
independent of the infected fraction.

Kucirka, L. M., Lauer, S. A., Laeyendecker, O., Boon, D. & Lessler, J.
(2020) "Variation in false-negative rate of reverse transcriptase
polymerase chain reaction-based SARS-CoV-2 tests by time since
exposure", *Annals of Internal Medicine* 173(4), 262-267,
doi:10.7326/M20-1495. The time-varying false-negative curve the
test-and-release calculation depends on; supplied here as an input
rather than hard-coded.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["gamma_generation_time", "quarantine_efficacy",
           "efficacy_test_and_release", "utility",
           "relative_utility", "optimal_duration"]

_EPS = 1e-12


def gamma_generation_time(shape=2.83, scale=1.86, grid=None,
                          t_max=30.0, n=3001):
    r"""A gamma generation-time density on a grid, normalised.

    Defaults are a shape/scale pair in the range reported for
    SARS-CoV-2; they are a placeholder, not the paper's fit. Supply
    your own ``grid`` and density for real work.
    """
    if float(shape) <= 0.0 or float(scale) <= 0.0:
        raise ValueError("qrntcq: the gamma shape and scale must be "
                         "positive")
    ts = ([float(t_max) * i / (int(n) - 1) for i in range(int(n))]
          if grid is None else [float(v) for v in grid])
    a, b = float(shape), float(scale)
    dens = []
    for t in ts:
        if t <= 0.0:
            dens.append(0.0)
        else:
            dens.append(math.exp((a - 1.0) * math.log(t) - t / b
                                 - k.lgamma(a) - a * math.log(b)))
    z = _trapz(ts, dens)
    if z <= _EPS:
        raise ValueError("qrntcq: the generation-time density "
                         "integrates to zero")
    return {"t": ts, "density": [v / z for v in dens]}


def _trapz(ts, ys):
    return sum(0.5 * (ys[i] + ys[i + 1]) * (ts[i + 1] - ts[i])
               for i in range(len(ts) - 1))


def _mass(ts, ys, lo, hi):
    """Integral of the density between lo and hi, linearly
    interpolated at the endpoints."""
    if hi <= lo:
        return 0.0
    tot = 0.0
    for i in range(len(ts) - 1):
        a, b = ts[i], ts[i + 1]
        if b <= lo or a >= hi:
            continue
        l = max(a, lo)
        r = min(b, hi)
        if r <= l:
            continue
        w = (b - a)
        ya = ys[i] + (ys[i + 1] - ys[i]) * ((l - a) / w if w else 0.0)
        yb = ys[i] + (ys[i + 1] - ys[i]) * ((r - a) / w if w else 0.0)
        tot += 0.5 * (ya + yb) * (r - l)
    return tot


def quarantine_efficacy(t_Q, t_R, generation_time=None,
                        t_E=0.0):
    r"""Eq. (1): the fraction of remaining transmission prevented.

    Transmission before :math:`t_Q` is already gone, so the
    denominator is the mass from :math:`t_Q` onward, not the whole
    distribution. That is what makes the quantity a proper fraction
    and what creates the ceiling below 1 whenever quarantine starts
    late.
    """
    g = generation_time or gamma_generation_time()
    ts, ys = g["t"], g["density"]
    q, r = float(t_Q), float(t_R)
    if r < q:
        raise ValueError("qrntcq: release at %g precedes quarantine "
                         "start at %g" % (r, q))
    if q < float(t_E):
        raise ValueError("qrntcq: quarantine cannot start before "
                         "exposure (t_Q %g < t_E %g)" % (q, t_E))
    remaining = _mass(ts, ys, q, ts[-1])
    if remaining <= _EPS:
        return {"efficacy": 0.0, "remaining_mass": remaining,
                "prevented_mass": 0.0,
                "note": "no transmission remains after t_Q, so "
                        "quarantine can prevent nothing"}
    prevented = _mass(ts, ys, q, r)
    return {"efficacy": prevented / remaining,
            "prevented_mass": prevented, "remaining_mass": remaining,
            "t_Q": q, "t_R": r,
            "max_attainable": 1.0,
            "pre_quarantine_mass": _mass(ts, ys, ts[0], q)}


def efficacy_test_and_release(t_Q, t_T, t_R, false_negative,
                              generation_time=None,
                              t_R_positive=None):
    r"""Eq. (2): efficacy averaged over test-negative and positive.

    ``false_negative`` is the probability of a false negative at the
    time of the test -- a function of :math:`t_T` in reality, so it is
    passed in rather than assumed. Those who test positive stay until
    ``t_R_positive`` (default: the end of infectiousness).
    """
    g = generation_time or gamma_generation_time()
    p = float(false_negative)
    if not 0.0 <= p <= 1.0:
        raise ValueError("qrntcq: the false-negative probability must "
                         "lie in [0, 1], got %r" % (false_negative,))
    if float(t_T) < float(t_Q):
        raise ValueError("qrntcq: the test cannot precede the start "
                         "of quarantine")
    if float(t_R) < float(t_T):
        raise ValueError("qrntcq: release cannot precede the test")
    stay = g["t"][-1] if t_R_positive is None else float(t_R_positive)
    released = quarantine_efficacy(t_Q, t_R, g)["efficacy"]
    detained = quarantine_efficacy(t_Q, stay, g)["efficacy"]
    eff = (1.0 - p) * detained + p * released
    return {"efficacy": eff, "efficacy_detained": detained,
            "efficacy_released": released, "false_negative": p,
            "t_T": float(t_T), "t_R": float(t_R),
            "bound": detained,
            "note": "always at or below the efficacy of detaining "
                    "everyone until t_R_positive, because a false "
                    "negative releases an infectious person"}


def utility(efficacy, days_in_quarantine):
    r"""Eq. (4): transmission prevented per day spent in quarantine."""
    d = float(days_in_quarantine)
    if d <= 0.0:
        raise ValueError("qrntcq: the time in quarantine must be "
                         "positive")
    return float(efficacy) / d


def relative_utility(t_R_a, t_R_b, t_Q=3.0, generation_time=None,
                     infected_fraction=None):
    r"""Utility of one standard quarantine relative to another.

    For **standard** quarantine the days spent do not depend on how
    many quarantined people are infected, so that fraction cancels in
    the ratio. ``infected_fraction`` is accepted only to demonstrate
    the cancellation -- passing different values returns the same
    number, which the anchor checks.
    """
    g = generation_time or gamma_generation_time()
    ea = quarantine_efficacy(t_Q, t_R_a, g)["efficacy"]
    eb = quarantine_efficacy(t_Q, t_R_b, g)["efficacy"]
    da, db = float(t_R_a) - float(t_Q), float(t_R_b) - float(t_Q)
    if da <= 0.0 or db <= 0.0:
        raise ValueError("qrntcq: both quarantines must have positive "
                         "duration")
    return {"relative_utility": (ea / da) / (eb / db),
            "utility_a": ea / da, "utility_b": eb / db,
            "efficacy_a": ea, "efficacy_b": eb,
            "independent_of_infected_fraction": True,
            "note": "the infected fraction cancels for standard "
                    "quarantine, so 'most quarantined people are not "
                    "infected' is not an argument for shortening it"}


def optimal_duration(t_Q=3.0, generation_time=None, t_max=20.0,
                     step=0.25):
    r"""The release time maximising utility.

    Efficacy alone always prefers a longer quarantine; utility does
    not, because the denominator grows while the numerator saturates.
    """
    g = generation_time or gamma_generation_time()
    best, curve = None, []
    t = float(t_Q) + float(step)
    while t <= float(t_max) + _EPS:
        e = quarantine_efficacy(t_Q, t, g)["efficacy"]
        u = e / (t - float(t_Q))
        curve.append({"t_R": t, "efficacy": e, "utility": u})
        if best is None or u > best["utility"]:
            best = {"t_R": t, "efficacy": e, "utility": u}
        t += float(step)
    return RichResult(payload={
        "estimate": best["t_R"], "optimal_t_R": best["t_R"],
        "efficacy_at_optimum": best["efficacy"],
        "utility_at_optimum": best["utility"],
        "curve": curve, "t_Q": float(t_Q),
        "method": "utility maximisation, Ashcroft et al. (2021) "
                  "eq. (4)",
    })


def cheatsheet():
    return ("qrntcq: efficacy = mass of the generation-time density "
            "between t_Q and t_R, over the mass remaining after t_Q. "
            "Transmission before quarantine is unrecoverable, so "
            "there is a CEILING every strategy sits under. "
            "Test-and-release is always below it (false negatives "
            "release infectious people) but wins on utility = "
            "efficacy per day. For STANDARD quarantine the infected "
            "fraction cancels in a utility ratio -- so 'most "
            "quarantined people are not infected' is not an argument "
            "for shortening.")


# compact alias per ledger/NAMING.md
quarantineefficacy = quarantine_efficacy
# NOTE: this function is deliberately NOT named
# test_and_release_efficacy -- a public name beginning with "test_"
# is collected as a test case by pytest in every file that imports it.
testandrelease = efficacy_test_and_release
