# morie.fn -- function file (rootcoder007/morie)
"""Kulldorff spatial scan statistic."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["scanstat", "bivand2013_chapter_10_equation_3"]


def scanstat(O, E, zones, highonly=True):
    """Poisson likelihood-ratio scan over candidate clusters.

    A scan statistic compares the relative risk inside a moving window
    with the relative risk outside it, and reports the window that
    maximises the likelihood ratio.  For a Poisson model with observed
    counts O and expected counts E, writing O_z and E_z for the totals
    inside window z and O_+ and E_+ for the study-area totals, the
    statistic is

        max_{z in Z_i}  (O_z/E_z)^{O_z}
                        ((O_+ - O_z)/(E_+ - E_z))^{O_+ - O_z}

    the maximum running over all candidate windows.  Because the
    detection is a maximum over many windows, this routine reports the
    per-zone log statistic as well, and by default restricts attention to
    windows whose internal risk exceeds the overall risk, which is the
    one-sided alternative the test is built for.

    Parameters
    ----------
    O : array-like
        Observed counts per region.
    E : array-like
        Expected counts per region, strictly positive.
    zones : sequence of sequences
        Candidate windows, each a list of zero-based region indices.
    highonly : bool
        Score only windows with O_z/E_z > O_+/E_+, as the one-sided test
        requires; other windows score -inf.

    Returns
    -------
    RichResult
        ``loglr``, ``best``, ``maxloglr``, ``bestzone``, ``Oz``, ``Ez``,
        ``rrin``, ``rrout``, ``Otot``, ``Etot``, ``nzone``.

    References
    ----------
    Bivand, R. S., Pebesma, E. and Gomez-Rubio, V. (2013),
    Applied Spatial Data Analysis with R, 2nd edn, Springer (Use R!).  Equation (10.3), p. 354, states the Poisson scan statistic
    max_{z in Z_i} (O_z/E_z)^{O_z} ((O_+ - O_z)/(E_+ - E_z))^{O_+ - O_z},
    attributing it to Kulldorff and Nagarwalla (1995) and describing the
    windows Z_i as circles centred at region i containing up to a fixed
    proportion of the population.  Read from the corpus PDF
    (bivand2013.pdf, p. 354).
    """
    O = C.vec(O)
    E = C.vec(E)
    n = len(O)
    if len(E) != n:
        raise ValueError("O and E must have the same length")
    if any(v <= 0.0 for v in E):
        raise ValueError("expected counts must be strictly positive")
    if any(v < 0.0 for v in O):
        raise ValueError("observed counts must be non-negative")
    Ot = sum(O)
    Et = sum(E)
    rr = Ot / Et
    ll = []
    Ozs = []
    Ezs = []
    rin = []
    rout = []
    for zs in zones:
        idx = [int(t) for t in zs]
        if any(t < 0 or t >= n for t in idx):
            raise ValueError("zone index out of range")
        oz = sum(O[t] for t in idx)
        ez = sum(E[t] for t in idx)
        Ozs.append(oz)
        Ezs.append(ez)
        oo = Ot - oz
        eo = Et - ez
        rin.append(oz / ez if ez > 0.0 else float("nan"))
        rout.append(oo / eo if eo > 0.0 else float("nan"))
        if ez <= 0.0 or eo <= 0.0:
            ll.append(float("-inf"))
            continue
        if highonly and oz / ez <= rr:
            ll.append(float("-inf"))
            continue
        v = 0.0
        if oz > 0.0:
            v += oz * (math.log(oz) - math.log(ez))
        if oo > 0.0:
            v += oo * (math.log(oo) - math.log(eo))
        ll.append(v)
    if not ll:
        raise ValueError("no candidate zones supplied")
    mx = max(ll)
    bi = ll.index(mx)
    return RichResult(payload={
        "loglr": ll, "best": bi, "maxloglr": mx,
        "bestzone": [int(t) for t in zones[bi]], "Oz": Ozs, "Ez": Ezs,
        "rrin": rin, "rrout": rout, "Otot": Ot, "Etot": Et,
        "nzone": len(ll),
        "method": "Kulldorff spatial scan statistic (Bivand et al. 2013 eq. 10.3)"})


bivand2013_chapter_10_equation_3 = scanstat


def cheatsheet():
    return "bivand201310e3: Kulldorff spatial scan statistic."
