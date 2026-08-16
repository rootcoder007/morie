"""Steady-state volume of distribution by the Oie-Tozer model.

Volume of distribution is not a volume. It is the proportionality
constant between the amount of drug in the body and its concentration in
plasma, so a drug that hides in fat has a "volume" of several hundred
litres in a seventy-kilogram person and one that stays in the blood has
about three. What the Oie-Tozer model does is take that number apart
into physiology and chemistry:

    Vss = Vp (1 + Re_i)
          + fu Vp (Ve/Vp - Re_i)
          + Vr (fu / fut)

Three terms, and each one is a different place the drug can be. The
first is plasma plus the protein that leaks out of it -- Re_i is the
ratio of extravascular to intravascular albumin, and it is there whether
the drug binds anything or not. The second is the drug free in plasma
distributing into extracellular water, which is why it carries fu and
why its sign depends on whether extracellular water is bigger than the
albumin term. The third is everything else: tissue. It carries the ratio
of the free fractions, and it is the term that makes volumes of
distribution large, because fut for a lipophilic base can be a
thousandth.

The physiological constants for a human are Vp = 0.0436, Ve = 0.151 and
Vr = 0.380 litres per kilogram, with Re_i = 1.4. Those four numbers
alone, with fu = fut = 1, give Vss = 0.5746 l/kg -- which is total body
water to two figures, and is the check that the arithmetic is assembled
right rather than merely plausible.

The equation runs both ways and both directions are here. Forward, from
the two free fractions to the volume. Backward, from a MEASURED volume
to fut, which is the use that makes the model interesting: fut is not
measurable in a person, and solving for it turns a pharmacokinetic
observation into a statement about tissue binding. The inverse is closed
form, not a search, so the round trip is exact.

What this module does NOT do is predict fut from structure. Lombardo and
colleagues do that with a regression on ElogD(7.4), the ionised fraction
at pH 7.4 and fu, and the coefficients of that regression are data --
they belong to the papers, not here. The route exists and takes the
coefficients from the caller; with none supplied it says so rather than
inventing them.

References
  Oie, S. and Tozer, T.N. (1979) "Effect of altered plasma protein
    binding on apparent volume of distribution." Journal of
    Pharmaceutical Sciences 68(9), 1203-1205.
    doi:10.1002/jps.2600680948. The model.
  Lombardo, F., Obach, R.S., Shalaeva, M.Y. and Gao, F. (2002)
    "Prediction of volume of distribution values in humans for neutral
    and basic drugs using physicochemical measurements and plasma
    protein binding data." Journal of Medicinal Chemistry 45(13),
    2867-2876. The descriptor regression this module takes coefficients
    for rather than shipping.
  Waters, N.J. and Lombardo, F. (2010) "Use of the Oie-Tozer model in
    understanding mechanisms and determinants of drug distribution."
    Drug Metabolism and Disposition 38(7), 1094-1102.
    doi:10.1124/dmd.110.032722. The physiological constants used here
    and the inverse solution for fut.
  Lombardo, F., Berellini, G. and Obach, R.S. (2019) "An accurate in
    vitro prediction of human VDss based on the Oie-Tozer equation and
    primary physicochemical descriptors. 3." Drug Metabolism and
    Disposition 47(12), 1380-1387. The large-dataset assessment.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["vdcal", "volume_of_distribution", "oie_tozer", "fut_from_vss",
           "HUMAN", "DIRECTIONS", "cheatsheet"]

DIRECTIONS = ("vss", "fut")

# Human physiology, in litres per kilogram, with the extravascular to
# intravascular albumin ratio. Waters and Lombardo (2010).
HUMAN = {"Vp": 0.0436, "Ve": 0.151, "Vr": 0.380, "Re_i": 1.4}


def _phys(par):
    p = dict(HUMAN)
    if par:
        p.update(par)
    for k in ("Vp", "Ve", "Vr"):
        if p[k] <= 0.0:
            raise ValueError("%s must be positive" % k)
    if p["Re_i"] < 0.0:
        raise ValueError("the albumin ratio cannot be negative")
    return p


def oie_tozer(fu, fut, par=None):
    """The volume of distribution, in litres per kilogram.

    Returns the total and the three terms separately, because the
    interesting question about a large volume is almost always WHICH
    term made it large.
    """
    p = _phys(par)
    fu = float(fu)
    fut = float(fut)
    if not 0.0 < fu <= 1.0:
        raise ValueError("the plasma free fraction must lie in (0, 1]")
    if not 0.0 < fut <= 1.0:
        raise ValueError("the tissue free fraction must lie in (0, 1]")
    plasma = p["Vp"] * (1.0 + p["Re_i"])
    extra = fu * p["Vp"] * (p["Ve"] / p["Vp"] - p["Re_i"])
    tissue = p["Vr"] * (fu / fut)
    return _w.csum((plasma, extra, tissue)), plasma, extra, tissue


def fut_from_vss(vss, fu, par=None):
    """Solve the model backwards for the tissue free fraction.

    Closed form, not a search: the volume is linear in fu/fut, so
    inverting it is arithmetic. A measured volume smaller than the
    plasma-and-albumin floor cannot be produced by this model at any
    tissue binding, and that is an error rather than a negative fut.
    """
    p = _phys(par)
    fu = float(fu)
    if not 0.0 < fu <= 1.0:
        raise ValueError("the plasma free fraction must lie in (0, 1]")
    plasma = p["Vp"] * (1.0 + p["Re_i"])
    extra = fu * p["Vp"] * (p["Ve"] / p["Vp"] - p["Re_i"])
    rest = float(vss) - plasma - extra
    if rest <= 0.0:
        raise ValueError("this volume is below what plasma and "
                         "extracellular water alone account for; no "
                         "tissue binding can produce it")
    fut = p["Vr"] * fu / rest
    if fut > 1.0:
        raise ValueError("the implied tissue free fraction exceeds one, "
                         "which would mean the tissue concentrates the "
                         "drug less than water does")
    return fut


def _fut_from_descriptors(elogd, fi, fu, coefficients):
    """The Lombardo descriptor route, on caller-supplied coefficients.

    log(1/fut) = a + b ElogD(7.4) + c fi(7.4) + d log(1/fu)

    The four coefficients are fitted quantities that belong to the
    papers. This module will not guess them: pass them, or use the
    measured route.
    """
    if not coefficients:
        raise ValueError("the descriptor route needs the regression "
                         "coefficients; they are fitted values from "
                         "Lombardo et al. and are not shipped here")
    c = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    c.update(coefficients)
    if not 0.0 < float(fu) <= 1.0:
        raise ValueError("the plasma free fraction must lie in (0, 1]")
    y = _w.csum((c["a"], c["b"] * float(elogd), c["c"] * float(fi),
                 c["d"] * math.log(1.0 / float(fu))))
    fut = math.exp(-y)
    if fut <= 0.0 or fut > 1.0:
        raise ValueError("the fitted tissue free fraction fell outside "
                         "(0, 1]; the coefficients and the descriptors "
                         "do not belong to the same model")
    return fut


def volume_of_distribution(smiles, ppb, fut=None, vss=None,
                           direction="vss", weight=70.0, par=None,
                           elogd=None, fi=None, coefficients=None):
    """Volume of distribution at steady state, forwards or backwards.

    Parameters
    ----------
    smiles : str or None
        Carried through untouched. The model is physiological, not
        structural: nothing here reads the structure, and pretending
        otherwise would be the fabrication this module exists to avoid.
    ppb : float
        Plasma protein binding as the FREE fraction fu, in (0, 1]. A
        drug quoted as "99% bound" has fu = 0.01.
    fut : float or None
        The tissue free fraction, for the forward direction. When it is
        absent and descriptors are supplied, the Lombardo route is used
        instead.
    vss : float or None
        A measured volume, in litres per kilogram, for the inverse
        direction.
    direction : str
        A member of DIRECTIONS.
    weight : float
        Body mass, used only to report the volume in litres as well as
        litres per kilogram.
    par : dict or None
        Overrides for the physiological constants.
    elogd, fi, coefficients : float, float, dict or None
        The Lombardo descriptor route and its fitted coefficients.

    Returns
    -------
    RichResult
        The volume in both units, the three terms, the tissue free
        fraction used or solved for, and the route taken.

    References
    ----------
    Oie and Tozer (1979) J Pharm Sci 68(9), 1203-1205; Waters and
    Lombardo (2010) Drug Metab Dispos 38(7), 1094-1102.
    """
    if direction not in DIRECTIONS:
        raise ValueError("direction must be one of %r" % (DIRECTIONS,))
    fu = float(ppb)
    p = _phys(par)
    route = "given"
    if direction == "fut":
        if vss is None:
            raise ValueError("the inverse direction needs a measured "
                             "volume")
        ft = fut_from_vss(vss, fu, par)
        route = "inverse"
        total, plasma, extra, tissue = oie_tozer(fu, ft, par)
    else:
        if fut is None:
            if elogd is None or fi is None:
                raise ValueError("give a tissue free fraction, or the "
                                 "descriptors and coefficients to fit "
                                 "one")
            ft = _fut_from_descriptors(elogd, fi, fu, coefficients)
            route = "descriptors"
        else:
            ft = float(fut)
        total, plasma, extra, tissue = oie_tozer(fu, ft, par)
    w = float(weight)
    if w <= 0.0:
        raise ValueError("body mass must be positive")
    return RichResult(payload={
        "vss": total,
        "vss_litres": total * w,
        "estimate": total,
        "se": float("nan"),
        "plasma_term": plasma,
        "extracellular_term": extra,
        "tissue_term": tissue,
        "fu": fu,
        "fut": ft,
        "binding_ratio": fu / ft,
        "weight": w,
        "Vp": p["Vp"],
        "Ve": p["Ve"],
        "Vr": p["Vr"],
        "Re_i": p["Re_i"],
        "smiles": smiles,
        "direction": direction,
        "route": route,
        "method": "Oie-Tozer steady-state volume of distribution",
    })


vdcal = volume_of_distribution


def cheatsheet():
    return ("vdcal: Oie-Tozer steady-state volume of distribution. "
            "directions " + ", ".join(DIRECTIONS)
            + "; human Vp 0.0436, Ve 0.151, Vr 0.380 l/kg, Re/I 1.4")
