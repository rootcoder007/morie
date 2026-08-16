"""Plasma half-life from volume of distribution and clearance.

Half-life is a derived quantity, not a measured one, and that is the
whole point. Clearance is the body's capacity to eliminate the drug and
volume of distribution is how widely the drug spreads; half-life is what
falls out of the two:

    t_half = ln(2) * V / CL

Reading it the other way round -- treating half-life as the primitive
and inferring clearance from it -- is the standard mistake, because a
long half-life can mean poor clearance OR wide distribution, and those
have opposite consequences. A drug with a huge volume and good clearance
has a long half-life and still leaves the body efficiently.

Which V, and therefore which half-life, depends on the model:

  "one_compartment"  V is the single apparent volume. One exponential,
                     one half-life, and everything downstream is easy.
                     Almost no real drug behaves this way after an
                     intravenous dose, and using it anyway is how the
                     distribution phase gets mistaken for elimination.

  "two_compartment"  Central volume V1, peripheral V2, elimination
                     clearance CL and inter-compartmental clearance Q.
                     The concentration is a sum of two exponentials
                     whose rate constants alpha and beta are the roots
                     of

                         lambda^2 - (k10 + k12 + k21) lambda
                                  + k10 k21 = 0

                     with k10 = CL/V1, k12 = Q/V1, k21 = Q/V2. The
                     TERMINAL half-life ln(2)/beta is the one usually
                     quoted, and on its own it is misleading: if the
                     terminal phase carries only a few per cent of the
                     area it says almost nothing about accumulation.

  "effective"        ln(2) * Vss / CL, with Vss = V1 + V2. This is the
                     mean-residence-time half-life, and it is the one
                     that predicts accumulation on repeated dosing --
                     which is the question half-life is usually being
                     asked in order to answer. It is reported alongside
                     the terminal value for the two-compartment route
                     precisely so the gap between them is visible.

The `smiles` argument is carried through untouched. The identity above
is a statement about V and CL and does not involve the structure; the
argument exists because the ledger's signature has it, and it is
recorded in the result for provenance rather than quietly ignored. If a
structure-based PREDICTION of V or CL is wanted, that is a different
module and it should not be hidden inside this one.

References
  Rowland, M. and Tozer, T.N. (2011) "Clinical Pharmacokinetics and
    Pharmacodynamics: Concepts and Applications," 4th edition. Wolters
    Kluwer. The half-life identity, the two-compartment roots, and mean
    residence time.
  Gibaldi, M. and Perrier, D. (1982) "Pharmacokinetics," 2nd edition.
    Marcel Dekker. Chapter 2: the biexponential disposition model.
  Boxenbaum, H. and Battle, M. (1995) "Effective half-life in clinical
    pharmacokinetics." Journal of Clinical Pharmacology 35(8), 763-766.
    The effective half-life and why the terminal one misleads.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["half_life", "halft", "two_compartment_rates", "ROUTES",
           "cheatsheet"]

ROUTES = ("one_compartment", "two_compartment", "effective")
LN2 = 0.6931471805599453


def two_compartment_rates(V1, V2, CL, Q):
    """alpha and beta, the roots of the disposition quadratic.

    Solved with the numerically stable form of the quadratic formula --
    the naive (-b +- sqrt(disc)) / 2 cancels catastrophically when the
    two rates are far apart, which is exactly the case where the
    terminal half-life matters most.
    """
    if V1 <= 0.0 or V2 <= 0.0 or CL <= 0.0 or Q <= 0.0:
        raise ValueError("volumes and clearances must be positive")
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    b = k10 + k12 + k21
    c = k10 * k21
    disc = b * b - 4.0 * c
    if disc < 0.0:
        disc = 0.0
    root = math.sqrt(disc)
    # The larger root first, then the smaller by c / alpha: computing
    # the small root as (b - root) / 2 subtracts two nearly equal
    # numbers and loses most of its digits.
    alpha = 0.5 * (b + root)
    beta = c / alpha if alpha > 0.0 else 0.0
    return {"alpha": alpha, "beta": beta, "k10": k10, "k12": k12,
            "k21": k21}


def half_life(smiles=None, Vd=None, Cl=None, route="one_compartment",
              V1=None, V2=None, Q=None, dose=None):
    """Plasma half-life from volume of distribution and clearance.

    Parameters
    ----------
    smiles : str or None
        Structure, carried through for provenance. Not used in the
        arithmetic; see the module docstring.
    Vd : float
        Volume of distribution, in litres. The steady-state volume for
        the one-compartment and effective routes.
    Cl : float
        Clearance, in litres per hour.
    route : str
        A member of ROUTES.
    V1, V2, Q : float
        Central volume, peripheral volume and inter-compartmental
        clearance, for the two-compartment route. Vd is then taken as
        V1 + V2 unless it is given.
    dose : float or None
        An intravenous dose, in the same mass units as the
        concentrations you intend to compare against. When supplied the
        result carries the biexponential coefficients A and B.

    Returns
    -------
    RichResult
        The half-life, the elimination rate constant, mean residence
        time, and for the two-compartment route both phase half-lives,
        the fraction of area in each, and the effective half-life.

    References
    ----------
    Rowland and Tozer (2011) 4th ed.; Gibaldi and Perrier (1982) ch. 2;
    Boxenbaum and Battle (1995) J. Clin. Pharmacol. 35(8), 763-766.
    """
    if route not in ROUTES:
        raise ValueError("route must be one of %r" % (ROUTES,))
    if Cl is None or float(Cl) <= 0.0:
        raise ValueError("Cl must be positive")
    CL = float(Cl)

    if route == "two_compartment":
        if V1 is None or V2 is None or Q is None:
            raise ValueError("the two-compartment route needs V1, V2 "
                             "and Q")
        V1 = float(V1)
        V2 = float(V2)
        Q = float(Q)
        r = two_compartment_rates(V1, V2, CL, Q)
        a, b = r["alpha"], r["beta"]
        vss = V1 + V2 if Vd is None else float(Vd)
        # MRT is the primitive and the effective half-life is ln(2)
        # times it. Writing the latter as LN2 * vss / CL instead groups
        # the arithmetic as (LN2 * vss) / CL, which differs from
        # LN2 * (vss / CL) in the last bit -- and the identity between
        # the two reported numbers then fails by an ulp.
        mrt = vss / CL
        # Coefficients of the unit-dose biexponential, from the standard
        # partial fractions: C(t) = A exp(-alpha t) + B exp(-beta t).
        Aunit = (a - r["k21"]) / (V1 * (a - b)) if a != b else 0.0
        Bunit = (r["k21"] - b) / (V1 * (a - b)) if a != b else 0.0
        auc_a = Aunit / a if a > 0.0 else 0.0
        auc_b = Bunit / b if b > 0.0 else 0.0
        auc = auc_a + auc_b
        payload = {
            "estimate": LN2 / b,
            "half_life": LN2 / b,
            "terminal_half_life": LN2 / b,
            "distribution_half_life": LN2 / a,
            "effective_half_life": LN2 * mrt,
            "alpha": a,
            "beta": b,
            "k10": r["k10"],
            "k12": r["k12"],
            "k21": r["k21"],
            "A_unit": Aunit,
            "B_unit": Bunit,
            "auc_unit": auc,
            "fraction_area_terminal": auc_b / auc if auc > 0.0 else
            float("nan"),
            "Vss": vss,
            "V1": V1,
            "V2": V2,
            "Q": Q,
            "mean_residence_time": mrt,
        }
        if dose is not None:
            payload["A"] = float(dose) * Aunit
            payload["B"] = float(dose) * Bunit
            payload["auc"] = float(dose) * auc
            payload["dose"] = float(dose)
    else:
        if Vd is None or float(Vd) <= 0.0:
            raise ValueError("Vd must be positive")
        V = float(Vd)
        k = CL / V
        mrt = V / CL
        payload = {
            "estimate": LN2 / k,
            "half_life": LN2 / k,
            "terminal_half_life": LN2 / k,
            "effective_half_life": LN2 * mrt,
            "k_elimination": k,
            "Vss": V,
            "mean_residence_time": mrt,
        }
        if dose is not None:
            payload["A_unit"] = 1.0 / V
            payload["auc_unit"] = 1.0 / CL
            payload["auc"] = float(dose) / CL
            payload["dose"] = float(dose)

    payload["Cl"] = CL
    payload["route"] = route
    payload["smiles"] = smiles
    payload["se"] = float("nan")
    payload["method"] = "plasma half-life from volume and clearance"
    return RichResult(payload=payload)


halft = half_life


def cheatsheet():
    return "halft: plasma half-life. routes " + ", ".join(ROUTES)
