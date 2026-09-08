"""Fragment growing, judged by efficiency rather than by potency.

A fragment that binds at a millimolar dissociation constant looks like
nothing next to a micromolar lead, and that comparison is the mistake
fragment-based discovery exists to correct. A twelve-atom fragment
binding at 1 mM is doing more per atom than a forty-atom compound
binding at 1 uM, and it is the per-atom figure that says which one has
room to grow.

Ligand efficiency is that figure:

    LE = -dG / HAC,   dG = R T ln Kd

with HAC the heavy-atom count. Binding free energy is negative, so the
minus sign makes LE positive and larger is better. There is a shortcut
form, LE = 1.37 pKd / HAC, and the 1.37 is not a fitted constant: it is
2.303 R T at 298.15 K in kcal per mole, so the two agree at room
temperature and diverge as soon as you leave it. Both routes are here
and the temperature is a parameter, because a binding measured at 310 K
and reported through a 298 K constant is off by four percent and nobody
notices.

Growing a fragment is then an arithmetic question. Adding a group of
d_HAC atoms that contributes d_dG of binding has GROUP EFFICIENCY
GE = -d_dG / d_HAC, and the grown compound's ligand efficiency is
exactly the atom-weighted average of the two:

    LE_new = (LE_parent HAC_parent + GE d_HAC) / (HAC_parent + d_HAC)

which is worth writing down because it settles the decision rule
without argument: an addition whose group efficiency beats the parent's
ligand efficiency RAISES it, one that does not LOWERS it, and a
tenfold potency gain bought with fifteen atoms is a step backwards even
though the number in the assay improved. That identity is checked here
rather than asserted.

The other metrics answer "efficient in what?", and they disagree on
purpose:

  LLE   = pKd - logP        potency you did not buy with grease.
  LELP  = logP / LE         lipophilicity per unit of efficiency;
                            unlike the others, SMALLER is better.
  BEI   = pKd / (MW/1000)   per unit of mass rather than per atom, so
                            a heavy halogen counts against you.
  SEI   = pKd / (PSA/100)   per unit of polar surface.

Nothing here computes a descriptor from a structure. HAC, logP, MW and
polar surface area come in as numbers, because deriving them needs a
chemistry toolkit and inventing them would be worse than asking.

References
  Hopkins, A.L., Groom, C.R. and Alex, A. (2004) "Ligand efficiency: a
    useful metric for lead selection." Drug Discovery Today 9(10),
    430-431. doi:10.1016/S1359-6446(04)03069-7. Ligand efficiency.
  Verdonk, M.L. and Rees, D.C. (2008) "Group efficiency: a guideline
    for hits-to-leads chemistry." ChemMedChem 3(8), 1179-1180. Group
    efficiency and the decision rule above.
  Murray, C.W. and Rees, D.C. (2009) "The rise of fragment-based drug
    discovery." Nature Chemistry 1(3), 187-192. The growing strategy.
  Shuker, S.B., Hajduk, P.J., Meadows, R.P. and Fesik, S.W. (1996)
    "Discovering high-affinity ligands for proteins: SAR by NMR."
    Science 274(5292), 1531-1534. The linking strategy the growing one
    is contrasted with.
  Leeson, P.D. and Springthorpe, B. (2007) "The influence of drug-like
    concepts on decision-making in medicinal chemistry." Nature Reviews
    Drug Discovery 6(11), 881-890. LLE.
  Keseru, G.M. and Makara, G.M. (2009) "The influence of lead discovery
    strategies on the properties of drug candidates." Nature Reviews
    Drug Discovery 8(3), 203-212. LELP.
  Abad-Zapatero, C. and Metz, J.T. (2005) "Ligand efficiency indices as
    guideposts for drug discovery." Drug Discovery Today 10(7),
    464-469. BEI and SEI.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["frgrow", "fragment_growing", "binding_energy",
           "ligand_efficiency", "group_efficiency", "metrics",
           "ENERGY_ROUTES", "R_KCAL", "T_STANDARD", "LE_SHORTCUT",
           "cheatsheet"]

ENERGY_ROUTES = ("rt", "shortcut")

# The gas constant in kcal per mole per kelvin, and the temperature the
# shortcut constant belongs to.
R_KCAL = 0.0019872041
T_STANDARD = 298.15
# 2.303 R T at 298.15 K, to two decimals: the constant the literature
# writes as 1.37. It is derived, not fitted, which is why the two energy
# routes agree at this temperature and nowhere else.
LE_SHORTCUT = 1.37


def binding_energy(kd, temperature=T_STANDARD):
    """Binding free energy in kcal per mole, negative for binding.

    Kd is a concentration in molar. A Kd of one molar gives zero, which
    is the reference the whole scale hangs from.
    """
    kd = float(kd)
    if kd <= 0.0:
        raise ValueError("a dissociation constant must be positive")
    t = float(temperature)
    if t <= 0.0:
        raise ValueError("the temperature must be positive")
    return R_KCAL * t * math.log(kd)


def ligand_efficiency(kd, hac, route="rt", temperature=T_STANDARD):
    """Binding energy per heavy atom, positive and larger-is-better.

    "rt"       the definition: -R T ln(Kd) / HAC.
    "shortcut" the literature's 1.37 pKd / HAC, which IS the definition
               at 298.15 K and drifts from it elsewhere.
    """
    if route not in ENERGY_ROUTES:
        raise ValueError("route must be one of %r" % (ENERGY_ROUTES,))
    n = float(hac)
    if n <= 0.0:
        raise ValueError("the heavy-atom count must be positive")
    if route == "rt":
        return -binding_energy(kd, temperature) / n
    return LE_SHORTCUT * (-math.log10(float(kd))) / n


def group_efficiency(kd_parent, hac_parent, kd_grown, hac_grown,
                     route="rt", temperature=T_STANDARD):
    """The efficiency of the atoms that were added, on their own.

    Not the grown compound's efficiency -- the ADDED group's. A group
    that contributes nothing has a group efficiency of zero however
    potent the parent was, and one that makes the compound bind worse
    has a negative one.
    """
    dn = float(hac_grown) - float(hac_parent)
    if dn <= 0.0:
        raise ValueError("growing must add heavy atoms; use the parent's "
                         "own efficiency for a compound that added none")
    if route == "rt":
        dg = (binding_energy(kd_grown, temperature)
              - binding_energy(kd_parent, temperature))
        return -dg / dn
    dp = (-math.log10(float(kd_grown))) - (-math.log10(float(kd_parent)))
    return LE_SHORTCUT * dp / dn


def metrics(kd, hac, logp=None, mw=None, psa=None, route="rt",
            temperature=T_STANDARD):
    """The efficiency metrics that the supplied descriptors allow.

    A metric whose descriptor is absent is reported as None rather than
    as zero: a compound with no measured logP has no ligand-lipophilicity
    efficiency, and a zero there would rank it as the best in the series.
    """
    kd = float(kd)
    pkd = -math.log10(kd)
    le = ligand_efficiency(kd, hac, route, temperature)
    out = {"pkd": pkd, "dg": binding_energy(kd, temperature), "le": le,
           "lle": None, "lelp": None, "bei": None, "sei": None}
    if logp is not None:
        out["lle"] = pkd - float(logp)
        # LELP is the one metric where smaller is better, and it is also
        # the one that can divide by zero -- a compound whose binding
        # energy per atom is nil has no defined lipophilicity per unit
        # of it.
        out["lelp"] = float(logp) / le if le != 0.0 else None
    if mw is not None:
        m = float(mw)
        if m <= 0.0:
            raise ValueError("molecular weight must be positive")
        out["bei"] = pkd / (m / 1000.0)
    if psa is not None:
        p = float(psa)
        if p <= 0.0:
            raise ValueError("polar surface area must be positive")
        out["sei"] = pkd / (p / 100.0)
    return out


def _nan_if_none(v):
    return float("nan") if v is None else float(v)


def fragment_growing(fragment, linker_lib, route="rt",
                     temperature=T_STANDARD):
    """Score a set of grown analogues against the fragment they came from.

    Parameters
    ----------
    fragment : sequence
        The parent: (kd, hac, logp, mw, psa). The last three may be
        None.
    linker_lib : sequence of sequences
        One row per grown analogue, in the same shape, optionally with a
        sixth entry naming it.
    route : str
        A member of ENERGY_ROUTES.
    temperature : float
        Kelvin. Only the "rt" route uses it; the shortcut is fixed at
        298.15 K by construction, which is the point of offering both.

    Returns
    -------
    RichResult
        The parent's metrics, each analogue's metrics and group
        efficiency, the ranking by group efficiency, and which additions
        actually improved the ligand efficiency they inherited.

    References
    ----------
    Hopkins et al. (2004) Drug Discov Today 9(10), 430-431; Verdonk and
    Rees (2008) ChemMedChem 3(8), 1179-1180.
    """
    def unpack(row):
        kd = float(row[0])
        hac = float(row[1])
        lp = None if len(row) < 3 or row[2] is None else float(row[2])
        mw = None if len(row) < 4 or row[3] is None else float(row[3])
        ps = None if len(row) < 5 or row[4] is None else float(row[4])
        nm = str(row[5]) if len(row) >= 6 and row[5] is not None else ""
        return kd, hac, lp, mw, ps, nm

    pkd, phac, plp, pmw, pps, pnm = unpack(fragment)
    parent = metrics(pkd, phac, plp, pmw, pps, route, temperature)

    rows = []
    for row in linker_lib:
        kd, hac, lp, mw, ps, nm = unpack(row)
        m = metrics(kd, hac, lp, mw, ps, route, temperature)
        ge = group_efficiency(pkd, phac, kd, hac, route, temperature)
        # The identity the decision rule rests on: the grown compound's
        # efficiency is the atom-weighted average of the parent's and
        # the added group's. Recomputing it here rather than trusting it
        # is what makes the check in the tests meaningful.
        blend = ((parent["le"] * phac + ge * (hac - phac))
                 / hac)
        rows.append({"name": nm, "kd": kd, "hac": hac,
                     "d_hac": hac - phac, "ge": ge, "blend": blend,
                     "improved": ge > parent["le"], "metrics": m})

    order = sorted(range(len(rows)),
                   key=lambda i: (-rows[i]["ge"], i))
    improved = [i for i in range(len(rows)) if rows[i]["improved"]]
    best = order[0] if order else -1
    return RichResult(payload={
        "parent_le": parent["le"],
        "parent_pkd": parent["pkd"],
        "parent_dg": parent["dg"],
        "parent_lle": parent["lle"],
        "parent_lelp": parent["lelp"],
        "parent_bei": parent["bei"],
        "parent_sei": parent["sei"],
        "name": [r["name"] for r in rows],
        "kd": [r["kd"] for r in rows],
        "hac": [r["hac"] for r in rows],
        "d_hac": [r["d_hac"] for r in rows],
        "group_efficiency": [r["ge"] for r in rows],
        "le": [r["metrics"]["le"] for r in rows],
        "le_from_blend": [r["blend"] for r in rows],
        "pkd": [r["metrics"]["pkd"] for r in rows],
        "dg": [r["metrics"]["dg"] for r in rows],
        # Inside a per-analogue COLUMN an absent metric has to be a
        # number, because a column is a vector and a vector cannot hold
        # an absence. It is not-a-number, not zero: zero would rank the
        # compound first on LELP and last on everything else. The
        # parent's scalars above keep the honest None.
        "lle": [_nan_if_none(r["metrics"]["lle"]) for r in rows],
        "lelp": [_nan_if_none(r["metrics"]["lelp"]) for r in rows],
        "bei": [_nan_if_none(r["metrics"]["bei"]) for r in rows],
        "sei": [_nan_if_none(r["metrics"]["sei"]) for r in rows],
        "improved": improved,
        "n_improved": len(improved),
        "ranking": order,
        "best": best,
        "estimate": rows[best]["ge"] if rows else float("nan"),
        "se": float("nan"),
        "n": len(rows),
        "temperature": float(temperature),
        "route": route,
        "method": "fragment growing by ligand and group efficiency",
    })


frgrow = fragment_growing


def cheatsheet():
    return ("frgrow: fragment growing by ligand and group efficiency. "
            "routes " + ", ".join(ENERGY_ROUTES)
            + "; LE = -RT ln(Kd)/HAC, GE on the added atoms only")
