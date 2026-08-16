"""GlideScore: the Glide-style empirical docking score.

Glide's scoring function is a weighted sum of eight terms, and the
published form fixes only the first two weights:

    GScore = 0.065 EvdW + 0.130 Coul + Lipo + HBond + Metal
             + BuryP + RotB + Site

EvdW is the van der Waals energy and Coul the Coulomb energy, both
computed with REDUCED net ionic charges on formally charged groups --
metals, carboxylates, guanidiniums -- because a full formal charge on a
solvent-exposed carboxylate overwhelms everything else in a
non-solvated calculation. The remaining six enter with unit weight and
carry their own internal scales:

  Lipo    rewards hydrophobic contact.
  HBond   split by charge state: neutral-neutral, neutral-charged and
          charged-charged are weighted differently, because a salt
          bridge and a hydroxyl-to-carbonyl bond are not the same
          event.
  Metal   only interactions with anionic acceptors count, and only when
          the apo metal carries a positive net charge.
  BuryP   penalises a polar group buried without a partner.
  RotB    penalises freezing a rotatable bond.
  Site    rewards polar but non-hydrogen-bonding atoms sitting in a
          hydrophobic region.

The two coefficients above are the published ones and are the defaults
here. The internal weights of the other six are NOT published in a form
this module could quote, so they are parameters with unit defaults and
the docstring says so rather than inventing numbers -- the ledger calls
this module a proxy for exactly that reason, and a proxy that is honest
about which of its constants are real is worth more than one that is
not.

What IS computed from geometry here: the van der Waals energy from a
Lennard-Jones potential over supplied radii and well depths; the Coulomb
energy from supplied partial charges, with the choice of a constant or a
distance-dependent dielectric made explicit; the lipophilic contact
count from a distance ramp; and the hydrogen bonds sorted into their
three charge classes. What is supplied: everything that needs chemical
perception a coordinate list does not carry -- which groups are buried,
which atoms are anionic, how many bonds are frozen.

A word on the distance-dependent dielectric, since it is the one place
a docking score quietly changes its physics. With a constant
permittivity the Coulomb term falls off as 1/r; with the
distance-dependent form used through most of the docking literature it
falls off as 1/r squared, which crudely models the screening of a
solvent that is not being simulated. The two give different rankings and
the choice travels in the result.

References
  Friesner, R.A., Banks, J.L., Murphy, R.B., Halgren, T.A., Klicic,
    J.J., Mainz, D.T., Repasky, M.P., Knoll, E.H., Shelley, M., Perry,
    J.K., Shaw, D.E., Francis, P. and Shenkin, P.S. (2004) "Glide: a new
    approach for rapid, accurate docking and scoring. 1. Method and
    assessment of docking accuracy." Journal of Medicinal Chemistry
    47(7), 1739-1749. doi:10.1021/jm0306430. The GScore expression and
    the 0.065 and 0.130 coefficients.
  Halgren, T.A., Murphy, R.B., Friesner, R.A., Beard, H.S., Frye, L.L.,
    Pollard, W.T. and Banks, J.L. (2004) "Glide: a new approach for
    rapid, accurate docking and scoring. 2. Enrichment factors in
    database screening." Journal of Medicinal Chemistry 47(7),
    1750-1759. The screening behaviour of the same function.
  Friesner, R.A., Murphy, R.B., Repasky, M.P., Frye, L.L., Greenwood,
    J.R., Halgren, T.A., Sanschagrin, P.C. and Mainz, D.T. (2006)
    "Extra precision Glide: docking and scoring incorporating a model of
    hydrophobic enclosure for protein-ligand complexes." Journal of
    Medicinal Chemistry 49(21), 6177-6196. The XP successor.
  Eldridge, M.D., Murray, C.W., Auton, T.R., Paolini, G.V. and Mee, R.P.
    (1997) "Empirical scoring functions: I." Journal of Computer-Aided
    Molecular Design 11(5), 425-445. The contact-ramp idea the
    lipophilic term here uses.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["glides", "glide_score_proxy", "glide_score", "vdw_term",
           "coulomb_term", "lipophilic_term", "hbond_term",
           "DIELECTRICS", "HBOND_CLASSES", "COEFFICIENTS", "WEIGHTS",
           "cheatsheet"]

DIELECTRICS = ("constant", "distance")
HBOND_CLASSES = ("neutral_neutral", "neutral_charged", "charged_charged")

# The two published GScore coefficients, Friesner et al. (2004).
COEFFICIENTS = {"vdw": 0.065, "coulomb": 0.130}

# Unit defaults for the six terms whose internal weights the papers do
# not state in a quotable form. They are parameters, not constants; a
# caller with a calibration should pass it.
WEIGHTS = {"lipo": 1.0, "hbond": 1.0, "metal": 1.0, "buryp": 1.0,
           "rotb": 1.0, "site": 1.0,
           "hbond_neutral_neutral": 1.0, "hbond_neutral_charged": 1.0,
           "hbond_charged_charged": 1.0}

# Coulomb's constant in kcal/mol per elementary charge squared per
# angstrom -- the unit system a docking score works in.
COULOMB_K = 332.0637


def _ipow(x, k):
    """x to a small integer power, by repeated multiplication.

    Not the language's power operator: R uses repeated squaring for an
    integer exponent and Python calls the C library's pow, and they
    disagree in the last bit.
    """
    p = 1.0
    for _ in range(int(k)):
        p = p * x
    return p


def vdw_term(pairs, m=6, n=12):
    """Lennard-Jones energy over (distance, r0, eps) contacts.

    Parameterised by the minimum, so r0 is where the well sits and eps
    is how deep it is. The reduced ionic charges the paper describes
    affect the Coulomb term, not this one.
    """
    terms = []
    for r, r0, eps in pairs:
        r = float(r)
        if r <= 0.0:
            terms.append(float("inf"))
            continue
        q = r0 / r
        terms.append(eps * ((float(m) / (n - m)) * _ipow(q, n)
                            - (float(n) / (n - m)) * _ipow(q, m)))
    return _w.csum(terms) if terms else 0.0, terms


def coulomb_term(pairs, dielectric="constant", epsilon=1.0):
    """Coulomb energy over (distance, q_i, q_j) contacts.

    With a constant permittivity the energy falls off as 1/r. With the
    distance-dependent form -- the usual choice in docking, where no
    solvent is present -- the permittivity is epsilon times r and the
    energy falls off as 1/r squared. That is a different physics, not a
    different constant, so it is a route and not a parameter tweak.
    """
    if dielectric not in DIELECTRICS:
        raise ValueError("dielectric must be one of %r" % (DIELECTRICS,))
    if epsilon <= 0.0:
        raise ValueError("the permittivity must be positive")
    terms = []
    for r, qi, qj in pairs:
        r = float(r)
        if r <= 0.0:
            terms.append(float("inf"))
            continue
        den = epsilon * r if dielectric == "constant" else epsilon * r * r
        terms.append(COULOMB_K * float(qi) * float(qj) / den)
    return _w.csum(terms) if terms else 0.0, terms


def lipophilic_term(distances, r1=4.1, r2=7.1):
    """A ramped contact count over lipophilic atom pairs.

    One inside r1, zero past r2, linear between -- the contact ramp of
    the empirical-scoring literature. Glide's own lipophilic term is not
    published in a form this module could reproduce, so this is a
    STAND-IN with stated parameters, and it is the reason the ledger
    calls this module a proxy.
    """
    if r2 <= r1:
        raise ValueError("the outer radius must exceed the inner")
    terms = []
    for r in distances:
        r = float(r)
        if r <= r1:
            terms.append(1.0)
        elif r >= r2:
            terms.append(0.0)
        else:
            terms.append((r2 - r) / (r2 - r1))
    return _w.csum(terms) if terms else 0.0, terms


def hbond_term(bonds, weights=None):
    """Hydrogen bonds summed within their three charge classes.

    `bonds` gives (class, strength). The classes are weighted separately
    because a charged-charged bond is a salt bridge and a neutral pair
    is not; collapsing them to one number is the modelling error this
    split exists to prevent.
    """
    w = dict(WEIGHTS)
    if weights:
        w.update(weights)
    by = {}
    for c in HBOND_CLASSES:
        by[c] = 0.0
    terms = []
    for c, s in bonds:
        c = str(c)
        if c not in by:
            raise ValueError("hydrogen bond class must be one of %r"
                             % (HBOND_CLASSES,))
        v = w["hbond_" + c] * float(s)
        by[c] += v
        terms.append(v)
    return (_w.csum(terms) if terms else 0.0), terms, by


def glide_score(vdw=0.0, coulomb=0.0, lipo=0.0, hbond=0.0, metal=0.0,
                buryp=0.0, rotb=0.0, site=0.0, coefficients=None,
                weights=None):
    """Assemble the eight terms into a GScore.

    The two published coefficients multiply the van der Waals and
    Coulomb energies; the other six enter through weights that default
    to one. Returns the total and every weighted contribution, so the
    sum can be checked against its parts.
    """
    co = dict(COEFFICIENTS)
    if coefficients:
        co.update(coefficients)
    w = dict(WEIGHTS)
    if weights:
        w.update(weights)
    parts = {
        "vdw": co["vdw"] * float(vdw),
        "coulomb": co["coulomb"] * float(coulomb),
        "lipo": w["lipo"] * float(lipo),
        "hbond": w["hbond"] * float(hbond),
        "metal": w["metal"] * float(metal),
        "buryp": w["buryp"] * float(buryp),
        "rotb": w["rotb"] * float(rotb),
        "site": w["site"] * float(site),
    }
    order = ("vdw", "coulomb", "lipo", "hbond", "metal", "buryp", "rotb",
             "site")
    total = _w.csum(parts[k] for k in order)
    return total, parts, order


def _dist(a, b):
    return math.sqrt(_w.csum((a[t] - b[t]) * (a[t] - b[t])
                             for t in range(3)))


def glide_score_proxy(receptor, ligand_pose, radii=(), depths=(),
                      charges=(), lipophilic=(), hbonds=(),
                      dielectric="constant", epsilon=1.0, m=6, n=12,
                      r1=4.1, r2=7.1, metal=0.0, buryp=0.0, n_rot=0,
                      rot_penalty=0.35, site=0.0, coefficients=None,
                      weights=None, cutoff=None):
    """Score a pose in the Glide form.

    Parameters
    ----------
    receptor, ligand_pose : sequence of sequences
        Atom rows: x, y, z, type.
    radii, depths, charges : sequence of pairs
        (type, value). Radii combine arithmetically, well depths
        geometrically; the charge is the atom's partial charge, already
        REDUCED on formally charged groups as the paper requires -- that
        reduction is a chemical judgement and is the caller's.
    lipophilic : sequence
        Atom types treated as lipophilic.
    hbonds : sequence
        (class, strength) per hydrogen bond, as `hbond_term` takes them.
    metal, buryp, site : float
        The three terms a coordinate list cannot supply on its own.
    n_rot : int
        Frozen rotatable bonds; the penalty is `rot_penalty` each and is
        POSITIVE, so it raises the score of a floppy ligand.
    cutoff : float or None
        Drop contacts beyond this separation.

    Returns
    -------
    RichResult
        The GScore, each weighted contribution, and the raw energies.

    References
    ----------
    Friesner et al. (2004) J Med Chem 47(7), 1739-1749.
    """
    def look(table, key, what):
        for k, v in table:
            if k == key:
                return float(v)
        raise ValueError("no %s for atom type %r" % (what, key))

    rec = [([float(a[0]), float(a[1]), float(a[2])], str(a[3]))
           for a in receptor]
    lig = [([float(a[0]), float(a[1]), float(a[2])], str(a[3]))
           for a in ligand_pose]
    lipset = [str(t) for t in lipophilic]

    vp = []
    cp = []
    lp = []
    for rx, rt in rec:
        for lx, lt in lig:
            d = _dist(rx, lx)
            if cutoff is not None and d > float(cutoff):
                continue
            vp.append((d, look(radii, rt, "radius")
                       + look(radii, lt, "radius"),
                       math.sqrt(look(depths, rt, "well depth")
                                 * look(depths, lt, "well depth"))))
            cp.append((d, look(charges, rt, "charge"),
                       look(charges, lt, "charge")))
            if rt in lipset and lt in lipset:
                lp.append(d)
    e_vdw, vdw_terms = vdw_term(vp, m, n)
    e_cou, cou_terms = coulomb_term(cp, dielectric, epsilon)
    e_lip, lip_terms = lipophilic_term(lp, r1, r2)
    e_hb, hb_terms, hb_by = hbond_term(hbonds, weights)
    e_rot = float(rot_penalty) * int(n_rot)

    total, parts, order = glide_score(e_vdw, e_cou, e_lip, e_hb, metal,
                                      buryp, e_rot, site, coefficients,
                                      weights)
    return RichResult(payload={
        "gscore": total,
        "estimate": total,
        "se": float("nan"),
        "parts": [parts[k] for k in order],
        "part_names": list(order),
        "vdw_energy": e_vdw,
        "coulomb_energy": e_cou,
        "lipophilic_count": e_lip,
        "hbond_total": e_hb,
        "hbond_by_class": [hb_by[c] for c in HBOND_CLASSES],
        "rot_penalty": e_rot,
        "metal": float(metal),
        "buryp": float(buryp),
        "site": float(site),
        "vdw_terms": vdw_terms,
        "coulomb_terms": cou_terms,
        "lipophilic_terms": lip_terms,
        "hbond_terms": hb_terms,
        "n_contacts": len(vp),
        "n_lipophilic": len(lp),
        "n_hbond": len(hb_terms),
        "n_rot": int(n_rot),
        "dielectric": dielectric,
        "epsilon": float(epsilon),
        "method": "Glide-style empirical docking score",
    })


glides = glide_score_proxy


def cheatsheet():
    return ("glides: Glide-style empirical docking score. dielectrics "
            + ", ".join(DIELECTRICS)
            + "; GScore = 0.065 vdW + 0.130 Coulomb + six weighted terms")
