"""GoldScore: the genetic-algorithm docking fitness.

GoldScore is the older of GOLD's two main fitness functions and it is a
different animal from ChemScore. ChemScore was fitted by regression
against measured affinities and tries to predict them. GoldScore was
tuned to predict the POSE -- where the ligand sits -- and its terms are
energies rather than fitted contributions:

    fitness = -( S_hbond_ext + w S_vdw_ext + S_vdw_int + S_torsion_int )

Four components: the protein-ligand hydrogen bond energy, the
protein-ligand van der Waals energy, the ligand's internal van der Waals
energy and its torsional strain. The fitness is the NEGATIVE of their
sum, so that larger is better, and the external van der Waals term is
multiplied by w = 1.375 -- an empirical correction, stated as such in
the GOLD documentation, whose purpose is to encourage protein-ligand
hydrophobic contact. It is a weight on one term only; leaving it at one
is a different scoring function and the module lets you see that.

The van der Waals terms are Lennard-Jones, and WHICH Lennard-Jones is
the interesting choice. Writing a potential with minimum depth eps at
separation r0 in general (m, n) form,

    E(r) = eps [ (m/(n-m)) (r0/r)^n - (n/(n-m)) (r0/r)^m ]

GOLD applies 6-12 internally and 4-8 externally. The 4-8 is deliberately
softer: a 12 rises so steeply that a genetic algorithm cannot get a
ligand past a slightly-too-close protein atom to find the pose on the
other side of it, and a docking search needs to be able to squeeze.

For binding sites with a loop known to move, GOLD parameterises two
SPLIT potentials, written "4-8 2-4" and "4-8 1-2": the long-range half
is the ordinary 4-8, and below the minimum the short-range half takes
over. The guide's condition is that the two halves agree at the
change-over and that the minimum stays put, which is exactly what the
general form above gives when both halves are built from the same r0
and eps -- both are -eps at r0. So the split potential is not a separate
formula here, it is the same formula with a softer exponent pair inside
the minimum, which is why it is continuous by construction rather than
by a fudge.

Everything that is DATA rather than method -- atom radii, well depths,
hydrogen bond energies, torsion potentials -- comes in as a parameter.
GOLD keeps those in its gold.params file and they are not reproduced
here; the module takes them from the caller, and the combining rule for
a pair (arithmetic on the radii, geometric on the depths) is stated
rather than assumed.

References
  Jones, G., Willett, P., Glen, R.C., Leach, A.R. and Taylor, R. (1997)
    "Development and validation of a genetic algorithm for flexible
    docking." Journal of Molecular Biology 267(3), 727-748.
    doi:10.1006/jmbi.1996.0897. GOLD and its fitness function.
  Jones, G., Willett, P. and Glen, R.C. (1995) "Molecular recognition of
    receptor sites using a genetic algorithm with a description of
    desolvation." Journal of Molecular Biology 245(1), 43-53. The
    hydrogen-bond fitting-point mechanism.
  Verdonk, M.L., Cole, J.C., Hartshorn, M.J., Murray, C.W. and Taylor,
    R.D. (2003) "Improved protein-ligand docking using GOLD." Proteins
    52(4), 609-623. doi:10.1002/prot.10465.
  Cambridge Crystallographic Data Centre, "GOLD User Guide," section 8.3
    for the four components and the 1.375 external van der Waals factor,
    and section 5.4 for the 6-12 internal, 4-8 external defaults and the
    two split potentials.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["goldsc", "gold_score", "lennard_jones", "split_potential",
           "vdw_energy", "hbond_energy", "torsion_energy", "POTENTIALS",
           "VDW_WEIGHT", "cheatsheet"]

# The exponent pairs GOLD parameterises. The first two are the plain
# potentials; the split ones keep the 4-8 outside the minimum and soften
# the inside.
POTENTIALS = ("4-8", "6-12", "split_2-4", "split_1-2")

# The empirical factor on the external van der Waals term, GOLD User
# Guide 8.3.1. It exists to encourage hydrophobic contact.
VDW_WEIGHT = 1.375

_EXPONENTS = {"4-8": (4, 8), "6-12": (6, 12), "split_2-4": (4, 8),
              "split_1-2": (4, 8)}
_INNER = {"split_2-4": (2, 4), "split_1-2": (1, 2)}


def _ipow(x, k):
    """x to a small non-negative integer power, by repeated multiplication.

    Not the language's power operator: R raises to an integer exponent by
    repeated squaring while Python calls the C library's pow, and the two
    disagree in the last bit. A potential that is about to be summed over
    thousands of contacts cannot afford to be a different function in the
    two arms.
    """
    p = 1.0
    for _ in range(int(k)):
        p = p * x
    return p


def lennard_jones(r, r0, eps, m=6, n=12):
    """A general (m, n) Lennard-Jones energy.

    Minimum of -eps at r0, zero at infinity, and singular at zero. The
    parameterisation is by the MINIMUM rather than by the zero crossing
    because that is what a docking parameter file stores: a contact
    radius and a well depth.
    """
    r = float(r)
    if r <= 0.0:
        return float("inf")
    if n <= m:
        raise ValueError("the repulsive exponent must exceed the "
                         "attractive one")
    q = r0 / r
    return eps * ((float(m) / (n - m)) * _ipow(q, n)
                  - (float(n) / (n - m)) * _ipow(q, m))


def split_potential(r, r0, eps, outer=(4, 8), inner=(2, 4)):
    """The soft split potential: 4-8 outside the minimum, softer inside.

    Both halves are built from the same minimum position and depth, so
    both equal -eps at the change-over and the minimum does not move --
    the continuity the guide asks for falls out of the construction
    instead of being imposed afterwards.
    """
    if float(r) >= r0:
        return lennard_jones(r, r0, eps, outer[0], outer[1])
    return lennard_jones(r, r0, eps, inner[0], inner[1])


def _pair(r, r0, eps, potential):
    if potential not in POTENTIALS:
        raise ValueError("potential must be one of %r" % (POTENTIALS,))
    if potential in _INNER:
        return split_potential(r, r0, eps, _EXPONENTS[potential],
                               _INNER[potential])
    m, n = _EXPONENTS[potential]
    return lennard_jones(r, r0, eps, m, n)


def _lookup(table, key, what):
    for k, v in table:
        if k == key:
            return float(v)
    raise ValueError("no %s for atom type %r" % (what, key))


def vdw_energy(pairs, radii, depths, potential="4-8", cutoff=None):
    """Sum a Lennard-Jones potential over a list of contacts.

    `pairs` gives (distance, type_i, type_j). The combining rule is
    arithmetic on the radii and geometric on the well depths, which is
    the ordinary Lorentz-Berthelot convention and is stated here rather
    than hidden: a parameter file that stores combined values directly
    should be passed through `pairs` with its own radii instead.

    A cutoff drops contacts beyond it entirely rather than tapering
    them. That is a discontinuity, and it is the caller's decision, so
    it is off by default.
    """
    terms = []
    kept = 0
    for r, ti, tj in pairs:
        if cutoff is not None and float(r) > float(cutoff):
            continue
        r0 = _lookup(radii, ti, "radius") + _lookup(radii, tj, "radius")
        eps = math.sqrt(_lookup(depths, ti, "well depth")
                        * _lookup(depths, tj, "well depth"))
        terms.append(_pair(r, r0, eps, potential))
        kept += 1
    return (_w.csum(terms) if terms else 0.0), terms, kept


def hbond_energy(bonds, max_distance=2.5):
    """Sum the tabulated hydrogen bond energies of the close pairs.

    `bonds` gives (distance, energy) for each donor-hydrogen to acceptor
    fitting-point pair. GOLD counts a bond towards the fitness only when
    that distance is below `max_distance`, and anneals the threshold
    down over a run so that poor bonds are tolerated early and not at
    the end. The threshold is therefore a parameter of the CALL, not a
    constant: a fitness computed at the starting threshold is not the
    same number as one computed at the finishing threshold, and the
    guide is explicit that only the final one means anything.
    """
    terms = []
    for r, e in bonds:
        if float(r) < float(max_distance):
            terms.append(float(e))
    return (_w.csum(terms) if terms else 0.0), terms, len(terms)


def torsion_energy(torsions):
    """The ligand's internal torsional strain.

    The cosine form a docking parameter file stores, one term per
    rotatable bond: A (1 + cos(n phi - phi0)), with the angle in degrees
    on the way in.
    """
    terms = [float(A) * (1.0 + math.cos(float(n) * math.radians(float(p))
                                        - float(f)))
             for p, A, n, f in torsions]
    return (_w.csum(terms) if terms else 0.0), terms


def _dist(a, b):
    return math.sqrt(_w.csum((a[t] - b[t]) * (a[t] - b[t])
                             for t in range(3)))


def gold_score(receptor, ligand, radii=(), depths=(), hbonds=(),
               internal=(), torsions=(), potential="4-8",
               internal_potential="6-12", vdw_weight=VDW_WEIGHT,
               max_distance=2.5, cutoff=None):
    """The GoldScore fitness of a pose.

    Parameters
    ----------
    receptor, ligand : sequence of sequences
        Atom rows: x, y, z, type. Every cross pair enters the external
        van der Waals sum.
    radii, depths : sequence of pairs
        (type, value) for the contact radius and the well depth. The
        pair values are combined arithmetically and geometrically
        respectively.
    hbonds : sequence
        (distance, energy) per candidate hydrogen bond.
    internal : sequence
        (distance, type_i, type_j) for the ligand's own non-bonded
        pairs.
    torsions : sequence
        (phi, A, n, phi0) per rotatable bond.
    potential, internal_potential : str
        Members of POTENTIALS. GOLD's defaults are 4-8 outside and 6-12
        inside.
    vdw_weight : float
        The factor on the external van der Waals term. GOLD's 1.375.
    max_distance : float
        The hydrogen-bond distance threshold.
    cutoff : float or None
        Drop external contacts beyond this separation.

    Returns
    -------
    RichResult
        The fitness, each component, and the per-contact energies.

    References
    ----------
    Jones et al. (1997) J Mol Biol 267(3), 727-748; CCDC GOLD User
    Guide 8.3 and 5.4.
    """
    rec = [([float(a[0]), float(a[1]), float(a[2])], str(a[3]))
           for a in receptor]
    lig = [([float(a[0]), float(a[1]), float(a[2])], str(a[3]))
           for a in ligand]
    pairs = []
    for rx, rt in rec:
        for lx, lt in lig:
            pairs.append((_dist(rx, lx), rt, lt))
    e_ext, ext_terms, n_ext = vdw_energy(pairs, radii, depths, potential,
                                         cutoff)
    e_int, int_terms, n_int = vdw_energy(internal, radii, depths,
                                         internal_potential, cutoff)
    e_hb, hb_terms, n_hb = hbond_energy(hbonds, max_distance)
    e_to, to_terms = torsion_energy(torsions)

    w = float(vdw_weight)
    total = e_hb + w * e_ext + e_int + e_to
    return RichResult(payload={
        "fitness": -total,
        "energy": total,
        "hbond": e_hb,
        "vdw_external": e_ext,
        "vdw_external_weighted": w * e_ext,
        "vdw_internal": e_int,
        "torsion": e_to,
        "internal": e_int + e_to,
        "external_terms": ext_terms,
        "internal_terms": int_terms,
        "hbond_terms": hb_terms,
        "torsion_terms": to_terms,
        "n_external": n_ext,
        "n_internal": n_int,
        "n_hbond": n_hb,
        "n_receptor": len(rec),
        "n_ligand": len(lig),
        "estimate": -total,
        "se": float("nan"),
        "vdw_weight": w,
        "max_distance": float(max_distance),
        "potential": potential,
        "internal_potential": internal_potential,
        "method": "GoldScore genetic-algorithm docking fitness",
    })


goldsc = gold_score


def cheatsheet():
    return ("goldsc: GoldScore docking fitness. potentials "
            + ", ".join(POTENTIALS)
            + "; external van der Waals weighted 1.375 (CCDC GOLD)")
