"""PMF: knowledge-based potentials of mean force for protein-ligand pairs.

An empirical scoring function like ChemScore fits a handful of
coefficients to measured affinities. A knowledge-based one asks a
different question: forget affinities, just look at where atoms
ACTUALLY sit in the several thousand protein-ligand complexes in the
Protein Data Bank, and read the energy off the statistics. If a
particular kind of protein atom is found next to a particular kind of
ligand atom more often at 3.5 angstroms than chance would put it there,
that separation is favourable, and how much more often says how
favourable.

Formally, inverting the Boltzmann distribution gives a Helmholtz free
interaction energy for each atom-type pair as a function of separation:

    A_ij(r) = -k_B T ln[ f_j(r) rho_ij(r) / rho_ij(bulk) ]

rho_ij(r) is the observed number density of the pair at separation r,
rho_ij(bulk) is its density in the reference state, and f_j(r) is the
ligand volume correction, which exists because the ligand's own atoms
occupy part of the shell and the raw density understates how crowded the
available space really is. The score of a pose is the sum of A_ij over
every pair inside a cutoff.

The important thing about this module is what it does NOT ship. The PMF
potential is a TABLE -- one curve per atom-type pair, derived from a
structure database -- and a table copied out of a paper is data, not
method. So the table is an OUTPUT here: give the module a set of
observed contacts from complexes you have, and it derives the potential;
give it the derived potential and a pose, and it scores. Nothing is
hard-coded that a reader could not reproduce from their own structures.

Two decisions are visible rather than buried. First, the reference
state: "bulk" uses the observed overall density of that pair inside the
cutoff, which is what makes A vanish where the pair is exactly as common
as it is on average; "uniform" uses the density a completely
structureless distribution would give. They differ, and which one you
picked changes every number. Second, an UNOBSERVED bin. A pair never
seen at some separation has zero density and a logarithm of minus
infinity, and treating that as infinite repulsion is a statement the
data does not support -- it may just be a rare pair. Those bins are
capped at a stated value and COUNTED, so a score resting on twenty
capped bins cannot look like a score resting on none.

References
  Muegge, I. and Martin, Y.C. (1999) "A general and fast scoring
    function for protein-ligand interactions: a simplified potential
    approach." Journal of Medicinal Chemistry 42(5), 791-804.
    doi:10.1021/jm980536j. The PMF score, the inverted-Boltzmann form
    above, and the pair-specific cutoffs.
  Muegge, I. (2001) "Effect of ligand volume correction on PMF scoring."
    Journal of Computational Chemistry 22(4), 418-425. The volume
    correction f_j(r) and its effect on the affinity correlation across
    225 complexes.
  Muegge, I. (2006) "PMF scoring revisited." Journal of Medicinal
    Chemistry 49(20), 5895-5902. The later reparameterisation.
  Sippl, M.J. (1990) "Calculation of conformational ensembles from
    potentials of mean force." Journal of Molecular Biology 213(4),
    859-883. The inverted-Boltzmann argument this rests on.

A note on what is sourced. The functional form and the role of each
factor above are from the papers. The particular realisation of the
volume correction as an excluded-volume ratio, and the capping rule for
unobserved bins, are this module's -- they are described in the sources
in words rather than given as formulas, so they are named here as
choices rather than attributed.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["pmfsc", "pmf_potential", "pmf_score", "derive_potential",
           "shell_volume", "bin_index", "REFERENCES", "CORRECTIONS",
           "DEFAULT_CUTOFF", "cheatsheet"]

REFERENCES = ("bulk", "uniform")
CORRECTIONS = ("none", "excluded_volume")

# The separation beyond which pairs are not counted. Muegge and Martin
# use pair-specific cutoffs; those are data this module does not ship,
# so this is a single default the caller is expected to override with
# their own table.
DEFAULT_CUTOFF = 12.0


def shell_volume(r1, r2):
    """The volume of the spherical shell between two radii.

    Written as the difference of two cubes rather than as a thin-shell
    approximation, because the bins here are wide enough that
    4 pi r^2 dr is visibly wrong at the inner ones -- and the inner bins
    are exactly where the interesting structure lives.
    """
    if r2 < r1:
        raise ValueError("the outer radius must not be inside the inner")
    return (4.0 / 3.0) * math.pi * (r2 * r2 * r2 - r1 * r1 * r1)


def bin_index(r, r_max, n_bins):
    """Which bin a separation falls in, or -1 if it is past the cutoff.

    A distance exactly on an edge goes to the UPPER bin, so the bins are
    half-open on the left and the assignment cannot depend on which side
    of an edge the arithmetic happens to land.
    """
    r = float(r)
    if r < 0.0:
        raise ValueError("a separation cannot be negative")
    if r >= r_max:
        return -1
    k = int(math.floor(r * n_bins / r_max))
    if k < 0:
        k = 0
    if k >= n_bins:
        k = n_bins - 1
    return k


def _key(a, b):
    return a + "|" + b


def derive_potential(observations, n_complexes=1, r_max=DEFAULT_CUTOFF,
                     n_bins=24, reference="bulk", correction="none",
                     occupied=None, kT=1.0, cap=6.0):
    """Turn observed contacts into a potential, one curve per type pair.

    Parameters
    ----------
    observations : sequence
        (type_i, type_j, separation) for every contact seen in the
        training complexes. Order within a pair matters only in that the
        two type labels are kept as given: a protein-carbon to
        ligand-oxygen pair is not the same statistic as its reverse, and
        collapsing them would be a modelling choice, not a convenience.
    n_complexes : int
        How many complexes the observations came from. It scales every
        density identically and therefore cancels out of a bulk-
        referenced potential -- which is worth knowing, and is the
        subject of one of this module's checks.
    r_max, n_bins : float, int
        The radial grid.
    reference : str
        A member of REFERENCES.
    correction : str
        A member of CORRECTIONS.
    occupied : sequence or None
        Per-bin ligand-occupied volume, for the excluded-volume
        correction.
    kT : float
        The energy scale. The default of one leaves the potential in
        units of kT; pass Boltzmann's constant times your temperature to
        get energies.
    cap : float
        The magnitude, in units of kT, at which an unobserved bin is
        held. Unobserved is not the same as forbidden.

    Returns
    -------
    dict
        Keyed by "type_i|type_j": counts, density, the reference
        density, the potential A(r) and the count of capped bins.
    """
    if reference not in REFERENCES:
        raise ValueError("reference must be one of %r" % (REFERENCES,))
    if correction not in CORRECTIONS:
        raise ValueError("correction must be one of %r" % (CORRECTIONS,))
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("need at least one radial bin")
    if r_max <= 0.0:
        raise ValueError("the cutoff must be positive")
    nc = float(n_complexes)
    if nc <= 0.0:
        raise ValueError("need at least one complex")

    edges = [r_max * t / n_bins for t in range(n_bins + 1)]
    vol = [shell_volume(edges[t], edges[t + 1]) for t in range(n_bins)]
    fcorr = [1.0] * n_bins
    if correction == "excluded_volume":
        if occupied is None:
            raise ValueError("the excluded-volume correction needs the "
                             "occupied volumes")
        for t in range(n_bins):
            free = vol[t] - float(occupied[t])
            if free <= 0.0:
                raise ValueError("bin %d is entirely occupied by ligand "
                                 "atoms; the correction is undefined "
                                 "there" % t)
            # The available space is smaller than the shell, so the true
            # density is HIGHER than the raw count suggests, by exactly
            # this ratio.
            fcorr[t] = vol[t] / free

    counts = {}
    for ti, tj, r in observations:
        k = bin_index(r, r_max, n_bins)
        if k < 0:
            continue
        key = _key(str(ti), str(tj))
        if key not in counts:
            counts[key] = [0] * n_bins
        counts[key][k] += 1

    out = {}
    total_vol = shell_volume(0.0, r_max)
    for key in sorted(counts):
        c = counts[key]
        dens = [c[t] / (nc * vol[t]) for t in range(n_bins)]
        n_tot = sum(c)
        if reference == "bulk":
            ref = n_tot / (nc * total_vol)
        else:
            ref = 1.0 / total_vol
        a = []
        capped = 0
        for t in range(n_bins):
            x = fcorr[t] * dens[t]
            if ref <= 0.0 or x <= 0.0:
                # Unobserved, not forbidden. Hold it at the cap and say
                # so rather than letting a logarithm of zero decide.
                a.append(float(kT) * cap)
                capped += 1
            else:
                a.append(-float(kT) * math.log(x / ref))
        out[key] = {"counts": c, "density": dens, "reference": ref,
                    "potential": a, "capped": capped, "n": n_tot,
                    "edges": edges, "volume": vol, "correction": fcorr}
    return out


def pmf_score(pairs, potential, r_max=DEFAULT_CUTOFF, n_bins=24,
              missing=0.0):
    """Score a pose against a derived potential.

    `pairs` gives (type_i, type_j, separation). A pair whose type
    combination is absent from the potential contributes `missing` and
    is counted separately -- silently scoring it as zero would let a
    pose made entirely of unparameterised atoms come out looking
    average.
    """
    terms = []
    used = 0
    beyond = 0
    unknown = 0
    for ti, tj, r in pairs:
        k = bin_index(r, r_max, n_bins)
        if k < 0:
            beyond += 1
            continue
        key = _key(str(ti), str(tj))
        if key not in potential:
            unknown += 1
            terms.append(float(missing))
            continue
        terms.append(potential[key]["potential"][k])
        used += 1
    return (_w.csum(terms) if terms else 0.0), used, beyond, unknown


def _dist(a, b):
    return math.sqrt(_w.csum((a[t] - b[t]) * (a[t] - b[t])
                             for t in range(3)))


def pmf_potential(receptor, ligand, potential=None, observations=None,
                  n_complexes=1, r_max=DEFAULT_CUTOFF, n_bins=24,
                  reference="bulk", correction="none", occupied=None,
                  kT=1.0, cap=6.0, missing=0.0):
    """Derive a potential if needed, then score the pose.

    Parameters
    ----------
    receptor, ligand : sequence of sequences
        Atom rows: x, y, z, type.
    potential : dict or None
        A potential from `derive_potential`. When omitted it is derived
        from `observations`, so a caller with a training set and a pose
        can do both in one call.
    observations : sequence or None
        Training contacts, as `derive_potential` takes them.

    Returns
    -------
    RichResult
        The score, the potential used, and the counts that say how much
        of the pose the potential actually covered.

    References
    ----------
    Muegge and Martin (1999) J Med Chem 42(5), 791-804; Muegge (2001)
    J Comput Chem 22(4), 418-425.
    """
    if potential is None:
        if observations is None:
            raise ValueError("give either a derived potential or the "
                             "observations to derive one from")
        potential = derive_potential(observations, n_complexes, r_max,
                                     n_bins, reference, correction,
                                     occupied, kT, cap)
    rec = [([float(a[0]), float(a[1]), float(a[2])], str(a[3]))
           for a in receptor]
    lig = [([float(a[0]), float(a[1]), float(a[2])], str(a[3]))
           for a in ligand]
    pairs = []
    for rx, rt in rec:
        for lx, lt in lig:
            pairs.append((rt, lt, _dist(rx, lx)))
    score, used, beyond, unknown = pmf_score(pairs, potential, r_max,
                                             n_bins, missing)
    capped = sum(potential[k]["capped"] for k in potential)
    return RichResult(payload={
        "score": score,
        "estimate": score,
        "se": float("nan"),
        "n_scored": used,
        "n_beyond_cutoff": beyond,
        "n_unparameterised": unknown,
        "n_pairs": len(pairs),
        "n_types": len(potential),
        "n_capped_bins": capped,
        "potential": potential,
        "r_max": float(r_max),
        "n_bins": int(n_bins),
        "kT": float(kT),
        "cap": float(cap),
        "reference": reference,
        "correction": correction,
        "method": "knowledge-based PMF scoring",
    })


pmfsc = pmf_potential


def cheatsheet():
    return ("pmfsc: knowledge-based PMF scoring. references "
            + ", ".join(REFERENCES) + "; corrections "
            + ", ".join(CORRECTIONS)
            + "; the potential is derived, not shipped")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
pmfpotential = pmf_potential
