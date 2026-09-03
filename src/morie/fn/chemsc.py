"""ChemScore: the empirical protein-ligand docking score.

ChemScore estimates the free energy of binding as a sum of physical
contributions, each multiplied by a coefficient fitted by regression
against measured affinities for 82 protein-ligand complexes:

    dG = dG0 + v1 Sum(hbond) + v2 Sum(metal) + v3 Sum(lipophilic)
              + v4 Hrot

with v1 = -3.34, v2 = -6.03, v3 = -0.117 and v4 = 2.56. The signs are
the interesting part: hydrogen bonds, metal contacts and lipophilic
contact all make binding more favourable, and freezing a rotatable bond
makes it less -- the entropy you pay for holding the ligand still.

Every geometric term is a BLOCK FUNCTION of a deviation from ideal:

    B(d) = 1                            d <= d_ideal
         = (d_max - d) / (d_max - d_ideal)   d_ideal < d <= d_max
         = 0                            d > d_max

so a contact inside its tolerance counts fully, a contact outside its
maximum counts not at all, and in between it ramps down linearly. The
GOLD implementation convolves that block with a Gaussian to smooth the
two corners, which matters when the score is being optimised: a genetic
algorithm climbing a function with a kink gets stuck on the kink. Both
are here, selectable, because the unsmoothed block is what the 1997
paper defines and the smoothed one is what a docking run should use.
The convolution is done in closed form rather than numerically -- a
trapezoid against a Gaussian integrates to normal distribution and
density values, so there is no quadrature to disagree about.

A hydrogen bond contributes the PRODUCT of three of these: one on the
H...A distance, one on the D-H...A angle, and one on the H...A-X angle
at the acceptor. An acceptor with several attached heavy atoms
contributes the product over all of them. So a bond of ideal geometry
contributes exactly one and anything else contributes less, which is
why the term is a count of good hydrogen bonds rather than an energy.

Clashes and internal torsional strain are penalties added on top. They
exist to stop a docking run proposing a pose that scores well and
overlaps the protein.

Parameter provenance. Every coefficient, radius, angle and sigma below
is the default stated in the CCDC GOLD User Guide, which names each one
by its identifier in the ChemScore parameter file; those identifiers are
carried through into the constant names here so the two can be checked
against each other. Two things are NOT taken from a source and are
marked as this module's own: the shape of the clash penalty (the guide
gives its radii but renders the function itself as an image), and the
value of the intercept dG0 (the guide omits it, since a constant cannot
change a ranking). dG0 therefore defaults to zero and is a parameter, not
a hard-coded number.

References
  Eldridge, M.D., Murray, C.W., Auton, T.R., Paolini, G.V. and Mee, R.P.
    (1997) "Empirical scoring functions: I. The development of a fast
    empirical scoring function to estimate the binding affinity of
    ligands in receptor complexes." Journal of Computer-Aided Molecular
    Design 11(5), 425-445. doi:10.1023/A:1007996124545. The function,
    the block form and the regression against 82 complexes.
  Baxter, C.A., Murray, C.W., Clark, D.E., Westhead, D.R. and Eldridge,
    M.D. (1998) "Flexible docking using Tabu search and an empirical
    estimate of binding affinity." Proteins 33(3), 367-382. The clash
    and torsion penalties that turn the affinity estimate into a
    docking score.
  Cambridge Crystallographic Data Centre, "GOLD User Guide," sections
    8.4.1 to 8.4.6. Every default parameter value used here, each named
    by its ChemScore-file identifier.
  Verdonk, M.L., Cole, J.C., Hartshorn, M.J., Murray, C.W. and Taylor,
    R.D. (2003) "Improved protein-ligand docking using GOLD." Proteins
    52(4), 609-623. The GOLD implementation these defaults belong to.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["chemsc", "chemscore_dock", "chemscore", "block", "smooth_block",
           "hbond_term", "metal_term", "lipophilic_term", "rot_term",
           "clash_term", "torsion_term", "SMOOTHINGS", "COEFFICIENTS",
           "HBOND", "METAL", "LIPO", "CLASH", "cheatsheet"]

SMOOTHINGS = ("gaussian", "none")

# Regression coefficients, GOLD User Guide 8.4.3 to 8.4.5. The names are
# the identifiers in the ChemScore parameter file.
COEFFICIENTS = {
    "HBOND_COEFFICIENT": -3.34,
    "METAL_COEFFICIENT": -6.03,
    "LIPO_COEFFICIENT": -0.117,
    "ROT_COEFFICIENT": 2.56,
}

# Hydrogen-bond geometry, GOLD User Guide 8.4.3. Distances in angstroms,
# angles in degrees.
HBOND = {
    "R_IDEAL": 1.85,
    "DELTA_R_IDEAL": 0.25,
    "DELTA_R_MAX": 0.65,
    "HBOND_R_SIGMA": 0.1,
    "ALPHA_IDEAL": 180.0,
    "DELTA_ALPHA_IDEAL": 30.0,
    "DELTA_ALPHA_MAX": 80.0,
    "HBOND_ALPHA_SIGMA": 10.0,
    "BETA_IDEAL": 180.0,
    "DELTA_BETA_IDEAL": 70.0,
    "DELTA_BETA_MAX": 80.0,
    "HBOND_BETA_SIGMA": 10.0,
}

# Metal-binding geometry, GOLD User Guide 8.4.4.
METAL = {"METAL_R1": 2.6, "METAL_R2": 3.0, "METAL_R_SIGMA": 0.1}

# Lipophilic contact geometry, GOLD User Guide 8.4.4. The lipophilic
# term is scored over a much longer range than the metal one, which is
# the whole difference between the two parameterisations.
LIPO = {"LIPO_R1": 4.1, "LIPO_R2": 7.1, "LIPO_R_SIGMA": 0.1}

# Clash radii, GOLD User Guide 8.4.6.
CLASH = {
    "CLASH_RADIUS_HBOND": 1.6,
    "CLASH_RADIUS_METAL": 1.3,
    "CLASH_RADIUS_SULPHUR": 3.35,
    "CLASH_RADIUS_GENERAL": 3.10,
}


def block(d, d_ideal, d_max):
    """The ChemScore block function of a deviation from ideal.

    One inside the tolerance, zero past the maximum, a straight line
    between. Note it is a function of the DEVIATION, so it is already
    folded about zero and never sees a sign.
    """
    d = abs(float(d))
    if d_max <= d_ideal:
        raise ValueError("the maximum deviation must exceed the ideal "
                         "tolerance")
    if d <= d_ideal:
        return 1.0
    if d >= d_max:
        return 0.0
    return (d_max - d) / (d_max - d_ideal)


def smooth_block(d, d_ideal, d_max, sigma):
    """The block function convolved with a Gaussian, in closed form.

    A trapezoid against a normal density integrates to normal
    distribution and density values, so this is exact arithmetic and not
    a quadrature -- which matters here, because a numerical convolution
    would be the one part of the score that two implementations could
    disagree about while both being "right".

    With sigma at or below zero this is the plain block function, which
    is the honest limit rather than a special case.
    """
    if sigma <= 0.0:
        return block(d, d_ideal, d_max)
    d = abs(float(d))
    if d_max <= d_ideal:
        raise ValueError("the maximum deviation must exceed the ideal "
                         "tolerance")
    z1 = (d_ideal - d) / sigma
    z2 = (d_max - d) / sigma
    flat = _w.ncdf(z1)
    ramp = ((d_max - d) * (_w.ncdf(z2) - _w.ncdf(z1))
            - sigma * (_w.npdf(z1) - _w.npdf(z2))) / (d_max - d_ideal)
    v = flat + ramp
    # The convolution of a function bounded in [0, 1] is bounded in
    # [0, 1]; only rounding can put it outside, and letting that leak
    # into a product of three terms would be a slow poison.
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _B(d, d_ideal, d_max, sigma, smoothing):
    if smoothing == "none":
        return block(d, d_ideal, d_max)
    if smoothing == "gaussian":
        return smooth_block(d, d_ideal, d_max, sigma)
    raise ValueError("smoothing must be one of %r" % (SMOOTHINGS,))


def hbond_term(r, alpha, betas, smoothing="gaussian", par=None):
    """One donor-acceptor pair's contribution, at most one.

    Parameters
    ----------
    r : float
        The H...A distance.
    alpha : float
        The D-H...A angle, in degrees.
    betas : sequence
        The H...A-X angles, one per heavy atom attached to the acceptor.
        Their block values MULTIPLY: an acceptor with three neighbours
        has to satisfy all three directions, not the best one.
    smoothing : str
        A member of SMOOTHINGS.
    par : dict or None
        Overrides for the HBOND parameters.
    """
    p = dict(HBOND)
    if par:
        p.update(par)
    v = _B(r - p["R_IDEAL"], p["DELTA_R_IDEAL"], p["DELTA_R_MAX"],
           p["HBOND_R_SIGMA"], smoothing)
    v *= _B(alpha - p["ALPHA_IDEAL"], p["DELTA_ALPHA_IDEAL"],
            p["DELTA_ALPHA_MAX"], p["HBOND_ALPHA_SIGMA"], smoothing)
    for b in betas:
        v *= _B(b - p["BETA_IDEAL"], p["DELTA_BETA_IDEAL"],
                p["DELTA_BETA_MAX"], p["HBOND_BETA_SIGMA"], smoothing)
    return v


def _over(r, r1):
    """How far past the ideal separation a contact is, never negative.

    The metal and lipophilic terms are RANGES, not windows: the guide
    calls R1 the ideal separation and R2 "the maximum distance to be
    considered a binding interaction", so anything at or inside R1 is
    fully ideal and only the far side ramps down. Folding this about
    zero the way the hydrogen-bond deviations are folded would penalise
    a contact for being too close, which is the clash term's job and not
    this one's -- and it would also make a contact sitting exactly on R1
    score a hair under one whenever the coordinate arithmetic put it a
    single bit on the wrong side.
    """
    d = float(r) - r1
    return d if d > 0.0 else 0.0


def metal_term(r, smoothing="gaussian", par=None):
    """One acceptor-metal contact's contribution, at most one."""
    p = dict(METAL)
    if par:
        p.update(par)
    return _B(_over(r, p["METAL_R1"]), 0.0, p["METAL_R2"] - p["METAL_R1"],
              p["METAL_R_SIGMA"], smoothing)


def lipophilic_term(r, smoothing="gaussian", par=None):
    """One lipophilic atom pair's contribution, at most one."""
    p = dict(LIPO)
    if par:
        p.update(par)
    return _B(_over(r, p["LIPO_R1"]), 0.0, p["LIPO_R2"] - p["LIPO_R1"],
              p["LIPO_R_SIGMA"], smoothing)


def rot_term(fractions):
    """The frozen-rotatable-bond entropy term.

    `fractions` gives, for each FROZEN rotatable bond, the pair of
    non-lipophilic fractions on its two sides. A bond counts as frozen
    when atoms on both sides touch the protein; a bond that is still
    free costs nothing and does not appear.

    The expression is

        Hrot = 1 + (1 - 1/Nrot) Sum_r (P_nl + P'_nl) / 2

    which is one unit for the first frozen bond and progressively less
    for each one after it -- freezing a bond in an already rigid ligand
    costs less than freezing the first. With no frozen bonds it is zero,
    not one: there is no entropy to pay.

    The GOLD guide quotes the two fractions as percentages; they are
    taken here as fractions in [0, 1].
    """
    n = len(fractions)
    if n == 0:
        return 0.0
    s = _w.csum(0.5 * (float(a) + float(b)) for a, b in fractions)
    return 1.0 + (1.0 - 1.0 / n) * s


def clash_term(r, kind="general", slope=1.0, par=None):
    """The clash penalty for one too-close contact.

    The radii are the GOLD defaults: 1.6 for a hydrogen-bonding contact,
    1.3 for a metal coordination contact, 3.35 to a protein sulphur and
    3.10 for anything else. A hydrogen bond is allowed much closer than
    a general contact because it IS a close contact.

    The SHAPE of the penalty is this module's, not a source's: the guide
    states the radii and renders the function as an image. A linear ramp
    of `slope` per angstrom of overlap is used, which is monotone, zero
    at the radius and continuous there -- the three properties a docking
    run actually needs from it. Anyone with the published form should
    pass their own through and the rest of the score is unaffected.
    """
    p = dict(CLASH)
    if par:
        p.update(par)
    key = {"hbond": "CLASH_RADIUS_HBOND", "metal": "CLASH_RADIUS_METAL",
           "sulphur": "CLASH_RADIUS_SULPHUR",
           "general": "CLASH_RADIUS_GENERAL"}.get(kind)
    if key is None:
        raise ValueError("kind must be hbond, metal, sulphur or general")
    rc = p[key]
    r = float(r)
    return 0.0 if r >= rc else float(slope) * (rc - r)


def torsion_term(phi, A, n, phi0):
    """One rotatable bond's internal torsional strain.

    The threefold cosine form the ChemScore parameter file carries, with
    A, n and the phase read from lines like SP3_SP3_BOND. The angle is
    in degrees on the way in, because that is how a torsion is measured.
    """
    return float(A) * (1.0 + math.cos(float(n) * math.radians(float(phi))
                                      - float(phi0)))


def chemscore(hbonds=(), metals=(), lipophilic=(), rotatable=(),
              clashes=(), torsions=(), smoothing="gaussian", dg0=0.0,
              clash_slope=1.0, intra_coefficient=1.0, coefficients=None,
              par=None):
    """Assemble a ChemScore from its already-measured geometric terms.

    This is the function the docking front end calls once it has turned
    coordinates into contacts, and it is separately callable because a
    reader who wants to check the arithmetic should not have to build a
    protein first.

    Returns
    -------
    RichResult
        The free energy estimate, the fitness (its negative, plus the
        penalties), and every term separately so the total can be
        checked against its parts.
    """
    if smoothing not in SMOOTHINGS:
        raise ValueError("smoothing must be one of %r" % (SMOOTHINGS,))
    co = dict(COEFFICIENTS)
    if coefficients:
        co.update(coefficients)
    hp = None if par is None else par.get("hbond")
    mp = None if par is None else par.get("metal")
    lp = None if par is None else par.get("lipo")
    cp = None if par is None else par.get("clash")

    hb = [hbond_term(r, a, b, smoothing, hp) for r, a, b in hbonds]
    mt = [metal_term(r, smoothing, mp) for r in metals]
    lp_ = [lipophilic_term(r, smoothing, lp) for r in lipophilic]
    s_hb = _w.csum(hb) if hb else 0.0
    s_mt = _w.csum(mt) if mt else 0.0
    s_lp = _w.csum(lp_) if lp_ else 0.0
    h_rot = rot_term(rotatable)

    cl = [clash_term(r, k, clash_slope, cp) for r, k in clashes]
    to = [torsion_term(p, a, n, f) for p, a, n, f in torsions]
    s_cl = _w.csum(cl) if cl else 0.0
    s_to = _w.csum(to) if to else 0.0

    dg = (float(dg0) + co["HBOND_COEFFICIENT"] * s_hb
          + co["METAL_COEFFICIENT"] * s_mt
          + co["LIPO_COEFFICIENT"] * s_lp
          + co["ROT_COEFFICIENT"] * h_rot)
    # The fitness is the negative of the free energy so that bigger is
    # better, with the penalties subtracted from it -- a clash makes a
    # pose worse whichever sign convention the energy is carrying.
    fitness = -dg - s_cl - float(intra_coefficient) * s_to
    return RichResult(payload={
        "dg": dg,
        "fitness": fitness,
        "hbond": s_hb,
        "metal": s_mt,
        "lipophilic": s_lp,
        "h_rot": h_rot,
        "clash": s_cl,
        "torsion": s_to,
        "hbond_terms": hb,
        "metal_terms": mt,
        "lipophilic_terms": lp_,
        "clash_terms": cl,
        "torsion_terms": to,
        "n_hbond": len(hb),
        "n_metal": len(mt),
        "n_lipophilic": len(lp_),
        "n_rotatable": len(rotatable),
        "n_clash": len(cl),
        "estimate": dg,
        "se": float("nan"),
        "dg0": float(dg0),
        "smoothing": smoothing,
        "method": "ChemScore empirical docking",
    })


def _dist(a, b):
    return math.sqrt(_w.csum((a[t] - b[t]) * (a[t] - b[t])
                             for t in range(3)))


def _angle(a, b, c):
    """The angle at b, in degrees, formed by a-b-c."""
    u = [a[t] - b[t] for t in range(3)]
    v = [c[t] - b[t] for t in range(3)]
    nu = math.sqrt(_w.dot(u, u))
    nv = math.sqrt(_w.dot(v, v))
    if nu <= 0.0 or nv <= 0.0:
        return float("nan")
    cc = _w.dot(u, v) / (nu * nv)
    if cc > 1.0:
        cc = 1.0
    if cc < -1.0:
        cc = -1.0
    return math.degrees(math.acos(cc))


def chemscore_dock(receptor, ligand, smoothing="gaussian", dg0=0.0,
                   clash_slope=1.0, intra_coefficient=1.0,
                   rotatable=(), torsions=(), coefficients=None, par=None):
    """Score a pose from coordinates and atom roles.

    Parameters
    ----------
    receptor, ligand : sequence of sequences
        One row per atom: x, y, z, role, then the coordinates of the
        attached atom the geometry needs -- the donor's hydrogen for a
        donor, the acceptor's attached heavy atom for an acceptor. Roles
        are "donor", "acceptor", "metal", "lipophilic", "sulphur" and
        "other". Rows whose partner coordinates are absent contribute
        no directional term.
    rotatable : sequence
        Pairs of non-lipophilic fractions for each FROZEN rotatable
        bond, as `rot_term` takes them. Nothing in a set of coordinates
        says which bonds are frozen, so this is supplied rather than
        guessed.
    torsions : sequence
        (phi, A, n, phi0) for each rotatable bond's internal strain.

    Returns
    -------
    RichResult
        As `chemscore`, with the contact lists it built.

    References
    ----------
    Eldridge et al. (1997) J Comput Aided Mol Des 11(5), 425-445; CCDC
    GOLD User Guide 8.4.
    """
    def parse(rows):
        out = []
        for r in rows:
            xyz = [float(r[0]), float(r[1]), float(r[2])]
            role = str(r[3])
            att = None
            if len(r) >= 7 and r[4] is not None:
                att = [float(r[4]), float(r[5]), float(r[6])]
            out.append((xyz, role, att))
        return out

    rec = parse(receptor)
    lig = parse(ligand)

    hbonds = []
    metals = []
    lipo = []
    clashes = []
    for rx, rrole, ratt in rec:
        for lx, lrole, latt in lig:
            d = _dist(rx, lx)
            pair = None
            if rrole == "donor" and lrole == "acceptor":
                # The donor's hydrogen is the atom the distance and the
                # D-H...A angle are both measured from, not the donor
                # heavy atom.
                if ratt is not None:
                    hb_r = _dist(ratt, lx)
                    al = _angle(rx, ratt, lx)
                    be = [_angle(ratt, lx, latt)] if latt is not None else []
                    hbonds.append((hb_r, al, be))
                    pair = ("hbond", hb_r)
            elif rrole == "acceptor" and lrole == "donor":
                if latt is not None:
                    hb_r = _dist(latt, rx)
                    al = _angle(lx, latt, rx)
                    be = [_angle(latt, rx, ratt)] if ratt is not None else []
                    hbonds.append((hb_r, al, be))
                    pair = ("hbond", hb_r)
            elif rrole == "metal" and lrole == "acceptor":
                metals.append(d)
                pair = ("metal", d)
            elif rrole == "acceptor" and lrole == "metal":
                metals.append(d)
                pair = ("metal", d)
            elif rrole in ("lipophilic", "sulphur") and lrole == "lipophilic":
                lipo.append(d)
            if pair is not None:
                clashes.append((pair[1], pair[0]))
            else:
                clashes.append((d, "sulphur" if rrole == "sulphur"
                                else "general"))
    return chemscore(hbonds, metals, lipo, rotatable, clashes, torsions,
                     smoothing, dg0, clash_slope, intra_coefficient,
                     coefficients, par)


chemsc = chemscore_dock


def cheatsheet():
    return ("chemsc: ChemScore empirical docking. smoothings "
            + ", ".join(SMOOTHINGS)
            + "; coefficients hbond -3.34, metal -6.03, lipo -0.117, "
              "rot 2.56 (CCDC GOLD defaults)")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
chemscoredock = chemscore_dock
