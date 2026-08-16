"""The reactive pose filter: does this docked pose let the reaction
actually happen?

A covalent inhibitor binds twice. First it docks, held by the same
shape and charge complementarity as any other ligand; then its warhead
reacts with a cysteine and the complex stops being reversible. Docking
software scores the first step. It does not, on its own, check the
second -- and a pose can score beautifully while presenting the warhead
to the wrong side of the cysteine, where no chemistry is possible.

This is the filter that checks the second step, and it is pure
geometry. Three measurements, all exact, none of them fitted:

  THE DISTANCE from the cysteine's sulfur to the electrophilic carbon.
  The forming carbon-sulfur bond is about 1.8 angstroms; a pose whose
  warhead is six angstroms away is not about to react no matter what it
  scored.

  THE ATTACK ANGLE at the electrophilic carbon. A nucleophile does not
  approach an electrophile from any direction. Buergi and Dunitz
  established from crystal structures that attack on a carbonyl comes
  in at about 107 degrees to the carbon-oxygen axis, and that angle is
  a property of the orbitals, not of the particular molecule. Michael
  addition to a conjugated alkene is different: the sulfur attacks the
  beta carbon roughly perpendicular to the double bond, so the two
  chemistries need two criteria and this module has both as named
  routes rather than one blurred average.

  THE DIHEDRAL about the forming bond, reported for both routes because
  it distinguishes an approach that is merely at the right angle from
  one that is at the right angle on the right FACE.

WHAT IS PUBLISHED AND WHAT IS A SETTING. The Buergi-Dunitz angle is
published and is the default. Every tolerance -- how far off the ideal
angle still counts, how long a distance still counts -- is a parameter
with a stated default, because those are decisions about how permissive
a screen should be and they belong to whoever is running it.

The warhead is FOUND, not assumed: the module reads the bond orders and
identifies the electrophilic carbon itself, and if there is no warhead
it says so rather than measuring an angle at an arbitrary atom.

References
  Buergi, H.B., Dunitz, J.D. and Shefter, E. (1973) "Geometrical
    reaction coordinates. II. Nucleophilic addition to a carbonyl
    group." Journal of the American Chemical Society 95(15),
    5065-5067. doi:10.1021/ja00796a058. The 107 degree approach.
  Bianco, G., Forli, S., Goodsell, D.S. and Olson, A.J. (2016)
    "Covalent docking using autodock: Two-point attractor and flexible
    side chain methods." Protein Science 25(1), 295-301.
    doi:10.1002/pro.2733. The pose-generation problem this filters.
  Zhu, K., Borrelli, K.W., Greenwood, J.R., Day, T., Abel, R., Farid,
    R.S. and Harder, E. (2014) "Docking covalent inhibitors: a
    parameter free approach to pose prediction and scoring." Journal of
    Chemical Information and Modeling 54(7), 1932-1940.
    doi:10.1021/ci500118s. CovDock, which the ledger entry names.
"""

import math

from . import _w3num as _w
from .avalon import parse_smiles, _adjacency
from ._richresult import RichResult

__all__ = ["reactive_pose_filter", "find_warhead", "angle", "dihedral",
           "distance", "cheatsheet"]

# Buergi and Dunitz's approach angle for nucleophilic addition to a
# carbonyl, in degrees. A property of the orbitals, not a fitted value.
BURGI_DUNITZ = 107.0
_MODES = ("burgi_dunitz", "michael")


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(_w.csum(v * v for v in a))


def distance(a, b):
    """The straight-line distance between two points."""
    return _norm(_sub(a, b))


def angle(a, b, c):
    """The angle at b, in degrees, subtended by a and c.

    The cosine is clamped into its own range before the arc cosine:
    two nearly parallel vectors can produce a ratio a hair outside
    minus one to one purely by rounding, and an unclamped arc cosine
    turns that into a domain error rather than the zero degrees it
    obviously is.
    """
    u = _sub(a, b)
    v = _sub(c, b)
    nu = _norm(u)
    nv = _norm(v)
    if nu == 0.0 or nv == 0.0:
        raise ValueError("an angle needs three distinct points")
    t = _w.dot(u, v) / (nu * nv)
    if t > 1.0:
        t = 1.0
    if t < -1.0:
        t = -1.0
    return math.degrees(math.acos(t))


def dihedral(a, b, c, d):
    """The torsion about the b-c axis, in degrees, signed.

    Computed from the two plane normals with an arc tangent of two
    arguments rather than an arc cosine, so the sign survives -- and
    the sign is the whole point here, because it is what says which
    FACE the nucleophile is approaching from.
    """
    b1 = _sub(b, a)
    b2 = _sub(c, b)
    b3 = _sub(d, c)
    n2 = _norm(b2)
    if n2 == 0.0:
        raise ValueError("a torsion needs a defined axis")
    u = [v / n2 for v in b2]
    n1 = _cross(b1, b2)
    n3 = _cross(b2, b3)
    x = _w.dot(n1, n3)
    y = _w.dot(_cross(n1, n3), u) * -1.0
    return math.degrees(math.atan2(y, x))


def find_warhead(smiles, mode="burgi_dunitz"):
    """Locate the electrophilic carbon and the atom that orients it.

    For the Buergi-Dunitz route: a carbon holding a double bond to
    oxygen or nitrogen, or a triple bond to nitrogen. The electrophile
    is that carbon and the reference is the heteroatom, because the
    approach angle is measured to the carbon-heteroatom axis.

    For the Michael route: a carbon-carbon double bond with one end
    attached to a carbonyl carbon. The electrophile is the FAR end --
    the beta carbon, which is where the sulfur adds -- and the
    reference is the alpha carbon it is doubly bonded to. Getting these
    two the wrong way round would measure a real angle at the wrong
    atom, which is why the anchors check the indices and not just that
    something was found.

    Returns the electrophile, the reference, and a third atom for the
    torsion, or None when the molecule carries no such warhead.
    """
    if mode not in _MODES:
        raise ValueError("the mode is burgi_dunitz or michael")
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    n = len(el)
    adj = _adjacency(n, bonds)
    if mode == "burgi_dunitz":
        for i in range(n):
            if el[i] != "C" or arom[i]:
                continue
            for v, o, k in adj[i]:
                if (o == 2 and el[v] in ("O", "N")) or \
                        (o == 3 and el[v] == "N"):
                    third = None
                    for w, oo, kk in adj[i]:
                        if w != v:
                            third = w
                    if third is None:
                        continue
                    return i, v, third
        return None
    # Michael: an alkene conjugated to a carbonyl.
    carbonyl = [False] * n
    for i in range(n):
        if el[i] != "C":
            continue
        for v, o, k in adj[i]:
            if o == 2 and el[v] == "O":
                carbonyl[i] = True
    for a, b, o in bonds:
        if o != 2 or el[a] != "C" or el[b] != "C":
            continue
        for alpha, beta in ((a, b), (b, a)):
            for v, oo, kk in adj[alpha]:
                if v != beta and carbonyl[v]:
                    return beta, alpha, v
    return None


def reactive_pose_filter(pose, cys_residue, mode="burgi_dunitz",
                         d_min=1.5, d_max=3.5, ideal=None,
                         angle_tol=15.0, warhead=None):
    """Does this pose present its warhead so the cysteine can attack?

    Parameters
    ----------
    pose : mapping
        ``smiles`` for the ligand and ``coords``, one triple per atom
        in the order the SMILES lists them.
    cys_residue : mapping
        ``SG`` for the sulfur, and ``CB`` for the beta carbon if the
        torsion is wanted.
    mode : {"burgi_dunitz", "michael"}
        Which chemistry, and therefore which ideal angle.
    d_min, d_max : float
        The window the forming bond must fall in. The default upper
        bound is a near-attack conformation, not a bond.
    ideal : float or None
        The approach angle. None takes 107 degrees for the
        Buergi-Dunitz route and 90 for the Michael route, which is
        perpendicular to the alkene.
    angle_tol : float
        How far off the ideal still counts. A setting, not a constant.
    warhead : tuple or None
        The electrophile, reference and torsion atoms, if the caller
        would rather name them than have them found.

    Returns
    -------
    RichResult
        The measured geometry, each criterion separately, and whether
        the pose passes all of them.

    References
    ----------
    Buergi et al. (1973) J. Am. Chem. Soc. 95(15), 5065-5067; Bianco et
    al. (2016) Protein Science 25(1), 295-301.
    """
    if mode not in _MODES:
        raise ValueError("the mode is burgi_dunitz or michael")
    smiles = pose["smiles"]
    coords = [[float(v) for v in row] for row in pose["coords"]]
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    if len(coords) != len(el):
        raise ValueError("one coordinate triple per atom, in the order "
                         "the SMILES lists them")
    sg = [float(v) for v in cys_residue["SG"]]
    cb = cys_residue.get("CB") if hasattr(cys_residue, "get") else None
    if cb is not None:
        cb = [float(v) for v in cb]

    if warhead is None:
        warhead = find_warhead(smiles, mode)
    if warhead is None:
        return RichResult(payload={
            "passes": False,
            "reason": ("the ligand carries no %s warhead: there is no "
                       "electrophilic carbon for the cysteine to attack, "
                       "so there is no pose geometry to judge" % mode),
            "electrophile": None, "reference": None, "torsion_atom": None,
            "distance": None, "angle": None, "angle_error": None,
            "dihedral": None, "warhead_torsion": None,
            "distance_ok": False, "angle_ok": False,
            "d_min": float(d_min), "d_max": float(d_max),
            "angle_tol": float(angle_tol),
            "ideal": None, "mode": mode, "n_atoms": len(el),
            "method": "covalent near-attack geometry filter",
        })
    e, r, t = warhead
    if ideal is None:
        ideal = BURGI_DUNITZ if mode == "burgi_dunitz" else 90.0
    ideal = float(ideal)

    d = distance(sg, coords[e])
    th = angle(sg, coords[e], coords[r])
    di = None
    if cb is not None:
        di = dihedral(cb, sg, coords[e], coords[r])
    tor = dihedral(sg, coords[e], coords[r], coords[t])

    dok = d_min <= d <= d_max
    aok = abs(th - ideal) <= angle_tol
    return RichResult(payload={
        "passes": bool(dok and aok),
        "reason": "" if (dok and aok) else
                  ("the warhead is %s" % ("too far or too close"
                                          if not dok
                                          else "at the wrong angle")),
        "electrophile": e,
        "reference": r,
        "torsion_atom": t,
        "distance": d,
        "angle": th,
        "angle_error": th - ideal,
        "dihedral": di,
        "warhead_torsion": tor,
        "distance_ok": bool(dok),
        "angle_ok": bool(aok),
        "ideal": ideal,
        "d_min": float(d_min),
        "d_max": float(d_max),
        "angle_tol": float(angle_tol),
        "mode": mode,
        "n_atoms": len(el),
        "method": "covalent near-attack geometry filter",
    })


def cheatsheet():
    return ("rfppos: covalent pose filter. Sulfur-to-electrophile "
            "distance, Buergi-Dunitz or perpendicular attack angle, and "
            "the torsion; the warhead is found from the bond orders")
