"""Induced-fit docking: let the receptor's side chains move while the
ligand looks for a place to sit.

Rigid-receptor docking asks the wrong question. It takes a protein
structure crystallised with one ligand, or with none, and asks which
new ligand fits that exact shape -- when the thing being modelled is a
protein that rearranges its side chains around whatever binds it. A
ligand that would fit perfectly after a leucine rotates thirty degrees
is thrown out for a clash that would not exist in reality.

Sherman and colleagues' answer is a three-stage protocol, and this
module is that protocol:

  STAGE ONE, SOFTEN AND DOCK. Shrink the van der Waals radii before
  scoring. A softened receptor tolerates the near-clashes that a real
  side chain would relieve by moving, so poses survive stage one that a
  hard receptor would have discarded before the side chains ever got a
  chance. The scale factor is the published mechanism of the stage and
  it is a parameter here.

  STAGE TWO, REFINE THE SIDE CHAINS. For each surviving pose, rotate
  the flexible residues' chi angles and keep what scores best. The
  torsion is applied by Rodrigues' rotation about the bond axis, which
  is exact and preserves every bond length in the side chain -- a
  rotamer that stretched a bond would not be a rotamer.

  STAGE THREE, RESCORE HARD. Score the refined complex at full radii.
  This is the number that is comparable across poses, and it is
  reported separately from the softened one, because a complex that
  looks good only when softened is a complex that has not been shown to
  fit.

TWO SEARCHES, BOTH HERE. ``coordinate`` optimises one residue at a time
holding the others fixed, sweeping until nothing improves -- cheap, and
it can stop at a local optimum where two side chains would have to move
together. ``exhaustive`` tries every combination, which is the true
optimum and is exponential in the number of chi angles, so it is for
small cases and for checking that the cheap search did not go wrong.
The second exists to keep the first honest and the anchors use it that
way.

WHAT IS PUBLISHED AND WHAT IS A SETTING. The protocol and the softening
are Sherman's. The chi grid defaults to the staggered positions --
minus sixty, sixty and one hundred and eighty degrees -- which are
elementary conformational analysis rather than a fitted library; a
caller holding a real backbone-dependent rotamer library passes their
own angles and gets their own search. The energy is a Lennard-Jones
12-6 with the radii the caller supplies, which is a functional form and
not a force field: it is the right shape for a steric term and it is
not claimed to be anyone's parameterisation.

References
  Sherman, W., Day, T., Jacobson, M.P., Friesner, R.A. and Farid, R.
    (2006) "Novel procedure for modeling ligand/receptor induced fit
    effects." Journal of Medicinal Chemistry 49(2), 534-553.
    doi:10.1021/jm050540c. The three-stage protocol and the softening.
  Sherman, W., Beard, H.S. and Farid, R. (2006) "Use of an induced fit
    receptor structure in virtual screening." Chemical Biology and Drug
    Design 67(1), 83-84. doi:10.1111/j.1747-0285.2005.00327.x.
  Jones, J.E. (1924) "On the determination of molecular fields. II.
    From the equation of state of a gas." Proceedings of the Royal
    Society A 106(738), 463-477. The 12-6 potential.
  Rodrigues, O. (1840) "Des lois geometriques qui regissent les
    deplacements d'un systeme solide." Journal de Mathematiques Pures
    et Appliquees 5, 380-440. The rotation about an axis.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["flexible_receptor_dock", "rotate_about_axis", "energy",
           "apply_chi", "STAGGERED", "cheatsheet"]

# The staggered torsions. Elementary conformational analysis, not a
# fitted rotamer library.
STAGGERED = (-60.0, 60.0, 180.0)
_SEARCHES = ("coordinate", "exhaustive")


def rotate_about_axis(p, a, b, degrees):
    """Rodrigues' rotation of a point about the axis from a to b.

    Exact, and it preserves every distance to the axis -- which is what
    makes a torsion a torsion rather than a distortion. A zero-length
    axis has no direction to rotate about and is refused rather than
    normalised into a division by zero.
    """
    ax = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    n = math.sqrt(_w.csum(v * v for v in ax))
    if n == 0.0:
        raise ValueError("a torsion axis needs two distinct atoms")
    k = [v / n for v in ax]
    v = [p[0] - a[0], p[1] - a[1], p[2] - a[2]]
    t = math.radians(float(degrees))
    c = math.cos(t)
    s = math.sin(t)
    kv = _w.dot(k, v)
    cr = [k[1] * v[2] - k[2] * v[1],
          k[2] * v[0] - k[0] * v[2],
          k[0] * v[1] - k[1] * v[0]]
    return [a[d] + v[d] * c + cr[d] * s + k[d] * kv * (1.0 - c)
            for d in range(3)]


def apply_chi(coords, chi, degrees):
    """Turn one chi angle: rotate the atoms it moves, and only those.

    ``chi`` names the two atoms of the bond and the atoms distal to it.
    Everything else in the receptor stays exactly where it was, which is
    what makes the side chain flexible and the backbone not.
    """
    out = [list(p) for p in coords]
    a = out[int(chi["a"])]
    b = out[int(chi["b"])]
    for i in chi["moves"]:
        out[int(i)] = rotate_about_axis(out[int(i)], a, b, degrees)
    return out


def energy(rec, lig, rec_r, lig_r, scale=1.0, epsilon=1.0, cutoff=8.0):
    """Lennard-Jones 12-6 between the ligand and the receptor.

    ``scale`` shrinks the summed radii, which is stage one's softening:
    at a scale below one a contact has to be closer before it costs
    anything, so a pose that a hard receptor would reject survives to
    the stage where the side chains can move out of its way.

    Pairs beyond the cutoff are dropped. The cutoff is stated rather
    than hidden because it makes the energy a sum over a finite
    neighbourhood, which is what lets two arms of this package agree
    term for term.
    """
    terms = []
    for i in range(len(lig)):
        for j in range(len(rec)):
            dx = lig[i][0] - rec[j][0]
            dy = lig[i][1] - rec[j][1]
            dz = lig[i][2] - rec[j][2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > cutoff * cutoff:
                continue
            r = math.sqrt(r2)
            if r == 0.0:
                raise ValueError("two atoms are on top of each other, "
                                 "which is not a pose")
            sig = (float(lig_r[i]) + float(rec_r[j])) * float(scale)
            q = sig / r
            q6 = q * q * q * q * q * q
            terms.append(4.0 * float(epsilon) * (q6 * q6 - q6))
    return _w.csum(terms)


def _grid(chis, angles):
    """Every combination of the given angles over the given chi list."""
    out = [[]]
    for _ in range(len(chis)):
        nxt = []
        for pre in out:
            for a in angles:
                nxt.append(pre + [a])
        out = nxt
    return out


def flexible_receptor_dock(receptor, ligand, flex_residues,
                           angles=None, soft=0.7, epsilon=1.0,
                           cutoff=8.0, search="coordinate", passes=3,
                           n_keep=3):
    """Dock a ligand into a receptor whose side chains may move.

    Parameters
    ----------
    receptor : mapping
        ``coords`` and ``radii``, one per receptor atom.
    ligand : mapping
        ``coords`` and ``radii`` for a single pose, or ``poses`` for a
        list of candidate poses to try.
    flex_residues : sequence
        One entry per movable chi angle: ``a`` and ``b`` name the bond
        it turns about and ``moves`` lists the atoms distal to it.
    angles : sequence or None
        The chi values to try, as offsets from the input geometry. None
        is zero -- the structure as given -- together with the
        staggered positions, so the input rotamer is always in the
        search and flexibility can never score worse than rigidity.
    soft : float
        Stage one's radius scale.
    search : {"coordinate", "exhaustive"}
        See the module docstring on why both are here.

    Returns
    -------
    RichResult
        The chosen pose, the chosen chi angles, the softened and hard
        energies at each stage, and the refined receptor.

    References
    ----------
    Sherman et al. (2006) J. Med. Chem. 49(2), 534-553.
    """
    if search not in _SEARCHES:
        raise ValueError("the search is coordinate or exhaustive")
    rc = [[float(v) for v in p] for p in receptor["coords"]]
    rr = [float(v) for v in receptor["radii"]]
    if len(rc) != len(rr):
        raise ValueError("one radius per receptor atom")
    if "poses" in ligand:
        poses = [[[float(v) for v in p] for p in pose]
                 for pose in ligand["poses"]]
    else:
        poses = [[[float(v) for v in p] for p in ligand["coords"]]]
    lr = [float(v) for v in ligand["radii"]]
    for pose in poses:
        if len(pose) != len(lr):
            raise ValueError("one radius per ligand atom")
    if not poses:
        raise ValueError("a dock with no pose has nothing to score")
    chis = [dict(c) for c in flex_residues]
    for c in chis:
        for key in ("a", "b", "moves"):
            if key not in c:
                raise ValueError("a chi needs a, b and moves")
        if int(c["a"]) == int(c["b"]):
            raise ValueError("a torsion axis needs two distinct atoms")
    if angles is None:
        angles = [0.0] + list(STAGGERED)
    angles = [float(v) for v in angles]
    if 0.0 not in angles:
        # Without it the search could not return the structure it was
        # given, and "flexibility never hurts" would stop being true.
        angles = [0.0] + angles

    # Stage one: soften and rank the poses on the receptor as supplied.
    soft_e = [energy(rc, pose, rr, lr, soft, epsilon, cutoff)
              for pose in poses]
    order = sorted(range(len(poses)), key=lambda i: (soft_e[i], i))
    kept = order[:max(1, int(n_keep))]

    def build(chosen):
        out = rc
        for k in range(len(chis)):
            if chosen[k] != 0.0:
                out = apply_chi(out, chis[k], chosen[k])
        return out

    best = None
    for pi in kept:
        pose = poses[pi]
        if not chis:
            chosen = []
            e_soft = energy(rc, pose, rr, lr, soft, epsilon, cutoff)
            e_hard = energy(rc, pose, rr, lr, 1.0, epsilon, cutoff)
            cand = (e_hard, pi, chosen, e_soft, rc)
        elif search == "exhaustive":
            bestc = None
            for combo in _grid(chis, angles):
                rc2 = build(combo)
                e = energy(rc2, pose, rr, lr, soft, epsilon, cutoff)
                if bestc is None or e < bestc[0]:
                    bestc = (e, list(combo), rc2)
            rc2 = bestc[2]
            cand = (energy(rc2, pose, rr, lr, 1.0, epsilon, cutoff),
                    pi, bestc[1], bestc[0], rc2)
        else:
            chosen = [0.0] * len(chis)
            cur = build(chosen)
            e = energy(cur, pose, rr, lr, soft, epsilon, cutoff)
            for _ in range(int(passes)):
                moved = False
                for k in range(len(chis)):
                    for a in angles:
                        trial = list(chosen)
                        trial[k] = a
                        rc2 = build(trial)
                        e2 = energy(rc2, pose, rr, lr, soft, epsilon,
                                    cutoff)
                        if e2 < e:
                            e, chosen, cur = e2, trial, rc2
                            moved = True
                if not moved:
                    break
            cand = (energy(cur, pose, rr, lr, 1.0, epsilon, cutoff),
                    pi, chosen, e, cur)
        if best is None or cand[0] < best[0]:
            best = cand

    e_hard, pi, chosen, e_soft, refined = best
    rigid_soft = energy(rc, poses[pi], rr, lr, soft, epsilon, cutoff)
    rigid_hard = energy(rc, poses[pi], rr, lr, 1.0, epsilon, cutoff)
    return RichResult(payload={
        "pose_index": pi,
        "pose": poses[pi],
        "chi": chosen,
        "receptor": refined,
        "energy": e_hard,
        "energy_soft": e_soft,
        "rigid_energy": rigid_hard,
        "rigid_energy_soft": rigid_soft,
        "gain": rigid_hard - e_hard,
        "stage1": soft_e,
        "stage1_order": order,
        "kept": kept,
        "n_pose": len(poses),
        "n_chi": len(chis),
        "n_receptor": len(rc),
        "n_ligand": len(lr),
        "soft": float(soft),
        "cutoff": float(cutoff),
        "epsilon": float(epsilon),
        "angles": angles,
        "search": search,
        "method": "induced-fit docking: soften, refine side chains, "
                  "rescore hard",
    })


def cheatsheet():
    return ("flexrd: induced-fit docking. Soften the radii and rank "
            "poses, turn the side-chain chi angles by Rodrigues "
            "rotation, rescore at full radii")
