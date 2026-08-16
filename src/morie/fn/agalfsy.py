# morie.fn -- function file (rootcoder007/morie)
r"""Reinforcement-learning search for a ligand pose in a rigid receptor.

Docking as a Markov decision process. The ligand starts somewhere in an
18 angstrom box and the agent nudges it -- six translations of 0.1 A along
the axes, six rotations of one degree about them -- until a critic says the
pose has settled. Wang et al. train an A3C actor and critic on gridded
receptor and ligand channels; the network is the part that has to be
learned, but the ENVIRONMENT is fully specified arithmetic, and that is
what this module is: the state, the twelve actions, the reward, and the
termination rule, exactly as published.

The reward is a difference of exponentials in the distance to the reference
site,

.. math:: r_t = e^{-d(s^*, s_{t+1})/18} - e^{-d(s^*, s_t)/18},

so it is positive exactly when the step moved the ligand closer, and it is
bounded, which is why the scale 18 matches the box. A negative reward is
doubled before it is returned: overshooting is punished harder than
approaching is rewarded, which is what stops the agent orbiting the site.

The policy is a parameter, not a fixture. Pass ``policy`` to drive the
search with a trained network; leave it out and the built-in greedy policy
takes the action with the largest immediate reward, which needs the
reference site and is therefore a benchmark oracle rather than a predictor.
That distinction is reported in ``policy_kind`` rather than buried, because
an oracle-driven RMSD is not a docking result.

A caution that belongs with the receptor, not the search: when the rigid
receptor comes from AlphaFold rather than crystallography, docking-based
virtual screening degrades sharply. Scardino et al. measured a mean
enrichment factor at 1% of 8.8 on AlphaFold models against 20.5 on the
experimental structures, with several targets enriching not at all. The
pose search can be exactly right and the screen still fail, because the
side chains were never induced to fit.

References
----------
Wang, C., Chen, Y., Zhang, Y., Li, K., Lin, M., Pan, F., Wu, W. and
Zhang, J. (2022) "A reinforcement learning approach for protein-ligand
binding pose prediction", *BMC Bioinformatics* **23**, 368,
doi:10.1186/s12859-022-04912-7. The MDP implemented here: section
"Methods", the twelve-action space, the reward above, and the critic
stabilisation stopping rule (minimum 300 steps, window 50, range 0.3,
cap 600).

Scardino, V., Di Filippo, J. I. and Cavasotto, C. N. (2022) "How good are
AlphaFold models for docking-based virtual screening?", *iScience*
**26**(1), 105920, doi:10.1016/j.isci.2022.105920. The enrichment
figures quoted above.

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger,
O., Tunyasuvunakool, K. et al. (2021) "Highly accurate protein structure
prediction with AlphaFold", *Nature* **596**(7873), 583-589,
doi:10.1038/s41586-021-03819-3. The source of the predicted receptor.

Notes
-----
The ledger recorded this module's source as "hypothetical synthesis
(#158 + #164)", which named no paper and pointed at two unrelated ledger
rows (a functional confidence band and the Canadian Fire Weather Index).
It has been replaced with the work above, which specifies the method
actually implemented here.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["rl_pose_search"]

_EPS = 1e-12
_BOX = 18.0            # angstroms; the paper's box, and the reward scale
_TRANS = 0.1           # angstrom per translational step
_ROT = 1.0             # degree per rotational step
_MIN_STEPS = 300
_WINDOW = 50
_RANGE = 0.3
_MAX_STEPS = 600


def _coords(x, what):
    """Coerce to a list of 3-vectors."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    for row in x:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("%s: each atom needs exactly x, y, z" % what)
        out.append([float(v) for v in row])
    if not out:
        raise ValueError("%s: no atoms" % what)
    return out


def _centroid(P):
    n = len(P)
    return [sum(p[a] for p in P) / n for a in range(3)]


def _rmsd(A, B):
    r"""Pose distance. Same atom order, no superposition -- the agent is
    moving the ligand rigidly, so the correspondence is fixed and fitting
    it away would hide exactly the displacement being rewarded."""
    n = len(A)
    s = 0.0
    for i in range(n):
        for a in range(3):
            d = A[i][a] - B[i][a]
            s += d * d
    return math.sqrt(s / n)


def _rotate(P, axis, deg):
    """Rotate rigidly about the pose centroid, right-handed."""
    c = _centroid(P)
    t = math.radians(deg)
    ct, st = math.cos(t), math.sin(t)
    out = []
    for p in P:
        x, y, z = p[0] - c[0], p[1] - c[1], p[2] - c[2]
        if axis == 0:
            y, z = ct * y - st * z, st * y + ct * z
        elif axis == 1:
            x, z = ct * x + st * z, -st * x + ct * z
        else:
            x, y = ct * x - st * y, st * x + ct * y
        out.append([x + c[0], y + c[1], z + c[2]])
    return out


def _translate(P, axis, step):
    out = []
    for p in P:
        q = list(p)
        q[axis] += step
        out.append(q)
    return out


def _apply(P, a):
    """The twelve actions: 0-5 translations, 6-11 rotations."""
    if a < 6:
        return _translate(P, a // 2, _TRANS if a % 2 == 0 else -_TRANS)
    a -= 6
    return _rotate(P, a // 2, _ROT if a % 2 == 0 else -_ROT)


def _reward(site, before, after):
    r"""Equation of the paper: a difference of exponentials, negatives
    doubled. Positive exactly when the step reduced the distance."""
    r = (math.exp(-_rmsd(site, after) / _BOX)
         - math.exp(-_rmsd(site, before) / _BOX))
    return 2.0 * r if r < 0.0 else r


def rl_pose_search(receptor, ligand, site=None, policy=None, critic=None,
                   max_steps=_MAX_STEPS, min_steps=_MIN_STEPS,
                   window=_WINDOW, tol=_RANGE, box=_BOX, seed=2):
    r"""Search a ligand pose by the Wang et al. (2022) docking MDP.

    Parameters
    ----------
    receptor : array-like, shape (m, 3)
        Receptor atom coordinates, held RIGID. If these came from
        AlphaFold rather than an experiment, read ``note``.
    ligand : array-like, shape (n, 3)
        Starting ligand pose.
    site : array-like, shape (n, 3), optional
        Reference pose defining the true site. Required for the reward,
        and therefore for the built-in greedy policy; a trained ``policy``
        does not need it and then the search runs blind, as at inference.
    policy : callable, optional
        ``policy(ligand, receptor, step) -> int`` in 0..11. Without it the
        greedy oracle is used and ``policy_kind`` says so.
    critic : callable, optional
        ``critic(ligand, receptor, step) -> float``. Its running range over
        the last ``window`` steps is what stops the search. Without it the
        distance to the site stands in, which is again an oracle.

    Returns
    -------
    RichResult
        ``pose`` (final), ``rmsd``, ``dcc`` (centroid separation),
        ``success`` (DCC < 4 A, the paper's criterion), ``improved``,
        ``steps``, ``stop_reason``, ``reward_total``, and the trajectory.
    """
    R = _coords(receptor, "agalfsy receptor")
    L0 = _coords(ligand, "agalfsy ligand")
    S = _coords(site, "agalfsy site") if site is not None else None
    if S is not None and len(S) != len(L0):
        raise ValueError("agalfsy: the site pose has %d atoms and the ligand "
                         "%d -- the reward is an RMSD over matched atoms"
                         % (len(S), len(L0)))
    if policy is None and S is None:
        raise ValueError(
            "agalfsy: with no policy the search falls back to the greedy "
            "oracle, which maximises the published reward and therefore "
            "needs `site`. Supply a trained policy to run blind.")
    box = float(box)
    if not box > 0.0:
        raise ValueError("agalfsy: box must be positive")

    kind = ("supplied policy" if policy is not None
            else "greedy oracle (needs the answer; benchmark only)")
    start_c = _centroid(L0)
    L = [list(p) for p in L0]
    traj = []
    crit_hist = []
    total = 0.0
    stop = "max_steps"
    step = 0
    for step in range(1, int(max_steps) + 1):
        if policy is not None:
            a = int(policy(L, R, step))
            if not 0 <= a < 12:
                raise ValueError("agalfsy: policy returned action %d, the "
                                 "action space is 0..11" % a)
            nxt = _apply(L, a)
        else:
            best_a, best_r, nxt = 0, None, None
            for cand in range(12):
                trial = _apply(L, cand)
                r = _reward(S, L, trial)
                if best_r is None or r > best_r:
                    best_a, best_r, nxt = cand, r, trial
            a = best_a
        r = _reward(S, L, nxt) if S is not None else 0.0
        total += r
        L = nxt
        traj.append([step, a, r])

        # box exit: the paper terminates when the ligand leaves the box
        c = _centroid(L)
        if max(abs(c[i] - start_c[i]) for i in range(3)) > box / 2.0:
            stop = "left_box"
            break

        cv = (float(critic(L, R, step)) if critic is not None
              else (_rmsd(S, L) if S is not None else 0.0))
        crit_hist.append(cv)
        if step >= int(min_steps) and len(crit_hist) >= int(window):
            w = crit_hist[-int(window):]
            if (max(w) - min(w)) < float(tol):
                stop = "critic_stabilised"
                break

    final_rmsd = _rmsd(S, L) if S is not None else float("nan")
    start_rmsd = _rmsd(S, L0) if S is not None else float("nan")
    dcc = (math.sqrt(sum((_centroid(L)[i] - _centroid(S)[i]) ** 2
                         for i in range(3))) if S is not None
           else float("nan"))
    return RichResult(payload={
        "estimate": final_rmsd,
        "pose": L,
        "rmsd": final_rmsd,
        "rmsd_start": start_rmsd,
        "dcc": dcc,
        "success": bool(dcc < 4.0) if S is not None else None,
        "improved": (bool(final_rmsd < start_rmsd) if S is not None
                     else None),
        "steps": step,
        "stop_reason": stop,
        "reward_total": total,
        "trajectory": traj,
        "policy_kind": kind,
        "n_actions": 12,
        "translation_step": _TRANS,
        "rotation_step_deg": _ROT,
        "box": box,
        "method": ("A3C docking MDP of Wang et al. (2022): 12 discrete "
                   "actions (six 0.1 A translations, six 1 degree "
                   "rotations), reward exp(-d/18) differenced and negatives "
                   "doubled, stopping when the critic range falls below "
                   "0.3 over 50 steps after at least 300"),
        "note": ("policy_kind is the first thing to read. The greedy "
                 "fallback maximises the published reward, which is a "
                 "function of the true site, so its rmsd is an upper bound "
                 "on what a trained agent could do and not a docking "
                 "prediction. Separately, a rigid AlphaFold receptor is a "
                 "weak basis for screening even when the pose search is "
                 "exact: Scardino et al. (2022) report a mean enrichment "
                 "factor at 1% of 8.8 against 20.5 for experimental "
                 "structures, several targets enriching not at all."),
    })


def cheatsheet():
    return ("agalfsy: rl_pose_search(receptor, ligand, site) -> ligand pose "
            "by the A3C docking MDP of Wang et al. (2022), BMC Bioinf 23:368")


# compact alias per ledger/NAMING.md
alphazero_alphafold_synergy = rl_pose_search
