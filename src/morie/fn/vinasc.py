# morie.fn -- function file (rootcoder007/morie)
"""AutoDock Vina scoring function."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["autodock_vina_score"]

W_GAUSS1 = -0.0356
W_GAUSS2 = -0.00516
W_REPULSION = 0.840
W_HYDROPHOBIC = -0.0351
W_HBOND = -0.587
W_ROT = 0.0585
CUTOFF = 8.0


def autodock_vina_score(receptor, ligand_pose, n_rot=0):
    """Score a docked pose with the Vina free-energy function.

    Vina scores a pose as a sum over heavy-atom pairs of five terms, all
    of them functions of the *surface* distance ``d = r - R_i - R_j``
    rather than the centre distance, which is what lets one set of
    weights cover atoms of very different size.  The rotatable-bond
    term is not a sixth pair term: it divides, penalising the entropy a
    flexible ligand gives up on binding.

    Parameters
    ----------
    receptor : array-like, shape (n, 5)
        Rows ``[x, y, z, radius, type]``.  ``type`` is 1 for a
        hydrophobic atom, 2 for a hydrogen-bond donor or acceptor,
        0 otherwise.
    ligand_pose : array-like, shape (m, 5)
        Same layout, for the ligand pose being scored.
    n_rot : int, default 0
        Active rotatable bonds between heavy atoms in the ligand.

    Returns
    -------
    RichResult
        ``estimate`` (the score, kcal/mol), ``c_inter``, and the five
        unweighted term sums ``gauss1``, ``gauss2``, ``repulsion``,
        ``hydrophobic``, ``hbond``.

    Notes
    -----
    Pairs beyond ``r = 8`` angstrom contribute nothing, as in the paper.
    Hydrophobic contributions are counted only for hydrophobic-
    hydrophobic pairs and hydrogen bonding only for donor-acceptor
    pairs, which is the paper own typing rule.

    References
    ----------
    Trott, O. & Olson, A. J. (2010).  AutoDock Vina: improving the speed
    and accuracy of docking with a new scoring function, efficient
    optimization, and multithreading.  Journal of Computational
    Chemistry 31:455-461.  Open access; fetched.  The five term forms
    and their weights are table 1 of that paper, and equation (9) is
    ``g(c_inter) = c_inter / (1 + w N_rot)`` -- a quotient, not a sum.
    """
    R = C.mat(receptor)
    L = C.mat(ligand_pose)
    g1 = g2 = rep = hyd = hb = 0.0
    for a in R:
        for b in L:
            r = math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))
            if r > CUTOFF:
                continue
            d = r - a[3] - b[3]
            g1 += math.exp(-((d / 0.5) ** 2))
            g2 += math.exp(-(((d - 3.0) / 2.0) ** 2))
            if d < 0.0:
                rep += d * d
            if len(a) > 4 and len(b) > 4 and a[4] == 1.0 and b[4] == 1.0:
                if d < 0.5:
                    hyd += 1.0
                elif d < 1.5:
                    hyd += 1.5 - d
            if len(a) > 4 and len(b) > 4 and a[4] == 2.0 and b[4] == 2.0:
                if d < -0.7:
                    hb += 1.0
                elif d < 0.0:
                    hb += -d / 0.7
    c_inter = (W_GAUSS1 * g1 + W_GAUSS2 * g2 + W_REPULSION * rep
               + W_HYDROPHOBIC * hyd + W_HBOND * hb)
    score = c_inter / (1.0 + W_ROT * float(n_rot))
    return RichResult(payload={
        "estimate": score, "c_inter": c_inter, "gauss1": g1, "gauss2": g2,
        "repulsion": rep, "hydrophobic": hyd, "hbond": hb,
        "method": "AutoDock Vina scoring function"})


def cheatsheet():
    return "vinasc: AutoDock Vina scoring function."
