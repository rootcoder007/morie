"""Felsenstein pruning likelihood for trees (Felsenstein 1981)."""

import math

from ._richresult import RichResult

__all__ = ["felsen", "felsenstein_pruning"]

_BASES = "ACGT"


def _pij(t, pi):
    # F81 substitution probabilities, Felsenstein (1981) Eq. 7:
    # P_ij(t) = e^{-ut} delta_ij + (1 - e^{-ut}) pi_j   (u = 1)
    e = math.exp(-t)
    return [[e * (1.0 if i == j else 0.0) + (1.0 - e) * pi[j]
             for j in range(4)] for i in range(4)]


def _prune(node, site, pi):
    # returns the four conditional likelihoods L_s(node)
    if isinstance(node, str):
        s = _BASES.index(site[node])
        return [1.0 if k == s else 0.0 for k in range(4)]
    out = [1.0, 1.0, 1.0, 1.0]
    for child, t in node:
        Lc = _prune(child, site, pi)
        P = _pij(float(t), pi)
        for s in range(4):
            out[s] *= sum(P[s][j] * Lc[j] for j in range(4))
    return out


def felsen(tree, sites, pi=None):
    """
    Maximum-likelihood tree likelihood by the pruning algorithm.

    Felsenstein (1981): conditional likelihoods L_s(k) -- the
    likelihood of the data at or above point k given state s -- are
    1/0 indicators at the tips and combine by postorder traversal as
    L_s(k) = prod_children [ sum_j P_sj(v_child) L_j(child) ]; the
    site likelihood is L = sum_s pi_s L_s(root) (his Eq. 5), and
    sites multiply.  Substitution follows his Markov model (Eqs.
    6-7): P_ij(t) = e^{-ut} delta_ij + (1 - e^{-ut}) pi_j, the F81
    process whose stationary distribution is pi.  Because L is a
    probability distribution over site patterns, the likelihoods of
    all 4^m patterns sum exactly to one -- an identity verified in
    the tests.

    Sources
    -------
    Felsenstein, J. (1981). Evolutionary trees from DNA sequences:
    a maximum likelihood approach. *Journal of Molecular
    Evolution*, 17(6), 368-376, Eqs. 4-7 and the "pruning"
    algorithm (local copy fetched-wave3/Evolutionary trees from DNA
    sequences- A maximum likelihood approach.pdf).

    Parameters
    ----------
    tree : nested structure
        Leaf = taxon name (str); internal node = list of
        (child, branch_length) pairs.
    sites : sequence of dict
        Per site, {taxon: base} with bases in "ACGT".
    pi : sequence of 4 floats, optional
        Stationary base frequencies (default uniform 1/4).

    Returns
    -------
    RichResult
        Keys: loglik, site_likelihoods, n_sites.
    """
    if pi is None:
        pi = [0.25] * 4
    pi = [float(v) for v in pi]
    if abs(sum(pi) - 1.0) > 1e-9 or any(v <= 0 for v in pi):
        raise ValueError("pi must be positive and sum to 1")
    if not sites:
        raise ValueError("need at least one site")
    liks = []
    ll = 0.0
    for site in sites:
        L0 = _prune(tree, site, pi)
        L = sum(pi[s] * L0[s] for s in range(4))
        if L <= 0:
            raise ValueError("zero likelihood site (bad pattern?)")
        liks.append(L)
        ll += math.log(L)
    return RichResult(payload={
        "loglik": ll,
        "site_likelihoods": liks,
        "n_sites": len(sites),
        "pi": pi,
        "method": "Felsenstein (1981) pruning, F81 model (Eqs. 5-7)",
    })


# long descriptive alias (stub-era name)
felsenstein_pruning = felsen


def cheatsheet():
    return "felsen: tips 1/0; L_s = prod_c sum_j P_sj(v) L_j(c); L = sum pi L_root"
