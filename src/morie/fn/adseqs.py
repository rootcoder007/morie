# morie.fn -- function file (rootcoder007/morie)
"""EM for the ADMIXTURE ancestry likelihood."""

import math

from ._richresult import RichResult

__all__ = ["admixq", "admixture_seq"]


def _grid(G):
    return [[float(v) for v in row] for row in G]


def admixq(G, K=2, steps=50, Q0=None, P0=None):
    """Ancestry proportions Q and allele frequencies P by EM.

    For I unrelated individuals typed at J biallelic markers, g_ij in
    {0,1,2} counts copies of allele 1.  Population k contributes fraction
    q_ik of individual i's genome and carries allele-1 frequency p_kj.
    Under linkage equilibrium and Hardy-Weinberg within source
    populations the log-likelihood is, up to an additive constant,

        L(Q,P) = sum_i sum_j [ g_ij log(sum_k q_ik p_kj)
                             + (2 - g_ij) log(sum_k q_ik (1 - p_kj)) ]

    The EM updates that maximise it split each observed allele between
    the K sources by its posterior responsibility,

        a_ijk = q_ik p_kj / sum_l q_il p_lj
        b_ijk = q_ik (1 - p_kj) / sum_l q_il (1 - p_lj)
        q_ik <- (1/(2J)) sum_j [ g_ij a_ijk + (2 - g_ij) b_ijk ]
        p_kj <- sum_i g_ij a_ijk
                / sum_i [ g_ij a_ijk + (2 - g_ij) b_ijk ]

    Parameters
    ----------
    G : array-like, shape (I, J)
        Genotype counts in {0, 1, 2}.
    K : int
        Number of ancestral populations.
    steps : int
        Fixed EM iteration count; no tolerance early exit.
    Q0, P0 : array-like or None
        Starting values.  ``None`` uses the deterministic starts
        q_ik proportional to 1 + ((i + k) mod K) and
        p_kj = (2 + ((k * J + j) mod 7)) / 10, identical in every arm.

    Returns
    -------
    RichResult
        ``Q``, ``P``, ``loglik``, ``loglik0``, ``I``, ``J``, ``K``,
        ``steps``.

    References
    ----------
    Alexander, D. H., Novembre, J. and Lange, K. (2009), "Fast model-based
    estimation of ancestry in unrelated individuals", Genome Research
    19(9), 1655-1664.  The likelihood above is their Equation (2) (the
    model of structure, Pritchard et al. 2000), read from the open-access
    PMC rendering of the article.  ADMIXTURE itself maximises that
    likelihood by block relaxation with sequential quadratic programming;
    the routine here uses the EM algorithm for the same likelihood, which
    the paper describes as the alternative used by FRAPPE (Tang et al.
    2005).  The objective, not the search, is what is shared.
    """
    Gm = _grid(G)
    I = len(Gm)
    J = len(Gm[0]) if I else 0
    K = int(K)
    steps = int(steps)
    if I == 0 or J == 0:
        raise ValueError("G must be non-empty")
    if K < 1:
        raise ValueError("K must be at least 1")
    if any(len(r) != J for r in Gm):
        raise ValueError("G must be rectangular")
    if any(v < 0.0 or v > 2.0 for r in Gm for v in r):
        raise ValueError("genotype counts must lie in [0, 2]")
    if Q0 is None:
        Q = []
        for i in range(I):
            row = [1.0 + ((i + k) % K) for k in range(K)]
            s = sum(row)
            Q.append([v / s for v in row])
    else:
        Q = [[float(v) for v in r] for r in Q0]
    if P0 is None:
        P = [[(2.0 + ((k * J + j) % 7)) / 10.0 for j in range(J)]
             for k in range(K)]
    else:
        P = [[float(v) for v in r] for r in P0]

    def loglik(Q, P):
        tot = 0.0
        for i in range(I):
            for j in range(J):
                a = sum(Q[i][k] * P[k][j] for k in range(K))
                b = sum(Q[i][k] * (1.0 - P[k][j]) for k in range(K))
                g = Gm[i][j]
                if g > 0.0:
                    tot += g * math.log(a)
                if 2.0 - g > 0.0:
                    tot += (2.0 - g) * math.log(b)
        return tot

    ll0 = loglik(Q, P)
    for _ in range(steps):
        Qn = [[0.0] * K for _ in range(I)]
        num = [[0.0] * J for _ in range(K)]
        den = [[0.0] * J for _ in range(K)]
        for i in range(I):
            for j in range(J):
                g = Gm[i][j]
                sa = sum(Q[i][k] * P[k][j] for k in range(K))
                sb = sum(Q[i][k] * (1.0 - P[k][j]) for k in range(K))
                for k in range(K):
                    a = 0.0 if sa == 0.0 else Q[i][k] * P[k][j] / sa
                    b = 0.0 if sb == 0.0 else Q[i][k] * (1.0 - P[k][j]) / sb
                    ca = g * a
                    cb = (2.0 - g) * b
                    Qn[i][k] += ca + cb
                    num[k][j] += ca
                    den[k][j] += ca + cb
        Q = [[v / (2.0 * J) for v in row] for row in Qn]
        P = [[(0.5 if den[k][j] == 0.0 else num[k][j] / den[k][j])
              for j in range(J)] for k in range(K)]
    return RichResult(payload={
        "Q": Q, "P": P, "loglik": loglik(Q, P), "loglik0": ll0,
        "I": I, "J": J, "K": K, "steps": steps,
        "method": "EM for the ADMIXTURE likelihood (Alexander et al. 2009 eq. 2)"})


admixture_seq = admixq


admixtureseq = admixq


def cheatsheet():
    return "adseqs: EM for the ADMIXTURE ancestry likelihood."
