# morie.fn -- function file (rootcoder007/morie)
"""Pairwise identity-by-descent matrix (PLINK method of moments)."""

from ._richresult import RichResult

__all__ = ["ibd_matrix", "p_ibs_given_ibd"]


def p_ibs_given_ibd(X, Y, T):
    """P(IBS = i | IBD = z) for one SNP, PLINK Table 1.

    ``X`` and ``Y`` are the sample counts of the A and a alleles,
    ``T = X + Y`` twice the number of non-missing genotypes,
    ``p = X/T``, ``q = Y/T``.  The entries incorporate the finite-
    sample ascertainment corrections of Purcell et al. (2007),
    Table 1 (p. 566); rows z = 0, 1, 2, columns i = 0, 1, 2.
    """
    X = float(X)
    Y = float(Y)
    T = float(T)
    if T < 4 or X + Y != T:
        raise ValueError("need T = X + Y >= 4")
    p = X / T
    q = Y / T
    c3 = (T / (T - 1.0)) * (T / (T - 2.0)) * (T / (T - 3.0))
    c2 = (T / (T - 1.0)) * (T / (T - 2.0))
    p00 = 2.0 * p * p * q * q * ((X - 1) / X) * ((Y - 1) / Y) * c3
    p10 = (4.0 * p ** 3 * q * ((X - 1) / X) * ((X - 2) / X)
           + 4.0 * p * q ** 3 * ((Y - 1) / Y) * ((Y - 2) / Y)) * c3
    p20 = (p ** 4 * ((X - 1) / X) * ((X - 2) / X) * ((X - 3) / X)
           + q ** 4 * ((Y - 1) / Y) * ((Y - 2) / Y) * ((Y - 3) / Y)
           + 4.0 * p * p * q * q * ((X - 1) / X) * ((Y - 1) / Y)) * c3
    p01 = 0.0
    p11 = (2.0 * p * p * q * ((X - 1) / X)
           + 2.0 * p * q * q * ((Y - 1) / Y)) * c2
    p21 = (p ** 3 * ((X - 1) / X) * ((X - 2) / X)
           + q ** 3 * ((Y - 1) / Y) * ((Y - 2) / Y)
           + p * p * q * ((X - 1) / X)
           + p * q * q * ((Y - 1) / Y)) * c2
    return [[p00, p10, p20], [p01, p11, p21], [0.0, 0.0, 1.0]]


def ibd_matrix(G):
    """Genome-wide pairwise IBD estimates, PLINK's method of moments.

    For each pair of individuals the observed counts N(I = i) of IBS
    states i = 0, 1, 2 (IBS = 2 - |g1 - g2| for 0/1/2 genotype codes)
    over their jointly non-missing SNPs are matched to expected counts
    N(I = i | Z = z) = sum_m P(I = i | Z = z) from
    :func:`p_ibs_given_ibd`, and (Purcell et al. 2007, pp. 565-566)

        P(Z=0) = N(I=0) / N(I=0|Z=0),
        P(Z=1) = (N(I=1) - P(Z=0) N(I=1|Z=0)) / N(I=1|Z=1),
        P(Z=2) = (N(I=2) - P(Z=0) N(I=2|Z=0) - P(Z=1) N(I=2|Z=1))
                 / N(I=2|Z=2).

    Bounding (their p. 566): if P(Z=0) > 1, set (1, 0, 0); if
    P(Z=0) < 0, set P(Z=0) = 0 and divide P(Z=1), P(Z=2) by their
    sum S; negative P(Z=1) or P(Z=2) are likewise set to 0 with the
    remaining two renormalized to sum 1.  Then
    pi-hat = P(Z=1)/2 + P(Z=2), and if pi-hat^2 <= P(Z=2) the
    biologically constrained transform P*(Z=0) = (1-pi)^2,
    P*(Z=1) = 2 pi (1-pi), P*(Z=2) = pi^2 replaces the estimate.

    SNPs with fewer than 2 non-missing genotypes overall (T < 4
    alleles) or monomorphic (X or Y = 0, where every table row would
    involve 0/0) are skipped for all pairs.

    NOTE on attribution: the stub docstring this module replaces cited
    Browning and Browning (2010), which is the fastIBD hidden-Markov
    segment method, NOT what its formula sketch described; the
    method-of-moments genome-wide IBD matrix implemented here is
    Purcell et al. (2007). The misattribution is recorded per the
    wave-3 contract.

    Parameters
    ----------
    G : (n, m) array-like
        Genotype matrix coded 0/1/2, individuals by SNPs; other values
        are treated as missing.

    Returns
    -------
    RichResult
        Keys ``estimate`` (pi-hat matrix, diagonal 1), ``Z0``, ``Z1``,
        ``Z2`` (pairwise IBD state probabilities, diagonal (0,0,1)),
        ``ibs_counts`` (per-pair list [(i, j, N0, N1, N2)]),
        ``n_snps_used``, ``n``, ``m``, ``method``.

    References
    ----------
    Purcell, S., Neale, B., Todd-Brown, K., et al. (2007). PLINK: a
    tool set for whole-genome association and population-based linkage
    analyses. American Journal of Human Genetics 81(3), 559-575;
    Table 1 (P(I|Z) with sample-size corrections, p. 566) and the
    method-of-moments and bounding equations pp. 565-566
    (fetched-wave3 PDF Purcell-2007-PLINK-AJHG81-559.pdf).
    """
    rows = [[float(v) for v in row] for row in G]
    n = len(rows)
    if n < 2:
        raise ValueError("need at least 2 individuals")
    m = len(rows[0])
    valid = (0.0, 1.0, 2.0)
    # per-SNP allele counts over all non-missing genotypes
    tables = [None] * m
    used = 0
    for j in range(m):
        obs = [rows[i][j] for i in range(n) if rows[i][j] in valid]
        T = 2.0 * len(obs)
        Xa = T - sum(obs)   # count of the 0-coded (A) allele
        Ya = sum(obs)
        if T < 4 or Xa == 0 or Ya == 0:
            continue
        tables[j] = p_ibs_given_ibd(Xa, Ya, T)
        used += 1
    pihat = [[1.0] * n for _ in range(n)]
    Z0 = [[0.0] * n for _ in range(n)]
    Z1 = [[0.0] * n for _ in range(n)]
    Z2 = [[1.0] * n for _ in range(n)]
    for i in range(n):
        Z2[i][i] = 1.0
    counts_out = []
    for i in range(n):
        for k in range(i + 1, n):
            Nobs = [0.0, 0.0, 0.0]
            Nexp = [[0.0, 0.0, 0.0] for _ in range(3)]
            for j in range(m):
                if tables[j] is None:
                    continue
                g1, g2 = rows[i][j], rows[k][j]
                if g1 not in valid or g2 not in valid:
                    continue
                ibs = 2 - int(abs(g1 - g2))
                Nobs[ibs] += 1.0
                for z in range(3):
                    for s in range(3):
                        Nexp[z][s] += tables[j][z][s]
            if Nexp[0][0] <= 0 or Nexp[1][1] <= 0:
                raise ValueError("no informative SNPs for a pair")
            z0 = Nobs[0] / Nexp[0][0]
            z1 = (Nobs[1] - z0 * Nexp[0][1]) / Nexp[1][1]
            z2 = (Nobs[2] - z0 * Nexp[0][2] - z1 * Nexp[1][2]) / Nexp[2][2]
            # bounding, Purcell et al. 2007 p. 566
            if z0 > 1.0:
                z0, z1, z2 = 1.0, 0.0, 0.0
            elif z0 < 0.0:
                z0 = 0.0
                s = z1 + z2
                if s > 0:
                    z1, z2 = z1 / s, z2 / s
            if z1 < 0.0:
                z1 = 0.0
                s = z0 + z2
                if s > 0:
                    z0, z2 = z0 / s, z2 / s
            if z2 < 0.0:
                z2 = 0.0
                s = z0 + z1
                if s > 0:
                    z0, z1 = z0 / s, z1 / s
            pi = 0.5 * z1 + z2
            if pi * pi <= z2:
                z0 = (1.0 - pi) ** 2
                z1 = 2.0 * pi * (1.0 - pi)
                z2 = pi * pi
            pihat[i][k] = pihat[k][i] = pi
            Z0[i][k] = Z0[k][i] = z0
            Z1[i][k] = Z1[k][i] = z1
            Z2[i][k] = Z2[k][i] = z2
            counts_out.append((i, k, Nobs[0], Nobs[1], Nobs[2]))
    return RichResult(payload={
        "estimate": pihat, "Z0": Z0, "Z1": Z1, "Z2": Z2,
        "ibs_counts": counts_out, "n_snps_used": used,
        "n": int(n), "m": int(m),
        "method": "Pairwise IBD (Purcell 2007 PLINK method of moments, Table 1)",
    })


def cheatsheet():
    return "ibdmtx: PLINK MoM pairwise IBD; pi-hat = P(Z=1)/2 + P(Z=2) (Purcell 2007 Table 1)."


# compact alias per ledger/NAMING.md
ibdmatrix = ibd_matrix
ibdmtx = ibd_matrix
