# morie.fn -- function file (rootcoder007/morie)
"""SNP-BLUP additive genomic prediction."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["snp_blup"]


def snp_blup(y, M, lam=None, h2=None, freq=None):
    """SNP-BLUP: additive marker effects as random, GEBV = Z u_hat.

    Model (Meuwissen, Hayes and Goddard 2001, "BLUP estimation",
    p. 1822): y = 1n mu + Z u + e with u ~ N(0, sigma_u^2 I) and
    e ~ N(0, sigma_e^2 I), estimated from Henderson's mixed-model
    equations (Henderson 1975; the MME as printed in Montesinos-Lopez
    et al. 2022, ch. 5.2, with R = sigma_e^2 I and D = sigma_u^2 I):

        [ n     1' Z        ] [mu]   [ 1' y ]
        [ Z' 1  Z'Z + lam I ] [u ] = [ Z' y ],   lam = sigma_e^2 / sigma_u^2.

    Z is the VanRaden-centred marker matrix, Z_ij = M_ij - 2 p_j
    (VanRaden 2008 Method 1 centring as reproduced in
    Montesinos-Lopez et al. 2022 sec. 2.4; the same centring as
    morie.fn.vanr1).  Genomic estimated breeding values are
    GEBV = Z u_hat.

    If ``h2`` is given instead of ``lam``, the ratio is derived from
    the VanRaden variance split sigma_g^2 = 2 sum_j p_j (1 - p_j)
    sigma_u^2 (VanRaden 2008; Montesinos-Lopez et al. 2022 sec. 2.4),
    so with total variance sigma_P^2 = var(y):

        lam = sigma_e^2 / sigma_u^2
            = ((1 - h2) / h2) * 2 sum_j p_j (1 - p_j).

    Parameters
    ----------
    y : (n,) array-like
        Phenotypes.
    M : (n, m) array-like
        Genotype matrix coded 0/1/2 (individuals by markers).
    lam : float, optional
        Shrinkage ratio sigma_e^2 / sigma_u^2.  Exactly one of
        ``lam`` and ``h2`` must be given.
    h2 : float, optional
        Heritability used to derive ``lam`` as above.
    freq : (m,) array-like, optional
        Allele frequencies p_j; column means over 2 by default.

    Returns
    -------
    RichResult
        Keys ``estimate`` (GEBV vector), ``u`` (marker effects),
        ``mu``, ``lam``, ``freq``, ``n``, ``m``, ``method``.

    References
    ----------
    Meuwissen, T. H. E., Hayes, B. J. and Goddard, M. E. (2001).
    Prediction of total genetic value using genome-wide dense marker
    maps. Genetics 157(4), 1819-1829, sec. "BLUP estimation" p. 1822
    (fetched-wave3 PDF Meuwissen-Hayes-Goddard-2001).
    Henderson, C. R. (1975). Best linear unbiased estimation and
    prediction under a selection model. Biometrics 31(2), 423-447.
    VanRaden, P. M. (2008). Efficient methods to compute genomic
    predictions. Journal of Dairy Science 91(11), 4414-4423 (centring
    and variance split; read from Montesinos-Lopez et al. 2022,
    Multivariate Statistical Machine Learning Methods for Genomic
    Prediction, Springer, sec. 2.4 pp. 50-52 and ch. 5.2 pp. 146-148,
    local PDFs "Multivariate Statistical Machine Learnin [Pages
    35-70]" and "[Pages 141-170]").
    """
    y = np.asarray(y, dtype=float)
    Mm = np.asarray(M, dtype=float)
    if Mm.ndim == 1:
        Mm = Mm.reshape((-1, 1))
    n, m = Mm.shape
    if len(y) != n:
        raise ValueError("y and M row count differ")
    if (lam is None) == (h2 is None):
        raise ValueError("give exactly one of lam or h2")
    if freq is not None:
        p = np.asarray(freq, dtype=float)
    else:
        p = np.sum(Mm, axis=0) / (2.0 * n)
    sum2pq = 2.0 * float(np.sum(p * (1.0 - p)))
    if lam is None:
        h2 = float(h2)
        if not (0.0 < h2 < 1.0):
            raise ValueError("h2 must be in (0, 1)")
        lam = (1.0 - h2) / h2 * sum2pq
    lam = float(lam)
    if lam <= 0:
        raise ValueError("lam must be positive")
    Z = Mm - 2.0 * p
    # Henderson MME, coefficient matrix (m+1) x (m+1)
    C = np.zeros((m + 1, m + 1))
    C[0, 0] = float(n)
    zsum = np.sum(Z, axis=0)
    C[0, 1:] = zsum
    C[1:, 0] = zsum
    C[1:, 1:] = Z.T @ Z + lam * np.eye(m)
    rhs = np.zeros(m + 1)
    rhs[0] = float(np.sum(y))
    rhs[1:] = Z.T @ y
    sol = np.linalg.solve(C, rhs)
    mu = float(sol[0])
    u = sol[1:]
    gebv = Z @ u
    return RichResult(payload={
        "estimate": gebv, "u": u, "mu": mu, "lam": lam,
        "sum2pq": sum2pq, "freq": p, "n": int(n), "m": int(m),
        "method": "SNP-BLUP (Meuwissen 2001 BLUP; Henderson MME; VanRaden centring)",
    })


def cheatsheet():
    return "snpblr: SNP-BLUP MME [n 1'Z; Z'1 Z'Z+lam I]; GEBV = Z u_hat; lam from h2 via 2*sum pq."


# compact alias per ledger/NAMING.md
snpblup = snp_blup
snpblr = snp_blup
