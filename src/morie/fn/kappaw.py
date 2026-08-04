# morie.fn -- function file (rootcoder007/morie)
"""Weighted kappa.

Source: Cohen, J. (1968), "Weighted kappa: nominal scale agreement with
provision for scaled disagreement or partial credit", *Psychological
Bulletin* 70(4):213-220.  The 1968 article is paywalled and was NOT read
directly; the estimator was checked against the standard published
statement of it:

    kappa_w = 1 - sum_ij W_ij O_ij / sum_ij W_ij E_ij

where W is a matrix of DISAGREEMENT weights with a zero diagonal, O_ij
are the observed cell proportions and E_ij = p_i. p_.j are the expected
proportions from the product of the marginals.  The two conventional
weight choices, W_ij = |i - j| (linear) and W_ij = (i - j)^2
(quadratic), can be requested by name.

No standard error is returned.  The large-sample variance of weighted
kappa is due to Fleiss, Cohen & Everitt (1969), which could NOT be
obtained; rather than ship an unverified variance formula this module
reports only the point estimate and its two components.
"""

from ._richresult import RichResult

__all__ = ["weighted_kappa"]


def _levels(a, b):
    seen = []
    for v in list(a) + list(b):
        if v not in seen:
            seen.append(v)
    seen.sort()
    return seen


def _weight_matrix(weights, k):
    if isinstance(weights, str):
        name = weights.lower()
        if name.startswith("lin"):
            return [[float(abs(i - j)) for j in range(k)] for i in range(k)]
        if name.startswith("quad") or name.startswith("sq"):
            return [[float((i - j) ** 2) for j in range(k)] for i in range(k)]
        raise ValueError("weights must be 'linear', 'quadratic' or a k x k matrix")
    w = [[float(v) for v in row] for row in weights]
    if len(w) != k or any(len(row) != k for row in w):
        raise ValueError("weight matrix must be k x k over the pooled category set")
    return w


def weighted_kappa(rater1, rater2, weights="linear"):
    """Weighted kappa with a caller-supplied disagreement weight matrix.

    Parameters
    ----------
    rater1, rater2 : sequence
        Ordered-category ratings of the same n subjects.
    weights : {'linear', 'quadratic'} or k x k array-like
        Disagreement weights over the pooled, sorted category set.

    Returns
    -------
    RichResult
        ``kappa``, ``observed_disagreement``, ``expected_disagreement``,
        ``n``, ``n_categories``.
    """
    r1 = list(rater1)
    r2 = list(rater2)
    n = len(r1)
    if n == 0 or len(r2) != n:
        raise ValueError("rater1 and rater2 must be non-empty and of equal length")
    lv = _levels(r1, r2)
    k = len(lv)
    pos = {v: i for i, v in enumerate(lv)}
    w = _weight_matrix(weights, k)
    p = [[0.0] * k for _ in range(k)]
    for a, b in zip(r1, r2):
        p[pos[a]][pos[b]] += 1.0 / n
    row = [sum(p[i]) for i in range(k)]
    col = [sum(p[i][j] for i in range(k)) for j in range(k)]
    qo = sum(w[i][j] * p[i][j] for i in range(k) for j in range(k))
    qe = sum(w[i][j] * row[i] * col[j] for i in range(k) for j in range(k))
    if qe <= 0.0:
        raise ValueError("expected disagreement is zero; kappa_w is undefined")
    kap = 1.0 - qo / qe
    return RichResult(payload={
        "kappa": float(kap), "observed_disagreement": float(qo),
        "expected_disagreement": float(qe),
        "n": n, "n_categories": k,
        "method": "Cohen (1968) weighted kappa, 1 - sum(W O)/sum(W E)"})


def cheatsheet():
    return "kappaw: Cohen (1968) weighted kappa"


# compact alias per ledger/NAMING.md
kappawt = weighted_kappa
weightedkappa = weighted_kappa
