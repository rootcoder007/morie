# morie.fn -- function file (rootcoder007/morie)
"""Weighted two-way table with a design-corrected test of independence."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["survey_xtab"]


def survey_xtab(x, y, weights=None):
    """Weighted cross-tabulation and a first-order corrected chi-square.

    The cell entries are estimated population counts, so the ordinary
    Pearson statistic computed on them is wrong twice over: it is on the
    population scale rather than the sample scale, and it ignores the
    variance inflation the weights carry.  Rao and Scott's first-order
    correction fixes both by evaluating Pearson's statistic on the
    weighted *proportions* and multiplying by the effective sample size
    ``n_eff = (sum w)^2 / sum w^2`` in place of ``n``.

    With equal weights ``n_eff = n`` and the statistic reduces exactly to
    the uncorrected ``chisq.test`` statistic, which is the anchor.

    Formula: ``X2 = n_eff sum_ij (p_ij - p_i. p_.j)^2 / (p_i. p_.j)``,
    referred to chi-square on ``(r-1)(c-1)`` degrees of freedom.

    Parameters
    ----------
    x, y : array-like
        Row and column category labels, equal length.
    weights : array-like, optional
        Design weights; equal weights if omitted.

    Returns
    -------
    RichResult
        ``estimate`` (corrected statistic), ``statistic_naive`` (the same
        statistic at ``n_eff = n``), ``df``, ``p_value``, ``counts``
        (weighted cell totals, row-major), ``prop``, ``rows``, ``cols``,
        ``nrow``, ``ncol``, ``neff``, ``deff``, ``n``, ``method``.

    References
    ----------
    Rao, J. N. K. & Scott, A. J. (1984).  On chi-squared tests for
    multiway contingency tables with cell proportions estimated from
    survey data.  The Annals of Statistics 12(1):46-60.
    <https://doi.org/10.1214/aos/1176346391>
    Kish, L. (1965).  Survey Sampling.  Wiley, section 8.2 (the design
    effect of unequal weighting).
    """
    xa = [_lab(v) for v in list(x)]
    ya = [_lab(v) for v in list(y)]
    n = len(xa)
    if n == 0:
        raise ValueError("survey_xtab: x is empty")
    if len(ya) != n:
        raise ValueError("survey_xtab: x and y differ in length")
    w = [1.0] * n if weights is None else C.vec(weights)
    if len(w) != n:
        raise ValueError("survey_xtab: x and weights differ in length")
    rl = sorted(set(xa))
    cl = sorted(set(ya))
    r, c = len(rl), len(cl)
    if r < 2 or c < 2:
        raise ValueError("survey_xtab: need at least two rows and two columns")
    ri = {s: i for i, s in enumerate(rl)}
    ci = {s: j for j, s in enumerate(cl)}
    cnt = [0.0] * (r * c)
    for i in range(n):
        cnt[ri[xa[i]] * c + ci[ya[i]]] += w[i]
    tot = sum(cnt)
    p = [v / tot for v in cnt]
    pr = [sum(p[i * c + j] for j in range(c)) for i in range(r)]
    pc = [sum(p[i * c + j] for i in range(r)) for j in range(c)]
    stat = 0.0
    for i in range(r):
        for j in range(c):
            e = pr[i] * pc[j]
            if e > 0.0:
                stat += (p[i * c + j] - e) ** 2 / e
    sw = sum(w)
    sw2 = sum(v * v for v in w)
    neff = sw * sw / sw2
    df = (r - 1) * (c - 1)
    X2 = neff * stat
    return RichResult(payload={
        "estimate": float(X2), "statistic_naive": float(n * stat),
        "df": int(df), "p_value": float(1.0 - C.pchisq(X2, df)),
        "counts": cnt, "prop": p, "rows": rl, "cols": cl,
        "nrow": r, "ncol": c, "neff": float(neff),
        "deff": float(n / neff), "n": n,
        "method": "Rao-Scott first-order corrected Pearson chi-square [Rao & Scott 1984]"})


def _lab(v):
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    return str(v)


# CANONICAL TEST
# >>> x = [0, 0, 0, 0, 1, 1, 1, 1]
# >>> y = [0, 0, 1, 1, 0, 1, 1, 1]
# >>> r = survey_xtab(x, y, None)
# >>> assert abs(r["estimate"] - r["statistic_naive"]) < 1e-12  # equal weights
# >>> assert r["df"] == 1 and abs(r["deff"] - 1.0) < 1e-12
# >>> # == chisq.test(table(x, y), correct = FALSE)$statistic


def cheatsheet():
    return "svytbl(x, y, weights): weighted table plus Rao-Scott corrected chi-square."

# public names resolved by fn/_lazy_map.json
surveyxtab = survey_xtab
