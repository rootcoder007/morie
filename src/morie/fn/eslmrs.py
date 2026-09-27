"""Multivariate adaptive regression splines (ESL sec 9.4)."""

import math

from ._richresult import RichResult
from .linsys import _householder_ls

__all__ = ["esl_mars"]


def _hinge(x, t, sign):
    return max(x - t, 0.0) if sign > 0 else max(t - x, 0.0)


def _eval_term(term, row):
    v = 1.0
    for j, t, s in term:
        v *= _hinge(row[j], t, s)
    return v


def esl_mars(X, y, max_terms=11, degree=1, penalty=None, thresh=1e-8, query=None):
    r"""MARS: forward selection of reflected hinge pairs, backward deletion by GCV.

    Forward pass (ESL sec 9.4): starting from the constant, repeatedly add the
    pair :math:`B_m(x)(x_j - t)_+`, :math:`B_m(x)(t - x_j)_+` (t an observed value
    of :math:`x_j`, :math:`B_m` an existing term not already using :math:`x_j`, with at
    most ``degree`` factors) that most reduces the residual sum of squares,
    until ``max_terms`` terms or the relative gain falls below ``thresh``.
    Backward pass: delete, one at a time, the term whose removal increases
    RSS least, and keep the model minimising eq 9.20,
    :math:`GCV = RSS/(N(1 - M/N)^2)` with :math:`M = r + c(r-1)/2` (r terms, c = 2
    for an additive model and 3 otherwise; the RSS/N scaling does not
    change the minimiser). Candidates are scored by orthogonalising
    against the current basis; zero or collinear hinges are not added (once
    a variable has a reflected pair, a second pair on it spans only one new
    direction, because (t - x)_+ = (t - x) + (x - t)_+). The ``earth`` package
    follows the same criterion with extra speed and span heuristics, so its
    forward pass can pick different knots.

    Parameters
    ----------
    X : N x p nested sequence
    y : sequence of N floats
    max_terms : int
        Maximum terms in the forward pass (including the constant).
    degree : int
        Maximum interaction order.
    penalty : float, optional
        GCV knot cost c (default 2 if degree == 1 else 3).
    thresh : float
        Forward-pass stopping threshold on the relative RSS decrease.
    query : m x p nested sequence, optional

    Returns
    -------
    RichResult
        ``terms`` (selected; each a list of (variable, knot, sign)),
        ``coefficients`` (intercept first), ``gcv``, ``rss``, ``fitted``,
        ``prediction``, ``forward_terms``.

    References
    ----------
    Friedman, J. H. (1991). Multivariate adaptive regression splines. Annals
    of Statistics 19, 1-67.
    """
    rows = [[float(v) for v in r] for r in X]
    yy = [float(v) for v in y]
    N, p = len(rows), len(rows[0])
    if len(yy) != N or max_terms < 1 or degree < 1:
        raise ValueError("need matching X and y, max_terms >= 1 and degree >= 1")
    c = (2.0 if degree == 1 else 3.0) if penalty is None else float(penalty)
    terms = [[]]
    cols = [[1.0] * N]
    Q = [[1 / math.sqrt(N)] * N]
    ybar = sum(yy) / N
    e = [v - ybar for v in yy]
    rss = sum(v * v for v in e)
    rss0 = rss

    def ortho(v):
        v = v[:]
        for q in Q:
            d = sum(a * b for a, b in zip(q, v))
            v = [a - d * b for a, b in zip(v, q)]
        return v

    while len(terms) < max_terms:
        best = None
        for m, term in enumerate(terms):
            if len(term) >= degree:
                continue
            used = {j for j, _, _ in term}
            Bm = cols[m]
            for j in range(p):
                if j in used:
                    continue
                for t in sorted({rows[i][j] for i in range(N) if Bm[i] != 0}):
                    u = ortho([Bm[i] * max(rows[i][j] - t, 0.0) for i in range(N)])
                    v = ortho([Bm[i] * max(t - rows[i][j], 0.0) for i in range(N)])
                    nu = math.sqrt(sum(a * a for a in u))
                    red, keep = 0.0, []
                    if nu > 1e-9 * math.sqrt(N):
                        u = [a / nu for a in u]
                        du = sum(a * b for a, b in zip(u, e))
                        red += du * du
                        keep.append(1)
                        d = sum(a * b for a, b in zip(u, v))
                        v = [a - d * b for a, b in zip(v, u)]
                    nv = math.sqrt(sum(a * a for a in v))
                    if nv > 1e-9 * math.sqrt(N):
                        v = [a / nv for a in v]
                        dv = sum(a * b for a, b in zip(v, e))
                        red += dv * dv
                        keep.append(-1)
                    if keep and (best is None or red > best[0] + 1e-12):
                        best = (red, m, j, t, keep)
        if best is None or best[0] <= thresh * rss0:
            break
        _, m, j, t, keep = best
        for s in keep:
            if len(terms) >= max_terms:
                break
            col = [cols[m][i] * _hinge(rows[i][j], t, s) for i in range(N)]
            q = ortho(col)
            nq = math.sqrt(sum(a * a for a in q))
            if nq <= 1e-9 * math.sqrt(N):
                continue
            q = [a / nq for a in q]
            d = sum(a * b for a, b in zip(q, e))
            e = [a - d * b for a, b in zip(e, q)]
            Q.append(q)
            terms.append(terms[m] + [(j, t, s)])
            cols.append(col)
        rss = sum(v * v for v in e)

    def fit(keep):
        coef, r = _householder_ls([[cols[k][i] for k in keep] for i in range(N)], yy)
        return coef, r

    def gcv(r, nterms):
        M = nterms + c * (nterms - 1) / 2
        return r / N / (1 - M / N) ** 2 if M < N else math.inf

    active = list(range(len(terms)))
    coef, r = fit(active)
    best_set, best_g = active[:], gcv(r, len(active))
    while len(active) > 1:
        cand = None
        for k in active[1:]:
            trial = [a for a in active if a != k]
            rr = fit(trial)[1]
            if cand is None or rr < cand[0] - 1e-12:
                cand = (rr, k)
        active = [a for a in active if a != cand[1]]
        g = gcv(cand[0], len(active))
        if g < best_g - 1e-15:
            best_g, best_set = g, active[:]
    coef, r = fit(best_set)
    fitted = [sum(cf * cols[k][i] for cf, k in zip(coef, best_set)) for i in range(N)]
    Qr = rows if query is None else [[float(v) for v in rw] for rw in query]
    pred = [sum(cf * _eval_term(terms[k], rw) for cf, k in zip(coef, best_set)) for rw in Qr]
    return RichResult(
        title="MARS",
        summary_lines=[("terms", len(best_set)), ("gcv", best_g)],
        payload={
            "terms": [terms[k] for k in best_set],
            "coefficients": coef,
            "gcv": best_g,
            "rss": r,
            "fitted": fitted,
            "prediction": pred,
            "forward_terms": terms,
        },
    )


def cheatsheet():
    return "eslmrs: forward reflected hinge pairs (max RSS drop), backward deletion by GCV = RSS/(N (1 - (r + c(r-1)/2)/N)^2)"
