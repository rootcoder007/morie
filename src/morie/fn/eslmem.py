"""EM for a multivariate normal with missing values (ESL Ex. 17.9, eq 17.44)."""

import math

from ._richresult import RichResult
from .nlsgn import _inverse

__all__ = ["esl_mvn_em_missing"]


def esl_mvn_em_missing(X, max_iter=1000, tol=1e-12):
    r"""Maximum-likelihood mean and covariance of Gaussian data with values missing at random.

    E-step: for each row the missing coordinates :math:`m_i` are replaced by
    their conditional means given the observed ones; M-step (eq 17.44):
    :math:`\hat\mu_j = \sum_i\hat x_{ij}/N` and
    :math:`\hat\Sigma_{jj'} = \sum_i[(\hat x_{ij}-\hat\mu_j)(\hat x_{ij'}-\hat\mu_{j'}) + c_{i,jj'}]/N`,
    where :math:`c_{i,jj'}` is the conditional covariance of the missing
    coordinates (zero unless both are missing), the correction that makes
    this the MLE rather than single imputation (Little & Rubin 2002). Missing
    values are ``None`` or NaN.

    Parameters
    ----------
    X : N x p nested sequence with None/NaN for missing entries
    max_iter, tol
        EM controls (largest change in the parameters).

    Returns
    -------
    RichResult
        ``mean``, ``cov``, ``loglik`` (observed-data), ``iterations``,
        ``converged``.

    References
    ----------
    Little, R. J. A. & Rubin, D. B. (2002). Statistical Analysis with Missing
    Data (2nd ed.), sec. 11.2.
    """

    def miss(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    rows = [[None if miss(v) else float(v) for v in r] for r in X]
    n, p = len(rows), len(rows[0])
    for j in range(p):
        if all(r[j] is None for r in rows):
            raise ValueError(f"column {j} is entirely missing")
    mu = [sum(r[j] for r in rows if r[j] is not None) / sum(r[j] is not None for r in rows) for j in range(p)]
    S = [[1.0 if a == b else 0.0 for b in range(p)] for a in range(p)]
    for j in range(p):
        v = [r[j] for r in rows if r[j] is not None]
        S[j][j] = max(sum((x - mu[j]) ** 2 for x in v) / len(v), 1e-8)

    def estep(mu, S):
        xs, cs = [], []
        for r in rows:
            o = [j for j in range(p) if r[j] is not None]
            m = [j for j in range(p) if r[j] is None]
            x = [r[j] if r[j] is not None else mu[j] for j in range(p)]
            c = [[0.0] * p for _ in range(p)]
            if m and o:
                Soo = _inverse([[S[a][b] for b in o] for a in o])
                dev = [r[j] - mu[j] for j in o]
                for a in m:
                    reg = [sum(S[a][o[k]] * Soo[k][q] for k in range(len(o))) for q in range(len(o))]
                    x[a] = mu[a] + sum(g * d for g, d in zip(reg, dev))
                for a in m:
                    for b in m:
                        c[a][b] = S[a][b] - sum(
                            S[a][o[k]] * Soo[k][q] * S[o[q]][b] for k in range(len(o)) for q in range(len(o))
                        )
            elif m:
                for a in m:
                    for b in m:
                        c[a][b] = S[a][b]
            xs.append(x)
            cs.append(c)
        return xs, cs

    it, converged = 0, False
    for _ in range(max_iter):
        it += 1
        xs, cs = estep(mu, S)
        nmu = [sum(x[j] for x in xs) / n for j in range(p)]
        nS = [
            [sum((x[a] - nmu[a]) * (x[b] - nmu[b]) + c[a][b] for x, c in zip(xs, cs)) / n for b in range(p)]
            for a in range(p)
        ]
        delta = max(
            max(abs(a - b) for a, b in zip(nmu, mu)), max(abs(nS[a][b] - S[a][b]) for a in range(p) for b in range(p))
        )
        mu, S = nmu, nS
        if delta < tol:
            converged = True
            break
    ll = 0.0
    for r in rows:
        o = [j for j in range(p) if r[j] is not None]
        So = [[S[a][b] for b in o] for a in o]
        Si = _inverse(So)
        dev = [r[j] - mu[j] for j in o]
        # log det by Cholesky
        L = [[0.0] * len(o) for _ in o]
        for a in range(len(o)):
            for b in range(a + 1):
                s = So[a][b] - sum(L[a][k] * L[b][k] for k in range(b))
                L[a][b] = math.sqrt(s) if a == b else s / L[b][b]
        ld = 2 * sum(math.log(L[a][a]) for a in range(len(o)))
        q = sum(dev[a] * Si[a][b] * dev[b] for a in range(len(o)) for b in range(len(o)))
        ll += -0.5 * (len(o) * math.log(2 * math.pi) + ld + q)
    return RichResult(
        title="Multivariate normal EM with missing data",
        summary_lines=[("loglik", ll)],
        payload={"mean": mu, "cov": S, "loglik": ll, "iterations": it, "converged": converged},
    )


def cheatsheet():
    return "eslmem: EM, conditional means plus the c_ijj' covariance correction (ESL 17.44)"
