"""Simultaneous pairwise marginal independence for two multiple-response categorical variables.

Bilder & Loughin (2025), Analysis of Categorical Data with R, Sec 6.2, eq (6.13).
"""

from ._richresult import RichResult
from ._rrng_core import pchisq

__all__ = ["spmi"]


def _chisq22(a, b, c, d):
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    return n * (a * d - b * c) ** 2 / (r1 * r2 * c1 * c2)


def spmi(W, Y, add_constant=0.5):
    r"""X^2_S = sum_ij X^2_S,ij (6.13), the Pearson statistics of the I x J (W_i, Y_j) 2 x 2 tables.

    Zero cells are replaced by ``add_constant`` (as MRCV does). Tests of SPMI:
    Bonferroni, p = min(1, I J min_ij P(chi2_1 > X^2_S,ij)); second-order
    Rao-Scott, X^2_RS2 = I J X^2_S / sum(lambda^2) on I^2 J^2 / sum(lambda^2)
    df, where the lambda are the eigenvalues of D^{-1} Sigma, Sigma the
    estimated covariance of the n^{1/2}(pi_ij - pi_i+ pi_+j) and
    D = diag(pi_i+ pi_+j (1 - pi_i+)(1 - pi_+j)); sum(lambda^2) = tr((D^{-1} Sigma)^2)
    is computed without an eigendecomposition.

    Parameters
    ----------
    W : n x I nested sequence of 0/1
    Y : n x J nested sequence of 0/1
    add_constant : float

    Returns
    -------
    RichResult
        Keys: statistic, statistic_ij, p_ij, p_bonferroni, p_ij_bonferroni,
        rs2_statistic, rs2_df, rs2_p.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2004). JASA 99, 1033-1043.
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Sec 6.2, eq (6.13).

    Examples
    --------
    >>> W = [[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]]
    >>> Y = [[1], [0], [1], [0], [1], [0]]
    >>> len(spmi(W, Y)["statistic_ij"])
    2
    """
    W = [[int(v) for v in r] for r in W]
    Y = [[int(v) for v in r] for r in Y]
    n = len(W)
    if n < 2 or len(Y) != n:
        raise ValueError("W and Y need the same n >= 2 rows")
    ni, J = len(W[0]), len(Y[0])
    xs, ps = [], []
    for i in range(ni):
        row, prow = [], []
        for j in range(J):
            cells = [0.0, 0.0, 0.0, 0.0]
            for w, y in zip(W, Y):
                cells[2 * w[i] + y[j]] += 1
            if min(cells[0] + cells[1], cells[2] + cells[3], cells[0] + cells[2], cells[1] + cells[3]) == 0:
                raise ValueError(f"item pair ({i}, {j}) has a constant response")
            cells = [c if c > 0 else add_constant for c in cells]
            x2 = _chisq22(cells[0], cells[1], cells[2], cells[3])
            row.append(x2)
            prow.append(pchisq(x2, 1, lower_tail=False))
        xs.append(row)
        ps.append(prow)
    stat = sum(map(sum, xs))
    pmin = min(min(r) for r in ps)
    pr = [sum(w[i] for w in W) / n for i in range(ni)]
    pc = [sum(y[j] for y in Y) / n for j in range(J)]
    idx = [(i, j) for i in range(ni) for j in range(J)]
    fs = [[w[i] * y[j] - pr[i] * y[j] - pc[j] * w[i] for i, j in idx] for w, y in zip(W, Y)]
    L = len(idx)
    fbar = [sum(f[a] for f in fs) / n for a in range(L)]
    S = [[sum(f[a] * f[b] for f in fs) / n - fbar[a] * fbar[b] for b in range(L)] for a in range(L)]
    d = [pr[i] * pc[j] * (1 - pr[i]) * (1 - pc[j]) for i, j in idx]
    DS = [[S[a][b] / d[a] for b in range(L)] for a in range(L)]
    s2 = sum(DS[a][b] * DS[b][a] for a in range(L) for b in range(L))
    rs2 = ni * J * stat / s2
    df = ni * ni * J * J / s2
    return RichResult(
        title="Simultaneous pairwise marginal independence",
        summary_lines=[
            ("X2_S", stat),
            ("RS2 p", pchisq(rs2, df, lower_tail=False)),
            ("Bonferroni p", min(1.0, ni * J * pmin)),
        ],
        payload={
            "statistic": stat,
            "statistic_ij": xs,
            "p_ij": ps,
            "p_bonferroni": min(1.0, ni * J * pmin),
            "p_ij_bonferroni": [[min(1.0, ni * J * v) for v in r] for r in ps],
            "rs2_statistic": rs2,
            "rs2_df": df,
            "rs2_p": pchisq(rs2, df, lower_tail=False),
        },
    )


def cheatsheet():
    return "spmi: MRCV SPMI statistic X2_S with Bonferroni and second-order Rao-Scott tests. Bilder & Loughin Sec 6.2."
