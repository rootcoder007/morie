"""Probabilistic spatial voting: binary and multinomial choice probabilities from utilities.

Poole, K. T. and Rosenthal, H. (1985). A spatial model for legislative roll call analysis.
American Journal of Political Science 29, 357-384 (normal error). McFadden, D. (1974).
Conditional logit analysis of qualitative choice behavior. In Frontiers in Econometrics,
105-142 (multinomial logit). Enelow, J. M. and Hinich, M. J. (1984). The Spatial Theory of
Voting (probabilistic voting with uncertain ideal points).
"""

import math

from ._richresult import RichResult
from ._rrng_core import pt
from .svutil import voter_utility

__all__ = ["vote_probability"]

LINKS = ("normal", "logistic", "laplace", "student_t", "gompertz", "cloglog")


def link_cdf(u, link, df=5.0):
    if link == "normal":
        return 0.5 * math.erfc(-u / math.sqrt(2))
    if link == "logistic":
        return 1 / (1 + math.exp(-u)) if u >= 0 else math.exp(u) / (1 + math.exp(u))
    if link == "laplace":
        return 0.5 * math.exp(u) if u < 0 else 1 - 0.5 * math.exp(-u)
    if link == "student_t":
        return float(pt(u, float(df)))
    if link == "gompertz":
        return math.exp(-math.exp(-u))
    if link == "cloglog":
        return 1 - math.exp(-math.exp(u))
    raise ValueError("link must be one of " + ", ".join(LINKS))


def vote_probability(
    voters,
    alternatives,
    model="quadratic",
    link="normal",
    scale=1.0,
    df=5.0,
    rule="binary",
    ideal_cov=None,
    **utility_args,
):
    r"""Choice probabilities from spatial utilities.

    ``rule="binary"`` (two alternatives): P(choose 1) = F((U_1 - U_2) / scale) with F the
    normal (probit / NOMINATE), logistic, Laplace, Student-t(df), Gompertz exp(-exp(-u)) or
    complementary log-log CDF. ``rule="multinomial"``: P_j = exp(U_j / scale) / sum exp(U_k / scale)
    (conditional logit, McFadden 1974). ``rule="exponential"``: P_j proportional to
    exp(-d_j / scale), d_j the Euclidean distance (Luce choice with exponential decay).
    With ``ideal_cov`` (a covariance matrix of the voter's uncertain ideal point) and
    quadratic utility, the binary normal probability integrates the uncertainty exactly:
    U_1 - U_2 = 2 x.(z_1 - z_2) + ||z_2||^2 - ||z_1||^2 is linear in x, so
    P = Phi(E[Delta] / sqrt(scale^2 + 4 (z_1 - z_2)' Sigma (z_1 - z_2))).

    Parameters
    ----------
    voters, alternatives : points
    model : str
        Utility model (see :func:`voter_utility`).
    link : {"normal", "logistic", "laplace", "student_t", "gompertz", "cloglog"}
    scale : float
    df : float
        Degrees of freedom for the Student-t link.
    rule : {"binary", "multinomial", "exponential"}
    ideal_cov : matrix, optional
    **utility_args
        Passed to :func:`voter_utility` (weights, neutral, beta, ...).

    Returns
    -------
    RichResult
        Keys: probability (voters x alternatives), utility.

    References
    ----------
    Poole, K. T. and Rosenthal, H. (1985). American Journal of Political Science 29, 357-384.
    McFadden, D. (1974). In Frontiers in Econometrics, 105-142.

    Examples
    --------
    >>> round(vote_probability([0.0], [1.0, -1.0])["probability"][0][0], 12)
    0.5
    """
    U = voter_utility(voters, alternatives, model=model, **utility_args)["utility"]
    P = []
    if rule == "binary":
        if len(U[0]) != 2:
            raise ValueError("binary voting needs exactly two alternatives")
        Z = (
            [[float(t)] for t in alternatives]
            if isinstance(alternatives[0], (int, float))
            else [[float(t) for t in r] for r in alternatives]
        )
        X = (
            [[float(t)] for t in voters]
            if isinstance(voters[0], (int, float))
            else [[float(t) for t in r] for r in voters]
        )
        for _i, row in enumerate(U):
            s = float(scale)
            if ideal_cov is not None:
                if model != "quadratic" or link != "normal":
                    raise ValueError("ideal_cov needs model='quadratic' and link='normal'")
                dz = [a - b for a, b in zip(Z[0], Z[1])]
                S = [[float(v) for v in r] for r in ideal_cov]
                var = 0.0
                for a in range(len(dz)):
                    for b in range(len(dz)):
                        var += dz[a] * S[a][b] * dz[b]
                s = math.sqrt(s * s + 4 * var)
            p1 = link_cdf((row[0] - row[1]) / s, link, df)
            P.append([p1, 1 - p1])
    elif rule == "multinomial":
        for row in U:
            m = max(row)
            e = [math.exp((v - m) / scale) for v in row]
            t = 0.0
            for v in e:
                t += v
            P.append([v / t for v in e])
    elif rule == "exponential":
        Z = (
            [[float(t)] for t in alternatives]
            if isinstance(alternatives[0], (int, float))
            else [[float(t) for t in r] for r in alternatives]
        )
        X = (
            [[float(t)] for t in voters]
            if isinstance(voters[0], (int, float))
            else [[float(t) for t in r] for r in voters]
        )
        for x in X:
            e = [math.exp(-math.dist(x, z) / scale) for z in Z]
            t = 0.0
            for v in e:
                t += v
            P.append([v / t for v in e])
    else:
        raise ValueError('rule must be "binary", "multinomial" or "exponential"')
    return RichResult(
        title="Spatial vote probability", summary_lines=[("rule", rule)], payload={"probability": P, "utility": U}
    )


def cheatsheet():
    return (
        "svprob: binary (probit, logit, Laplace, t, Gompertz), multinomial and exponential spatial vote probabilities"
    )
