"""Spatial voting utilities: proximity, directional and mixed models.

Enelow, J. M. and Hinich, M. J. (1984). The Spatial Theory of Voting. Cambridge University
Press. Poole, K. T. and Rosenthal, H. (1985). A spatial model for legislative roll call
analysis. American Journal of Political Science 29, 357-384 (Gaussian utility). Rabinowitz, G.
and Macdonald, S. E. (1989). A directional theory of issue voting. American Political Science
Review 83, 93-121. Matthews, S. A. (1979). A simple direction model of electoral competition.
Public Choice 34, 141-156 (angular model). Grofman, B. (1985). The neglected role of the status
quo in models of issue voting. Journal of Politics 47, 230-237 (discounting). Merrill, S. and
Grofman, B. (1999). A Unified Theory of Voting. Cambridge University Press. Groseclose, T.
(2001). A model of candidate location when one candidate has a valence advantage. American
Journal of Political Science 45, 862-886.
"""

import math

from ._richresult import RichResult

__all__ = ["voter_utility"]

MODELS = (
    "quadratic",
    "linear",
    "cityblock",
    "gaussian",
    "dot",
    "angular",
    "rm",
    "mixed",
    "discount",
    "categorical_proximity",
    "categorical_directional",
)


def _pts(v):
    if v and isinstance(v[0], (int, float)):
        return [[float(t)] for t in v]
    return [[float(t) for t in r] for r in v]


def voter_utility(
    voters,
    alternatives,
    model="quadratic",
    weights=None,
    neutral=None,
    beta=1.0,
    mix=0.5,
    status_quo=None,
    discount=0.5,
    region=None,
    valence=None,
):
    r"""Utility of every voter for every alternative under a spatial model.

    With salience weights w_k (default 1), voter x and alternative z:

    - ``"quadratic"``: -sum_k w_k (x_k - z_k)^2 (Enelow and Hinich 1984);
    - ``"linear"``: -sqrt(sum_k w_k (x_k - z_k)^2) (Euclidean distance loss);
    - ``"cityblock"``: -sum_k w_k |x_k - z_k|;
    - ``"gaussian"``: beta exp(-1/2 sum_k w_k^2 (x_k - z_k)^2) (Poole and Rosenthal 1985);
    - ``"dot"``: (x - N) . (z - N) with neutral point N (default 0), the directional model;
    - ``"rm"``: the dot product minus beta max(0, ||z - N|| - region), Rabinowitz and
      Macdonald's penalty for alternatives outside the region of acceptability;
    - ``"angular"``: cos of the angle between x - N and z - N (Matthews 1979);
    - ``"mixed"``: (1 - mix) quadratic + mix dot, a convex mixture of proximity and direction
      (the unified model family of Merrill and Grofman 1999);
    - ``"discount"``: quadratic loss to SQ + discount (z - SQ), promises discounted toward the
      status quo (Grofman 1985);
    - ``"categorical_proximity"``: minus the number of issues where x and z differ;
    - ``"categorical_directional"``: sum_k x_k z_k on issues coded symmetrically about 0.

    ``valence`` (one value per alternative) is added to every voter's utility (Groseclose 2001).

    Parameters
    ----------
    voters, alternatives : points (lists of coordinates, or scalars for one dimension)
    model : str
    weights : sequence, optional
        Salience per dimension.
    neutral : sequence, optional
        Neutral point N for the directional models.
    beta, mix, status_quo, discount, region : model parameters
    valence : sequence, optional

    Returns
    -------
    RichResult
        Keys: utility (voters x alternatives), choice (best alternative per voter),
        total (summed utility per alternative, the utilitarian joint utility).

    References
    ----------
    Enelow, J. M. and Hinich, M. J. (1984). The Spatial Theory of Voting.
    Rabinowitz, G. and Macdonald, S. E. (1989). American Political Science Review 83, 93-121.

    Examples
    --------
    >>> voter_utility([0.0], [1.0, -2.0])["utility"]
    [[-1.0, -4.0]]
    """
    if model not in MODELS:
        raise ValueError("model must be one of " + ", ".join(MODELS))
    X, Z = _pts(voters), _pts(alternatives)
    d = len(X[0])
    if any(len(r) != d for r in X + Z):
        raise ValueError("voters and alternatives need the same dimension")
    w = [1.0] * d if weights is None else [float(v) for v in weights]
    N = [0.0] * d if neutral is None else [float(v) for v in neutral]
    SQ = [0.0] * d if status_quo is None else [float(v) for v in status_quo]

    def quad(x, z):
        s = 0.0
        for k in range(d):
            s += w[k] * (x[k] - z[k]) ** 2
        return -s

    def dot(x, z):
        s = 0.0
        for k in range(d):
            s += (x[k] - N[k]) * (z[k] - N[k])
        return s

    def norm(v):
        s = 0.0
        for k in range(d):
            s += (v[k] - N[k]) ** 2
        return math.sqrt(s)

    def u(x, z):
        if model == "quadratic":
            return quad(x, z)
        if model == "linear":
            return -math.sqrt(-quad(x, z))
        if model == "cityblock":
            s = 0.0
            for k in range(d):
                s += w[k] * abs(x[k] - z[k])
            return -s
        if model == "gaussian":
            s = 0.0
            for k in range(d):
                s += w[k] ** 2 * (x[k] - z[k]) ** 2
            return beta * math.exp(-0.5 * s)
        if model == "dot":
            return dot(x, z)
        if model == "rm":
            return dot(x, z) - (beta * max(0.0, norm(z) - float(region)) if region is not None else 0.0)
        if model == "angular":
            nx, nz = norm(x), norm(z)
            return dot(x, z) / (nx * nz) if nx > 0 and nz > 0 else 0.0
        if model == "mixed":
            return (1 - mix) * quad(x, z) + mix * dot(x, z)
        if model == "discount":
            return quad(x, [SQ[k] + discount * (z[k] - SQ[k]) for k in range(d)])
        if model == "categorical_proximity":
            return -float(sum(1 for k in range(d) if x[k] != z[k]))
        s = 0.0
        for k in range(d):
            s += x[k] * z[k]
        return s

    val = [0.0] * len(Z) if valence is None else [float(v) for v in valence]
    U = [[u(x, z) + val[j] for j, z in enumerate(Z)] for x in X]
    choice = [max(range(len(Z)), key=lambda j: (row[j], -j)) for row in U]
    total = []
    for j in range(len(Z)):
        s = 0.0
        for row in U:
            s += row[j]
        total.append(s)
    return RichResult(
        title="Spatial utility",
        summary_lines=[("model", model)],
        payload={"utility": U, "choice": choice, "total": total},
    )


def cheatsheet():
    return "svutil: proximity, directional, angular, mixed, discounting and categorical spatial utilities"
