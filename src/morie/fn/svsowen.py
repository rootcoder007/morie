"""Shapley-Owen value: voting power of positions in a two-dimensional spatial game.

Owen, G. and Shapley, L. S. (1989). Optimal location of candidates in ideological space.
International Journal of Game Theory 18, 339-356.
"""

import math

from ._richresult import RichResult

__all__ = ["shapley_owen"]


def shapley_owen(ideals):
    r"""Shapley-Owen value of each voter: the share of directions u (uniform on the half circle) for
    which the voter is the median of the projections x_i . u, i.e. pivotal in the ordering along u.

    The median voter changes only at directions perpendicular to a pair of ideal points, so the
    value is computed exactly by summing arc lengths between consecutive critical directions.
    Requires an odd number of voters. At the centre of a radially symmetric configuration a voter
    is the median in every direction and holds all the power.

    Parameters
    ----------
    ideals : list of (x, y), odd count

    Returns
    -------
    RichResult
        Keys: value (per voter, sums to 1), strongest (index of the largest value).

    References
    ----------
    Owen, G. and Shapley, L. S. (1989). International Journal of Game Theory 18, 339-356.

    Examples
    --------
    >>> [round(v, 6) for v in shapley_owen([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])["value"]]
    [1.0, 0.0, 0.0, 0.0, 0.0]
    """
    P = [(float(x), float(y)) for x, y in ideals]
    n = len(P)
    if n % 2 == 0 or n < 3:
        raise ValueError("needs an odd number (>= 3) of voters")
    ang = set()
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
            if dx or dy:
                ang.add(math.atan2(dx, -dy) % math.pi)
    ang = sorted(ang)
    arcs = ang + [ang[0] + math.pi] if ang else [0.0, math.pi]
    val = [0.0] * n
    for a, b in zip(arcs, arcs[1:]):
        mid = 0.5 * (a + b)
        u = (math.cos(mid), math.sin(mid))
        k = sorted(range(n), key=lambda i: (P[i][0] * u[0] + P[i][1] * u[1], i))[(n - 1) // 2]
        val[k] += (b - a) / math.pi
    return RichResult(
        title="Shapley-Owen value",
        summary_lines=[("strongest", max(range(n), key=lambda i: val[i]))],
        payload={"value": val, "strongest": max(range(n), key=lambda i: (val[i], -i))},
    )


def cheatsheet():
    return "svsowen: Shapley-Owen spatial voting power (median-in-direction measure)"

# alias kept from the retired placeholder of the same name
coalition_value = shapley_owen
