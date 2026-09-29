# morie.fn -- function file (rootcoder007/morie)
"""Baron-Kenny mediation -- front-end to bkmed."""

__all__ = ["causal_mediation_baron_kenny"]


def causal_mediation_baron_kenny(X, M, Y):
    r"""Baron-Kenny stepwise mediation under the (X, M, Y) argument order.

    Delegates to :func:`morie.fn.bkmed.baron_kenny`, which holds the
    three-regression procedure and reports each of the four steps
    separately (Baron & Kenny 1986, *J. Pers. Soc. Psychol.* 51(6),
    1173-1182).

    Examples
    --------
    >>> import math
    >>> X = [math.sin(k) for k in range(12)]
    >>> M = [0.5 * X[k] + 0.3 * math.cos(2 * k) for k in range(12)]
    >>> Y = [0.4 * X[k] + 0.8 * M[k] + 0.2 * math.sin(3 * k) for k in range(12)]
    >>> round(causal_mediation_baron_kenny(X, M, Y)["indirect"], 12)
    0.383604604142
    """
    from .bkmed import baron_kenny

    return baron_kenny(Y, X, M)


def cheatsheet():
    return "causmedb: Baron-Kenny mediation (front-end to bkmed)"
