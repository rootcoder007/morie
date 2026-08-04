"""Shared 3PL item response machinery for the k05 slice.

One place for the item response function, so the test characteristic
curve and the test information function cannot drift apart. Both are
sums over the same P_i(theta).
"""

from math import exp

__all__ = []


def _vec(x, name, n=None):
    if x is None:
        return None
    try:
        v = [float(t) for t in x]
    except TypeError:
        v = [float(x)]
    if n is not None and len(v) == 1:
        v = v * n
    if n is not None and len(v) != n:
        raise ValueError("%s must have one entry per item (got %d, expected %d)"
                         % (name, len(v), n))
    return v


def item_params(a, b, c=None, upper=None):
    """Coerce item parameters to equal-length lists; returns (a, b, c, u, n)."""
    bb = _vec(b, "b")
    n = len(bb)
    if n == 0:
        raise ValueError("need at least one item.")
    aa = _vec(a, "a", n)
    cc = _vec(c, "c", n) if c is not None else [0.0] * n
    uu = _vec(upper, "upper", n) if upper is not None else [1.0] * n
    for i in range(n):
        if not 0.0 <= cc[i] < uu[i] <= 1.0:
            raise ValueError("need 0 <= c_i < upper_i <= 1 for every item.")
    return aa, bb, cc, uu, n


def prob(theta, a, b, c, u, D=1.0):
    r"""Four-parameter logistic item response function.

    .. math:: P_i(\theta)=c_i+(u_i-c_i)\,
              \bigl[1+e^{-D a_i(\theta-b_i)}\bigr]^{-1}

    ``c`` is the lower asymptote (guessing), ``u`` the upper. With
    ``c=0, u=1`` this is the 2PL; with ``a`` shared, the 1PL. ``D``
    is the scaling constant -- 1.0 for the logistic metric, 1.702 to
    approximate the normal ogive.
    """
    z = D * a * (theta - b)
    # branch on the sign so exp never overflows for large |z|
    if z >= 0:
        p2 = 1.0 / (1.0 + exp(-z))
    else:
        e = exp(z)
        p2 = e / (1.0 + e)
    return c + (u - c) * p2


def info(theta, a, b, c, u, D=1.0):
    r"""Fisher information of one dichotomous item.

    The general definition for a two-category item is
    :math:`I=(P')^2/(PQ)`. Substituting the 4PL ``P`` and simplifying
    with :math:`P-c=(u-c)P^*` and :math:`u-P=(u-c)Q^*` gives

    .. math:: I_i(\theta)=D^2a_i^2\,
              \frac{(P_i-c_i)^2(u_i-P_i)^2}{(u_i-c_i)^2\,P_i(1-P_i)} ,

    which collapses to the familiar :math:`D^2a_i^2 P_i Q_i` when
    :math:`c_i=0` and :math:`u_i=1`, and to Birnbaum's 3PL form
    :math:`D^2a^2(Q/P)[(P-c)/(1-c)]^2` when :math:`u_i=1`.
    """
    p = prob(theta, a, b, c, u, D)
    q = 1.0 - p
    if p <= 0.0 or q <= 0.0:
        return 0.0
    num = (p - c) * (u - p)
    return (D * a) ** 2 * num * num / ((u - c) ** 2 * p * q)
