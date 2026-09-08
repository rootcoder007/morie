# morie.fn -- function file (rootcoder007/morie)
"""Joint density of the complete set of n order statistics."""

import math

from ._richresult import RichResult

__all__ = ['ostatjall', 'gibbons_joint_all_order']


def ostatjall(x, pdf):
    """Joint pdf of X_(1) < ... < X_(n): n! times the product of f.

    Section 2.2 (book p. 31): on the ordered region the density of the
    full vector of order statistics is

    .. math:: f_{(1),\\dots,(n)}(x_1,\\dots,x_n) = n! \\prod_i f(x_i),

    and 0 off that region.

    Parameters
    ----------
    x : sequence of float
        Candidate argument vector, length n.
    pdf : callable
        Parent density f_X.

    Returns
    -------
    RichResult
        keys ``pdf``, ``coef`` (n!), ``prod`` (product of densities),
        ``ordered`` (bool as int), ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 2.2, p. 31.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 1:
        raise ValueError("x must be non-empty.")
    ordered = all(xs[i] < xs[i + 1] for i in range(n - 1))
    prod = 1.0
    for v in xs:
        prod *= float(pdf(v))
    coef = float(math.factorial(n))
    return RichResult(
        payload={
            "pdf": float(coef * prod) if ordered else 0.0,
            "coef": coef,
            "prod": float(prod),
            "ordered": int(ordered),
            "n": n,
            "method": "n! prod f(x_i) on x_1 < ... < x_n (Gibbons Sec. 2.2)",
        }
    )


gibbons_joint_all_order = ostatjall
