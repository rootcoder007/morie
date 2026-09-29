# morie.fn -- function file (rootcoder007/morie)
"""Second-order spatial lag W^2 y."""

from .swops import lag_operator


def swlag2(W, y):
    r"""Second-order spatial lag ``W(Wy) = W^2 y`` (``spdep::lag.listw`` applied twice).

    Used as the extra instrument in spatial 2SLS (Kelejian and Prucha 1998).
    Thin front-end to :func:`morie.fn.swops.lag_operator` with ``power=2``.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial two-stage
    least squares procedure. *J. Real Estate Finance Econ.* 17, 99-121.

    Examples
    --------
    >>> swlag2([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]], [1.0, 2.0, 4.0])
    [2.5, 2.0, 2.5]
    """
    return lag_operator(W, y, 2)


swlag2_fn = swlag2


def cheatsheet() -> str:
    return "swlag2(W, y) -> second-order spatial lag W^2 y."
