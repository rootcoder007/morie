# morie.fn -- function file (rootcoder007/morie)
"""Linear Diophantine equation ax + by = c.

Classical number theory (the extended Euclidean algorithm).  Triage confirmed this names no owning source; the
standard definition is implemented and no citation is manufactured.
"""


from ._richresult import RichResult, with_describe_pointer

__all__ = ["diophantine"]


def _egcd(a, b):
    """Extended Euclid: return (g, x, y) with a x + b y = g = gcd(a, b).
    Iterative, so the recursion depth cannot bite on large inputs."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def diophantine(a, b, c):
    """Solve a x + b y = c over the integers.

    A solution exists iff gcd(a, b) divides c.  When it does, the
    extended Euclidean algorithm gives one particular solution and
    every other is x = x0 + t b/g, y = y0 - t a/g for integer t, so
    the step sizes are returned with the particular solution.

    Parameters
    ----------
    a, b, c : int coefficients; a and b not both zero.

    Returns
    -------
    RichResult with keys estimate (1.0 if solvable else 0.0),
    solvable, x, y, gcd, x_step, y_step, method.  x and y are None
    when there is no solution.
    """
    ai, bi, ci = int(a), int(b), int(c)
    if ai == 0 and bi == 0:
        raise ValueError("a and b cannot both be zero")
    g, x, y = _egcd(abs(ai), abs(bi))
    if ai < 0:
        x = -x
    if bi < 0:
        y = -y
    solvable = (ci % g == 0)
    if solvable:
        m = ci // g
        x0, y0 = x * m, y * m
        xs, ys = bi // g, -(ai // g)
    else:
        x0 = y0 = xs = ys = None
    return with_describe_pointer(RichResult(payload={
        "estimate": 1.0 if solvable else 0.0, "solvable": solvable,
        "x": x0, "y": y0, "gcd": g, "x_step": xs, "y_step": ys,
        "method": "linear Diophantine equation (extended Euclid)",
    }), "diophs")


def cheatsheet():
    return "diophs: Solve linear Diophantine ax+by=c"


# compact alias per ledger/NAMING.md
diophant = diophantine
