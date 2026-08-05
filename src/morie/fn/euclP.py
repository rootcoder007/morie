# morie.fn -- function file (rootcoder007/morie)
"""Polynomial GCD via Euclid."""

from ._richresult import RichResult

__all__ = ["polynomial_gcd"]


def _trim(c, tol):
    while len(c) > 1 and abs(c[-1]) <= tol:
        c = c[:-1]
    return c


def _divmod_poly(a, b, tol):
    """Polynomial long division; a, b ascending-order coefficients."""
    a = list(a)
    db = len(b) - 1
    lead = b[db]
    q = [0.0] * max(len(a) - db, 1)
    while len(a) - 1 >= db and not (len(a) == 1 and abs(a[0]) <= tol):
        da = len(a) - 1
        f = a[da] / lead
        if da - db < len(q):
            q[da - db] = f
        for i in range(db + 1):
            a[da - db + i] -= f * b[i]
        a = _trim(a, tol)
        if len(a) - 1 < db:
            break
    return q, a


def polynomial_gcd(p, q, tol=1e-10):
    """
    Polynomial GCD via Euclid

    Formula: repeat polynomial division, gcd(a, b) = gcd(b, a mod b),
    until the remainder vanishes; the last non-zero remainder, made
    monic, is the greatest common divisor.  This is Algorithm E of
    Knuth, TAOCP Vol. 2, sec. 4.6.1, run over the reals with a numerical
    tolerance in place of exact zero testing.

    Coefficients are ASCENDING: ``[c0, c1, c2]`` means
    c0 + c1 x + c2 x^2.

    Parameters
    ----------
    p, q : array-like
        Coefficient vectors in ascending order.
    tol : float
        Magnitude at or below which a leading coefficient counts as zero.

    Returns
    -------
    result : dict
        Keys: estimate (degree of the gcd), gcd, degree, steps, n, method.

    References
    ----------
    Knuth (1997), The Art of Computer Programming, Vol. 2:
    Seminumerical Algorithms, 3rd ed., sec. 4.6.1, Addison-Wesley.
    """
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    a = _trim([float(v) for v in p], tol)
    b = _trim([float(v) for v in q], tol)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("empty input: p and q must have coefficients")
    zero_a = len(a) == 1 and abs(a[0]) <= tol
    zero_b = len(b) == 1 and abs(b[0]) <= tol
    if zero_a and zero_b:
        raise ValueError("gcd(0, 0) is undefined")
    if zero_b:
        a, b = b, a
        zero_a, zero_b = zero_b, zero_a
    steps = 0
    if zero_a:
        g = b
    else:
        if len(a) < len(b):
            a, b = b, a
        while not (len(b) == 1 and abs(b[0]) <= tol):
            _, r = _divmod_poly(a, b, tol)
            a, b = b, _trim(r, tol)
            steps += 1
            if steps > 10000:  # pragma: no cover
                raise ValueError("Euclid's algorithm failed to terminate")
        g = a
    lead = g[-1]
    if abs(lead) <= tol:
        g = [1.0]
    else:
        g = [v / lead for v in g]
    return RichResult(payload={
        "estimate": float(len(g) - 1),
        "gcd": g,
        "degree": len(g) - 1,
        "steps": steps,
        "n": len(g),
        "method": "Polynomial GCD via Euclid",
    })


def cheatsheet():
    return "euclP: Polynomial GCD via Euclid"


# compact alias per ledger/NAMING.md
polynomialgcd = polynomial_gcd
