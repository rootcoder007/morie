# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""
Brent's method for root finding.

Combines bisection, secant, and inverse quadratic interpolation.
"""


__all__ = ["brtmh"]


def brtmh(f, a, b, tol=1e-6, max_iter=100, full_output=False):
    """
    Brent's method for root finding.

    Robust hybrid method combining bisection, secant, and inverse quadratic
    interpolation. Faster than bisection, more reliable than Newton's method.

    Parameters
    ----------
    f : callable
        Function f(x).
    a : float
        Left bracket (f(a) * f(b) < 0 assumed).
    b : float
        Right bracket.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 100).
    full_output : bool, optional
        If True, return (root, info_dict).

    Returns
    -------
    root : float
        Estimated root.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_residual'.

    References
    ----------
    Brent, R. P. (1973). Algorithms for Minimization Without Derivatives.
    Prentice Hall.

    Examples
    --------
    >>> from morie.fn import brtmh
    >>> f = lambda x: x**3 - 2
    >>> root = brtmh(f, 1.0, 2.0)
    >>> from morie.fn import _array_core as np
    >>> np.isclose(root, 2**(1/3), atol=1e-6)
    True
    """
    # Brent (1973), ch. 4, procedure zero: b is the best estimate, c the contrapoint
    # (f(b) and f(c) of opposite sign), a the previous b.
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    c, fc = a, fa
    d = e = b - a
    eps = 2.220446049250313e-16
    for iteration in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0:
            if full_output:
                return b, {"iterations": iteration + 1, "converged": True, "final_residual": abs(fb)}
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s_ = fb / fa
            if a == c:  # secant
                p_ = 2.0 * xm * s_
                q_ = 1.0 - s_
            else:  # inverse quadratic interpolation
                q_ = fa / fc
                r_ = fb / fc
                p_ = s_ * (2.0 * xm * q_ * (q_ - r_) - (b - a) * (r_ - 1.0))
                q_ = (q_ - 1.0) * (r_ - 1.0) * (s_ - 1.0)
            if p_ > 0:
                q_ = -q_
            p_ = abs(p_)
            if 2.0 * p_ < min(3.0 * xm * q_ - abs(tol1 * q_), abs(e * q_)):
                e, d = d, p_ / q_
            else:  # interpolation would leave the bracket or shrink too slowly: bisect
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
    if full_output:
        return b, {"iterations": max_iter, "converged": False, "final_residual": abs(fb)}
    return b

def cheatsheet() -> str:
    return "brtmh: brtmh(f, a, b, tol, max_iter, full_output) -> Brent's method for root finding."
