# morie.fn -- function file (rootcoder007/morie)
"""BFGS secant update of the Hessian and its inverse."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bfgsupd", "bfgs"]


def bfgsupd(H, s, y, inverse=True):
    """One rank-two BFGS update from a step and a gradient change.

    Quasi-Newton methods never form the Hessian; they maintain an
    approximation that satisfies the secant condition B_{k+1} s = y with
    s the step taken and y the change in gradient, and choose among the
    infinitely many such matrices the one closest to the current
    approximation in a weighted Frobenius norm.  That choice gives the
    rank-two updates

        B_{k+1} = B - B s s' B / (s' B s) + y y' / (y' s)
        H_{k+1} = (I - rho s y') H (I - rho y s') + rho s s',
                  rho = 1 / (y' s)

    the second being the same update written directly for the inverse, so
    a search direction is a matrix-vector product rather than a solve.
    The curvature condition y' s > 0 is what keeps the update positive
    definite; it is checked rather than assumed.

    Parameters
    ----------
    H : array-like, shape (p, p)
        Current approximation -- the inverse Hessian when ``inverse`` is
        true, the Hessian itself otherwise.
    s : array-like
        Step x_{k+1} - x_k.
    y : array-like
        Gradient change g_{k+1} - g_k.
    inverse : bool
        Update the inverse Hessian (the H form) rather than B.

    Returns
    -------
    RichResult
        ``M``, ``rho``, ``curvature``, ``secant``, ``p``, ``inverse``.

    References
    ----------
    The update is due independently to Broyden, C. G. (1970), Journal of
    the Institute of Mathematics and Its Applications 6(1), 76-90;
    Fletcher, R. (1970), The Computer Journal 13(3), 317-322; Goldfarb,
    D. (1970), Mathematics of Computation 24(109), 23-26; and Shanno,
    D. F. (1970), Mathematics of Computation 24(111), 647-656.  Standard
    published form -- it is stated identically in every numerical
    optimisation text, e.g. Nocedal and Wright, Numerical Optimization,
    2nd edn, Equations (6.17) and (6.19).  None of the four 1970 papers
    was in the local corpus and none was read for this implementation.
    """
    M = C.mat(H)
    p = len(M)
    if len(M[0]) != p:
        raise ValueError("H must be square")
    s = C.vec(s)
    y = C.vec(y)
    if len(s) != p or len(y) != p:
        raise ValueError("s and y must match the dimension of H")
    ys = sum(y[i] * s[i] for i in range(p))
    if ys <= 0.0:
        raise ValueError("curvature condition y's > 0 is violated")
    if inverse:
        rho = 1.0 / ys
        L = [[(1.0 if i == j else 0.0) - rho * s[i] * y[j]
              for j in range(p)] for i in range(p)]
        R = [[(1.0 if i == j else 0.0) - rho * y[i] * s[j]
              for j in range(p)] for i in range(p)]
        T = C.matmul(C.matmul(L, M), R)
        N = [[T[i][j] + rho * s[i] * s[j] for j in range(p)]
             for i in range(p)]
        sec = C.matvec(N, y)
        gap = max(abs(sec[i] - s[i]) for i in range(p))
    else:
        rho = 1.0 / ys
        Bs = C.matvec(M, s)
        sBs = sum(s[i] * Bs[i] for i in range(p))
        if sBs <= 0.0:
            raise ValueError("s'Bs must be strictly positive")
        N = [[M[i][j] - Bs[i] * Bs[j] / sBs + y[i] * y[j] / ys
              for j in range(p)] for i in range(p)]
        sec = C.matvec(N, s)
        gap = max(abs(sec[i] - y[i]) for i in range(p))
    return RichResult(payload={
        "M": N, "rho": rho, "curvature": ys, "secant": gap, "p": p,
        "inverse": bool(inverse),
        "method": "BFGS rank-two secant update (Broyden-Fletcher-Goldfarb-Shanno 1970)"})


bfgs = bfgsupd


def cheatsheet():
    return "bfgsop: BFGS secant update of the Hessian and its inverse."
