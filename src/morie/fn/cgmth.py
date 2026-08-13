# morie.fn -- function file (rootcoder007/morie)
"""
Conjugate gradient method for unconstrained optimization.

First-order iterative method using conjugate directions, with the
Polak-Ribiere-Polyak choice of beta and its max(beta, 0) safeguard.

The name PRP carries three papers and this module uses all three, so
all three are cited: Hestenes & Stiefel for the method, Polak & Ribiere
and Polyak for the beta they arrived at independently in the same year.
"""

from . import _array_core as np

__all__ = ["cgmth"]


def cgmth(f, grad_f, x0, tol=1e-6, max_iter=1000, full_output=False):
    """
    Conjugate gradient method for unconstrained minimization.

    Uses conjugate directions to minimize f(x). Converges in at most n steps
    for quadratic functions.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    x0 : ndarray
        Initial point.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 1000).
    full_output : bool, optional
        If True, return (x_min, info_dict).

    Returns
    -------
    x_min : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_value'.

    References
    ----------
    Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients
    for solving linear systems. Journal of Research of the National Bureau
    of Standards, 49(6), 409-436. -- the method itself, for linear
    systems.

    Polak, E., & Ribiere, G. (1969). Note sur la convergence de methodes
    de directions conjuguees. Revue francaise d'informatique et de
    recherche operationnelle, Serie rouge, 3(R1), 35-43. -- equation
    3.20, the beta used below, written there as
    gamma_i = (||r_{i+1}||^2 - r'_{i+1} r_i) / ||r_i||^2 with
    r = -gradient.

    Polyak, B. T. (1969). The conjugate gradient method in extremal
    problems. USSR Computational Mathematics and Mathematical Physics,
    9(4), 94-112. doi:10.1016/0041-5553(69)90035-4. Russian original:
    Zh. vychisl. matem. i matem. fiz., 9(4), 807-821. -- the same beta,
    arrived at independently; the P of PRP.

    Shewchuk, J. R. (1994). An Introduction to the Conjugate Gradient
    Method Without the Agonizing Pain, Edition 1 1/4, School of Computer
    Science, Carnegie Mellon University, CMU-CS-94-125, section 14.1.
    -- for the max(beta, 0) safeguard only, which is not in Polak &
    Ribiere or Polyak: without it the method "can, in rare cases, cycle
    infinitely without converging".

    Nocedal, J., & Wright, S. J. (2006). Numerical Optimization, 2nd ed.,
    Springer, section 3.1. -- the Armijo sufficient-decrease condition
    f(x + alpha d) <= f(x) + c alpha g'd used by the backtracking line
    search below, with the conventional c = 1e-4.

    See morie.fn.cgnonl for the Fletcher & Reeves (1964) beta, its
    three-stage line search, and the restart rule.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import cgmth
    >>> f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    >>> gf = lambda x: np.array([2*(x[0]-2), 2*(x[1]-3)])
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = cgmth(f, gf, x0)
    >>> np.allclose(x_min, [2, 3], atol=1e-4)
    True
    """
    x = np.atleast_1d(x0).astype(float)
    g = grad_f(x)
    d = -g

    for iteration in range(max_iter):
        if np.linalg.norm(g) < tol:
            if full_output:
                return x, {"iterations": iteration, "converged": True, "final_value": f(x)}
            return x

        # Backtracking line search on the Armijo sufficient-decrease
        # condition (Nocedal & Wright sec. 3.1); c = 1e-4 is the
        # conventional value.
        alpha = 1.0
        c = 1e-4
        for _ in range(20):
            if f(x + alpha * d) <= f(x) + c * alpha * np.dot(g, d):
                break
            alpha *= 0.5

        x = x + alpha * d
        g_new = grad_f(x)

        # Polak-Ribiere (1969) eq. 3.20 and Polyak (1969), the PRP beta.
        # Written there with r = -gradient, so the signs cancel and it is
        # g'_{i+1}(g_{i+1} - g_i) / (g'_i g_i) in terms of the gradient.
        beta = np.dot(g_new, g_new - g) / (np.dot(g, g) + 1e-14)
        # max(beta, 0) is Shewchuk sec. 14.1, not PRP: it is what gives
        # the method a convergence guarantee.
        beta = max(0, beta)

        d = -g_new + beta * d
        g = g_new

    if full_output:
        return x, {"iterations": max_iter, "converged": False, "final_value": f(x)}
    return x
