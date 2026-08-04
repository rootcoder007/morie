# morie.fn -- function file (rootcoder007/morie)
"""Support vector regression with the epsilon-insensitive loss.

MVSML (2022) sec. 9.6 p.369 introduces SVR but states plainly that
"detailed SVR theory is not covered in this book", referring the
reader to Burges (1998) and Awad & Khanna (2015).  The equations
implemented here are therefore NOT taken from the book; they are
taken from the primary tutorial the field cites:

  Smola, A.J. & Scholkopf, B. (2004).  A tutorial on support vector
  regression.  Statistics and Computing 14:199-222.
  Fetched and read in full: eq. (4) the epsilon-insensitive loss,
  eq. (10) the dual, eq. (11) the support vector expansion, eq. (16)
  the offset b.

The dual is solved by projected gradient ascent with a fixed
iteration count and no tolerance-driven early exit, so both language
arms follow the identical arithmetic path.
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["svr_epsilon_insensitive"]


def _svr_dual(K, y, C, eps, n_iter):
    """Maximize (10) of Smola & Scholkopf (2004),

        -(1/2) sum_ij (a_i - a*_i)(a_j - a*_j) <x_i, x_j>
        - eps sum_i (a_i + a*_i) + sum_i y_i (a_i - a*_i)

    subject to sum_i (a_i - a*_i) = 0 and a_i, a*_i in [0, C].

    The equality constraint is kept by removing the component of the
    gradient along the constraint normal c = (1, ..., 1, -1, ..., -1)
    before each step; the box is imposed by clipping.
    """
    n = len(y)
    scale = max(abs(K[i][i]) for i in range(n)) or 1.0
    step = 1.0 / (n * scale)
    a = [0.0] * n
    b = [0.0] * n
    for _ in range(int(n_iter)):
        th = [a[i] - b[i] for i in range(n)]
        Kt = [sum(K[i][j] * th[j] for j in range(n)) for i in range(n)]
        ga = [-Kt[i] - eps + y[i] for i in range(n)]
        gb = [Kt[i] - eps - y[i] for i in range(n)]
        # project onto {g : sum_i (ga_i - gb_i) = 0}; ||c||^2 = 2n
        sh = (sum(ga) - sum(gb)) / (2.0 * n)
        ga = [g - sh for g in ga]
        gb = [g + sh for g in gb]
        a = [min(C, max(0.0, a[i] + step * ga[i])) for i in range(n)]
        b = [min(C, max(0.0, b[i] + step * gb[i])) for i in range(n)]
    return a, b


def svr_epsilon_insensitive(X, y, C, eps, n_iter=4000, kernel="linear",
                            gamma=None, degree=2, coef0=1.0):
    """Epsilon-insensitive support vector regression.

    The loss of eq. (4) is |xi|_eps = 0 if |xi| <= eps and |xi| - eps
    otherwise, so residuals inside the eps-tube are not penalized at
    all.  The primal minimizes (1/2)||w||^2 + C sum_i (xi_i + xi*_i)
    subject to y_i - <w, x_i> - b <= eps + xi_i and
    <w, x_i> + b - y_i <= eps + xi*_i, with xi, xi* >= 0.  Its dual is
    eq. (10) and the fitted weights follow the support vector
    expansion (11), w = sum_i (a_i - a*_i) x_i.  The offset is taken
    at the midpoint of the interval (16).

    Parameters
    ----------
    X : (n, p) array-like of inputs.
    y : (n,) array-like of continuous responses.
    C : float, the trade-off constant of the primal.
    eps : float, the half-width of the insensitive tube.
    n_iter : int, fixed number of projected-gradient steps.
    kernel, gamma, degree, coef0 : passed to the shared Gram matrix
        builder so the dual can be taken in a feature space.

    Returns
    -------
    RichResult with keys estimate (the mean absolute
    epsilon-insensitive loss over the training points), w, b,
    alpha, alpha_star, theta, support_vectors, fitted, loss,
    objective, method.

    References
    ----------
    Smola & Scholkopf (2004) eqs. (4), (10), (11), (16).
    """
    Xm = [[float(v) for v in row] for row in X]
    ys = [float(v) for v in y]
    n = len(ys)
    Cv = float(C)
    ev = float(eps)
    K = _gp.kernel_matrix(Xm, kernel=kernel, gamma=gamma, degree=degree,
                          coef0=coef0)
    a, b = _svr_dual(K, ys, Cv, ev, n_iter)
    th = [a[i] - b[i] for i in range(n)]
    p = len(Xm[0])
    w = [sum(th[i] * Xm[i][j] for i in range(n)) for j in range(p)]
    Kt = [sum(K[i][j] * th[j] for j in range(n)) for i in range(n)]
    # eq. (16): b lies between the two bounds; take their midpoint
    lo = [-ev + ys[i] - Kt[i] for i in range(n)
          if a[i] < Cv - 1e-12 or b[i] > 1e-12]
    hi = [ev + ys[i] - Kt[i] for i in range(n)
          if a[i] > 1e-12 or b[i] < Cv - 1e-12]
    b0 = 0.5 * ((max(lo) if lo else 0.0) + (min(hi) if hi else 0.0))
    fit = [Kt[i] + b0 for i in range(n)]
    loss = [max(0.0, abs(ys[i] - fit[i]) - ev) for i in range(n)]
    obj = (-0.5 * sum(th[i] * Kt[i] for i in range(n))
           - ev * sum(a[i] + b[i] for i in range(n))
           + sum(ys[i] * th[i] for i in range(n)))
    sv = [i for i in range(n) if abs(th[i]) > 1e-9]
    return with_describe_pointer(RichResult(payload={
        "estimate": float(sum(loss) / n), "w": w, "b": float(b0),
        "alpha": a, "alpha_star": b, "theta": th,
        "support_vectors": sv, "fitted": fit, "loss": loss,
        "objective": float(obj),
        "method": "epsilon-insensitive SVR (Smola & Scholkopf 2004 eq. 10)",
    }), "svmep")


def cheatsheet():
    return "svmep: Support vector regression, epsilon-insensitive loss"


# compact alias per ledger/NAMING.md
svrepsilon = svr_epsilon_insensitive
