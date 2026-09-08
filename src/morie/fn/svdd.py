r"""Support Vector Data Description: the smallest sphere containing the data.

Tax, D. M. J., & Duin, R. P. W. (2004) "Support Vector Data
Description", *Machine Learning* 54(1), 45-66.

One-class classification: model a single class well enough to reject
everything else, having seen (almost) nothing else. SVDD does it
geometrically -- find the smallest hypersphere enclosing the target
data. Minimise :math:`F(R, a) = R^2` subject to
:math:`\|x_i - a\|^2 \le R^2` (eqs. 1-2), softened with slack variables
so outliers in the training set do not inflate the sphere (eqs. 3-4):

.. math:: F(R, a, \xi) = R^2 + C \sum_i \xi_i,
          \qquad
          \|x_i - a\|^2 \le R^2 + \xi_i,\quad \xi_i \ge 0.

Lagrangian (eq. 5), stationarity in :math:`R`, :math:`a` and
:math:`\xi` (eqs. 6-8):

.. math:: \sum_i \alpha_i = 1, \qquad a = \sum_i \alpha_i x_i,
          \qquad C - \alpha_i - \gamma_i = 0,

the last of which, with :math:`\alpha_i, \gamma_i \ge 0`, gives the box
constraint :math:`0 \le \alpha_i \le C` (eq. 9). Resubstituting leaves
the dual (eq. 10):

.. math:: \max_\alpha\ \sum_i \alpha_i (x_i \cdot x_i)
          - \sum_{i,j} \alpha_i \alpha_j (x_i \cdot x_j),
          \qquad \sum_i \alpha_i = 1,\ 0 \le \alpha_i \le C.

The data appear only as inner products, so the kernel trick applies
directly; ``kernel="rbf"`` is the paper's recommended choice and makes
the description tighter than a sphere in input space.

The KKT conditions (eqs. 11-13) classify every training point, and are
what the fit actually returns:

.. math:: \|x_i - a\|^2 < R^2 \Rightarrow \alpha_i = 0
          \quad\text{(interior)},\qquad
          \|x_i - a\|^2 = R^2 \Rightarrow 0 < \alpha_i < C
          \quad\text{(on the boundary)},\qquad
          \|x_i - a\|^2 > R^2 \Rightarrow \alpha_i = C
          \quad\text{(bounded, outside)}.

:math:`R^2` is read off any *unbounded* support vector (eq. 15) --
support vectors at :math:`\alpha = C` sit outside the description and
must be excluded, which is a detail easy to get wrong and easy to
check.

Rejection: a test object :math:`z` is accepted when
:math:`\|z - a\|^2 \le R^2`, expanded through the kernel as eq. 14.

Two knobs, and they are the same knob. :math:`C` bounds each
:math:`\alpha_i`; since the :math:`\alpha` sum to 1, at most
:math:`1/C` of them can sit at the bound, so **at most a fraction
:math:`1/(CN)` of the training data can be rejected**. Setting
:math:`C = 1/(\nu N)` therefore makes :math:`\nu` the target outlier
fraction directly, the same reparameterisation as in :math:`\nu`-SVC
(the paper's eq. 17 discussion). Pass ``nu=`` instead of ``C=`` to use
it. ``C >= 1`` forbids outliers entirely and the fit reduces to the
exact minimum enclosing ball.
"""

import math

from . import _array_core as np

from ._richresult import RichResult
from ._svm import kernel_matrix

__all__ = ["svdd", "support_vector_data_description"]


def _mat(X, name):
    # A flat sequence of n numbers is n observations in ONE dimension, not
    # a single n-dimensional point. atleast_2d gives the second reading --
    # a (1, n) matrix -- which makes the description a sphere around one
    # point, so R^2 comes out zero and the fit is silently meaningless.
    # R's as.matrix() takes the first reading, which is also the
    # convention every SVDD reference assumes, so match it here.
    arr = np.asarray(X, dtype=float)
    flat = getattr(arr, "ndim", None) == 1 or (
        isinstance(X, (list, tuple)) and X
        and not isinstance(X[0], (list, tuple))
        and not hasattr(X[0], "__len__"))
    if flat:
        rows = [[float(v)] for v in np.asarray(X, dtype=float)]
    else:
        rows = [[float(v) for v in r]
                for r in np.atleast_2d(arr)]
    if not rows or not rows[0]:
        raise ValueError("svdd: %s must be a non-empty (n, p) matrix" % name)
    w = len(rows[0])
    for r in rows:
        if len(r) != w:
            raise ValueError("svdd: %s must be rectangular" % name)
    return rows


def svdd(X, C=None, nu=None, kernel="rbf", gamma=None, degree=3,
         coef0=1.0, tol=1e-10, max_iter=20000):
    r"""Fit a Support Vector Data Description.

    Parameters
    ----------
    X : array-like
        ``(n, p)`` target-class training data.
    C : float, optional
        The :math:`C` of eq. 3. ``C >= 1`` forbids outliers (the
        minimum enclosing ball); smaller :math:`C` admits up to
        :math:`1/(CN)` of them. Defaults to ``1.0`` if neither ``C``
        nor ``nu`` is given.
    nu : float, optional
        Target outlier fraction, applied as :math:`C = 1/(\nu N)`.
        Mutually exclusive with ``C``.
    kernel : {"rbf", "linear", "poly"}
        Inner product to use. The data enter eqs. 10, 14 and 15 only
        through inner products, so any kernel is admissible; ``"rbf"``
        is the paper's recommendation.
    gamma, degree, coef0 : float, int, float
        Kernel parameters. ``gamma`` defaults to ``1/p``.
    tol : float
        Convergence tolerance on the dual.
    max_iter : int
        Cap on pairwise updates.

    Returns
    -------
    RichResult
        ``estimate`` / ``alpha`` are the dual coefficients;
        ``R2`` and ``radius`` the squared and plain radius; ``center``
        the sphere centre in input space (only meaningful for
        ``kernel="linear"``, and ``None`` otherwise, since with a
        non-linear kernel the centre lives in feature space);
        ``support_`` the indices with :math:`\alpha_i > 0`,
        ``boundary_`` those with :math:`0 < \alpha_i < C` (eq. 12) and
        ``bounded_`` those at :math:`\alpha_i = C` (eq. 13, the
        rejected ones); ``degenerate``, True when *every* support
        vector is bounded so eq. 15 has no valid source for
        :math:`R^2` (raise ``C``); ``distance2`` the squared distance of each
        training point to the centre; ``outlier_fraction`` the observed
        rejection rate and ``outlier_bound`` the guaranteed ceiling
        :math:`1/(CN)`; and ``decision`` / ``predict``, callables that
        score and accept new data via eq. 14.

    Examples
    --------
    With ``C >= 1`` no point may be excluded, so the fit is the exact
    minimum enclosing ball -- for three points on a circle, its centre
    and radius::

        fit = svdd([[1, 0], [-1, 0], [0, 1]], C=1.0, kernel="linear")
        fit["center"]     # [0, 0]
        fit["radius"]     # 1.0

    References
    ----------
    Tax & Duin (2004), *Machine Learning* 54(1), 45-66: eqs. 1-15.
    """
    rows = _mat(X, "X")
    n = len(rows)
    p = len(rows[0])
    if C is not None and nu is not None:
        raise ValueError("svdd: pass C or nu, not both")
    if nu is not None:
        nu = float(nu)
        if not 0.0 < nu <= 1.0:
            raise ValueError("svdd: nu must lie in (0, 1], got %r" % (nu,))
        C = 1.0 / (nu * n)
    if C is None:
        C = 1.0
    C = float(C)
    if C <= 0.0:
        raise ValueError("svdd: C must be > 0, got %r" % (C,))
    if C * n < 1.0:
        raise ValueError("svdd: C = %g with n = %d makes sum(alpha) = 1 "
                         "infeasible under alpha_i <= C; need C >= 1/n"
                         % (C, n))
    if kernel not in ("rbf", "linear", "poly"):
        raise ValueError("svdd: kernel must be rbf, linear or poly, got %r"
                         % (kernel,))
    if gamma is None:
        gamma = 1.0 / p

    K = kernel_matrix(rows, kernel=kernel, gamma=gamma, degree=degree,
                      coef0=coef0)
    K = [[float(v) for v in r] for r in K]

    alpha = _solve_dual(K, C, n, tol, max_iter)

    # eq. 7 / eq. 14: ||x - a||^2 = K(x,x) - 2 sum_i a_i K(x,x_i)
    #                              + sum_ij a_i a_j K(x_i,x_j)
    aKa = 0.0
    for i in range(n):
        if alpha[i] == 0.0:
            continue
        ai = alpha[i]
        for j in range(n):
            if alpha[j] != 0.0:
                aKa += ai * alpha[j] * K[i][j]

    def dist2_row(krow, kxx):
        s = 0.0
        for i in range(n):
            if alpha[i] != 0.0:
                s += alpha[i] * krow[i]
        return kxx - 2.0 * s + aKa

    d2 = [dist2_row(K[i], K[i][i]) for i in range(n)]

    eps = 1e-8
    support = [i for i in range(n) if alpha[i] > eps]
    bounded = [i for i in range(n) if alpha[i] >= C - eps]
    boundary = [i for i in support if i not in bounded]

    # eq. 15: R^2 from any support vector with alpha < C. Bounded ones
    # lie OUTSIDE the description and must not set the radius.
    #
    # SV_{<C} can be empty. sum(alpha) = 1 with alpha_i <= C means at
    # least ceil(1/C) coefficients are non-zero, and when C is small
    # enough the optimum can put ALL of them at the bound -- then every
    # support vector is nominally outside its own description and eq. 15
    # has no valid source. That is a degenerate fit, not a rounding
    # problem: raise C (or lower nu) until an unbounded support vector
    # exists. It is flagged rather than smoothed over, because the
    # radius returned in that case does not mean what eq. 15 says it
    # means.
    degenerate = not boundary and bool(support)
    if boundary:
        R2 = sum(d2[i] for i in boundary) / len(boundary)
    elif support:
        R2 = max(d2[i] for i in support)
    else:
        R2 = 0.0
    R2 = max(0.0, R2)

    center = None
    if kernel == "linear":
        center = [0.0] * p
        for i in range(n):
            if alpha[i] == 0.0:
                continue
            for t in range(p):
                center[t] += alpha[i] * rows[i][t]

    def decision(Z):
        """||z - a||^2 - R^2 via eq. 14. Negative means accepted."""
        zr = _mat(Z, "Z")
        if len(zr[0]) != p:
            raise ValueError("svdd: test data has %d columns, training had "
                             "%d" % (len(zr[0]), p))
        Kz = kernel_matrix(zr, rows, kernel=kernel, gamma=gamma,
                           degree=degree, coef0=coef0)
        Kzz = kernel_matrix(zr, kernel=kernel, gamma=gamma, degree=degree,
                            coef0=coef0)
        return [dist2_row([float(v) for v in Kz[t]], float(Kzz[t][t])) - R2
                for t in range(len(zr))]

    def predict(Z):
        """True where the object is accepted as a member of the class."""
        return [v <= 0.0 for v in decision(Z)]

    n_out = sum(1 for i in range(n) if d2[i] > R2 + 1e-8)
    return RichResult(payload={
        "estimate": alpha,
        "alpha": alpha,
        "R2": float(R2),
        "radius": float(math.sqrt(R2)),
        "center": center,
        "support_": support,
        "boundary_": boundary,
        "bounded_": bounded,
        "n_support": len(support),
        "degenerate": bool(degenerate),
        "distance2": d2,
        "outlier_fraction": float(n_out) / n,
        "outlier_bound": min(1.0, 1.0 / (C * n)),
        "decision": decision,
        "predict": predict,
        "C": C,
        "kernel": kernel,
        "gamma": gamma,
        "n": n,
        "method": "SVDD (Tax & Duin 2004)",
    })


def _solve_dual(K, C, n, tol, max_iter):
    r"""Maximise sum_i a_i K_ii - sum_ij a_i a_j K_ij on the simplex
    intersected with the box, by pairwise (SMO-style) updates.

    Pairs are the right move because sum(alpha) = 1 (eq. 6) is an
    equality constraint: any feasible step must move two coordinates in
    opposite directions.
    """
    alpha = [1.0 / n] * n
    if C < 1.0 / n:
        raise ValueError("svdd: infeasible C")
    # Ka[i] = sum_j alpha_j K_ij, kept incrementally.
    Ka = [sum(alpha[j] * K[i][j] for j in range(n)) for i in range(n)]

    # gradient of the objective wrt alpha_i
    def grad(i):
        return K[i][i] - 2.0 * Ka[i]

    for _ in range(int(max_iter)):
        g = [grad(i) for i in range(n)]
        # Steepest feasible pair: raise the largest gradient that can
        # still rise, lower the smallest that can still fall.
        up = None
        dn = None
        for i in range(n):
            if alpha[i] < C - 1e-15 and (up is None or g[i] > g[up]):
                up = i
            if alpha[i] > 1e-15 and (dn is None or g[i] < g[dn]):
                dn = i
        if up is None or dn is None or g[up] - g[dn] <= tol:
            break
        i, j = up, dn
        # objective along alpha_i += d, alpha_j -= d is a downward
        # parabola; its unconstrained maximiser is
        denom = 2.0 * (K[i][i] - 2.0 * K[i][j] + K[j][j])
        if denom <= 1e-15:
            d = alpha[j] if g[i] > g[j] else 0.0
        else:
            d = (g[i] - g[j]) / denom
        d = min(d, C - alpha[i], alpha[j])
        if d <= 1e-15:
            break
        alpha[i] += d
        alpha[j] -= d
        for t in range(n):
            Ka[t] += d * (K[t][i] - K[t][j])
    # tidy numerical dust
    for i in range(n):
        if alpha[i] < 1e-12:
            alpha[i] = 0.0
        elif alpha[i] > C - 1e-12:
            alpha[i] = C
    s = sum(alpha)
    if s > 0:
        alpha = [v / s for v in alpha]
    return alpha


def cheatsheet():
    return ("svdd: smallest enclosing sphere, min R^2 + C sum xi "
            "(Tax & Duin 2004 eqs. 3-4). Dual: max sum a_i K_ii - "
            "sum a_i a_j K_ij, sum a = 1, 0 <= a_i <= C (eqs. 9-10). "
            "KKT eqs. 11-13 label interior / boundary / bounded; R^2 "
            "comes from an UNBOUNDED support vector (eq. 15). At most "
            "1/(CN) of the data can be rejected, so C = 1/(nu N) "
            "makes nu the outlier fraction. C >= 1 gives the exact MEB.")


# compact alias per ledger/NAMING.md
support_vector_data_description = svdd
