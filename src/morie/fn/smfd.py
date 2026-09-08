# morie.fn -- function file (rootcoder007/morie)
r"""P-splines: B-splines on many knots with a difference penalty.

**The idea.** Regression splines force a choice of knots that the data
rarely justify. Smoothing splines avoid the choice but carry a knot
per observation and a penalty on :math:`\int (f'')^2`. P-splines take
the middle road: a generous, evenly spaced B-spline basis, and a
penalty not on a derivative of the fitted curve but on *differences of
the coefficients themselves*,

.. math:: S = \lVert y - B a\rVert^2
          + \lambda \lVert \Delta^d a \rVert^2 ,

so the normal equations are

.. math:: (B'B + \lambda D_d' D_d)\,\hat a = B'y ,

with :math:`D_d` the :math:`d`-th order difference matrix. The
penalty is a plain quadratic form in the coefficients, which is why it
drops into any regression -- Gaussian, logistic, Poisson -- without new
mathematics.

**What the penalty order actually does.** As :math:`\lambda \to
\infty` the fit is forced into the null space of :math:`D_d`, which is
the polynomials of degree :math:`d-1`: :math:`d = 1` gives a constant,
:math:`d = 2` a straight line, :math:`d = 3` a parabola. That is an
exact statement, not a tendency, and it is the anchor here -- a
second-order penalty at huge :math:`\lambda` must reproduce the
ordinary least squares line through the data to machine precision.

**Choosing** :math:`\lambda`. The paper proposes criteria rather than
a single answer, and all of them are here: the effective dimension
:math:`\mathrm{tr}(H)` with :math:`H` the hat matrix, cross-validation
through the hat diagonal (each deletion residual is
:math:`e_i/(1-h_{ii})`, so leave-one-out costs one fit, not
:math:`n`), and AIC on the effective dimension. ``choose_lambda``
searches a grid and returns the whole trace, so the criterion's
sensitivity is visible instead of collapsed to one number.

**The basis.** Cox-de Boor recursion on an evenly spaced knot
sequence, extended by ``degree`` knots at each end. B-splines of any
degree sum to one at every interior point; ``partition_of_unity``
checks it, and a basis that fails that check is a basis with a knot
placed wrong.

References
----------
Eilers, P. H. C. & Marx, B. D. (1996) "Flexible Smoothing with
B-splines and Penalties", *Statistical Science* 11(2), 89-121,
doi:10.1214/ss/1038425655. Sec. 2 for the B-spline basis and the
Cox-de Boor construction; Sec. 3 for the difference penalty on
adjacent coefficients, the penalised normal equations
:math:`(B'B + \lambda D'D)a = B'y` reproduced above, and the argument
for a difference penalty in place of the derivative penalty; Sec. 4
for the polynomial limit as :math:`\lambda \to \infty` under a
:math:`d`-th order penalty; and Sec. 5-6 for the effective dimension
as the trace of the smoother matrix, the leave-one-out
cross-validation formula through the hat diagonal, and AIC as
criteria for choosing :math:`\lambda`.

O'Sullivan, F. (1986) "A Statistical Perspective on Ill-Posed Inverse
Problems", *Statistical Science* 1(4), 502-518,
doi:10.1214/ss/1177013525, for the penalised-B-spline idea that Eilers
and Marx simplify by penalising coefficient differences.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["knot_sequence", "bspline_basis", "partition_of_unity",
           "difference_matrix", "fit", "predict",
           "effective_dimension", "cross_validation", "aic",
           "choose_lambda"]


def knot_sequence(xmin, xmax, nseg=10, degree=3):
    r"""Evenly spaced knots, extended by ``degree`` at each end."""
    nseg, degree = int(nseg), int(degree)
    if nseg < 1:
        raise ValueError("smfd: nseg must be at least 1")
    if degree < 0:
        raise ValueError("smfd: degree cannot be negative")
    if not xmax > xmin:
        raise ValueError("smfd: xmax must exceed xmin")
    h = (float(xmax) - float(xmin)) / nseg
    return [float(xmin) + h * (k - degree)
            for k in range(nseg + 2 * degree + 1)]


def _bspline(x, k, degree, knots):
    """Cox-de Boor recursion for one basis function."""
    if degree == 0:
        last = k + 1 == len(knots) - 1
        if knots[k] <= x < knots[k + 1] or (last and x == knots[k + 1]):
            return 1.0
        return 0.0
    out = 0.0
    d1 = knots[k + degree] - knots[k]
    if d1 > 0.0:
        out += (x - knots[k]) / d1 * _bspline(x, k, degree - 1, knots)
    d2 = knots[k + degree + 1] - knots[k + 1]
    if d2 > 0.0:
        out += ((knots[k + degree + 1] - x) / d2
                * _bspline(x, k + 1, degree - 1, knots))
    return out


def bspline_basis(x, knots, degree=3):
    r"""The :math:`n \times p` B-spline design matrix."""
    degree = int(degree)
    p = len(knots) - degree - 1
    if p < 1:
        raise ValueError("smfd: the knot sequence is too short for "
                         "degree %d" % degree)
    return [[_bspline(float(v), k, degree, knots) for k in range(p)]
            for v in x]


def partition_of_unity(B, tol=1e-10):
    r"""B-splines of any degree sum to one at every interior point."""
    sums = [sum(row) for row in B]
    return {"sums": sums,
            "ok": all(abs(s - 1.0) < tol for s in sums),
            "worst": max(abs(s - 1.0) for s in sums) if sums else 0.0}


def difference_matrix(p, order=2):
    r""":math:`D_d`, whose null space is the degree :math:`d-1`
    polynomials."""
    p, order = int(p), int(order)
    if order < 0:
        raise ValueError("smfd: the penalty order cannot be negative")
    if order >= p:
        raise ValueError("smfd: a %d-th order penalty needs more than "
                         "%d coefficients" % (order, p))
    D = [[1.0 if i == j else 0.0 for j in range(p)] for i in range(p)]
    for _ in range(order):
        D = [[D[i + 1][j] - D[i][j] for j in range(p)]
             for i in range(len(D) - 1)]
    return D


def _solve(B, y, D, lam, weights=None):
    n, p = len(B), len(B[0])
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    A = [[sum(w[i] * B[i][r] * B[i][c] for i in range(n))
          for c in range(p)] for r in range(p)]
    for r in range(p):
        for c in range(p):
            A[r][c] += lam * sum(D[k][r] * D[k][c]
                                 for k in range(len(D)))
    b = [sum(w[i] * B[i][r] * y[i] for i in range(n))
         for r in range(p)]
    return [float(v) for v in np.linalg.solve(np.array(A),
                                              np.array(b))], A


def fit(x, y, nseg=10, degree=3, lam=1.0, order=2, weights=None):
    r"""Penalised least squares on the B-spline basis."""
    n = len(x)
    if n != len(y):
        raise ValueError("smfd: x and y must have the same length")
    if n < 2:
        raise ValueError("smfd: need at least two points")
    if lam < 0.0:
        raise ValueError("smfd: lambda cannot be negative")
    knots = knot_sequence(min(x), max(x), nseg, degree)
    B = bspline_basis(x, knots, degree)
    p = len(B[0])
    D = difference_matrix(p, order)
    a, A = _solve(B, y, D, float(lam), weights)
    fitted = [sum(B[i][k] * a[k] for k in range(p)) for i in range(n)]
    Ainv = np.linalg.inv(np.array(A))
    hat = [sum(B[i][r] * float(Ainv[r][c]) * B[i][c]
               for r in range(p) for c in range(p)) for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]
    rss = sum(v * v for v in resid)
    ed = sum(hat)
    return RichResult(payload={
        "estimate": ed, "coefficients": a, "fitted": fitted,
        "residuals": resid, "rss": rss, "hat_diagonal": hat,
        "effective_dimension": ed, "knots": knots, "degree": int(degree),
        "nseg": int(nseg), "order": int(order), "lam": float(lam),
        "n": n, "p": p,
        "sigma2": rss / max(n - ed, 1e-9),
        "method": "P-spline: (B'B + lambda D'D) a = B'y; Eilers & "
                  "Marx (1996) Sec. 3",
    })


def predict(fit_result, x):
    r"""Evaluate the fitted curve anywhere in the knot range."""
    B = bspline_basis(x, fit_result["knots"], fit_result["degree"])
    a = fit_result["coefficients"]
    return [sum(row[k] * a[k] for k in range(len(a))) for row in B]


def effective_dimension(fit_result):
    r"""The trace of the smoother matrix."""
    return fit_result["effective_dimension"]


def cross_validation(fit_result):
    r"""Leave-one-out through the hat diagonal, so one fit suffices."""
    n = fit_result["n"]
    tot = 0.0
    for i in range(n):
        denom = 1.0 - fit_result["hat_diagonal"][i]
        if abs(denom) < 1e-12:
            raise ValueError("smfd: h_ii = 1 at point %d, so the "
                             "deletion residual is undefined" % i)
        tot += (fit_result["residuals"][i] / denom) ** 2
    return {"cv": math.sqrt(tot / n), "press": tot}


def aic(fit_result):
    r"""AIC on the effective dimension rather than a parameter count."""
    n = fit_result["n"]
    rss = fit_result["rss"]
    if rss <= 0.0:
        return {"aic": float("-inf"), "effective_dimension":
                fit_result["effective_dimension"]}
    return {"aic": n * math.log(rss / n)
            + 2.0 * fit_result["effective_dimension"],
            "effective_dimension": fit_result["effective_dimension"]}


def choose_lambda(x, y, lambdas=None, criterion="cv", nseg=10,
                  degree=3, order=2):
    r"""Search a grid and return the whole trace, not just the winner."""
    if criterion not in ("cv", "aic"):
        raise ValueError("smfd: criterion must be 'cv' or 'aic', got "
                         "%r" % criterion)
    if lambdas is None:
        lambdas = [10.0 ** (k / 2.0) for k in range(-8, 13)]
    trace = []
    best = None
    for lam in lambdas:
        f = fit(x, y, nseg, degree, lam, order)
        score = (cross_validation(f)["cv"] if criterion == "cv"
                 else aic(f)["aic"])
        trace.append({"lam": float(lam), "score": score,
                      "effective_dimension":
                          f["effective_dimension"]})
        if best is None or score < best[0]:
            best = (score, lam, f)
    return RichResult(payload={
        "estimate": best[1], "lam": best[1], "score": best[0],
        "fit": best[2], "trace": trace, "criterion": criterion,
        "method": "lambda by %s over a grid; Eilers & Marx (1996) "
                  "Sec. 6" % criterion,
    })


def cheatsheet():
    return ("smfd: P-splines = a generous B-spline basis plus a "
            "DIFFERENCE penalty on the coefficients, so "
            "(B'B + lambda D'D) a = B'y. Order d of the penalty fixes "
            "the limit: as lambda -> infinity the fit becomes a "
            "polynomial of degree d-1 exactly (d=2 gives the OLS "
            "line). Effective dimension is tr(H); leave-one-out CV "
            "comes free from the hat diagonal.")


# compact alias per ledger/NAMING.md
penalized_spline = fit
