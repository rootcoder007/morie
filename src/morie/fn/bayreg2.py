# morie.fn -- function file (rootcoder007/morie)
r"""Robust regression with Student-t errors, as a scale mixture of normals.

Least squares gives every residual the same weight, so a single gross
outlier moves the fit by an amount proportional to its distance. Modelling
the errors as Student-t with :math:`\nu` degrees of freedom fixes that
without discarding anything: a t is a normal whose variance is itself
random, :math:`y_i \mid \lambda_i \sim N(x_i'\beta, \sigma^2/\lambda_i)`
with :math:`\lambda_i \sim \mathrm{Gamma}(\nu/2, \nu/2)`, and the EM
algorithm that follows is iteratively reweighted least squares with

.. math:: \hat\lambda_i = \frac{\nu + 1}{\nu + r_i^2/\sigma^2}.

An observation is downweighted in proportion to how badly it fits, which
is a decision the MODEL makes rather than a threshold anyone picks. As
:math:`\nu \to \infty` every weight tends to 1 and the fit returns to
least squares -- the comparison the returned ``weights`` are for.

References
----------
West, M. (1984) "Outlier models and prior distributions in Bayesian
linear regression", *Journal of the Royal Statistical Society B*
**46**(3), 431-439. The scale-mixture representation and its use for
outlier accommodation.

Geweke, J. (1993) "Bayesian treatment of the independent Student-t linear
model", *Journal of Applied Econometrics* **8**(S1), S19-S40,
doi:10.1002/jae.3950080504.

Lange, K. L., Little, R. J. A. and Taylor, J. M. G. (1989) "Robust
statistical modeling using the t distribution", *Journal of the American
Statistical Association* **84**(408), 881-896,
doi:10.1080/01621459.1989.10478852. The EM/IRLS algorithm used here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["student_t_regression"]

_EPS = 1e-12


def student_t_regression(X, y, nu=4.0, max_iter=200, tol=1e-10,
                         add_intercept=True):
    r"""EM for the Student-t linear model: IRLS with model-chosen weights."""
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    yv = [float(v) for v in k.vec(y)]
    n = len(Xm)
    if n == 0:
        raise ValueError("bayreg2: no observations")
    if len(yv) != n:
        raise ValueError("bayreg2: %d rows but %d responses" % (n, len(yv)))
    if add_intercept:
        Xm = [[1.0] + r for r in Xm]
    p = len(Xm[0])
    nu = float(nu)
    if nu <= 0.0:
        raise ValueError("bayreg2: the degrees of freedom must be positive")
    if n <= p:
        raise ValueError("bayreg2: %d observations cannot identify %d "
                         "coefficients" % (n, p))

    def wls(w):
        A = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in range(n))
              for b in range(p)] for a in range(p)]
        for a in range(p):
            A[a][a] += 1e-12
        b = [sum(w[i] * Xm[i][a] * yv[i] for i in range(n)) for a in range(p)]
        return k.cholsolve(A, b), A

    w = [1.0] * n
    beta, A = wls(w)
    s2 = 1.0
    it, converged = 0, False
    for it in range(1, int(max_iter) + 1):
        res = [yv[i] - sum(Xm[i][a] * beta[a] for a in range(p))
               for i in range(n)]
        s2 = sum(w[i] * res[i] * res[i] for i in range(n)) / n
        if s2 <= _EPS:
            s2 = _EPS
        w_new = [(nu + 1.0) / (nu + res[i] * res[i] / s2) for i in range(n)]
        beta_new, A = wls(w_new)
        shift = max(abs(beta_new[a] - beta[a]) for a in range(p))
        beta, w = beta_new, w_new
        if shift < tol:
            converged = True
            break

    res = [yv[i] - sum(Xm[i][a] * beta[a] for a in range(p))
           for i in range(n)]
    # weighted information matrix -> approximate standard errors
    cov = []
    for a in range(p):
        e = [1.0 if b == a else 0.0 for b in range(p)]
        cov.append(k.cholsolve(A, e))
    se = [math.sqrt(max(s2 * cov[a][a], 0.0)) for a in range(p)]
    loglik = 0.0
    for i in range(n):
        z = res[i] * res[i] / s2
        loglik += (math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0)
                   - 0.5 * math.log(math.pi * nu * s2)
                   - (nu + 1.0) / 2.0 * math.log1p(z / nu))

    return RichResult(payload={
        "estimate": beta, "coefficients": beta, "std_error": se,
        "weights": w, "residuals": res, "scale2": s2,
        "fitted": [yv[i] - res[i] for i in range(n)],
        "iterations": it, "converged": converged, "nu": nu,
        "loglik": loglik, "n": n, "p": p,
        "method": "Student-t linear model by EM as a scale mixture of "
                  "normals (West 1984; Geweke 1993; Lange, Little & "
                  "Taylor 1989)",
        "note": "the weight (nu+1)/(nu+r^2/s^2) is chosen by the model, not "
                "by a threshold; as nu grows every weight tends to 1 and the "
                "fit returns to least squares",
    })


def cheatsheet():
    return ("bayreg2: student_t_regression(X, y, nu) -> robust regression by "
            "EM on the Student-t scale mixture (West 1984, JRSS B 46(3), "
            "431-439; Lange, Little & Taylor 1989, JASA 84(408), 881-896)")
