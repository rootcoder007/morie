# morie.fn -- function file (rootcoder007/morie)
r"""Ordinary cokriging: predicting one variable with the help of another.

A soil survey measures lead at forty sites and zinc at four hundred,
because zinc is cheap to measure and lead is not. Kriging on lead alone
throws the zinc away. Cokriging keeps it: the two variables are modelled
jointly through a *linear model of coregionalisation*, in which every
direct and cross covariance is the same basic structure scaled by an
entry of a coregionalisation matrix,

.. math:: C_{ij}(h) = b_{ij}\,\rho(h) + n_{ij}\,\delta(h),

with :math:`B=(b_{ij})` and the nugget matrix :math:`N=(n_{ij})` both
positive semidefinite -- the condition that keeps the joint model from
implying a negative variance. Both are checked here rather than assumed,
because an indefinite :math:`B` produces confident predictions with
negative prediction variances and no error message.

The ordinary cokriging system carries one unbiasedness constraint per
variable, :math:`\sum\lambda=1` on the primary weights and
:math:`\sum\mu=0` on the secondary ones, so the secondary variable
contributes information without contributing its own mean:

.. math::
   \begin{pmatrix}C_{11}&C_{12}&1&0\\ C_{21}&C_{22}&0&1\\
   \mathbf 1'&\mathbf 0'&0&0\\ \mathbf 0'&\mathbf 1'&0&0\end{pmatrix}
   \begin{pmatrix}\lambda\\ \mu\\ m_1\\ m_2\end{pmatrix}
   = \begin{pmatrix}c_{10}\\ c_{20}\\ 1\\ 0\end{pmatrix}.

The primary-only ordinary kriging predictor is computed alongside and
returned, because the question a cokriging is run to answer is whether
the secondary variable bought anything: the cokriging variance can never
exceed the kriging variance, and the gap is the answer.

References
----------
Wackernagel, H. (2003) *Multivariate Geostatistics: An Introduction with
Applications*, 3rd ed., Springer, Ch. 24-25 (the linear model of
coregionalisation and the ordinary cokriging system),
doi:10.1007/978-3-662-05294-5.

Goovaerts, P. (1997) *Geostatistics for Natural Resources Evaluation*,
Oxford University Press, Sec. 6.2 (cokriging, and the two-constraint
form of the unbiasedness conditions).

Myers, D. E. (1982) "Matrix formulation of co-kriging", *Mathematical
Geology* **14**(3), 249-257, doi:10.1007/BF01032887.

Journel, A. G. and Huijbregts, C. J. (1978) *Mining Geostatistics*,
Academic Press, Ch. V (the permissibility conditions on a joint
covariance model).
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["cokriging"]

_EPS = 1e-12

_DEFAULT_LMC = {
    "model": "exponential",
    "range": 1.0,
    "b11": 1.0,
    "b22": 1.0,
    "b12": 0.0,
    "nugget11": 0.0,
    "nugget22": 0.0,
    "nugget12": 0.0,
}


def _rho(h, model, rng):
    """Correlogram of the basic structure; ``rho(0) = 1``."""
    if h <= 0.0:
        return 1.0
    if rng <= _EPS:
        return 0.0
    if model == "spherical":
        if h >= rng:
            return 0.0
        r = h / rng
        return 1.0 - (1.5 * r - 0.5 * r ** 3)
    if model == "exponential":
        return math.exp(-3.0 * h / rng)
    if model == "gaussian":
        return math.exp(-3.0 * (h / rng) ** 2)
    raise ValueError("crkbsg: model must be spherical, exponential or "
                     "gaussian, got %r" % (model,))


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def _solve(A, b):
    """Gaussian elimination with partial pivoting.

    The cokriging matrix is symmetric but NOT positive definite -- the
    Lagrange rows see to that -- so a Cholesky solve is not available.
    """
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[piv][col]):
                piv = r
        if abs(M[piv][col]) < 1e-300:
            raise ValueError("crkbsg: the cokriging system is singular -- "
                             "duplicated sample locations, or a "
                             "coregionalisation matrix of deficient rank")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / d
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


def cokriging(coords, y, z, s_predict, cross_variogram=None, coords_z=None):
    r"""Ordinary cokriging of a primary variable using a secondary one.

    Parameters
    ----------
    coords : array-like, shape ``(n1, d)``
        Locations at which the primary variable was measured.
    y : array-like, length ``n1``
        Primary variable -- the one being predicted.
    z : array-like, length ``n2``
        Secondary variable.
    s_predict : array-like, shape ``(m, d)`` or ``(d,)``
        Target locations.
    cross_variogram : dict, optional
        The linear model of coregionalisation: ``model`` (``spherical``,
        ``exponential`` or ``gaussian``), ``range``, the coregionalisation
        entries ``b11``, ``b22``, ``b12``, and the nugget entries
        ``nugget11``, ``nugget22``, ``nugget12``. Missing keys take the
        defaults in ``_DEFAULT_LMC``; ``b12 = 0`` means the secondary
        variable carries no information and cokriging reduces to kriging.
    coords_z : array-like, optional
        Locations of the secondary measurements. Defaults to ``coords``
        (the isotopic case, in which both variables are sampled at the
        same sites).

    Returns
    -------
    RichResult
        ``prediction`` and ``variance``/``std_error`` at each target, the
        primary-only ``kriging_prediction`` and ``kriging_variance`` for
        comparison, the ``variance_reduction`` between them, and the
        ``weights_primary``/``weights_secondary``/``lagrange`` of the
        last target solved.
    """
    C1 = [[float(v) for v in row] for row in k.mat(coords)]
    yv = [float(v) for v in k.vec(y)]
    zv = [float(v) for v in k.vec(z)]
    C2 = C1 if coords_z is None else [[float(v) for v in row]
                                      for row in k.mat(coords_z)]
    n1, n2 = len(C1), len(C2)
    if n1 == 0:
        raise ValueError("crkbsg: no primary observations")
    if len(yv) != n1:
        raise ValueError("crkbsg: %d primary locations but %d values"
                         % (n1, len(yv)))
    if len(zv) != n2:
        raise ValueError("crkbsg: %d secondary locations but %d values"
                         % (n2, len(zv)))
    d = len(C1[0])
    if any(len(r) != d for r in C1) or any(len(r) != d for r in C2):
        raise ValueError("crkbsg: all coordinates must have the same "
                         "dimension")

    par = dict(_DEFAULT_LMC)
    if cross_variogram:
        for key in cross_variogram:
            if key not in _DEFAULT_LMC:
                raise ValueError("crkbsg: unknown cross_variogram key %r"
                                 % (key,))
            par[key] = cross_variogram[key]
    model = str(par["model"])
    rng = float(par["range"])
    b11, b22, b12 = float(par["b11"]), float(par["b22"]), float(par["b12"])
    n11 = float(par["nugget11"])
    n22 = float(par["nugget22"])
    n12 = float(par["nugget12"])
    if rng <= 0.0:
        raise ValueError("crkbsg: the range must be positive")
    # permissibility: an indefinite B gives negative prediction variances
    if b11 < 0.0 or b22 < 0.0 or b11 * b22 < b12 * b12 - 1e-12:
        raise ValueError("crkbsg: the coregionalisation matrix is not "
                         "positive semidefinite (b11*b22 = %.6g < b12^2 = "
                         "%.6g)" % (b11 * b22, b12 * b12))
    if n11 < 0.0 or n22 < 0.0 or n11 * n22 < n12 * n12 - 1e-12:
        raise ValueError("crkbsg: the nugget matrix is not positive "
                         "semidefinite")

    tg = k.mat(s_predict)
    if tg and not isinstance(tg[0], (list, tuple)):
        tg = [list(tg)]
    targets = [[float(v) for v in row] for row in tg]
    if any(len(t) != d for t in targets):
        raise ValueError("crkbsg: targets must have dimension %d" % d)

    def cov(a, b_, bij, nij):
        h = _dist(a, b_)
        return bij * _rho(h, model, rng) + (nij if h <= _EPS else 0.0)

    m = n1 + n2 + 2
    A = [[0.0] * m for _ in range(m)]
    for i in range(n1):
        for j in range(n1):
            A[i][j] = cov(C1[i], C1[j], b11, n11)
        for j in range(n2):
            A[i][n1 + j] = cov(C1[i], C2[j], b12, n12)
        A[i][n1 + n2] = 1.0
    for i in range(n2):
        for j in range(n1):
            A[n1 + i][j] = cov(C2[i], C1[j], b12, n12)
        for j in range(n2):
            A[n1 + i][n1 + j] = cov(C2[i], C2[j], b22, n22)
        A[n1 + i][n1 + n2 + 1] = 1.0
    for j in range(n1):
        A[n1 + n2][j] = 1.0
    for j in range(n2):
        A[n1 + n2 + 1][n1 + j] = 1.0

    # primary-only ordinary kriging, for the comparison the run is for
    mk = n1 + 1
    Ak = [[0.0] * mk for _ in range(mk)]
    for i in range(n1):
        for j in range(n1):
            Ak[i][j] = A[i][j]
        Ak[i][n1] = 1.0
        Ak[n1][i] = 1.0

    c11_0 = b11 + n11
    pred, var, kpred, kvar = [], [], [], []
    lam = mu = None
    lagr = [0.0, 0.0]
    for t in targets:
        rhs = ([cov(C1[i], t, b11, n11) for i in range(n1)]
               + [cov(C2[i], t, b12, n12) for i in range(n2)] + [1.0, 0.0])
        sol = _solve(A, rhs)
        lam = sol[:n1]
        mu = sol[n1:n1 + n2]
        lagr = [sol[n1 + n2], sol[n1 + n2 + 1]]
        pred.append(sum(lam[i] * yv[i] for i in range(n1))
                    + sum(mu[i] * zv[i] for i in range(n2)))
        v = c11_0 - (sum(lam[i] * rhs[i] for i in range(n1))
                     + sum(mu[i] * rhs[n1 + i] for i in range(n2))
                     + lagr[0])
        var.append(max(v, 0.0))

        rk = [cov(C1[i], t, b11, n11) for i in range(n1)] + [1.0]
        sk = _solve(Ak, rk)
        wk = sk[:n1]
        kpred.append(sum(wk[i] * yv[i] for i in range(n1)))
        vk = c11_0 - (sum(wk[i] * rk[i] for i in range(n1)) + sk[n1])
        kvar.append(max(vk, 0.0))

    return RichResult(payload={
        "estimate": pred, "prediction": pred,
        "variance": var, "std_error": [math.sqrt(v) for v in var],
        "kriging_prediction": kpred, "kriging_variance": kvar,
        "variance_reduction": [kvar[i] - var[i] for i in range(len(var))],
        "weights_primary": lam, "weights_secondary": mu, "lagrange": lagr,
        "targets": targets,
        "coregionalisation": [[b11, b12], [b12, b22]],
        "nugget_matrix": [[n11, n12], [n12, n22]],
        "model": model, "range": rng,
        "n_primary": n1, "n_secondary": n2,
        "method": "ordinary cokriging under a linear model of "
                  "coregionalisation, with the two-constraint unbiasedness "
                  "system (Wackernagel 2003 Ch. 24-25; Goovaerts 1997 "
                  "Sec. 6.2)",
        "note": "the cokriging variance can never exceed the primary-only "
                "kriging variance; variance_reduction is what the secondary "
                "variable bought, and it is exactly zero when b12 = 0",
    })


def cheatsheet():
    return ("crkbsg: cokriging(coords, y, z, s_predict, cross_variogram) -> "
            "ordinary cokriging prediction and variance under a linear "
            "model of coregionalisation (Wackernagel 2003, Ch. 24-25)")
