# morie.fn -- function file (rootcoder007/morie)
"""Radial basis function interpolation in any dimension: polyharmonic, thin-plate, cubic, quintic, multiquadric
family, Gaussian and Wendland kernels with polynomial augmentation and smoothing, anisotropy, multiscale residual
fitting, Rippa leave-one-out shape selection and grid prediction."""

from __future__ import annotations

import itertools
import math

from . import _array_core as np
from ._qpcore import solve, ssum
from ._richresult import RichResult

__all__ = ["radial_basis", "rbf_interpolate", "rbf_loocv", "rbf_multiscale", "rbf_grid"]


def _pts(a):
    return [tuple(float(v) for v in (r if isinstance(r, (list, tuple)) else [r])) for r in a]


def _d(a, b):
    return math.sqrt(ssum((x - y) ** 2 for x, y in zip(a, b)))


def radial_basis(r, kernel: str = "thin_plate", *, epsilon: float = 1.0, order: int = 3, support: float = 1.0):
    r"""Radial basis function values ``phi(r)``.

    ``linear`` ``-r``; ``cubic`` ``r^3``; ``quintic`` ``-r^5``;
    ``thin_plate`` ``r^2 log r``; ``polyharmonic`` ``r^k`` for odd ``k =
    order`` (sign ``(-1)^((k+1)/2)``) and ``r^k log r`` for even ``k``
    (sign ``(-1)^(k/2 + 1)``); ``multiquadric`` ``-sqrt(1 + (e r)^2)``;
    ``inverse_multiquadric`` ``1/sqrt(1 + (e r)^2)``; ``inverse_quadratic``
    ``1/(1 + (e r)^2)``; ``gaussian`` ``exp(-(e r)^2)``; ``wendland`` (C2,
    positive definite up to three dimensions) ``(1 - r/s)_+^4 (4 r/s + 1)``.
    Signs follow Fasshauer (2007) so every conditionally positive definite
    kernel has the right order.

    References
    ----------
    Fasshauer, G. E. (2007). *Meshfree Approximation Methods with MATLAB*.
    World Scientific.
    Wendland, H. (1995). Piecewise polynomial, positive definite and compactly
    supported radial functions of minimal degree. *Advances in Computational
    Mathematics*, 4(1), 389-396.

    Examples
    --------
    >>> [round(v, 6) for v in radial_basis([0.0, 0.5, 1.0], "wendland")]
    [1.0, 0.1875, 0.0]
    """
    out = []
    for x in (float(v) for v in (r if isinstance(r, (list, tuple)) else [r])):
        e = epsilon * x
        if kernel == "linear":
            v = -x
        elif kernel == "cubic":
            v = x**3
        elif kernel == "quintic":
            v = -(x**5)
        elif kernel == "thin_plate":
            v = x * x * math.log(x) if x > 0 else 0.0
        elif kernel == "polyharmonic":
            k = int(order)
            if k % 2:
                v = (-1) ** ((k + 1) // 2) * x**k
            else:
                v = (-1) ** (k // 2 + 1) * x**k * math.log(x) if x > 0 else 0.0
        elif kernel == "multiquadric":
            v = -math.sqrt(1 + e * e)
        elif kernel == "inverse_multiquadric":
            v = 1 / math.sqrt(1 + e * e)
        elif kernel == "inverse_quadratic":
            v = 1 / (1 + e * e)
        elif kernel == "gaussian":
            v = math.exp(-e * e)
        elif kernel == "wendland":
            t = x / support
            v = (1 - t) ** 4 * (4 * t + 1) if t < 1 else 0.0
        else:
            raise ValueError("unknown kernel")
        out.append(v)
    return out


def _monomials(dim, degree):
    if degree < 0:
        return []
    return [e for k in range(degree + 1) for e in itertools.combinations_with_replacement(range(dim), k)]


def _poly_row(p, mons):
    return [math.prod(p[i] for i in m) if m else 1.0 for m in mons]


def _aniso(P, transform):
    if transform is None:
        return P
    A = [[float(v) for v in r] for r in transform]
    return [tuple(ssum(A[i][j] * p[j] for j in range(len(p))) for i in range(len(A))) for p in P]


def rbf_interpolate(
    X,
    y,
    Xnew,
    *,
    kernel: str = "thin_plate",
    epsilon: float = 1.0,
    degree: int = 1,
    smoothing: float = 0.0,
    order: int = 3,
    support: float = 1.0,
    transform=None,
) -> RichResult:
    r"""RBF interpolant ``s(x) = sum_i w_i phi(|x - x_i|) + sum_j c_j p_j(x)`` of scattered data in any dimension.

    Solves ``[[Phi + lambda I, P], [P', 0]] [w; c] = [y; 0]`` with polynomial
    terms of total ``degree`` (``-1`` for none; at least the kernel's order
    of conditional positive definiteness is needed: 1 for thin-plate, cubic,
    linear, multiquadric; 2 for quintic) and ``smoothing`` ``lambda`` (0
    interpolates; ``lambda > 0`` gives the regularised smoothing fit).
    ``transform`` (a matrix) maps coordinates before distances are taken:
    anisotropic RBFs. Returns predictions at ``Xnew``, the coefficients and the
    assembled system (solved by Gaussian elimination with partial pivoting).

    References
    ----------
    Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
    Fasshauer, G. E. (2007). *Meshfree Approximation Methods with MATLAB*,
    chapters 6-8. World Scientific.

    Examples
    --------
    >>> r = rbf_interpolate([(0.0,), (1.0,), (2.0,)], [0.0, 1.0, 0.0], [(0.5,)], kernel="cubic")
    >>> round(r.prediction[0], 6)
    0.6875
    """
    P = _aniso(_pts(X), transform)
    Q = _aniso(_pts(Xnew), transform)
    yv = [float(v) for v in y]
    n, dim = len(P), len(P[0])
    mons = _monomials(dim, degree)
    m = len(mons)

    def phi(a, b):
        return radial_basis([_d(a, b)], kernel, epsilon=epsilon, order=order, support=support)[0]

    A = [[phi(P[i], P[j]) + (smoothing if i == j else 0.0) for j in range(n)] + _poly_row(P[i], mons) for i in range(n)]
    A += [[_poly_row(P[i], mons)[k] for i in range(n)] + [0.0] * m for k in range(m)]
    rhs = yv + [0.0] * m
    sol = [float(v) for v in solve(A, rhs)]
    w, c = sol[:n], sol[n:]
    pred = [
        ssum(w[i] * phi(q, P[i]) for i in range(n)) + ssum(cc * t for cc, t in zip(c, _poly_row(q, mons))) for q in Q
    ]
    return RichResult(payload={"prediction": pred, "weights": w, "poly_coef": c, "system": A, "rhs": rhs})


def rbf_loocv(X, y, epsilons, *, kernel: str = "gaussian", degree: int = -1, smoothing: float = 0.0) -> RichResult:
    r"""Leave-one-out error of RBF interpolation for each shape parameter by Rippa's (1999) formula, and the best one.

    ``e_k = a_k / (A^{-1})_{kk}`` with ``a = A^{-1} [y; 0]`` for the full
    (augmented) system matrix ``A``: exact leave-one-out residuals without
    refitting. Reports the RMS error per ``epsilon`` and the minimiser.

    References
    ----------
    Rippa, S. (1999). An algorithm for selecting a good value for the
    parameter c in radial basis function interpolation. *Advances in
    Computational Mathematics*, 11(2), 193-210.

    Examples
    --------
    >>> r = rbf_loocv([(0.0,), (1.0,), (2.0,), (3.0,)], [0.0, 1.0, 0.0, -1.0], [0.5, 1.0])
    >>> r.best in (0.5, 1.0)
    True
    """
    P = _pts(X)
    n = len(P)
    rms = []
    for e in (float(v) for v in epsilons):
        fit = rbf_interpolate(P, y, [], kernel=kernel, epsilon=e, degree=degree, smoothing=smoothing)
        A = fit["system"]
        Ai = [[float(v) for v in r] for r in np.linalg.inv(np.asarray(A, dtype=float)).tolist()]
        rhs = fit["rhs"]
        a = [ssum(Ai[i][j] * rhs[j] for j in range(len(rhs))) for i in range(n)]
        err = [a[k] / Ai[k][k] for k in range(n)]
        rms.append(math.sqrt(ssum(v * v for v in err) / n))
    eps = [float(v) for v in epsilons]
    k = min(range(len(eps)), key=lambda i: (rms[i], i))
    return RichResult(payload={"epsilon": eps, "rmse": rms, "best": eps[k]})


def rbf_multiscale(X, y, Xnew, supports, *, degree: int = -1) -> RichResult:
    r"""Multiscale (hierarchical) RBF approximation with Wendland kernels of decreasing support (Floater and Iske 1996).

    Level ``l`` interpolates the residual of the previous levels with a
    Wendland kernel of support ``supports[l]`` at all data sites; the result
    is the sum of the levels. Returns predictions and the residual RMS after
    each level at the data.

    References
    ----------
    Floater, M. S. and Iske, A. (1996). Multistep scattered data
    interpolation using compactly supported radial basis functions. *Journal
    of Computational and Applied Mathematics*, 73(1-2), 65-78.

    Examples
    --------
    >>> r = rbf_multiscale([(0.0,), (1.0,), (2.0,)], [1.0, 2.0, 0.0], [(1.0,)], [3.0, 1.5])
    >>> round(r.prediction[0], 6)
    2.0
    """
    P = _pts(X)
    Q = _pts(Xnew)
    res = [float(v) for v in y]
    pred = [0.0] * len(Q)
    trail = []
    for s in supports:
        fit = rbf_interpolate(P, res, P + Q, kernel="wendland", support=float(s), degree=degree)
        at_data, at_new = fit["prediction"][: len(P)], fit["prediction"][len(P) :]
        res = [r - v for r, v in zip(res, at_data)]
        pred = [a + b for a, b in zip(pred, at_new)]
        trail.append(math.sqrt(ssum(v * v for v in res) / len(res)))
    return RichResult(payload={"prediction": pred, "residual_rms": trail})


def rbf_grid(X, y, xs, ys, **kwargs) -> RichResult:
    r"""Evaluate an RBF surface (:func:`rbf_interpolate` options) on the grid ``xs`` x ``ys``; rows follow ``ys``.

    Examples
    --------
    >>> g = rbf_grid([(0, 0), (1, 0), (0, 1), (1, 1)], [0.0, 1.0, 1.0, 2.0], [0.5], [0.5], kernel="thin_plate")
    >>> round(g.surface[0][0], 6)
    1.0
    """
    pts = [(float(a), float(b)) for b in ys for a in xs]
    fit = rbf_interpolate(X, y, pts, **kwargs)
    nx = len(xs)
    surf = [fit["prediction"][k * nx : (k + 1) * nx] for k in range(len(ys))]
    return RichResult(payload={"surface": surf, "x": [float(v) for v in xs], "y": [float(v) for v in ys]})


def cheatsheet() -> str:
    return "radial_basis / rbf_interpolate / rbf_loocv / rbf_multiscale / rbf_grid -> radial basis function surfaces."
