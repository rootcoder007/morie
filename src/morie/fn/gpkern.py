# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Composing Gaussian process kernels.

Duvenaud, Lloyd, Grosse, Tenenbaum and Ghahramani (2013), "Structure
discovery in nonparametric regression through compositional kernel
search", ICML 2013 (PMLR 28(3):1166-1174), arXiv:1302.4922, and
Rasmussen and Williams (2006), *Gaussian Processes for Machine
Learning*, MIT Press, section 4.2.4.

Positive semidefiniteness is closed under the operations used here:
sums, products and any input warping k(u(x), u(x')).  That closure is
the whole reason a grammar over kernels is well posed, so the
smallest eigenvalue of every returned Gram matrix is reported and the
tests check it is non-negative.

    k_sum  = sum_i k_i(x, x')
    k_prod = prod_i k_i(x, x')
    k_warp = k(u(x), u(x')) with u(x) = (sin x, cos x) (periodic warp)
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gp_kernel_compose"]


def _sqdist(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s


def _rbf(A, B, ell, var):
    return [[var * math.exp(-0.5 * _sqdist(A[i], B[j]) / (ell * ell)) for j in range(len(B))] for i in range(len(A))]


def _warp(A):
    out = []
    for row in A:
        w = []
        for v in row:
            w.append(math.sin(v))
            w.append(math.cos(v))
        out.append(w)
    return out


def gp_kernel_compose(X, Y=None, kernel_spec=None):
    """Gram matrix of a composed kernel and its positive-definiteness.

    Parameters
    ----------
    X : n x d inputs.
    Y : m x d inputs; X by default.
    kernel_spec : dict-like with
        op    : "sum" or "prod",
        parts : list of dicts each with type ("rbf" or "warp"),
                lengthscale and variance.
    """
    A = core.mat(X)
    if len(A) == 0:
        raise ValueError("gp_kernel_compose: X is empty")
    B = A if Y is None else core.mat(Y)
    if len(B[0]) != len(A[0]):
        raise ValueError("gp_kernel_compose: X and Y have different dimensions")
    if kernel_spec is None:
        kernel_spec = {"op": "sum", "parts": [{"type": "rbf", "lengthscale": 1.0, "variance": 1.0}]}
    get = (lambda o, k, d=None: o.get(k, d)) if hasattr(kernel_spec, "get") else (lambda o, k, d=None: getattr(o, k, d))
    op = get(kernel_spec, "op", "sum")
    if op not in ("sum", "prod"):
        raise ValueError("gp_kernel_compose: op must be sum or prod")
    parts = get(kernel_spec, "parts")
    if not parts:
        raise ValueError("gp_kernel_compose: kernel_spec has no parts")
    K = None
    for p in parts:
        pg = (lambda o, k, d=None: o.get(k, d)) if hasattr(p, "get") else (lambda o, k, d=None: getattr(o, k, d))
        typ = pg(p, "type", "rbf")
        ell = float(pg(p, "lengthscale", 1.0))
        var = float(pg(p, "variance", 1.0))
        if ell <= 0 or var <= 0:
            raise ValueError("gp_kernel_compose: lengthscale and variance must be positive")
        if typ == "rbf":
            Kp = _rbf(A, B, ell, var)
        elif typ == "warp":
            Kp = _rbf(_warp(A), _warp(B), ell, var)
        else:
            raise ValueError("gp_kernel_compose: unknown kernel type " + str(typ))
        if K is None:
            K = Kp
        else:
            for i in range(len(A)):
                for j in range(len(B)):
                    K[i][j] = K[i][j] + Kp[i][j] if op == "sum" else K[i][j] * Kp[i][j]
    if Y is None:
        vals, _ = core.jacobi(K)
        lo = vals[0]
    else:
        lo = float("nan")
    diag = [K[i][i] for i in range(min(len(A), len(B)))]
    return RichResult(
        title="Composed GP kernel",
        summary_lines=[("rows", len(A)), ("cols", len(B)), ("op", op)],
        payload={
            "estimate": K[0][0],
            "K": K,
            "diagonal": diag,
            "min_eigenvalue": lo,
            "is_psd": 1 if (lo != lo or lo > -1e-10) else 0,
            "n": len(A),
            "method": "sum/product/warp composition of RBF kernels, Duvenaud et al. (2013); Rasmussen & Williams (2006) sect. 4.2.4",
        },
    )


def cheatsheet():
    return "gpkern: compose GP kernels"
