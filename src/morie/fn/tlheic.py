# morie.fn -- function file (rootcoder007/morie)
r"""Estimating the efficient influence curve numerically.

Constructing a TMLE requires the canonical gradient of the target
parameter. For many estimation problems that object exists only
implicitly -- as a projection onto a tangent space with no closed form
-- and deriving it is the part of the work that stops a method being
used. This chapter's answer is to estimate the efficient influence
curve **from the definition** instead of solving for it.

**The definition is a derivative.** For a pathwise differentiable
:math:`\Psi` and a score :math:`s` in the tangent space,

.. math:: \frac{d}{d\epsilon}\Psi\big(P_\epsilon\big)\Big|_{\epsilon=0}
          = E_P\big[D^*(P)(O)\, s(O)\big],

with :math:`P_\epsilon` a path through :math:`P` with score
:math:`s`. So the gradient is identified by how the parameter *moves*
when the distribution is perturbed -- and a numerical derivative along
a chosen path gives one inner product with :math:`D^*`. Perturbing
along enough paths, and projecting onto the tangent space, recovers
:math:`D^*` itself.

**The representation is what makes it computable.** Any cadlag
:math:`h` of finite variation norm is
:math:`h(x) = \sum_S \int \prod_{j\in S} I(x_j \ge u_j)\, dh_S(u_S)`
-- exactly the HAL basis. Restricting to the class :math:`H_M` of such
functions with variation norm below :math:`M`, the projection becomes
a lasso-constrained regression, and the estimated gradient is a linear
combination of indicator basis functions. The tangent-space projection
is then a numerical problem rather than an analytic one.

**A check that costs nothing and catches everything.** A gradient must
satisfy the derivative identity along *any* path, not the ones used to
fit it. ``verify_gradient`` perturbs along a held-out direction and
compares the numerical derivative with the inner product -- and
because the ATE's efficient influence curve is known in closed form,
the anchor compares the numerical estimate against it directly.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 8 (a
machine-learning based estimator of an efficient influence curve which
avoids the need for its analytic computation; many estimation problems
in which the object exists only in implicit form and is extremely hard
to compute from it; the representation of a cadlag function of finite
variation norm as a sum over subsets of integrals against products of
indicator basis functions; the restriction to H_M, the subset with
variation norm below M; and the numerical computation of the
projection of an initial gradient onto the tangent space).

Bickel, P. J., Klaassen, C. A. J., Ritov, Y. & Wellner, J. A. (1993)
*Efficient and Adaptive Estimation for Semiparametric Models*, Johns
Hopkins University Press. Tangent spaces, pathwise differentiability
and canonical gradients.

Carone, M., Diaz, I. & van der Laan, M. J. (2018) "Higher-Order
Targeted Loss-Based Estimation", in *Targeted Learning in Data
Science*, Springer, 483-510, doi:10.1007/978-3-319-65304-4_26.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["numerical_derivative", "gradient_inner_product",
           "estimate_eic", "verify_gradient"]

_EPS = 1e-12


def numerical_derivative(psi_of_P, weights, score, h=1e-5):
    r""":math:`\frac{d}{d\epsilon}\Psi(P_\epsilon)` along the path
    :math:`p_\epsilon \propto (1 + \epsilon s)p`.

    A weighted empirical distribution tilted by the score, which is
    the simplest concrete path with that score.
    """
    w = [float(v) for v in k.vec(weights)]
    s = [float(v) for v in k.vec(score)]
    if len(w) != len(s):
        raise ValueError("tlheic: %d weights but %d score values"
                         % (len(w), len(s)))
    m = sum(w[i] * s[i] for i in range(len(w))) / sum(w)

    def tilt(e):
        v = [w[i] * (1.0 + e * (s[i] - m)) for i in range(len(w))]
        if any(x <= 0.0 for x in v):
            raise ValueError("tlheic: the perturbation left the "
                             "simplex; use a smaller h")
        t = sum(v)
        return [x / t for x in v]

    return (psi_of_P(tilt(h)) - psi_of_P(tilt(-h))) / (2.0 * h)


def gradient_inner_product(D, score, weights=None):
    r""":math:`E_P[D\, s]`, the quantity the derivative must equal."""
    d = [float(v) for v in k.vec(D)]
    s = [float(v) for v in k.vec(score)]
    if len(d) != len(s):
        raise ValueError("tlheic: %d gradient values but %d score "
                         "values" % (len(d), len(s)))
    w = [1.0 / len(d)] * len(d) if weights is None \
        else [float(v) for v in k.vec(weights)]
    t = sum(w)
    return sum(w[i] * d[i] * s[i] for i in range(len(d))) / t


def estimate_eic(psi_of_P, basis, weights=None, h=1e-5, ridge=1e-8):
    r"""Recover :math:`D^*` from directional derivatives.

    Each basis direction gives one equation
    :math:`E[D^*s_j] = \partial_\epsilon\Psi`; solving the resulting
    linear system in the same basis yields the gradient, with the
    mean-zero constraint imposed by centring every direction.
    """
    B = [[float(v) for v in r] for r in k.mat(basis)]
    n = len(B)
    p = len(B[0])
    w = [1.0 / n] * n if weights is None \
        else [float(v) for v in k.vec(weights)]
    tot = sum(w)
    w = [v / tot for v in w]
    C = []
    for j in range(p):
        col = [B[i][j] for i in range(n)]
        m = sum(w[i] * col[i] for i in range(n))
        C.append([col[i] - m for i in range(n)])
    rhs = [numerical_derivative(psi_of_P, w, C[j], h)
           for j in range(p)]
    G = [[sum(w[i] * C[a][i] * C[b][i] for i in range(n))
          for b in range(p)] for a in range(p)]
    for a in range(p):
        G[a][a] += float(ridge)
    coef = _solve(G, rhs)
    D = [sum(coef[j] * C[j][i] for j in range(p)) for i in range(n)]
    m = sum(w[i] * D[i] for i in range(n))
    D = [v - m for v in D]
    return RichResult(payload={
        "estimate": D, "D": D, "coefficients": coef,
        "n_directions": p, "mean": sum(w[i] * D[i]
                                       for i in range(n)),
        "method": "numerical estimation of the efficient influence "
                  "curve; van der Laan & Rose (2018) Chap. 8",
        "note": "no analytic derivation: the gradient is identified "
                "by how the parameter MOVES under perturbation",
    })


def _solve(A, b):
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) < 1e-14:
            raise ValueError("tlheic: the direction system is "
                             "singular; the basis is degenerate")
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0.0:
                f = M[i][c]
                M[i] = [M[i][j] - f * M[c][j] for j in range(n + 1)]
    return [M[i][n] for i in range(n)]


def verify_gradient(psi_of_P, D, score, weights=None, h=1e-5,
                    tol=1e-4):
    r"""Check the derivative identity along a HELD-OUT direction.

    A gradient fitted to some directions will satisfy the identity
    along those by construction; the test only means something on a
    direction that was not used.
    """
    n = len(k.vec(D))
    w = [1.0 / n] * n if weights is None \
        else [float(v) for v in k.vec(weights)]
    lhs = numerical_derivative(psi_of_P, w, score, h)
    rhs = gradient_inner_product(D, score, w)
    return {"derivative": lhs, "inner_product": rhs,
            "difference": abs(lhs - rhs),
            "verified": abs(lhs - rhs) < float(tol),
            "note": "must hold along ANY path, including ones not "
                    "used to fit the gradient"}


def cheatsheet():
    return ("tlheic: for many parameters the efficient influence curve "
            "exists only IMPLICITLY and deriving it is what stops the "
            "method being used. Estimate it from the DEFINITION "
            "instead: d/d_eps Psi(P_eps) = E[D* s], so perturbing "
            "along directions and reading off how the parameter moves "
            "identifies D*. Represent it in the HAL indicator basis "
            "with a variation-norm bound, and the tangent-space "
            "projection becomes a numerical regression. Verify on a "
            "HELD-OUT direction -- the fitted ones satisfy it by "
            "construction.")


# compact alias per ledger/NAMING.md
eicestimator = estimate_eic
