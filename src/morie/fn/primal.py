r"""Chambolle-Pock primal-dual hybrid gradient.

Chambolle, A., & Pock, T. (2011) "A First-Order Primal-Dual Algorithm
for Convex Problems with Applications to Imaging", *Journal of
Mathematical Imaging and Vision* **40**(1), 120-145.

Solves the saddle-point problem

.. math:: \min_x \max_y \; \langle Kx, y\rangle + G(x) - F^*(y)

by Algorithm 1 (their eq. 8), alternating a dual ascent, a primal
descent, and an extrapolation:

.. math::
    y^{n+1} &= \mathrm{prox}_{\sigma F^*}(y^n + \sigma K \bar{x}^n)\\
    x^{n+1} &= \mathrm{prox}_{\tau G}(x^n - \tau K^* y^{n+1})\\
    \bar{x}^{n+1} &= x^{n+1} + \theta (x^{n+1} - x^n)

The step sizes must satisfy :math:`\tau\sigma\lVert K\rVert^2 < 1`
(their Theorem 1); with :math:`\theta = 1` this converges at
:math:`O(1/N)` on the partial primal-dual gap. The condition is
enforced here rather than assumed, because violating it does not
produce a warning -- it produces a divergent sequence that still
returns numbers.

The extrapolation :math:`\bar{x}` is the whole trick: with
:math:`\theta = 0` the method reduces to plain Arrow-Hurwicz, which is
not convergent under these step sizes.

Routes
------
``theta`` exposes the relaxation: 1 is the paper's convergent choice, 0
recovers Arrow-Hurwicz. ``prox_f_star`` and ``prox_g`` are supplied by
the caller, so any :math:`F, G` pair works; :func:`tv_denoise_1d` wires
up the paper's own example, total-variation denoising, where
:math:`F^*` is the projection onto the :math:`\ell_\infty` ball.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["chambolle_pock", "tv_denoise_1d", "primal"]


def chambolle_pock(K, Kt, prox_f_star, prox_g, x0, y0, tau=None, sigma=None,
                   theta=1.0, norm_K=None, max_iter=500, tol=1e-10):
    r"""Algorithm 1 of Chambolle & Pock (2011).

    Parameters
    ----------
    K, Kt : callable
        The linear operator and its adjoint.
    prox_f_star, prox_g : callable
        ``prox_f_star(y, sigma)`` and ``prox_g(x, tau)``.
    x0, y0 : array-like
        Primal and dual starting points.
    tau, sigma : float, optional
        Step sizes. Default to ``1/norm_K`` each, which satisfies the
        condition with equality margin.
    theta : float
        Extrapolation parameter; 1 is the paper's choice.
    norm_K : float, optional
        An upper bound on ``||K||``. Estimated by power iteration when
        omitted.
    """
    x = [float(v) for v in np.atleast_1d(np.asarray(x0, dtype=float))]
    y = [float(v) for v in np.atleast_1d(np.asarray(y0, dtype=float))]
    theta = float(theta)

    if norm_K is None:
        # Power iteration on K*K to bound ||K||.
        v = [1.0] * len(x)
        nrm = 1.0
        for _ in range(100):
            w = Kt(K(v))
            nrm2 = math.sqrt(sum(float(t) * float(t) for t in w))
            if nrm2 <= 0.0:
                break
            v = [float(t) / nrm2 for t in w]
            nrm = math.sqrt(nrm2)
        norm_K = max(nrm, 1e-12)
    norm_K = float(norm_K)

    if tau is None:
        tau = 1.0 / norm_K
    if sigma is None:
        sigma = 1.0 / norm_K
    tau, sigma = float(tau), float(sigma)
    if tau <= 0.0 or sigma <= 0.0:
        raise ValueError(
            "chambolle_pock: tau and sigma must be positive, got %r and %r"
            % (tau, sigma))
    prod = tau * sigma * norm_K * norm_K
    if prod >= 1.0 + 1e-12:
        raise ValueError(
            "chambolle_pock: Theorem 1 requires tau*sigma*||K||^2 < 1, got "
            "%.6g. The iteration diverges outside this range while still "
            "returning finite numbers, so this is refused rather than "
            "warned about." % prod)

    xbar = list(x)
    it = 0
    converged = False
    for it in range(1, int(max_iter) + 1):
        Kx = K(xbar)
        y = list(prox_f_star([y[i] + sigma * float(Kx[i])
                              for i in range(len(y))], sigma))
        Kty = Kt(y)
        x_new = list(prox_g([x[j] - tau * float(Kty[j])
                             for j in range(len(x))], tau))
        xbar = [x_new[j] + theta * (x_new[j] - x[j]) for j in range(len(x))]
        step = math.sqrt(sum((x_new[j] - x[j]) ** 2 for j in range(len(x))))
        x = x_new
        if step <= tol:
            converged = True
            break

    return RichResult(payload={
        "estimate": x,
        "x": x,
        "y": y,
        "tau": tau,
        "sigma": sigma,
        "theta": theta,
        "norm_K": norm_K,
        "step_condition": prod,
        "iterations": int(it),
        "converged": bool(converged),
        "method": "Chambolle-Pock primal-dual hybrid gradient "
                  "(Chambolle & Pock 2011, Algorithm 1)",
    })


def tv_denoise_1d(signal, lam=1.0, max_iter=1000, tol=1e-12, theta=1.0):
    r"""1-D total-variation denoising, the paper's Sec. 6.1 example.

    Minimises :math:`\tfrac12\lVert x - b\rVert^2 + \lambda\lVert
    \nabla x\rVert_1`. Here :math:`K = \nabla` is the forward
    difference, :math:`G(x) = \tfrac12\lVert x-b\rVert^2` with prox
    :math:`(x + \tau b)/(1+\tau)`, and :math:`F^*` is the indicator of
    the :math:`\ell_\infty` ball of radius :math:`\lambda`, whose prox
    is the clip to :math:`[-\lambda, \lambda]`.
    """
    b = [float(v) for v in np.atleast_1d(np.asarray(signal, dtype=float))]
    n = len(b)
    if n < 2:
        raise ValueError("tv_denoise_1d: need at least two samples")
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("tv_denoise_1d: lam must be non-negative")

    def K(x):
        return [x[i + 1] - x[i] for i in range(n - 1)]

    def Kt(y):
        # Adjoint of the forward difference, with the sign convention
        # that makes <Kx, y> = <x, Kt y> exactly.
        out = [0.0] * n
        for i in range(n - 1):
            out[i] -= y[i]
            out[i + 1] += y[i]
        return out

    def prox_fs(y, s):
        return [max(-lam, min(lam, v)) for v in y]

    def prox_g(x, t):
        return [(x[j] + t * b[j]) / (1.0 + t) for j in range(n)]

    # ||grad||^2 <= 4 for the forward difference, so ||K|| <= 2.
    res = chambolle_pock(K, Kt, prox_fs, prox_g, b, [0.0] * (n - 1),
                         theta=theta, norm_K=2.0, max_iter=max_iter, tol=tol)
    d = dict(res)
    d["lambda"] = lam
    d["signal"] = b
    d["objective"] = (0.5 * sum((d["x"][j] - b[j]) ** 2 for j in range(n))
                      + lam * sum(abs(v) for v in K(d["x"])))
    return RichResult(payload=d)


def cheatsheet():
    return ("primal: Chambolle-Pock, y = prox_{s F*}(y + s K xbar), "
            "x = prox_{t G}(x - t K* y), xbar = x + theta (x - x_prev); "
            "requires tau sigma ||K||^2 < 1.")


primal = chambolle_pock
