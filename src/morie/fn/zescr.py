"""Spatial cure rate model"""

import math

from ._containers import SpatialResult
from ._qpcore import inverse
from .zesfr import _newton, _prep


def spatial_cure_rate(time, event, X, region, W, *, tau=1.0, tol=1e-10):
    r"""Promotion-time (bounded cumulative hazard) cure model with ICAR regional effects, penalised likelihood.

    Population survival ``S_pop(t) = exp(-theta_i F(t))`` with ``theta_i =
    exp(x_i' beta + u_r(i))`` the mean number of latent causes and ``F(t) =
    1 - exp(-e^c t^rho)`` a Weibull promotion-time distribution; the cure
    fraction is ``exp(-theta_i)`` (Yakovlev and Tsodikov 1996; Chen, Ibrahim
    and Sinha 1999). Regional effects ``u`` carry the ICAR penalty ``(tau/2)
    u'(D - W)u`` (Cooner, Banerjee, Carlin and Sinha 2007 in a Bayesian
    setting). The penalised log-likelihood ``sum_i [d_i (eta_i + log f(t_i))
    - theta_i F(t_i)] - (tau/2) u'Qu`` is maximised over ``(log rho, c, beta,
    u)`` by Newton-Raphson on its analytic gradient; standard errors from
    the inverse negative Hessian. ``statistic`` is the penalised
    log-likelihood; ``cure_fraction`` is ``exp(-theta_i)`` per subject.

    References
    ----------
    Chen, M.-H., Ibrahim, J. G. and Sinha, D. (1999). A new Bayesian model
    for survival data with a surviving fraction. *JASA* 94, 909-919.
    Cooner, F., Banerjee, S., Carlin, B. P. and Sinha, D. (2007). Flexible
    cure rate modeling under latent activation schemes. *JASA* 102, 560-572.

    Examples
    --------
    >>> import math
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> reg = [i % 3 for i in range(36)]
    >>> x = [[math.sin(i)] for i in range(36)]
    >>> t = [5.0 if i % 4 == 0 else 0.2 + (i * 7 % 11) / 8 for i in range(36)]
    >>> ev = [0 if (i % 4 == 0 or i % 9 == 1) else 1 for i in range(36)]
    >>> r = spatial_cure_rate(t, ev, x, reg, W)
    >>> round(r.extra["shape"], 8)
    2.38865845
    """
    t, dl, Xm, reg, Q = _prep(time, event, X, region, W)
    n, p, R = len(t), len(Xm[0]), len(Q)
    lt = [math.log(v) for v in t]

    def parts(th):
        a, c, b, u = th[0], th[1], th[2 : 2 + p], th[2 + p :]
        rho = math.exp(a)
        eta = [sum(Xm[i][k] * b[k] for k in range(p)) + u[reg[i]] for i in range(n)]
        G = [math.exp(c + rho * lt[i]) for i in range(n)]
        return a, c, rho, u, eta, G

    def obj(th):
        a, c, rho, u, eta, G = parts(th)
        ll = sum(
            dl[i] * (eta[i] + a + c + (rho - 1.0) * lt[i] - G[i]) - math.exp(eta[i]) * (1.0 - math.exp(-G[i]))
            for i in range(n)
        )
        return ll - 0.5 * tau * sum(u[i] * Q[i][j] * u[j] for i in range(R) for j in range(R))

    def grad(th):
        a, c, rho, u, eta, G = parts(th)
        th_ = [math.exp(v) for v in eta]
        F = [1.0 - math.exp(-g) for g in G]
        dF = [math.exp(-g) * g for g in G]  # dF/dc; dF/da = dF * rho log t
        ga = sum(dl[i] * (1.0 + rho * lt[i] - G[i] * rho * lt[i]) - th_[i] * dF[i] * rho * lt[i] for i in range(n))
        gc = sum(dl[i] * (1.0 - G[i]) - th_[i] * dF[i] for i in range(n))
        ge = [dl[i] - th_[i] * F[i] for i in range(n)]
        gb = [sum(ge[i] * Xm[i][k] for i in range(n)) for k in range(p)]
        gu = [
            sum(ge[i] for i in range(n) if reg[i] == r) - tau * sum(Q[r][j] * u[j] for j in range(R)) for r in range(R)
        ]
        return [ga, gc] + gb + gu

    x, f, H, it = _newton(grad, obj, [0.0] * (2 + p + R), tol=tol)
    V = inverse([[-v for v in r] for r in H])
    se = [math.sqrt(V[i][i]) if V[i][i] > 0 else float("nan") for i in range(len(x))]
    b, u = x[2 : 2 + p], x[2 + p :]
    cure = [math.exp(-math.exp(sum(Xm[i][k] * b[k] for k in range(p)) + u[reg[i]])) for i in range(n)]
    return SpatialResult(
        name="zescr",
        statistic=f,
        extra={
            "shape": math.exp(x[0]),
            "log_scale": x[1],
            "coefficients": b,
            "regional_effects": u,
            "se": se,
            "cure_fraction": cure,
            "iterations": it,
            "tau": tau,
        },
    )


spat = spatial_cure_rate


def cheatsheet() -> str:
    return "spatial_cure_rate(time, event, X, region, W, tau) -> promotion-time cure model with ICAR effects."
