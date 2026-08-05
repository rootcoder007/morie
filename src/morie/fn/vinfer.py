# morie.fn -- function file (rootcoder007/morie)
"""Mean-field variational inference by coordinate ascent on the ELBO.

SOURCE.  Jordan, M.I., Ghahramani, Z., Jaakkola, T.S. and Saul, L.K.
(1999), "An Introduction to Variational Methods for Graphical Models",
*Machine Learning* 37(2):183-233, doi:10.1023/A:1007665907178.

The mean-field family factorises the variational posterior,
q(z) = prod_j q_j(z_j), and the paper's Section 3 gives the coordinate
update that maximises the lower bound with respect to one factor while
the others are held fixed:

    log q*_j(z_j) = E_{q_{-j}}[ log p(x, z) ] + const.

The bound itself is

    ELBO(q) = E_q[log p(x, z)] - E_q[log q(z)]  <=  log p(x),

and each coordinate update is guaranteed not to decrease it, which is
asserted here (``elbo_monotone``) rather than assumed.

MODEL.  The general update above is not executable without a joint, so
the joint is the conjugate univariate Normal-Gamma of the classical
worked mean-field example:

    x_n | mu, tau ~ N(mu, tau^-1),  n = 1 ... N
    mu | tau      ~ N(mu0, (lambda0 tau)^-1)
    tau           ~ Gamma(a0, b0)

with the factorisation q(mu, tau) = q(mu) q(tau).  The updates are then
in closed form, q(mu) = N(mu_N, lambda_N^-1) and q(tau) = Gamma(a_N, b_N)
with

    mu_N     = (lambda0 mu0 + N xbar) / (lambda0 + N)
    lambda_N = (lambda0 + N) E[tau]
    a_N      = a0 + (N + 1)/2
    b_N      = b0 + (1/2) E_mu[ sum_n (x_n - mu)^2 + lambda0 (mu - mu0)^2 ]
    E[tau]   = a_N / b_N.

Only this joint is implemented.  That restriction is this
implementation's scope choice, stated rather than attributed: a
general-purpose ``log_p`` callable cannot cross the Python/R boundary
and would make the two arms untestable against each other.

ANCHOR (exact, not a parity check).  Take the improper limit
lambda0 = a0 = b0 = 0.  Then mu_N = xbar exactly, and the fixed point
solves t = (N + 1) / (S + 1/t) with S = sum (x_n - xbar)^2, whose
solution is t = N / S -- the maximum likelihood precision.  Both are
checked in closed form.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["variational_inference"]

_MODELS = ("normal-gamma", "gaussian-gamma", "normalgamma")


def variational_inference(log_p="normal-gamma", q_family="meanfield", x=None,
                          mu0=0.0, lambda0=0.0, a0=0.0, b0=0.0,
                          max_iter=200, tol=1e-12):
    """Coordinate-ascent mean-field VI for the Normal-Gamma joint.

    Parameters
    ----------
    log_p : str
        Name of the joint.  Only ``"normal-gamma"`` is implemented.
    q_family : str
        Variational family.  Only ``"meanfield"`` is implemented.
    x : array-like
        The observed sample.
    mu0, lambda0, a0, b0 : float
        Normal-Gamma prior hyperparameters.  All zero is the improper
        limit used by the closed-form anchor.
    max_iter : int
        Maximum coordinate sweeps.
    tol : float
        Stop when E[tau] moves by less than this.

    Returns
    -------
    RichResult
        ``mu_n``, ``lambda_n``, ``a_n``, ``b_n``, ``e_tau``, ``e_mu``,
        ``var_mu``, ``elbo``, ``elbo_path``, ``elbo_monotone``,
        ``iterations``, ``converged``, ``n``.

    Raises
    ------
    ValueError
        Fewer than two observations, an unknown model or family, a
        negative hyperparameter, or a non-positive tolerance.

    References
    ----------
    Jordan, M.I., Ghahramani, Z., Jaakkola, T.S. and Saul, L.K. (1999).
    Machine Learning 37(2):183-233.  doi:10.1023/A:1007665907178.
    """
    if str(log_p).strip().lower() not in _MODELS:
        raise ValueError("variational_inference: only the normal-gamma joint is implemented")
    if str(q_family).strip().lower() not in ("meanfield", "mean-field"):
        raise ValueError("variational_inference: only the mean-field family is implemented")
    xv = core.vec(x)
    n = len(xv)
    if n < 2:
        raise ValueError("variational_inference: need at least two observations")
    for nm, v in (("lambda0", lambda0), ("a0", a0), ("b0", b0)):
        if float(v) < 0.0:
            raise ValueError("variational_inference: %s must be non-negative" % nm)
    if float(tol) <= 0.0:
        raise ValueError("variational_inference: tol must be positive")
    mu0 = float(mu0)
    lambda0 = float(lambda0)
    a0 = float(a0)
    b0 = float(b0)
    xbar = 0.0
    for v in xv:
        xbar += v
    xbar /= n
    ss = 0.0
    for v in xv:
        ss += (v - xbar) * (v - xbar)
    mu_n = (lambda0 * mu0 + n * xbar) / (lambda0 + n)
    a_n = a0 + 0.5 * (n + 1.0)
    e_tau = 1.0
    lam_n = (lambda0 + n) * e_tau
    b_n = b0
    path = []
    it = 0
    converged = False
    for it in range(1, int(max_iter) + 1):
        # q(mu) update -- Jordan et al. (1999) Sec. 3 applied to the joint
        lam_n = (lambda0 + n) * e_tau
        var_mu = 1.0 / lam_n
        # q(tau) update
        quad = ss + n * (xbar - mu_n) * (xbar - mu_n) + n * var_mu
        quad += lambda0 * ((mu_n - mu0) * (mu_n - mu0) + var_mu)
        b_n = b0 + 0.5 * quad
        new_tau = a_n / b_n
        # ELBO up to the additive constants that do not involve q
        e_log_tau = core.digamma(a_n) - math.log(b_n)
        elbo = 0.5 * n * e_log_tau - 0.5 * new_tau * quad
        elbo += 0.5 * e_log_tau - 0.5 * math.log(2.0 * math.pi)
        elbo += (a0 - 1.0) * e_log_tau - b0 * new_tau
        elbo -= (-0.5 * math.log(2.0 * math.pi * var_mu) - 0.5)
        elbo -= (a_n * math.log(b_n) - core.lgamma(a_n)
                 + (a_n - 1.0) * e_log_tau - a_n)
        path.append(elbo)
        if abs(new_tau - e_tau) < float(tol):
            e_tau = new_tau
            converged = True
            break
        e_tau = new_tau
    lam_n = (lambda0 + n) * e_tau
    var_mu = 1.0 / lam_n
    mono = True
    for i in range(1, len(path)):
        if path[i] < path[i - 1] - 1e-10:
            mono = False
    return RichResult(
        title="Mean-field variational inference (Normal-Gamma)",
        summary_lines=[("obs", n), ("E[tau]", e_tau), ("iterations", it)],
        payload={
            "estimate": e_tau,
            "mu_n": mu_n,
            "lambda_n": lam_n,
            "a_n": a_n,
            "b_n": b_n,
            "e_tau": e_tau,
            "e_mu": mu_n,
            "var_mu": var_mu,
            "elbo": path[-1] if path else float("nan"),
            "elbo_path": path,
            "elbo_monotone": 1.0 if mono else 0.0,
            "iterations": it,
            "converged": 1.0 if converged else 0.0,
            "n": n,
            "method": "Coordinate-ascent mean-field VI, Normal-Gamma joint (Jordan et al. 1999 Sec. 3)",
        },
    )


def cheatsheet():
    return "vinfer: mean-field variational inference by coordinate ascent (Jordan et al. 1999)"
