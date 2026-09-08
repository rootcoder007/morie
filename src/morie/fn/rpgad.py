r"""Rényi differential privacy and its conversion to :math:`(\varepsilon,
\delta)`-DP.

Mironov, I. (2017) "Rényi Differential Privacy", *30th IEEE Computer
Security Foundations Symposium (CSF)*, 263-275.

A mechanism is :math:`(\alpha, \varepsilon)`-RDP when the Rényi
divergence of order :math:`\alpha` between its outputs on adjacent
inputs is at most :math:`\varepsilon`. The reason to bother: RDP
composes by *addition* of the :math:`\varepsilon` at fixed
:math:`\alpha` (Proposition 1), so a long composition can be tracked
exactly along a curve and converted to :math:`(\varepsilon, \delta)`
once, at the end -- which is much tighter than converting at every step.

Proposition 3 is the conversion:

.. math:: (\alpha, \varepsilon)\text{-RDP} \Longrightarrow
          \Big(\varepsilon + \frac{\log(1/\delta)}{\alpha - 1},
          \delta\Big)\text{-DP}
          \qquad \text{for any } 0 < \delta < 1.

Because that bound holds *for every* :math:`\alpha` at which the
mechanism's RDP curve is known, the useful number is the minimum over
the available orders. Pass a vector of ``alpha`` and matching
``epsilon_R`` and this returns that minimum along with the order that
achieved it.

Routes
------
``epsilon_R`` may be given directly, or derived from a named mechanism
via ``mechanism``:

``"gaussian"`` (Corollary 3)
    A sensitivity-1 Gaussian mechanism :math:`G_\sigma f` satisfies
    :math:`(\alpha, \alpha/(2\sigma^2))`-RDP; the curve is a straight
    line in :math:`\alpha`. With sensitivity :math:`\Delta` it is
    :math:`\alpha \Delta^2/(2\sigma^2)`, from Proposition 7's
    :math:`D_\alpha(N(0,\sigma^2) \Vert N(\mu,\sigma^2)) =
    \alpha\mu^2/(2\sigma^2)`.

``"laplace"`` (Corollary 2)
    A sensitivity-1 Laplace mechanism :math:`L_\lambda f` satisfies
    :math:`(\alpha, \varepsilon_\alpha)`-RDP with

    .. math:: \varepsilon_\alpha = \frac{1}{\alpha - 1}
        \log\!\Big[\frac{\alpha}{2\alpha - 1}
        e^{(\alpha-1)/\lambda}
        + \frac{\alpha - 1}{2\alpha - 1} e^{-\alpha/\lambda}\Big].

``n_compositions`` multiplies the curve, which is Proposition 1's
additivity for identical mechanisms.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["rdp_to_eps_delta", "rdp_gaussian", "rdp_laplace", "rpgad"]

_MECHANISMS = ("gaussian", "laplace")


def rdp_gaussian(alpha, sigma, sensitivity=1.0):
    r"""Corollary 3: :math:`\varepsilon_\alpha = \alpha\Delta^2/(2\sigma^2)`."""
    a = float(alpha)
    s = float(sigma)
    d = float(sensitivity)
    if s <= 0.0:
        raise ValueError("rdp_gaussian: sigma must be positive, got %r" % (s,))
    if d < 0.0:
        raise ValueError(
            "rdp_gaussian: sensitivity must be non-negative, got %r" % (d,))
    return a * d * d / (2.0 * s * s)


def rdp_laplace(alpha, lam, sensitivity=1.0):
    r"""Corollary 2, the Laplace RDP curve at order ``alpha``."""
    a = float(alpha)
    lm = float(lam) / float(sensitivity) if float(sensitivity) != 0 else \
        float("inf")
    if a <= 1.0:
        raise ValueError(
            "rdp_laplace: alpha must exceed 1, got %r" % (alpha,))
    if lm <= 0.0:
        raise ValueError("rdp_laplace: lambda must be positive, got %r" % (lam,))
    # Evaluated in log space. The first term carries exp((a-1)/lam),
    # which overflows a float for large alpha -- and large alpha is
    # exactly where the curve is read, since eps_alpha -> 1/lam (the
    # pure-DP guarantee) as alpha -> infinity. log-sum-exp keeps that
    # limit reachable instead of raising.
    log_t1 = math.log(a / (2.0 * a - 1.0)) + (a - 1.0) / lm
    log_t2 = math.log((a - 1.0) / (2.0 * a - 1.0)) - a / lm
    hi = log_t1 if log_t1 > log_t2 else log_t2
    lo = log_t2 if log_t1 > log_t2 else log_t1
    return (hi + math.log1p(math.exp(lo - hi))) / (a - 1.0)


def rdp_to_eps_delta(alpha, epsilon_R=None, delta=1e-5, mechanism=None,
                     sigma=None, lam=None, sensitivity=1.0,
                     n_compositions=1):
    r"""Convert an RDP curve to :math:`(\varepsilon, \delta)`-DP.

    Parameters
    ----------
    alpha : float or array-like
        One Rényi order, or several. Every order must exceed 1;
        Proposition 3 divides by ``alpha - 1``.
    epsilon_R : float or array-like, optional
        The RDP epsilon at each order. Omit it and give ``mechanism``
        instead to use one of the paper's closed-form curves.
    delta : float
        The target delta, strictly inside (0, 1).
    mechanism : {"gaussian", "laplace"}, optional
        Derive ``epsilon_R`` from Corollary 3 or Corollary 2.
    sigma : float
        Noise scale, ``mechanism="gaussian"``.
    lam : float
        Noise scale, ``mechanism="laplace"``.
    sensitivity : float
        Query sensitivity; the guarantee is only as good as this bound.
    n_compositions : int
        Number of identical mechanisms composed. Proposition 1 makes
        the RDP curves add, so this scales the curve.

    Returns
    -------
    RichResult
        ``estimate`` is the best (smallest) epsilon over the supplied
        orders; ``best_alpha`` is the order that achieved it.
    """
    av = np.atleast_1d(np.asarray(alpha, dtype=float))
    orders = [float(v) for v in av]
    if not orders:
        raise ValueError("rdp_to_eps_delta: alpha must hold at least one order")
    for a in orders:
        if a <= 1.0:
            raise ValueError(
                "rdp_to_eps_delta: every alpha must exceed 1 (Proposition 3 "
                "divides by alpha - 1), got %r" % (a,))
    d = float(delta)
    if not (0.0 < d < 1.0):
        raise ValueError(
            "rdp_to_eps_delta: delta must lie strictly in (0, 1), got %r"
            % (d,))
    k = int(n_compositions)
    if k < 1:
        raise ValueError(
            "rdp_to_eps_delta: n_compositions must be at least 1, got %r"
            % (n_compositions,))

    if epsilon_R is not None:
        ev = np.atleast_1d(np.asarray(epsilon_R, dtype=float))
        eps_r = [float(v) for v in ev]
        if len(eps_r) == 1 and len(orders) > 1:
            eps_r = eps_r * len(orders)
        if len(eps_r) != len(orders):
            raise ValueError(
                "rdp_to_eps_delta: got %d alpha but %d epsilon_R"
                % (len(orders), len(eps_r)))
        mech = "supplied"
    else:
        if mechanism is None:
            raise ValueError(
                "rdp_to_eps_delta: give either epsilon_R or a mechanism")
        mech = str(mechanism).lower()
        if mech not in _MECHANISMS:
            raise ValueError(
                "rdp_to_eps_delta: mechanism must be one of %s, got %r"
                % (", ".join(_MECHANISMS), mechanism))
        if mech == "gaussian":
            if sigma is None:
                raise ValueError(
                    "rdp_to_eps_delta: mechanism='gaussian' needs sigma")
            eps_r = [rdp_gaussian(a, sigma, sensitivity) for a in orders]
        else:
            if lam is None:
                raise ValueError(
                    "rdp_to_eps_delta: mechanism='laplace' needs lam")
            eps_r = [rdp_laplace(a, lam, sensitivity) for a in orders]

    # Proposition 1: composing k identical mechanisms adds their curves.
    eps_r = [k * e for e in eps_r]

    log_inv_delta = math.log(1.0 / d)
    eps = [e + log_inv_delta / (a - 1.0) for a, e in zip(orders, eps_r)]

    best = 0
    for i in range(1, len(eps)):
        if eps[i] < eps[best]:
            best = i

    return RichResult(payload={
        "estimate": float(eps[best]),
        "epsilon": float(eps[best]),
        "best_alpha": float(orders[best]),
        "epsilons": eps,
        "alphas": orders,
        "rdp_epsilons": eps_r,
        "delta": d,
        "mechanism": mech,
        "sensitivity": float(sensitivity),
        "n_compositions": k,
        "method": "RDP -> (eps, delta)-DP, Mironov (2017) Proposition 3",
    })


def cheatsheet():
    return ("rpgad: Mironov 2017 Prop 3, eps = eps_R + log(1/delta)/(alpha-1), "
            "minimised over alpha; curves gaussian a*D^2/(2 sigma^2) (Cor 3) "
            "and laplace (Cor 2); composition adds curves (Prop 1).")


rpgad = rdp_to_eps_delta

# public names resolved by fn/_lazy_map.json
rdptoepsdelta = rdp_to_eps_delta
