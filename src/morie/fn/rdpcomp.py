r"""Rényi DP of the Sampled Gaussian Mechanism, and its composition.

Mironov, I., Talwar, K., & Zhang, L. (2019) "Rényi Differential Privacy
of the Sampled Gaussian Mechanism", arXiv:1908.10530.

The Sampled Gaussian Mechanism (Definition 3) draws each record
independently with probability :math:`q`, applies :math:`f`, and adds
:math:`N(0, \sigma^2 I)`. It is the mechanism underneath DP-SGD, and the
whole point of analysing it in RDP is that subsampling *amplifies*
privacy: the same :math:`\sigma` buys a much smaller
:math:`\varepsilon` when :math:`q` is small.

Theorem 4 reduces the :math:`d`-dimensional problem, for a function of
sensitivity 1, to a one-dimensional Rényi divergence against a mixture:

.. math:: \varepsilon \le D_\alpha\big(N(0,\sigma^2) \,\big\|\,
          (1-q)N(0,\sigma^2) + qN(1,\sigma^2)\big) =: A_\alpha,

and the reverse direction :math:`B_\alpha`; Corollary 7 shows
:math:`A_\alpha \ge B_\alpha` for :math:`\alpha \ge 1`, so
:math:`A_\alpha` is the binding one and the only one that need be
computed.

For **integer** :math:`\alpha` the paper's Case I expands the mixture
binomially and integrates each term in closed form. Since

.. math:: E_{z \sim \mu_0}\Big[\big(\tfrac{\mu_1(z)}{\mu_0(z)}\big)^k\Big]
          = \exp\!\Big(\frac{k^2 - k}{2\sigma^2}\Big),

the whole expectation is a finite sum:

.. math:: A_\alpha = \sum_{k=0}^{\alpha} \binom{\alpha}{k}
          (1-q)^{\alpha-k} q^{k}
          \exp\!\Big(\frac{k(k-1)}{2\sigma^2}\Big),
          \qquad
          \varepsilon(\alpha) = \frac{\log A_\alpha}{\alpha - 1}.

That is exact, not a bound on a bound, which is why integer orders are
the ones worth using.

Sanity that falls out of the formula, and is asserted in the anchors: at
:math:`q = 1` every term but :math:`k = \alpha` vanishes and
:math:`\varepsilon(\alpha) = \alpha/(2\sigma^2)` -- exactly Corollary 3
of Mironov (2017), the unsubsampled Gaussian. Two papers, one number.

Fractional :math:`\alpha` is the paper's Case II, an infinite series
split at the inflection point :math:`z_1 = \tfrac12 + \sigma^2
\ln(q^{-1}-1)`. It is **not** implemented here; a fractional order
raises rather than silently returning the integer formula evaluated
off-support, which would be wrong and quiet about it.

Composition is Proposition 1 of Mironov (2017): RDP curves add, so
:math:`T` steps of the same sampled Gaussian cost
:math:`T \cdot \varepsilon(\alpha)` at each order. That is the DP-SGD
accountant: track the curve across steps, convert once at the end.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["rdp_sampled_gaussian", "rdp_compose", "rdpcomp"]


def rdp_sampled_gaussian(alpha, q, sigma):
    r"""Exact RDP of the sampled Gaussian at an integer order.

    Parameters
    ----------
    alpha : int
        Rényi order, an integer strictly greater than 1.
    q : float
        Sampling rate in (0, 1].
    sigma : float
        Noise multiplier, for a sensitivity-1 function.
    """
    a = float(alpha)
    if a != math.floor(a):
        raise ValueError(
            "rdp_sampled_gaussian: alpha must be an integer -- the closed "
            "form is the paper's Case I binomial expansion. Fractional "
            "orders are Case II, an infinite series, which is not "
            "implemented; got %r" % (alpha,))
    ai = int(a)
    if ai <= 1:
        raise ValueError(
            "rdp_sampled_gaussian: alpha must exceed 1, got %r" % (alpha,))
    qq = float(q)
    if not (0.0 < qq <= 1.0):
        raise ValueError(
            "rdp_sampled_gaussian: q must lie in (0, 1], got %r" % (q,))
    s = float(sigma)
    if s <= 0.0:
        raise ValueError(
            "rdp_sampled_gaussian: sigma must be positive, got %r" % (s,))

    # A_alpha = sum_k C(a,k) (1-q)^(a-k) q^k exp(k(k-1)/(2 sigma^2)).
    # Summed in log space: the exp term reaches exp(a^2/(2 sigma^2)),
    # which overflows for the large alpha used in tight accounting.
    log_terms = []
    for k in range(ai + 1):
        lg = (math.lgamma(ai + 1.0) - math.lgamma(k + 1.0)
              - math.lgamma(ai - k + 1.0))
        if qq == 1.0:
            # log(0) for every k < alpha; only k = alpha survives.
            if k < ai:
                continue
            lt = lg + k * math.log(qq)
        else:
            if k == 0:
                lt = lg + (ai - k) * math.log1p(-qq)
            else:
                lt = lg + (ai - k) * math.log1p(-qq) + k * math.log(qq)
        lt += k * (k - 1.0) / (2.0 * s * s)
        log_terms.append(lt)

    hi = max(log_terms)
    acc = sum(math.exp(t - hi) for t in log_terms)
    log_A = hi + math.log(acc)
    return log_A / (a - 1.0)


def rdp_compose(alpha, q, sigma, steps=1):
    """Proposition 1: identical mechanisms add their RDP curves."""
    t = int(steps)
    if t < 1:
        raise ValueError("rdp_compose: steps must be at least 1, got %r"
                         % (steps,))
    return t * rdp_sampled_gaussian(alpha, q, sigma)


def rdpcomp(q, sigma, alpha=None, steps=1, delta=None):
    r"""RDP curve of ``steps`` sampled Gaussians, optionally converted.

    Parameters
    ----------
    q : float
        Sampling rate.
    sigma : float
        Noise multiplier.
    alpha : int or array-like of int, optional
        Orders at which to evaluate. Defaults to the accountant's usual
        integer ladder 2..64.
    steps : int
        Number of composed steps.
    delta : float, optional
        When given, also convert via Mironov (2017) Proposition 3,
        :math:`\varepsilon = \varepsilon_R + \log(1/\delta)/(\alpha-1)`,
        minimised over the supplied orders.

    Returns
    -------
    RichResult
        ``rdp_epsilons`` is the composed curve; with ``delta``,
        ``estimate`` is the converted epsilon and ``best_alpha`` the
        order attaining it.
    """
    if alpha is None:
        orders = list(range(2, 65))
    else:
        av = np.atleast_1d(np.asarray(alpha, dtype=float))
        orders = [int(v) for v in av]
    if not orders:
        raise ValueError("rdpcomp: alpha must hold at least one order")

    t = int(steps)
    curve = [rdp_compose(a, q, sigma, steps=t) for a in orders]

    payload = {
        "estimate": float(min(curve)),
        "rdp_epsilons": curve,
        "alphas": [float(a) for a in orders],
        "q": float(q),
        "sigma": float(sigma),
        "steps": t,
        "method": "RDP of the Sampled Gaussian Mechanism "
                  "(Mironov, Talwar & Zhang 2019, Thm 4 / Case I); "
                  "composition by Mironov (2017) Prop 1",
    }
    if delta is not None:
        d = float(delta)
        if not (0.0 < d < 1.0):
            raise ValueError(
                "rdpcomp: delta must lie strictly in (0, 1), got %r" % (d,))
        log_inv = math.log(1.0 / d)
        eps = [e + log_inv / (a - 1.0) for a, e in zip(orders, curve)]
        best = 0
        for i in range(1, len(eps)):
            if eps[i] < eps[best]:
                best = i
        payload["estimate"] = float(eps[best])
        payload["epsilon"] = float(eps[best])
        payload["best_alpha"] = float(orders[best])
        payload["epsilons"] = eps
        payload["delta"] = d
    return RichResult(payload=payload)


def cheatsheet():
    return ("rdpcomp: sampled Gaussian RDP, A_alpha = sum_k C(a,k) "
            "(1-q)^(a-k) q^k exp(k(k-1)/(2 sigma^2)), eps = log(A)/(a-1) "
            "(MTZ 2019 Case I); integer alpha only; composes by addition.")

# public names resolved by fn/_lazy_map.json
rdp_subsampled_composition = rdp_sampled_gaussian
