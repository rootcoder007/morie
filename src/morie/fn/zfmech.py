# morie.fn -- function file (rootcoder007/morie)
r"""Zero-concentrated differential privacy and the Gaussian mechanism.

**The definition.** A mechanism :math:`M` is :math:`(\xi,\rho)`-zCDP if
for every pair of neighbouring inputs and *every* :math:`\alpha \in
(1,\infty)`,

.. math:: D_\alpha\big(M(x)\,\|\,M(x')\big) \le \xi + \rho\alpha ,

with :math:`D_\alpha` the Rényi divergence. :math:`\rho`-zCDP is the
:math:`\xi = 0` case. The linear-in-:math:`\alpha` bound is the whole
point: pure :math:`\varepsilon`-DP demands a bound uniform in
:math:`\alpha`, approximate DP allows a failure probability, and zCDP
sits between them by letting the bound grow, but only linearly.

**Why the Gaussian mechanism is the prototype.** For a query of
sensitivity :math:`\Delta` released as :math:`N(q(x), \sigma^2)`, the
Rényi divergence between two neighbouring outputs is *exactly*
:math:`\alpha\Delta^2/(2\sigma^2)` -- linear in :math:`\alpha` with no
slack at all. So the mechanism is :math:`(\Delta^2/2\sigma^2)`-zCDP
and the definition is tight for it simultaneously at every
:math:`\alpha`. ``renyi_divergence_gaussian`` returns that closed form
and the anchor checks it against a numerical integral of the
divergence, which is what would catch a factor of two.

**The three conversions, all kept.**

``from_pure_dp``
    :math:`\varepsilon`-DP implies :math:`\tfrac12\varepsilon^2`-zCDP
    (Proposition 1.4).
``to_approx_dp``
    :math:`\rho`-zCDP implies
    :math:`(\rho + 2\sqrt{\rho\log(1/\delta)},\ \delta)`-DP for every
    :math:`\delta > 0` (Proposition 1.3).
``group_privacy``
    :math:`\rho`-zCDP gives :math:`k^2\rho`-zCDP for groups of size
    :math:`k` -- quadratic, not linear, and exactly tight
    (Proposition 1.9).

The conversions do not round-trip, and the module does not pretend
they do: going :math:`\varepsilon \to \rho \to (\varepsilon',\delta)`
lands at a different :math:`\varepsilon'`, and ``round_trip`` reports
the loss instead of hiding it.

**Composition and post-processing.** :math:`\rho` adds under
composition (Lemma 1.7) and is untouched by post-processing (Lemma
1.8). Composition being *additive in* :math:`\rho` is what makes zCDP
worth using: :math:`k` Gaussian releases at :math:`\rho` each cost
:math:`k\rho`, whereas the same accounting under approximate DP needs
an advanced-composition theorem and a :math:`\delta` budget.

References
----------
Bun, M. & Steinke, T. (2016) "Concentrated Differential Privacy:
Simplifications, Extensions, and Lower Bounds", in *Theory of
Cryptography (TCC 2016-B)*, Lecture Notes in Computer Science 9985,
635-658, doi:10.1007/978-3-662-53641-4_24 (arXiv:1605.02065).
Definition 1.1 for :math:`(\xi,\rho)`-zCDP and the equivalent
moment-generating form (2); Definition 1.2 for the privacy loss random
variable; Definition 1.5 for sensitivity; Proposition 1.6 for the
Gaussian mechanism being :math:`(\Delta^2/2\sigma^2)`-zCDP and the
remark that the defining inequality is exactly tight for it at every
:math:`\alpha`; Proposition 1.3 and Proposition 1.4 for the
conversions to and from approximate and pure differential privacy;
Lemma 1.7 for composition; Lemma 1.8 for post-processing; and
Proposition 1.9 for the :math:`k^2\rho` group privacy bound.

Dwork, C., Kenthapadi, K., McSherry, F., Mironov, I. & Naor, M. (2006)
"Our Data, Ourselves: Privacy via Distributed Noise Generation", in
*Advances in Cryptology (EUROCRYPT 2006)*, Lecture Notes in Computer
Science 4004, 486-503, doi:10.1007/11761679_29, for the Gaussian
mechanism itself.
"""

import math

from . import _array_core as np
from . import survrsf as _rsf
from ._richresult import RichResult

__all__ = ["renyi_divergence_gaussian", "zcdp_of_gaussian",
           "sigma_for_rho", "gaussian_mechanism", "compose",
           "group_privacy", "to_approx_dp", "from_pure_dp",
           "round_trip", "postprocessing"]


def _check_rho(rho):
    if rho <= 0.0:
        raise ValueError("zfmech: rho must be positive, got %r" % rho)


def renyi_divergence_gaussian(mu0, mu1, sigma, alpha):
    r""":math:`D_\alpha` between two Gaussians of equal variance.

    Equals :math:`\alpha(\mu_0-\mu_1)^2/(2\sigma^2)` -- linear in
    :math:`\alpha`, which is what makes the Gaussian mechanism sit
    exactly on the zCDP boundary rather than inside it.
    """
    if sigma <= 0.0:
        raise ValueError("zfmech: sigma must be positive")
    if not 1.0 < float(alpha) < float("inf"):
        raise ValueError("zfmech: alpha must lie in (1, inf), got %r"
                         % alpha)
    d = float(mu0) - float(mu1)
    return float(alpha) * d * d / (2.0 * sigma * sigma)


def zcdp_of_gaussian(sensitivity, sigma):
    r"""Proposition 1.6: :math:`\rho = \Delta^2/(2\sigma^2)`."""
    if sigma <= 0.0:
        raise ValueError("zfmech: sigma must be positive")
    if sensitivity < 0.0:
        raise ValueError("zfmech: sensitivity cannot be negative")
    d = float(sensitivity)
    return d * d / (2.0 * sigma * sigma)


def sigma_for_rho(sensitivity, rho):
    r"""The noise scale that buys a given :math:`\rho`."""
    _check_rho(rho)
    if sensitivity < 0.0:
        raise ValueError("zfmech: sensitivity cannot be negative")
    return float(sensitivity) / math.sqrt(2.0 * float(rho))


def gaussian_mechanism(value, sensitivity, rho, seed=0, n=1):
    r"""Release :math:`q(x)` under :math:`\rho`-zCDP."""
    sigma = sigma_for_rho(sensitivity, rho)
    rng = _rsf._Rng(seed)
    out = []
    for _ in range(int(n)):
        u1 = max(rng.next(), 1e-12)
        u2 = rng.next()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(
            2.0 * math.pi * u2)
        out.append(float(value) + sigma * z)
    return {"release": out if n > 1 else out[0], "sigma": sigma,
            "rho": float(rho), "sensitivity": float(sensitivity)}


def compose(rhos):
    r"""Lemma 1.7: :math:`\rho` simply adds."""
    rs = [float(v) for v in rhos]
    if any(v < 0.0 for v in rs):
        raise ValueError("zfmech: every rho must be non-negative")
    return {"rho": sum(rs), "k": len(rs),
            "note": "additive, with no delta budget and no advanced "
                    "composition theorem"}


def group_privacy(rho, k):
    r"""Proposition 1.9: groups of size :math:`k` cost
    :math:`k^2\rho`."""
    _check_rho(rho)
    k = int(k)
    if k < 1:
        raise ValueError("zfmech: the group size must be at least 1")
    return {"rho": float(k) ** 2 * float(rho), "k": k,
            "growth": "quadratic in k, and exactly tight"}


def to_approx_dp(rho, delta):
    r"""Proposition 1.3: :math:`(\rho + 2\sqrt{\rho\log(1/\delta)},
    \delta)`-DP."""
    _check_rho(rho)
    d = float(delta)
    if not 0.0 < d < 1.0:
        raise ValueError("zfmech: delta must lie in (0, 1), got %r"
                         % delta)
    eps = float(rho) + 2.0 * math.sqrt(float(rho) * math.log(1.0 / d))
    return {"epsilon": eps, "delta": d, "rho": float(rho)}


def from_pure_dp(epsilon):
    r"""Proposition 1.4: :math:`\varepsilon`-DP is
    :math:`\tfrac12\varepsilon^2`-zCDP."""
    e = float(epsilon)
    if e < 0.0:
        raise ValueError("zfmech: epsilon cannot be negative")
    return {"rho": 0.5 * e * e, "epsilon": e}


def round_trip(epsilon, delta):
    r"""What the two conversions cost when chained.

    :math:`\varepsilon \to \rho \to (\varepsilon', \delta)` does not
    return :math:`\varepsilon`; the gap is reported rather than
    quietly absorbed.
    """
    rho = from_pure_dp(epsilon)["rho"]
    back = to_approx_dp(rho, delta) if rho > 0.0 else {"epsilon": 0.0}
    return {"epsilon_in": float(epsilon), "rho": rho,
            "epsilon_out": back["epsilon"],
            "inflation": back["epsilon"] - float(epsilon),
            "delta": float(delta)}


def postprocessing(rho):
    r"""Lemma 1.8: any function of the output keeps the same
    :math:`\rho`."""
    _check_rho(rho)
    return {"rho": float(rho),
            "note": "invariant -- unlike Dwork and Rothblum's mCDP, "
                    "which is not closed under post-processing"}


def cheatsheet():
    return ("zfmech: rho-zCDP means D_alpha(M(x)||M(x')) <= rho alpha "
            "for EVERY alpha > 1. Gaussian mechanism: rho = "
            "Delta^2/(2 sigma^2), and the bound is exactly tight. "
            "Composition adds rho; post-processing leaves it alone; "
            "groups of size k cost k^2 rho, not k rho. Conversions: "
            "eps-DP -> eps^2/2 zCDP, and rho-zCDP -> (rho + "
            "2 sqrt(rho log(1/delta)), delta)-DP. Chaining them does "
            "NOT return the original epsilon.")


# compact alias per ledger/NAMING.md
zero_concentrated_dp = zcdp_of_gaussian
