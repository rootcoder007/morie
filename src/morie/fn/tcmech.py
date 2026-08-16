"""Truncated concentrated differential privacy for an unbounded query.

A query with no bound on its output has no sensitivity, and a mechanism
with no sensitivity has no privacy guarantee at all. Clipping is what
buys one: hold every contribution inside [-C, C] and a single record can
move the answer by at most the amount C allows. The clipping is not a
detail, it IS the privacy argument, so the number of records it actually
bound is reported -- a C so loose that nothing was clipped is a C that
was chosen to look harmless.

The accounting is concentrated differential privacy, stated through the
Renyi divergence between the mechanism's output on neighbouring inputs:

    (xi, rho)-zCDP:   D_alpha(M(x) || M(x')) <= xi + rho alpha
                      for all alpha in (1, infinity)

TRUNCATED CDP restricts that quantifier to alpha in (1, omega). Bounding
the divergence only up to a finite order is a weaker promise, and it is
the right one for mechanisms whose divergence is well behaved near one
and blows up further out; the price is that the conversion to
(epsilon, delta) can no longer optimise alpha freely, and this module
shows that price rather than hiding it.

Three published facts do all the work:

  Gaussian mechanism   releasing N(q(x), sigma^2) for a sensitivity-Delta
                       query satisfies (Delta^2 / 2 sigma^2)-zCDP, and
                       the bound is tight at every alpha.
  conversion           rho-zCDP implies
                       (rho + 2 sqrt(rho log(1/delta)), delta)-DP.
  pure DP              epsilon-DP implies (epsilon^2 / 2)-zCDP.

Running the conversion backwards gives the rho a target (epsilon, delta)
can afford, and the Gaussian proposition turns that into a noise scale.
The inversion is a fixed-step bisection on a monotone function rather
than a closed form, so it takes the same number of steps on every input
and cannot iterate differently in two implementations.

With a finite omega the free-alpha conversion is unavailable whenever
the optimising alpha would exceed it, and the module falls back to the
fixed-order Renyi bound, epsilon = rho omega + log(1/delta)/(omega - 1).
That branch is Mironov's RDP conversion at a fixed order, cited below;
it is not from the tCDP paper and is labelled here rather than passed
off as such.

What this module does NOT implement is the sinh-normal mechanism, which
is the reason tCDP was introduced -- its parameterisation is not
something this implementation could reproduce faithfully from a
description, so it is absent rather than guessed at.

References
  Bun, M., Dwork, C., Rothblum, G.N. and Steinke, T. (2018) "Composable
    and versatile privacy via truncated CDP." Proceedings of the 50th
    Annual ACM SIGACT Symposium on Theory of Computing (STOC), 74-86.
    doi:10.1145/3188745.3188946. Truncated CDP.
  Bun, M. and Steinke, T. (2016) "Concentrated differential privacy:
    simplifications, extensions, and lower bounds." Theory of
    Cryptography Conference (TCC), 635-658. arXiv:1605.02065.
    Definition 1.1 (zCDP), Proposition 1.3 (the conversion),
    Proposition 1.4 (pure DP to zCDP), Definition 1.5 (sensitivity) and
    Proposition 1.6 (the Gaussian mechanism) are quoted above.
  Mironov, I. (2017) "Renyi differential privacy." IEEE Computer
    Security Foundations Symposium (CSF), 263-275. The fixed-order
    conversion used on the truncated branch.
  Dwork, C. and Rothblum, G.N. (2016) "Concentrated differential
    privacy." arXiv:1603.01887.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["tcmech", "truncated_cdp_mechanism", "eps_from_rho",
           "rho_from_eps", "epsilon_floor", "gaussian_sigma",
           "rho_from_sigma", "rho_from_pure_dp", "compose", "cheatsheet"]


def eps_from_rho(rho, delta, omega=None):
    """The (epsilon, delta) guarantee a rho-tCDP mechanism gives.

    With no truncation this is Bun and Steinke's Proposition 1.3,
    exactly. With a finite omega the optimising order may lie outside
    the range where the divergence is bounded, and the fixed-order
    Renyi conversion is used instead -- which is strictly worse, as it
    must be, because a weaker premise cannot give a stronger promise.
    """
    rho = float(rho)
    delta = float(delta)
    if rho < 0.0:
        raise ValueError("rho cannot be negative")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly inside (0, 1)")
    l = math.log(1.0 / delta)
    if rho == 0.0:
        return 0.0
    free = rho + 2.0 * math.sqrt(rho * l)
    if omega is None:
        return free
    w = float(omega)
    if w <= 1.0:
        raise ValueError("the truncation order must exceed one")
    # The order that minimises the fixed-order bound is 1 + sqrt(l/rho);
    # inside the truncation it reproduces the free conversion, outside
    # it the best available order is omega itself.
    star = 1.0 + math.sqrt(l / rho)
    if star <= w:
        return free
    return rho * w + l / (w - 1.0)


def epsilon_floor(delta, omega=None):
    """The epsilon a truncated guarantee cannot get below, at any rho.

    On the fixed-order branch the bound is rho omega + log(1/delta) /
    (omega - 1), and the second term does not depend on rho at all. So a
    tight truncation puts a FLOOR under epsilon that no amount of noise
    removes -- spending less budget shrinks the first term towards zero
    and leaves the second exactly where it was.

    This is a real property of truncating the divergence, not a defect
    of the implementation, and it is the thing that has to be checked
    before inverting: a caller asking for an epsilon below the floor is
    asking for something the definition cannot supply, and the honest
    answer is to say which floor and why.
    """
    if omega is None:
        return 0.0
    w = float(omega)
    if w <= 1.0:
        raise ValueError("the truncation order must exceed one")
    return math.log(1.0 / float(delta)) / (w - 1.0)


def rho_from_eps(epsilon, delta, omega=None, iters=200):
    """The largest rho whose guarantee still meets a target (eps, delta).

    A fixed-step bisection on a strictly increasing function, so it runs
    the same number of steps whatever the inputs. The upper bracket is
    widened by doubling first, which is a fixed schedule and not a
    search.
    """
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    fl = epsilon_floor(delta, omega)
    if epsilon <= fl:
        raise ValueError(
            "no rho can reach epsilon %r at delta %r with truncation "
            "%r: the fixed-order bound has an irreducible term "
            "log(1/delta)/(omega - 1) = %r that does not depend on rho. "
            "Loosen the truncation, loosen delta, or ask for a larger "
            "epsilon." % (epsilon, delta, omega, fl))
    lo = 0.0
    hi = 1.0
    for _ in range(60):
        if eps_from_rho(hi, delta, omega) >= epsilon:
            break
        hi = hi * 2.0
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if eps_from_rho(mid, delta, omega) <= epsilon:
            lo = mid
        else:
            hi = mid
    return lo


def gaussian_sigma(sensitivity, rho):
    """The noise scale a rho budget buys, from Proposition 1.6.

    rho = Delta^2 / (2 sigma^2), so sigma = Delta / sqrt(2 rho).
    """
    s = float(sensitivity)
    r = float(rho)
    if s <= 0.0:
        raise ValueError("the sensitivity must be positive")
    if r <= 0.0:
        raise ValueError("rho must be positive to release anything")
    return s / math.sqrt(2.0 * r)


def rho_from_sigma(sensitivity, sigma):
    """Proposition 1.6 the other way round."""
    s = float(sensitivity)
    g = float(sigma)
    if g <= 0.0:
        raise ValueError("the noise scale must be positive")
    return s * s / (2.0 * g * g)


def rho_from_pure_dp(epsilon):
    """Proposition 1.4: epsilon-DP implies (epsilon^2 / 2)-zCDP."""
    e = float(epsilon)
    if e < 0.0:
        raise ValueError("epsilon cannot be negative")
    return 0.5 * e * e


def compose(rhos, omegas=None):
    """Composition: the rhos add and the truncation is the tightest one.

    Renyi divergence is additive over independent releases at each
    order, so the budgets add. A composed mechanism is only bounded at
    orders where EVERY part is bounded, which is why the truncation
    takes the minimum -- taking the maximum would claim a guarantee at
    orders one of the parts never had.
    """
    total = _w.csum(float(r) for r in rhos) if rhos else 0.0
    if omegas is None:
        return total, None
    live = [float(w) for w in omegas if w is not None]
    return total, (min(live) if live else None)


def truncated_cdp_mechanism(y, f_value, C, epsilon, delta, omega=None,
                            seed=0, n_release=1):
    """Release a clipped query under a target (epsilon, delta) budget.

    Parameters
    ----------
    y : sequence
        The per-record contributions. Clipping them to [-C, C] is what
        gives the query a sensitivity at all.
    f_value : float
        The query value being protected. Reported alongside the private
        answer so the noise actually added is visible.
    C : float
        The clipping bound, and hence the sensitivity of one record.
    epsilon, delta : float
        The target guarantee.
    omega : float or None
        The truncation order. None is untruncated zCDP.
    n_release : int
        How many releases the budget is split across. The per-release
        rho is the total over this, because the rhos add.

    Returns
    -------
    RichResult
        The private answer, the noise scale, the budget in rho, how many
        records the clipping bound, and the guarantee actually achieved.

    References
    ----------
    Bun and Steinke (2016) TCC, Propositions 1.3, 1.4, 1.6; Bun, Dwork,
    Rothblum and Steinke (2018) STOC, 74-86.
    """
    vals = [float(v) for v in y]
    c = float(C)
    if c <= 0.0:
        raise ValueError("the clipping bound must be positive")
    k = int(n_release)
    if k < 1:
        raise ValueError("there must be at least one release")
    clipped = []
    n_clipped = 0
    for v in vals:
        if v > c:
            clipped.append(c)
            n_clipped += 1
        elif v < -c:
            clipped.append(-c)
            n_clipped += 1
        else:
            clipped.append(v)

    rho_total = rho_from_eps(epsilon, delta, omega)
    rho_each = rho_total / k
    sigma = gaussian_sigma(c, rho_each)
    rng = _core._SplitMix64(seed)
    noise = float(rng.normal(0.0, sigma))
    private = float(f_value) + noise
    achieved = eps_from_rho(rho_total, delta, omega)
    return RichResult(payload={
        "private_value": private,
        "estimate": private,
        "se": sigma,
        "noise": noise,
        "sigma": sigma,
        "rho_total": rho_total,
        "rho_per_release": rho_each,
        "epsilon_target": float(epsilon),
        "epsilon_achieved": achieved,
        "delta": float(delta),
        "omega": omega,
        "truncation_binds": (omega is not None
                             and eps_from_rho(rho_total, delta, omega)
                             > eps_from_rho(rho_total, delta, None)
                             - 1e-15
                             and eps_from_rho(rho_total, delta, omega)
                             != eps_from_rho(rho_total, delta, None)),
        "clipped": clipped,
        "n_clipped": n_clipped,
        "n": len(vals),
        "sensitivity": c,
        "n_release": k,
        "f_value": float(f_value),
        "method": "truncated CDP Gaussian mechanism",
    })


tcmech = truncated_cdp_mechanism


def cheatsheet():
    return ("tcmech: truncated CDP Gaussian mechanism. clip to bound the "
            "sensitivity, rho from the target (eps, delta), sigma from "
            "Delta / sqrt(2 rho)")
