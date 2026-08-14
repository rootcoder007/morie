# morie.fn -- function file (rootcoder007/morie)
r"""Pitman-Yor: one extra parameter, a different tail.

The Dirichlet process breaks its stick with :math:`V_k \sim
\mathrm{Beta}(1,\theta)` -- the same distribution at every :math:`k`.
The two-parameter family lets the beta parameters **move with the
index**:

.. math:: \tilde Y_n \sim \mathrm{Beta}(1-\alpha,\ \theta+n\alpha),
          \qquad \tilde V_1 = \tilde Y_1,\quad
          \tilde V_n = \Big(\prod_{j<n}(1-\tilde Y_j)\Big)\tilde Y_n,

for :math:`0\le\alpha<1` and :math:`\theta>-\alpha`. Setting
:math:`\alpha=0` gives back Beta(1, θ) at every index, so the DP is
the boundary case, not a different construction.

**The form is forced, not chosen.** Pitman and Yor's Proposition 4 is
a characterisation: a size-biased permutation admits this residual
allocation form with *independent* factors **if and only if** the
:math:`\tilde Y_n` are beta with exactly these parameters. So the
parameter range :math:`0\le\alpha<1,\ \theta>-\alpha` is dictated by
the requirement, and ``check_parameters`` enforces it rather than
treating it as a convention.

**What the extra parameter buys is the tail.** The DP's weights decay
geometrically, so the number of distinct values in :math:`n` draws
grows like :math:`\theta\log n`. With :math:`\alpha>0` the beta
parameters drift and the weights decay **polynomially**, giving a
number of clusters growing like :math:`n^{\alpha}` -- which is what
makes Pitman-Yor the right prior for data with many rare types
(vocabulary, species, surnames) where a DP systematically
under-predicts new types. ``expected_clusters`` computes both from
the same recursion, so the difference is a number rather than a
slogan.

**The discount also changes the predictive rule.** An occupied
cluster's weight becomes :math:`(n_j-\alpha)/(\theta+n)`: every
existing cluster is *discounted* by :math:`\alpha`, and the mass
removed is handed to the new-cluster term. Small clusters lose
proportionally more, which is exactly the rich-get-richer behaviour
being tempered.

References
----------
Pitman, J. & Yor, M. (1997) "The Two-Parameter Poisson-Dirichlet
Distribution Derived from a Stable Subordinator", *The Annals of
Probability* 25(2), 855-900, doi:10.1214/aop/1024404422. [PDF
supplied by Vee.] Definition 1: for 0 <= alpha < 1 and theta >
-alpha, independent Y_n ~ Beta(1 - alpha, theta + n alpha) with
V_1 = Y_1 and V_n = (1 - Y_1)...(1 - Y_{n-1}) Y_n, whose ranked
values define PD(alpha, theta); Proposition 2, that the V_n sum to 1
almost surely and the tilde-V are a size-biased permutation of the
ranked values; and Proposition 4, that a size-biased permutation
admits this residual allocation form with independent factors IF AND
ONLY IF the Y_n are beta of exactly this form -- so the parameter set
is dictated. Also that PD(0, theta) is Kingman's one-parameter
Poisson-Dirichlet distribution, and that PD(alpha, 0) arises as the
asymptotic distribution of ranked excursion lengths of a Markov chain
whose recurrence time is in the domain of attraction of a stable law
of index alpha.

Sethuraman, J. (1994) "A Constructive Definition of Dirichlet
Priors", *Statistica Sinica* 4(2), 639-650. The alpha = 0 case;
implemented in :mod:`slowdp`.

Ishwaran, H. & James, L. F. (2001) "Gibbs Sampling Methods for
Stick-Breaking Priors", *JASA* 96(453), 161-173,
doi:10.1198/016214501750332758. The stick-breaking family that
contains both. NOTE: not held locally.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["check_parameters", "stick_breaking_py",
           "predictive_weights", "expected_clusters",
           "tail_comparison"]

_EPS = 1e-12


def check_parameters(alpha, theta):
    r"""Definition 1's range: :math:`0\le\alpha<1`,
    :math:`\theta>-\alpha`.

    Dictated by Proposition 4, not a convention -- outside it the
    residual factors are not independent betas and the object is not
    PD(alpha, theta).
    """
    a, t = float(alpha), float(theta)
    if not 0.0 <= a < 1.0:
        raise ValueError("pmpfit: the discount must satisfy "
                         "0 <= alpha < 1, got %r" % (alpha,))
    if t <= -a:
        raise ValueError("pmpfit: the concentration must satisfy "
                         "theta > -alpha = %r, got %r" % (-a, theta))
    return {"alpha": a, "theta": t, "is_dirichlet": a == 0.0,
            "note": "alpha = 0 is exactly the Dirichlet process"}


def _beta(rng, a, b):
    r"""Beta(a, b) by the ratio of two gamma draws."""
    def gamma(shape):
        if shape < 1.0:
            u = max(float(rng.uniform()), 1e-15)
            return gamma(shape + 1.0) * u ** (1.0 / shape)
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            u1 = min(max(float(rng.uniform()), 1e-12), 1 - 1e-12)
            u2 = min(max(float(rng.uniform()), 1e-12), 1 - 1e-12)
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(
                2.0 * math.pi * u2)
            v = (1.0 + c * z) ** 3
            if v <= 0.0:
                continue
            u = max(float(rng.uniform()), 1e-15)
            if math.log(u) < 0.5 * z * z + d - d * v + d * math.log(v):
                return d * v
    g1 = gamma(a)
    g2 = gamma(b)
    return g1 / (g1 + g2) if (g1 + g2) > _EPS else 0.5


def stick_breaking_py(alpha, theta, K, rng=None, seed=0):
    r""":math:`\tilde Y_n\sim\mathrm{Beta}(1-\alpha,\theta+n\alpha)`.

    The index enters the beta parameters, which is the entire
    difference from the DP.
    """
    p = check_parameters(alpha, theta)
    a, t = p["alpha"], p["theta"]
    n = int(K)
    if n < 1:
        raise ValueError("pmpfit: at least one stick is needed")
    r = rng if rng is not None else np.random.default_rng(seed)
    w, rest, Ys = [], 1.0, []
    for kk in range(1, n + 1):
        y = _beta(r, 1.0 - a, t + kk * a) if a > 0.0 else \
            1.0 - max(float(r.uniform()), 1e-15) ** (1.0 / t)
        Ys.append(y)
        w.append(y * rest)
        rest *= (1.0 - y)
    return {"weights": w, "Y": Ys, "remaining": rest,
            "kept_mass": sum(w), "alpha": a, "theta": t,
            "note": "Beta(1-alpha, theta + n alpha): the parameters "
                    "DRIFT with n, which is what fattens the tail"}


def predictive_weights(counts, alpha, theta):
    r"""Occupied clusters discounted by :math:`\alpha`.

    :math:`(n_j-\alpha)/(\theta+n)` per occupied cluster and
    :math:`(\theta+K\alpha)/(\theta+n)` for a new one -- the mass
    taken off each cluster is exactly what funds the new-table term.
    """
    c = [float(v) for v in k.vec(counts)]
    p = check_parameters(alpha, theta)
    a, t = p["alpha"], p["theta"]
    if any(v <= 0.0 for v in c):
        raise ValueError("pmpfit: an occupied cluster must have a "
                         "positive count")
    n = sum(c)
    K = len(c)
    if any(v <= a for v in c) and a > 0.0:
        raise ValueError("pmpfit: a cluster of size <= alpha would "
                         "get a negative weight; alpha must be "
                         "smaller than every cluster size")
    occ = [(v - a) / (t + n) for v in c]
    new = (t + K * a) / (t + n)
    return {"occupied": occ, "new": new,
            "total": sum(occ) + new, "n": n, "K": K,
            "discount_transferred": K * a / (t + n),
            "note": "each existing cluster is discounted by alpha, "
                    "and the removed mass funds the new cluster"}


def expected_clusters(n, alpha, theta):
    r"""Expected distinct values in :math:`n` draws, exactly.

    :math:`E[K_n] = \sum_{i=0}^{n-1}
    (\theta+K\alpha)/(\theta+i)` in expectation -- computed here by
    the exact recursion, so the log-vs-power growth is measured
    rather than quoted.
    """
    p = check_parameters(alpha, theta)
    a, t = p["alpha"], p["theta"]
    N = int(n)
    if N < 1:
        raise ValueError("pmpfit: n must be at least 1")
    ek = 0.0
    for i in range(N):
        ek += (t + ek * a) / (t + i)
    return {"expected": ek, "n": N, "alpha": a, "theta": t,
            "regime": "power law n^alpha" if a > 0.0
            else "logarithmic theta log n",
            "note": "the DP's count grows like theta log n; a "
                    "positive discount makes it grow like n^alpha"}


def tail_comparison(n, theta=1.0, alphas=(0.0, 0.3, 0.6)):
    r"""How many clusters each discount predicts at the same
    :math:`\theta`."""
    out = {}
    for a in alphas:
        out[float(a)] = expected_clusters(n, a, theta)["expected"]
    keys = sorted(out)
    return RichResult(payload={
        "estimate": out, "expected_clusters": out, "n": int(n),
        "theta": float(theta),
        "monotone_in_alpha": all(out[keys[i]] <= out[keys[i + 1]]
                                 for i in range(len(keys) - 1)),
        "method": "two-parameter Poisson-Dirichlet; Pitman & Yor "
                  "(1997) Definition 1",
        "note": "more discount, more distinct types at the same n -- "
                "which is why vocabulary-like data want alpha > 0",
    })


def cheatsheet():
    return ("pmpfit: the DP breaks its stick with Beta(1, theta) at "
            "EVERY index; Pitman-Yor lets the parameters DRIFT -- "
            "Y_n ~ Beta(1 - alpha, theta + n alpha) for 0 <= alpha < 1 "
            "and theta > -alpha, with alpha = 0 recovering the DP "
            "exactly. The form is FORCED: Proposition 4 says a "
            "size-biased permutation has independent residual factors "
            "iff the betas take this form. The payoff is the TAIL: "
            "geometric decay becomes polynomial, so the distinct-type "
            "count grows like n^alpha instead of theta log n -- which "
            "is why vocabularies and species counts want alpha > 0. "
            "Predictively, each cluster is discounted by alpha and the "
            "removed mass funds the new-cluster term.")


# compact alias per ledger/NAMING.md
pitman_yor = stick_breaking_py

# public names resolved by fn/_lazy_map.json
pmp_fit = stick_breaking_py
pmpfit = stick_breaking_py
