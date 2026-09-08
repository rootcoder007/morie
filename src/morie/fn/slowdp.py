# morie.fn -- function file (rootcoder007/morie)
r"""Truncating a Dirichlet process, and knowing what you dropped.

Sethuraman's constructive definition builds a Dirichlet process out of
two independent sequences: locations :math:`\theta_k` drawn from the
base measure and **stick-breaking** weights

.. math:: p_k = V_k\prod_{l<k}(1 - V_l), \qquad
          V_k \sim \mathrm{Beta}(1, \alpha),

giving :math:`G = \sum_{k=1}^{\infty} p_k\delta_{\theta_k}`. The
weights sum to one almost surely, and the construction is what makes
the DP simulable at all: an infinite object written as a program.

**Truncation is the practical step and the honest one.** Keeping
:math:`K` sticks leaves a remainder, and its expectation is available
in closed form. Since :math:`E[1-V] = \alpha/(1+\alpha)` and the
:math:`V_k` are independent,

.. math:: E\Big[1 - \sum_{k\le K} p_k\Big]
          = \Big(\frac{\alpha}{1+\alpha}\Big)^{K},

so the truncation error is **geometric in K and worsens with
:math:`\alpha`** -- a more diffuse process needs more sticks for the
same fidelity. ``truncation_error`` returns that number and
``sticks_for_tolerance`` inverts it, so the level is chosen rather
than guessed.

**"Slow-decreasing" names the failure mode.** The weights decay
geometrically *in expectation*, but they are not ordered: an
individual draw can put a large stick far out in the sequence, and a
truncation chosen from the mean alone will occasionally cut a
component that matters. ``decay_diagnostics`` reports the realised
tail against the expected one so that gap is visible.

**Renormalising the kept sticks is a choice with a consequence.** It
restores a proper distribution and silently moves the discarded mass
onto the survivors -- fine when the tail is negligible, misleading
when it is not, which is exactly why the error is reported alongside.

References
----------
Sethuraman, J. (1994) "A Constructive Definition of Dirichlet
Priors", *Statistica Sinica* 4(2), 639-650. The stick-breaking
construction: independent Beta(1, alpha) variables V_k and locations
from the base measure give p_k = V_k prod_{l<k}(1 - V_l) with sum p_k
= 1 almost surely, and the resulting random measure is a Dirichlet
process. NOTE: the local scan has no text layer; it was read by
rendering at 200 dpi and running tesseract, which confirms the title,
journal, volume and pages above.

Ferguson, T. S. (1973) "A Bayesian Analysis of Some Nonparametric
Problems", *The Annals of Statistics* 1(2), 209-230,
doi:10.1214/aos/1176342360. The Dirichlet process itself.

Ishwaran, H. & James, L. F. (2001) "Gibbs Sampling Methods for
Stick-Breaking Priors", *Journal of the American Statistical
Association* 96(453), 161-173, doi:10.1198/016214501750332758. The
truncated stick-breaking prior and the analysis of the truncation
level. NOTE: not held locally -- the geometric tail expectation
implemented here is derived directly from Sethuraman's construction
and checked by simulation, not quoted from this paper.

Neal, R. M. (2000) "Markov Chain Sampling Methods for Dirichlet
Process Mixture Models", *Journal of Computational and Graphical
Statistics* 9(2), 249-265, doi:10.1080/10618600.2000.10474879.
[Technical Report 9815, University of Toronto, supplied locally.]
The sampling context in which a truncation level has to be chosen.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["stick_breaking", "truncation_error",
           "sticks_for_tolerance", "decay_diagnostics",
           "truncated_dp"]

_EPS = 1e-12


def _beta_1_alpha(rng, alpha):
    r"""Beta(1, alpha) by inverse transform: :math:`1-u^{1/\alpha}`."""
    u = float(rng.uniform())
    u = min(max(u, 1e-15), 1.0 - 1e-15)
    return 1.0 - u ** (1.0 / float(alpha))


def stick_breaking(alpha, K, rng=None, seed=0):
    r""":math:`p_k = V_k\prod_{l<k}(1-V_l)`, :math:`V_k\sim
    \mathrm{Beta}(1,\alpha)`."""
    a = float(alpha)
    n = int(K)
    if a <= 0.0:
        raise ValueError("slowdp: alpha must be positive")
    if n < 1:
        raise ValueError("slowdp: at least one stick is needed")
    r = rng if rng is not None else np.random.default_rng(seed)
    p, rest = [], 1.0
    Vs = []
    for _ in range(n):
        v = _beta_1_alpha(r, a)
        Vs.append(v)
        p.append(v * rest)
        rest *= (1.0 - v)
    return {"weights": p, "V": Vs, "remaining": rest,
            "kept_mass": sum(p), "K": n, "alpha": a,
            "note": "the remaining stick is the mass truncation "
                    "throws away"}


def truncation_error(alpha, K):
    r""":math:`E[1-\sum_{k\le K}p_k] = (\alpha/(1+\alpha))^{K}`.

    Geometric in :math:`K`, and worse for larger :math:`\alpha`.
    """
    a = float(alpha)
    n = int(K)
    if a <= 0.0 or n < 1:
        raise ValueError("slowdp: need alpha > 0 and K >= 1")
    e = (a / (1.0 + a)) ** n
    return {"expected_tail": e, "kept": 1.0 - e, "alpha": a, "K": n,
            "per_stick_factor": a / (1.0 + a),
            "note": "a more diffuse process needs more sticks for "
                    "the same fidelity"}


def sticks_for_tolerance(alpha, tol=1e-3):
    r"""Smallest :math:`K` with expected tail below ``tol``."""
    a = float(alpha)
    t = float(tol)
    if a <= 0.0:
        raise ValueError("slowdp: alpha must be positive")
    if not 0.0 < t < 1.0:
        raise ValueError("slowdp: the tolerance must lie in (0,1)")
    f = a / (1.0 + a)
    K = int(math.ceil(math.log(t) / math.log(f)))
    return {"K": max(1, K), "expected_tail":
            truncation_error(a, max(1, K))["expected_tail"],
            "tolerance": t,
            "note": "chosen from the closed form, not guessed"}


def decay_diagnostics(weights, alpha):
    r"""Realised tail against the expected one.

    The weights decay geometrically IN EXPECTATION and are not
    ordered, so a single draw can put a large stick far out.
    """
    p = [float(v) for v in k.vec(weights)]
    K = len(p)
    if K < 1:
        raise ValueError("slowdp: no weights given")
    exp_tail = truncation_error(alpha, K)["expected_tail"]
    realised = max(0.0, 1.0 - sum(p))
    biggest_late = max(range(K), key=lambda i: p[i])
    return {"realised_tail": realised, "expected_tail": exp_tail,
            "ratio": realised / exp_tail if exp_tail > _EPS
            else float("inf"),
            "largest_index": biggest_late,
            "monotone": all(p[i] >= p[i + 1] - _EPS
                            for i in range(K - 1)),
            "note": "the sticks are NOT ordered; a late large stick "
                    "is exactly what a mean-based truncation misses"}


def truncated_dp(alpha, K, base_sampler=None, rng=None, seed=0,
                 renormalise=True):
    r"""A truncated DP draw, with the discarded mass reported."""
    r = rng if rng is not None else np.random.default_rng(seed)
    sb = stick_breaking(alpha, K, r)
    p = list(sb["weights"])
    tail = sb["remaining"]
    if renormalise:
        z = sum(p)
        if z <= _EPS:
            raise ValueError("slowdp: the kept sticks carry no mass")
        p = [v / z for v in p]
    atoms = ([base_sampler(r) for _ in range(int(K))]
             if base_sampler is not None else list(range(int(K))))
    return RichResult(payload={
        "estimate": p, "weights": p, "atoms": atoms,
        "discarded_mass": tail,
        "expected_discarded": truncation_error(alpha,
                                               K)["expected_tail"],
        "renormalised": bool(renormalise), "K": int(K),
        "alpha": float(alpha),
        "method": "truncated stick-breaking; Sethuraman (1994)",
        "note": "renormalising moves the discarded mass onto the "
                "survivors, which is why the amount is returned",
    })


def cheatsheet():
    return ("slowdp: Sethuraman writes the DP as a PROGRAM -- "
            "p_k = V_k prod(1 - V_l) with V_k ~ Beta(1, alpha) and "
            "atoms from the base measure, summing to 1 almost surely. "
            "Truncating at K leaves an expected tail of EXACTLY "
            "(alpha/(1+alpha))^K: geometric in K, and worse for larger "
            "alpha, so a diffuse process needs more sticks. Invert it "
            "to CHOOSE K rather than guess. But the decay is only in "
            "EXPECTATION and the sticks are NOT ordered -- a single "
            "draw can put a large stick late, which is what "
            "'slow-decreasing' names. Renormalising the survivors "
            "silently absorbs the discarded mass, so report it.")


# compact alias per ledger/NAMING.md
dp_truncation = truncated_dp

# public names resolved by fn/_lazy_map.json
slow_dp_truncate = truncated_dp
slowdptruncate = truncated_dp
