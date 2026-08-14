# morie.fn -- function file (rootcoder007/morie)
r"""Pólya trees: a prior that can be continuous.

The Dirichlet process is discrete with probability one -- every draw
is a countable sum of point masses. For clustering that is a feature;
for putting a prior on a *density* it is a defect, and the usual
workaround is to convolve with a kernel and model a mixture instead.

**A Pólya tree fixes it directly, and the DP is the special case.**
Take a nested sequence of binary partitions: :math:`\Omega = B_0\cup
B_1`, then :math:`B_0 = B_{00}\cup B_{01}`, and so on, so level
:math:`m` is indexed by binary strings :math:`\varepsilon` of length
:math:`m`. Attach independent

.. math:: Y_\varepsilon \sim \mathrm{Beta}(\alpha_{\varepsilon 0},\
          \alpha_{\varepsilon 1}),

and give a set its accumulated product of branch probabilities. The
Pólya tree includes the DP as a special case, and -- unlike the DP --
**an appropriate choice of the parameters generates continuous
distributions with probability 1**.

**Which choice, and why it matters.** Taking
:math:`\alpha_\varepsilon = c\,m^2` at level :math:`m` makes the
branch probabilities concentrate near :math:`1/2` fast enough down
the tree that the limit is absolutely continuous; taking
:math:`\alpha_\varepsilon = c` (constant) gives the DP behaviour
instead. ``level_parameters`` exposes the rule and
``continuity_regime`` names the consequence, because a module that
silently picked one would be hiding the only decision that matters.

**Centring is by construction.** With
:math:`\alpha_{\varepsilon 0} = \alpha_{\varepsilon 1}` the branch
probabilities have mean :math:`1/2`, so the tree is centred on the
partitioning measure -- which is how a Pólya tree is centred on a
parametric family and then allowed to depart from it.

**The partition is a modelling choice with teeth.** The tree is
defined *relative to* its partition, so two Pólya trees with
different partitions are different priors, and a draw is only as
smooth as the partition is fine. ``finite_tree`` therefore truncates
at a stated level and reports it rather than pretending to be
infinite.

References
----------
Muller, P. & Quintana, F. A. (2004) "Nonparametric Bayesian Data
Analysis", *Statistical Science* 19(1), 95-110,
doi:10.1214/088342304000000017. [PDF supplied by Vee.] Sec. 2.3:
Polya trees as a generalisation of the DP which includes DP models as
a special case; that in contrast to the DP an appropriate choice of
the PT parameters allows one to generate CONTINUOUS distributions
with probability 1; the definition requiring a nested sequence of
(binary, without loss of generality) partitions with pi_1 = {B_0,
B_1}, B_0 = B_00 union B_01 and so on, so that the partition at level
m is indexed by binary strings of length m; and F ~ PT(Pi, A) defined
by nonnegative constants A = {alpha_eps} and independent
Y_eps ~ Beta(alpha_eps0, alpha_eps1).

Lavine, M. (1992) "Some Aspects of Polya Tree Distributions for
Statistical Modelling", *The Annals of Statistics* 20(3), 1222-1235,
doi:10.1214/aos/1176348767; Lavine, M. (1994) "More Aspects of Polya
Tree Distributions for Statistical Modelling", *The Annals of
Statistics* 22(3), 1161-1176, doi:10.1214/aos/1176325623. The
proposal and its development, including the parameter choices
governing absolute continuity. NOTE: neither is held locally -- the
construction implemented here follows the Muller-Quintana review,
which is, and the c m^2 rule is flagged as coming from that
literature rather than derived here.

Ferguson, T. S. (1974) "Prior Distributions on Spaces of Probability
Measures", *The Annals of Statistics* 2(4), 615-629,
doi:10.1214/aos/1176342752. Tailfree processes, of which both the DP
and the Pólya tree are instances.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["level_parameters", "continuity_regime", "finite_tree",
           "set_probability", "partition_index", "tree_density"]

_EPS = 1e-12
_RULES = ("m_squared", "constant", "linear")


def level_parameters(level, c=1.0, rule="m_squared"):
    r""":math:`\alpha_\varepsilon` at level :math:`m`.

    ``m_squared`` (:math:`cm^2`) is the choice that buys absolute
    continuity; ``constant`` reproduces DP-like behaviour.
    """
    m = int(level)
    if m < 1:
        raise ValueError("poltrx: levels are numbered from 1")
    if float(c) <= 0.0:
        raise ValueError("poltrx: c must be positive")
    if rule == "m_squared":
        a = float(c) * m * m
    elif rule == "constant":
        a = float(c)
    elif rule == "linear":
        a = float(c) * m
    else:
        raise ValueError("poltrx: rule must be one of %s, got %r"
                         % (", ".join(_RULES), rule))
    return {"alpha": a, "level": m, "rule": rule}


def continuity_regime(rule):
    r"""What the parameter rule implies about the draws."""
    if rule not in _RULES:
        raise ValueError("poltrx: rule must be one of %s, got %r"
                         % (", ".join(_RULES), rule))
    table = {
        "m_squared": ("absolutely continuous",
                      "branch probabilities concentrate near 1/2 "
                      "fast enough that the limit has a density"),
        "constant": ("discrete, DP-like",
                     "the DP is the special case; draws are "
                     "atomic"),
        "linear": ("borderline",
                   "between the two; growth is not fast enough to "
                   "guarantee a density"),
    }
    kind, why = table[rule]
    return {"rule": rule, "draws": kind, "reason": why}


def partition_index(x, level, lo=0.0, hi=1.0):
    r"""Which set at level :math:`m` contains :math:`x`.

    Returns the binary string, which IS the address in the tree.
    """
    m = int(level)
    a, b = float(lo), float(hi)
    v = float(x)
    if not a <= v <= b:
        raise ValueError("poltrx: x = %r lies outside the "
                         "partitioned interval [%r, %r]" % (x, lo, hi))
    bits = []
    for _ in range(m):
        mid = 0.5 * (a + b)
        if v < mid:
            bits.append(0)
            b = mid
        else:
            bits.append(1)
            a = mid
    return {"epsilon": tuple(bits), "interval": (a, b), "level": m}


def finite_tree(levels, c=1.0, rule="m_squared", rng=None, seed=0):
    r"""Draw the branch probabilities down to a stated level.

    Truncated and honest about it: a Pólya tree is defined relative
    to its partition, so the depth is part of the model.
    """
    M = int(levels)
    if M < 1:
        raise ValueError("poltrx: at least one level is needed")
    r = rng if rng is not None else np.random.default_rng(seed)
    Y = {}
    for m in range(1, M + 1):
        a = level_parameters(m, c, rule)["alpha"]
        for idx in range(2 ** (m - 1)):
            eps = tuple(int(b) for b in
                        bin(idx)[2:].zfill(m - 1)) if m > 1 else ()
            # Beta(a, a) by two gamma draws
            def gamma(shape):
                if shape < 1.0:
                    u = max(float(r.uniform()), 1e-15)
                    return gamma(shape + 1.0) * u ** (1.0 / shape)
                d = shape - 1.0 / 3.0
                cc = 1.0 / math.sqrt(9.0 * d)
                while True:
                    u1 = min(max(float(r.uniform()), 1e-12),
                             1 - 1e-12)
                    u2 = min(max(float(r.uniform()), 1e-12),
                             1 - 1e-12)
                    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(
                        2.0 * math.pi * u2)
                    v = (1.0 + cc * z) ** 3
                    if v <= 0.0:
                        continue
                    u = max(float(r.uniform()), 1e-15)
                    if math.log(u) < (0.5 * z * z + d - d * v
                                      + d * math.log(v)):
                        return d * v
            g0, g1 = gamma(a), gamma(a)
            Y[eps] = g0 / (g0 + g1) if (g0 + g1) > _EPS else 0.5
    return {"Y": Y, "levels": M, "rule": rule, "c": float(c),
            "n_nodes": len(Y),
            "note": "truncated at level %d; the partition depth is "
                    "part of the model, not an approximation to hide"
                    % M}


def set_probability(epsilon, tree):
    r"""Accumulated product of branch probabilities down to
    :math:`B_\varepsilon`."""
    eps = tuple(int(b) for b in epsilon)
    if len(eps) > tree["levels"]:
        raise ValueError("poltrx: the tree was truncated at level "
                         "%d, so it says nothing about level %d"
                         % (tree["levels"], len(eps)))
    p = 1.0
    for m in range(len(eps)):
        parent = eps[:m]
        y = tree["Y"][parent]
        p *= (y if eps[m] == 0 else 1.0 - y)
    return {"probability": p, "epsilon": eps, "level": len(eps)}


def tree_density(tree, level=None, lo=0.0, hi=1.0):
    r"""The piecewise-constant density the truncated tree implies."""
    M = int(tree["levels"]) if level is None else int(level)
    n = 2 ** M
    width = (float(hi) - float(lo)) / n
    probs, dens, edges = [], [], []
    for idx in range(n):
        eps = tuple(int(b) for b in bin(idx)[2:].zfill(M))
        p = set_probability(eps, tree)["probability"]
        probs.append(p)
        dens.append(p / width)
        edges.append((float(lo) + idx * width,
                      float(lo) + (idx + 1) * width))
    return RichResult(payload={
        "estimate": dens, "density": dens, "probabilities": probs,
        "edges": edges, "level": M, "total": sum(probs),
        "method": "Polya tree; Muller & Quintana (2004) Sec. 2.3, "
                  "after Lavine (1992, 1994)",
        "note": "the probabilities at a level sum to 1 by "
                "construction, whatever the branch draws were",
    })


def cheatsheet():
    return ("poltrx: the DP is DISCRETE with probability one, which "
            "is fine for clustering and wrong for a prior on a "
            "DENSITY. A Polya tree fixes it directly and CONTAINS the "
            "DP as a special case: nested binary partitions, with "
            "independent Y_eps ~ Beta(alpha_eps0, alpha_eps1) at each "
            "node and a set's probability the accumulated product "
            "down the tree. The parameter rule is the whole decision "
            "-- alpha = c m^2 concentrates the branches near 1/2 fast "
            "enough to give an absolutely continuous draw, while "
            "constant alpha gives DP-like atoms. Equal alphas centre "
            "the tree on the partitioning measure. The PARTITION is "
            "part of the prior, so the depth is stated, not hidden.")


# compact alias per ledger/NAMING.md
polya_tree = finite_tree

# public names resolved by fn/_lazy_map.json
polya_tree_extended = finite_tree
