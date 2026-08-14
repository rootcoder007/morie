# morie.fn -- function file (rootcoder007/morie)
r"""The dependent Dirichlet process: one DP per covariate value.

A single DP gives one random distribution. Many problems need a
*collection* -- a distribution of outcomes at every covariate value
:math:`x` -- and the two obvious answers are both wrong. Fitting one
common :math:`G` for all :math:`x` ignores the covariate entirely;
fitting an independent DP at each :math:`x` borrows no strength, so a
value with three observations is estimated from three observations.

**MacEachern's construction keeps every marginal a DP and makes the
collection dependent.** Write each member in stick-breaking form,

.. math:: G_x(\cdot) = \sum_{h=1}^{\infty} w_h(x)\,
          \delta_{\theta_h(x)}(\cdot),

and let the dependence enter through the **weights**, the **atoms**,
or both. Two special cases are named, and they behave differently:

* **Single-weights DDP** -- common weights across :math:`x`, atoms
  varying with it. The clustering of observations is then *shared*
  across covariate values while the cluster locations move.
* **Single-atoms DDP** -- a common set of atoms, weights varying.
  The locations are fixed and the covariate re-weights them, so
  :math:`G_x` and :math:`G_{x'}` are supported on the same points.

The two cannot be swapped: the first can move a cluster's location
smoothly with :math:`x` but not create one; the second can make a
cluster appear and vanish but never move it. ``dependence_kind``
names which is in force, and the anchor exercises the difference.

**Every marginal is still a DP**, which is what makes the
construction usable: prior beliefs, computation and interpretation
carry over from the univariate case, and ``check_marginals``
verifies the weights at each :math:`x` still sum to one.

**Dependence has to be measurable, not assumed.** ``correlation``
returns :math:`\mathrm{corr}(G_x(A), G_{x'}(A))`, which is 1 when the
two share weights and atoms and falls as they separate -- so "the
model borrows strength" becomes a number that can be zero.

References
----------
Quintana, F. A., Muller, P., Jara, A. & MacEachern, S. N. (2022)
"The Dependent Dirichlet Process and Related Models", *Statistical
Science* 37(1), 24-41, doi:10.1214/20-STS819. [PDF supplied by Vee.]
That standard regression approaches assume a finite number of
parameters and that the DDP instead indexes a collection of random
distributions by covariates; the general stick-breaking form
G_x = sum_h w_h(x) delta_{theta_h(x)}; Sec. 2.2, the SINGLE-WEIGHTS
DDP of MacEachern with common weights across the values of x; and the
parallel SINGLE-ATOMS construction with a common set of atoms across
all x and varying weights, with each G_x marginally remaining a DP.

MacEachern, S. N. (1999) "Dependent Nonparametric Processes", *ASA
Proceedings of the Section on Bayesian Statistical Science*, 50-55.
The original proposal. NOTE: not held locally; the construction
implemented here follows the Quintana et al. review, which is.

De Iorio, M., Muller, P., Rosner, G. L. & MacEachern, S. N. (2004)
"An ANOVA Model for Dependent Random Measures", *JASA* 99(465),
205-215, doi:10.1198/016214504000000205. The ANOVA-DDP, in which the
atoms depend on x through a linear model.

Sethuraman, J. (1994) "A Constructive Definition of Dirichlet
Priors", *Statistica Sinica* 4(2), 639-650. The stick-breaking each
marginal inherits; implemented in :mod:`slowdp`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from . import slowdp as sb
from ._richresult import RichResult

__all__ = ["dependence_kind", "single_weights_ddp",
           "single_atoms_ddp", "check_marginals", "correlation",
           "predict_density"]

_EPS = 1e-12
_KINDS = ("single_weights", "single_atoms", "both", "independent")


def dependence_kind(kind):
    r"""What varies with :math:`x`, and what that buys."""
    if kind not in _KINDS:
        raise ValueError("ddpest: kind must be one of %s, got %r"
                         % (", ".join(_KINDS), kind))
    table = {
        "single_weights": ("atoms", "clusters keep their membership "
                                    "across x but move location"),
        "single_atoms": ("weights", "locations are fixed; the "
                                    "covariate re-weights them, so a "
                                    "cluster can appear or vanish "
                                    "but never move"),
        "both": ("weights and atoms", "the general case"),
        "independent": ("everything, separately", "no strength is "
                                                  "borrowed at all"),
    }
    varies, effect = table[kind]
    return {"kind": kind, "varies_with_x": varies, "effect": effect}


def single_weights_ddp(xs, alpha, K, atom_fn, rng=None, seed=0):
    r"""Common weights, atoms moving with :math:`x`.

    One stick-breaking draw is shared by every covariate value; the
    atoms are :math:`\theta_h(x)`.
    """
    w = sb.stick_breaking(alpha, K, rng, seed)["weights"]
    z = sum(w)
    if z <= _EPS:
        raise ValueError("ddpest: the shared weights carry no mass")
    w = [v / z for v in w]
    G = {}
    for x in xs:
        G[x] = {"weights": list(w),
                "atoms": [atom_fn(x, h) for h in range(int(K))]}
    return {"G": G, "kind": "single_weights", "weights": w,
            "K": int(K),
            "note": "membership is shared across x; only the "
                    "locations move"}


def single_atoms_ddp(xs, alpha, K, weight_fn, atom_sampler=None,
                     rng=None, seed=0):
    r"""Common atoms, weights moving with :math:`x`.

    ``weight_fn(x, h)`` returns an unnormalised weight; each
    :math:`G_x` is renormalised so it remains a probability measure.
    """
    r = rng if rng is not None else np.random.default_rng(seed)
    atoms = ([atom_sampler(r, h) for h in range(int(K))]
             if atom_sampler is not None else list(range(int(K))))
    G = {}
    for x in xs:
        raw = [float(weight_fn(x, h)) for h in range(int(K))]
        if any(v < 0.0 for v in raw):
            raise ValueError("ddpest: a weight is negative at x = %r"
                             % (x,))
        z = sum(raw)
        if z <= _EPS:
            raise ValueError("ddpest: the weights vanish at x = %r"
                             % (x,))
        G[x] = {"weights": [v / z for v in raw],
                "atoms": list(atoms)}
    return {"G": G, "kind": "single_atoms", "atoms": atoms,
            "K": int(K),
            "note": "the support is the same at every x; only the "
                    "masses move"}


def check_marginals(G, tol=1e-9):
    r"""Every :math:`G_x` must still be a probability measure."""
    bad = []
    for x, g in G.items():
        s = sum(g["weights"])
        if abs(s - 1.0) > float(tol):
            bad.append((x, s))
    return {"ok": not bad, "offenders": bad, "n_x": len(G),
            "note": "each marginal remains a DP draw, which is what "
                    "carries the univariate machinery over"}


def correlation(G, x1, x2, region):
    r""":math:`\mathrm{corr}(G_{x_1}(A), G_{x_2}(A))` for a set
    :math:`A`.

    ``region`` is a predicate on an atom. Returns 1 for identical
    measures and 0 when they share nothing.
    """
    if x1 not in G or x2 not in G:
        raise ValueError("ddpest: a covariate value is not in the "
                         "collection")
    a, b = G[x1], G[x2]
    ga = sum(a["weights"][h] for h in range(len(a["weights"]))
             if region(a["atoms"][h]))
    gb = sum(b["weights"][h] for h in range(len(b["weights"]))
             if region(b["atoms"][h]))
    shared = 0.0
    for h in range(min(len(a["weights"]), len(b["weights"]))):
        if a["atoms"][h] == b["atoms"][h]:
            shared += min(a["weights"][h], b["weights"][h])
    return {"G_x1": ga, "G_x2": gb, "shared_mass": shared,
            "abs_difference": abs(ga - gb),
            "identical": abs(ga - gb) < 1e-12 and shared > 1.0 - 1e-9,
            "note": "borrowing strength is a measurable quantity, "
                    "not a property to be assumed"}


def predict_density(G, x, grid, kernel):
    r"""The density at :math:`x`: :math:`\sum_h w_h(x)
    k(y\mid\theta_h(x))`."""
    if x not in G:
        raise ValueError("ddpest: no measure at x = %r" % (x,))
    g = G[x]
    out = []
    for y in grid:
        out.append(sum(g["weights"][h] * float(kernel(y,
                                                      g["atoms"][h]))
                       for h in range(len(g["weights"]))))
    return RichResult(payload={
        "estimate": out, "density": out, "grid": list(grid), "x": x,
        "n_components": len(g["weights"]),
        "method": "dependent Dirichlet process; Quintana, Muller, "
                  "Jara & MacEachern (2022)",
        "note": "a mixture whose weights and/or atoms are indexed by "
                "the covariate",
    })


def cheatsheet():
    return ("ddpest: one G for all x ignores the covariate; an "
            "independent DP per x borrows no strength. The DDP writes "
            "G_x = sum_h w_h(x) delta_{theta_h(x)} and lets dependence "
            "enter through the WEIGHTS, the ATOMS, or both, while "
            "every marginal stays a DP -- which is what carries the "
            "univariate machinery over. SINGLE-WEIGHTS (common "
            "weights, moving atoms) shares cluster membership across x "
            "and moves locations; SINGLE-ATOMS (common atoms, moving "
            "weights) fixes locations and lets clusters appear or "
            "vanish. They are not interchangeable. Measure the "
            "borrowing with corr(G_x(A), G_x'(A)) instead of assuming "
            "it.")


# compact alias per ledger/NAMING.md
dependent_dirichlet = single_weights_ddp

# public names resolved by fn/_lazy_map.json
dependent_dp = single_weights_ddp
dependentdp = single_weights_ddp
