# morie.fn -- function file (rootcoder007/morie)
r"""Auxiliary-variable slice sampling inside a Gibbs sweep.

**The trick.** Sampling from :math:`f(x)` is hard; sampling uniformly
from the region *under* its graph is easy. Introduce a height
:math:`u` and sample the pair:

.. math:: u \mid x \sim \mathrm{U}(0, f(x)), \qquad
          x \mid u \sim \mathrm{U}(S_u), \quad
          S_u = \{x : f(x) \ge u\}.

The marginal of :math:`x` under this joint *is* :math:`f`, so the
chain needs no proposal distribution, no tuning constant, and no
accept/reject step. Damien, Wakefield and Walker's contribution is
that this is exactly what rescues a Gibbs sampler whose conditionals
are non-conjugate: the awkward factor gets its own auxiliary
variable, and each resulting conditional is a uniform on an interval.

**Finding the slice.** :math:`S_u` is generally not known in closed
form. Neal's stepping-out procedure grows an interval of width
:math:`w` around the current point until both ends are below
:math:`u`, then shrinks it on rejection. Shrinkage handles a slice
that is not an interval: points in a valley fall below :math:`u`, get
rejected, and narrow the bracket instead of being accepted.

**Where that stops working, which matters.** Stepping out halts at the
*first* end point below :math:`u`. If two modes are separated by a
region whose density is below the slice level at every width step,
the bracket stops at the near side of the valley and the far mode is
never proposed. Overlapping modes are crossed routinely; well
separated ones are not crossed at all. The anchor demonstrates both:
modes two units apart mix in the right proportion, modes twelve units
apart give a chain that never leaves the one it started in. This is a
property of the method, not a defect in this implementation, and no
choice of :math:`w` repairs it -- tempering or a mixture proposal is
the answer if separated modes are expected.

**Why the step size cannot break it.** :math:`w` affects only how
many function evaluations a transition costs. Too small and the
interval is stepped out many times; too large and it is shrunk many
times. Neither changes the stationary distribution -- which is the
property that separates this from Metropolis, where a bad scale
silently biases the answer by collapsing the acceptance rate. The
anchor checks exactly this: the same target sampled at :math:`w`
values spanning two orders of magnitude gives the same moments, while
the evaluation count moves a lot.

**Unnormalised is enough.** Only ratios :math:`f(x)/u` matter, so the
target never needs its normalising constant -- the usual situation in
a Bayesian conditional.

Work on the log scale throughout: :math:`u \sim \mathrm{U}(0, f(x))`
becomes :math:`\log u = \log f(x) - \mathrm{Exp}(1)`, which does not
underflow when :math:`f` is a likelihood over many observations.

References
----------
Damien, P., Wakefield, J. & Walker, S. (1999) "Gibbs sampling for
Bayesian non-conjugate and hierarchical models by using auxiliary
variables", *Journal of the Royal Statistical Society: Series B*
61(2), 331-344, doi:10.1111/1467-9868.00179. The auxiliary-variable
construction above, its use to make non-conjugate full conditionals
uniform, and the hybrid Gibbs sweep in which some coordinates are
drawn directly and others by slice.

Neal, R. M. (2003) "Slice sampling", *Annals of Statistics* 31(3),
705-767, doi:10.1214/aos/1056562461. The stepping-out and shrinkage
procedures reproduced here, the log-scale formulation, and the result
that correctness does not depend on the width :math:`w`.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["slice_sample_1d", "slice_chain", "gibbs_slice",
           "effective_sample_size", "hybrid_gibbs_slice"]


def _rng(seed):
    return np.random.default_rng(int(seed))


def _expo(rng):
    """Exp(1) without leaving the native core."""
    u = rng.random()
    while u <= 0.0:
        u = rng.random()
    return -math.log(u)


def slice_sample_1d(logf, x0, rng, w=1.0, max_steps=50,
                    lower=float("-inf"), upper=float("inf")):
    r"""One univariate slice transition. Returns the draw and the cost.

    ``logf`` is the log of an unnormalised density. ``lower``/``upper``
    bound the support, which is how a positive parameter stays
    positive without any rejection.
    """
    if not (w > 0):
        raise ValueError("baygsl: the slice width must be positive")
    fx = logf(x0)
    if not (fx == fx) or fx == float("-inf"):
        raise ValueError("baygsl: the chain is at a point of zero "
                         "density, so no slice exists there")
    n_eval = 1
    logu = fx - _expo(rng)          # u ~ U(0, f(x)) on the log scale
    # stepping out
    r = rng.random()
    L = x0 - r * w
    R = L + w
    j = int(max_steps * rng.random())
    k = int(max_steps) - 1 - j
    while j > 0 and L > lower:
        n_eval += 1
        if logf(L) <= logu:
            break
        L -= w
        j -= 1
    while k > 0 and R < upper:
        n_eval += 1
        if logf(R) <= logu:
            break
        R += w
        k -= 1
    L = max(L, lower)
    R = min(R, upper)
    # shrinkage
    for _ in range(10000):
        x1 = L + rng.random() * (R - L)
        n_eval += 1
        if logf(x1) > logu:
            return {"x": x1, "n_eval": n_eval, "interval": (L, R)}
        if x1 < x0:
            L = x1
        else:
            R = x1
        if R - L < 1e-15:
            return {"x": x0, "n_eval": n_eval, "interval": (L, R)}
    raise ValueError("baygsl: the shrinkage loop did not terminate; "
                     "is logf returning a constant or NaN?")


def slice_chain(logf, x0, n=2000, w=1.0, burn=0, seed=1,
                lower=float("-inf"), upper=float("inf"), thin=1):
    r"""A univariate slice-sampling chain."""
    if int(n) < 1:
        raise ValueError("baygsl: need at least one draw")
    if int(thin) < 1:
        raise ValueError("baygsl: thin must be at least 1")
    rng = _rng(seed)
    x = float(x0)
    out, evals = [], 0
    total = int(burn) + int(n) * int(thin)
    for i in range(total):
        st = slice_sample_1d(logf, x, rng, w, lower=lower, upper=upper)
        x = st["x"]
        evals += st["n_eval"]
        if i >= int(burn) and (i - int(burn)) % int(thin) == 0:
            out.append(x)
    return {"draws": out, "n_eval": evals, "w": float(w),
            "evals_per_draw": evals / float(len(out))}


def effective_sample_size(x):
    r"""ESS from the initial-positive-sequence autocorrelations."""
    v = [float(t) for t in x]
    n = len(v)
    if n < 4:
        raise ValueError("baygsl: too few draws for an ESS")
    m = sum(v) / n
    d = [t - m for t in v]
    c0 = sum(t * t for t in d) / n
    if c0 <= 0:
        return float(n)
    s = 0.0
    for lag in range(1, min(n - 2, 1000)):
        c = sum(d[i] * d[i + lag] for i in range(n - lag)) / n
        r = c / c0
        if r < 0.05:
            break
        s += r
    return n / (1.0 + 2.0 * s)


def gibbs_slice(log_conditionals, x0, n=2000, w=None, burn=0, seed=1,
                bounds=None):
    r"""A Gibbs sweep in which every coordinate is drawn by slice.

    ``log_conditionals[k](value, state)`` returns the log of the
    unnormalised full conditional of coordinate ``k``.
    """
    p = len(x0)
    if len(log_conditionals) != p:
        raise ValueError("baygsl: %d conditionals for %d coordinates"
                         % (len(log_conditionals), p))
    if p == 0:
        raise ValueError("baygsl: no coordinates to sample")
    ws = [1.0] * p if w is None else ([float(w)] * p
                                      if not isinstance(w, (list,
                                                            tuple))
                                      else [float(t) for t in w])
    bd = [(float("-inf"), float("inf"))] * p if bounds is None \
        else [(float(a), float(b)) for a, b in bounds]
    rng = _rng(seed)
    state = [float(t) for t in x0]
    keep, evals = [], 0
    for i in range(int(burn) + int(n)):
        for k in range(p):
            def lf(v, k=k):
                s = list(state)
                s[k] = v
                return log_conditionals[k](v, s)
            st = slice_sample_1d(lf, state[k], rng, ws[k],
                                 lower=bd[k][0], upper=bd[k][1])
            state[k] = st["x"]
            evals += st["n_eval"]
        if i >= int(burn):
            keep.append(list(state))
    means = [sum(r[k] for r in keep) / len(keep) for k in range(p)]
    return RichResult(payload={
        "estimate": means, "draws": keep, "mean": means,
        "n_draws": len(keep), "n_eval": evals,
        "ess": [effective_sample_size([r[k] for r in keep])
                for k in range(p)],
        "method": "Gibbs sweep with a slice update per coordinate "
                  "(Damien, Wakefield & Walker 1999; Neal 2003)",
    })


def hybrid_gibbs_slice(log_conditionals, x0, n=2000, **kw):
    r"""Entry point: see :func:`gibbs_slice`."""
    return gibbs_slice(log_conditionals, x0, n, **kw)


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
gibbsslice = gibbs_slice
