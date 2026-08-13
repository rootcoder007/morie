# morie.fn -- function file (rootcoder007/morie)
r"""Bayesian optimisation of an expensive black-box function.

Mockus, J. (1975) "On Bayesian methods for seeking the extremum", in
*Optimization Techniques IFIP Technical Conference*, 400-404.

Snoek, J., Larochelle, H., & Adams, R. P. (2012) "Practical Bayesian
Optimization of Machine Learning Algorithms", *NIPS 25*.
arXiv:1206.2944

The setting is a function that costs too much to evaluate often and
whose gradient is unavailable. Put a Gaussian process prior on it, and
after each evaluation use the posterior to decide where to look next.
The posterior gives a mean :math:`\mu(x)` and a variance
:math:`\sigma^2(x)` everywhere; an *acquisition function* turns those
two numbers into a single score, and the next point is its arg max.

With :math:`x_{best} = \arg\min_{x_n} f(x_n)` and
:math:`\gamma(x) = \dfrac{f(x_{best}) - \mu(x)}{\sigma(x)}`, the paper
gives all three in closed form (Equations 1-3, minimisation
throughout):

``"pi"``
    probability of improvement (Kushner),
    :math:`a_{PI} = \Phi(\gamma)`.
``"ei"`` (the default)
    expected improvement,
    :math:`a_{EI} = \sigma\,[\gamma\,\Phi(\gamma) + \phi(\gamma)]`.
    Snoek et al. focus on it: "better-behaved than probability of
    improvement" and, unlike UCB, it "does not require its own tuning
    parameter".
``"lcb"``
    lower confidence bound (Srinivas et al.),
    :math:`a_{LCB} = \mu - \kappa\sigma`, minimised rather than
    maximised, with :math:`\kappa` trading exploitation against
    exploration.

**The kernel matters more than the acquisition.** Equation 4 is the ARD
squared exponential, the usual default; the paper argues against it
because "sample functions with this covariance function are
unrealistically smooth for practical optimization problems", and
proposes the ARD Matérn 5/2 of Equation 5,

.. math::

   K_{M52}(x, x') = \theta_0\left(1 + \sqrt{5 r^2} +
   \tfrac{5}{3} r^2\right)\exp\left\{-\sqrt{5 r^2}\right\},
   \qquad r^2 = \sum_d (x_d - x'_d)^2/\theta_d^2,

whose sample paths are twice differentiable -- the assumption
quasi-Newton methods make. Both are here, Matérn 5/2 by default, and
``length_scale`` may be a scalar or one value per dimension (the ARD
part: each input gets its own relevance).

The GP is exact rather than approximate: the posterior is the standard
conditional of a joint Gaussian, so with ``noise = 0`` it interpolates
the observations exactly.
"""

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = [
    "bayopt",
    "bayesian_optimization",
    "gp_posterior",
    "matern52",
    "squared_exponential",
    "expected_improvement",
    "probability_of_improvement",
    "lower_confidence_bound",
    "acquire",
]

_KERNELS = ("matern52", "se")
_ACQ = ("ei", "pi", "lcb")


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _lengths(ls, d):
    if isinstance(ls, (int, float)):
        out = [float(ls)] * d
    else:
        out = [float(v) for v in ls]
    if len(out) != d:
        raise ValueError("bayopt: length_scale must be a scalar or one "
                         "value per dimension")
    if any(v <= 0 for v in out):
        raise ValueError("bayopt: length scales must be positive")
    return out


def _r2(a, b, ls):
    return sum((a[d] - b[d]) ** 2 / (ls[d] ** 2) for d in range(len(a)))


def matern52(a, b, amplitude=1.0, length_scale=1.0):
    r"""Equation 5: the ARD Matern 5/2 kernel."""
    ls = _lengths(length_scale, len(a))
    r2 = _r2(a, b, ls)
    s = math.sqrt(5.0 * r2)
    return amplitude * (1.0 + s + (5.0 / 3.0) * r2) * math.exp(-s)


def squared_exponential(a, b, amplitude=1.0, length_scale=1.0):
    r"""Equation 4: the ARD squared exponential kernel."""
    ls = _lengths(length_scale, len(a))
    return amplitude * math.exp(-0.5 * _r2(a, b, ls))


def _kernel(name):
    if name not in _KERNELS:
        raise ValueError("bayopt: kernel must be one of %s" % (_KERNELS,))
    return matern52 if name == "matern52" else squared_exponential


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("bayopt: the covariance matrix is not "
                                     "positive definite; add noise or "
                                     "spread the design points")
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _chol_solve(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / \
            L[i][i]
    return x


def gp_posterior(X, y, Xs, kernel="matern52", amplitude=1.0,
                 length_scale=1.0, noise=1e-8, mean=None):
    """Posterior mean and variance of the GP at each row of ``Xs``."""
    rows = [[float(v) for v in r] for r in X]
    if not rows:
        raise ValueError("bayopt: no observations")
    d = len(rows[0])
    if any(len(r) != d for r in rows):
        raise ValueError("bayopt: X is ragged")
    ys = [float(v) for v in y]
    if len(ys) != len(rows):
        raise ValueError("bayopt: one observation per design point")
    if noise < 0:
        raise ValueError("bayopt: noise must be non-negative")
    k = _kernel(kernel)
    m = (sum(ys) / len(ys)) if mean is None else float(mean)
    n = len(rows)
    K = [[k(rows[i], rows[j], amplitude, length_scale) +
          (noise if i == j else 0.0) for j in range(n)] for i in range(n)]
    L = _chol(K)
    alpha = _chol_solve(L, [v - m for v in ys])
    out_m, out_v = [], []
    for xs in Xs:
        q = [float(v) for v in xs]
        if len(q) != d:
            raise ValueError("bayopt: a query point has the wrong "
                             "dimension")
        ks = [k(q, rows[i], amplitude, length_scale) for i in range(n)]
        mu = m + sum(ks[i] * alpha[i] for i in range(n))
        v = _chol_solve(L, ks)
        var = k(q, q, amplitude, length_scale) - \
            sum(ks[i] * v[i] for i in range(n))
        out_m.append(mu)
        out_v.append(max(var, 0.0))
    return {"mean": out_m, "variance": out_v,
            "sd": [math.sqrt(v) for v in out_v]}


def probability_of_improvement(mu, sd, best, xi=0.0):
    r"""Equation 1: :math:`\Phi(\gamma)`."""
    if sd <= 0:
        return 0.0
    return _Phi((best - xi - mu) / sd)


def expected_improvement(mu, sd, best, xi=0.0):
    r"""Equation 2: :math:`\sigma[\gamma\Phi(\gamma) + \phi(\gamma)]`."""
    if sd <= 0:
        return 0.0
    g = (best - xi - mu) / sd
    return sd * (g * _Phi(g) + _phi(g))


def lower_confidence_bound(mu, sd, kappa=2.0):
    r"""Equation 3: :math:`\mu - \kappa\sigma`, to be minimised."""
    return mu - kappa * sd


def acquire(mu, sd, best, acq="ei", kappa=2.0, xi=0.0):
    """Score one candidate; higher is always better here.

    LCB is negated so that every route is a maximisation, which is what
    lets the loop treat them alike.
    """
    if acq not in _ACQ:
        raise ValueError("bayopt: acq must be one of %s" % (_ACQ,))
    if acq == "ei":
        return expected_improvement(mu, sd, best, xi)
    if acq == "pi":
        return probability_of_improvement(mu, sd, best, xi)
    return -lower_confidence_bound(mu, sd, kappa)


def bayopt(f, bounds, n_iter=20, n_init=5, acq="ei", kernel="matern52",
           amplitude=1.0, length_scale=1.0, noise=1e-8, kappa=2.0, xi=0.0,
           n_candidates=200, seed=0, X0=None, y0=None):
    """Minimise ``f`` over a box by Bayesian optimisation.

    ``bounds`` is a list of ``(lo, hi)`` per dimension. The acquisition
    is maximised over a random candidate set of ``n_candidates`` points
    each round, which is the simple inner search; the paper uses a
    gradient method there, which changes speed rather than definition.
    """
    if acq not in _ACQ:
        raise ValueError("bayopt: acq must be one of %s" % (_ACQ,))
    box = [(float(a), float(b)) for a, b in bounds]
    if not box:
        raise ValueError("bayopt: bounds are empty")
    if any(a >= b for a, b in box):
        raise ValueError("bayopt: each bound must have lo < hi")
    if n_iter < 1 or n_candidates < 1:
        raise ValueError("bayopt: n_iter and n_candidates must be "
                         "positive")
    if X0 is None and n_init < 2:
        raise ValueError("bayopt: at least two initial points are needed")
    d = len(box)
    st = [int(seed) & 0x7FFFFFFF or 1]

    def rnd():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)

    def draw():
        return [box[i][0] + rnd() * (box[i][1] - box[i][0])
                for i in range(d)]

    if X0 is not None:
        X = [[float(v) for v in r] for r in X0]
        Y = [float(v) for v in y0] if y0 is not None else [f(x) for x in X]
        if len(X) != len(Y):
            raise ValueError("bayopt: X0 and y0 have different lengths")
    else:
        X = [draw() for _ in range(int(n_init))]
        Y = [float(f(x)) for x in X]
    trace = []
    for _ in range(int(n_iter)):
        best = min(Y)
        cand = [draw() for _ in range(int(n_candidates))]
        post = gp_posterior(X, Y, cand, kernel, amplitude, length_scale,
                            noise)
        scores = [acquire(post["mean"][i], post["sd"][i], best, acq,
                          kappa, xi) for i in range(len(cand))]
        k = max(range(len(cand)), key=lambda i: scores[i])
        x_new = cand[k]
        X.append(x_new)
        Y.append(float(f(x_new)))
        trace.append({"x": x_new, "y": Y[-1], "acq": scores[k],
                      "best": min(Y)})
    b = min(range(len(Y)), key=lambda i: Y[i])
    return RichResult(payload={
        "estimate": X[b],
        "x_best": X[b],
        "y_best": Y[b],
        "X": X,
        "y": Y,
        "trace": trace,
        "acq": acq,
        "kernel": kernel,
        "n_eval": len(Y),
        "method": ("Bayesian optimisation (Mockus 1975; Snoek, "
                   "Larochelle & Adams 2012) with a %s kernel and the "
                   "%s acquisition" % (kernel, acq)),
        "note": ("minimisation throughout, as the paper writes it "
                 "(x_best = argmin); the acquisition is maximised over a "
                 "random candidate set rather than by a gradient method, "
                 "which changes the inner search's speed and not the "
                 "criterion"),
    })


bayesian_optimization = bayopt


def cheatsheet():
    return ("bayopt: Bayesian optimisation (Mockus 1975; Snoek et al. "
            "2012). A GP posterior gives mu(x) and sigma(x); with "
            "gamma = (f(x_best) - mu)/sigma the acquisitions are "
            "PI = Phi(gamma) (eq.1), EI = sigma[gamma Phi(gamma) + "
            "phi(gamma)] (eq.2) and LCB = mu - kappa sigma (eq.3). The "
            "kernel is the ARD Matern 5/2 of eq.5 by default, which the "
            "paper prefers over the squared exponential of eq.4 because "
            "the latter's sample paths are unrealistically smooth.")
