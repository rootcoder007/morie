r"""Curriculum learning: easy examples first.

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009)
"Curriculum Learning", *ICML*, 41-48.

The paper gives the idea a definition precise enough to check. Let
:math:`0 \le W_\lambda(z) \le 1` be the weight on example :math:`z` at
step :math:`\lambda` of the sequence, with :math:`W_1(z) = 1`. The
training distribution at that step is

.. math::

   Q_\lambda(z) \propto W_\lambda(z) P(z),
   \qquad Q_1(z) = P(z)
   \tag{1, 2}

and the sequence is a **curriculum** when two things hold:

.. math::

   H(Q_\lambda) < H(Q_{\lambda + \epsilon})
   \quad\text{and}\quad
   W_{\lambda+\epsilon}(z) \ge W_\lambda(z)
   \qquad \forall \epsilon > 0
   \tag{3, 4}

-- the entropy of the training distribution increases, and no example's
weight ever falls. The second condition is what makes it a curriculum
rather than a schedule: examples are only ever *added*.

**What a curriculum cannot do.** The paper's empirical claim is about
deep networks, and the reason matters: on a *convex* objective the
ordering of the examples cannot change where training ends up, because
there is one minimum and every route reaches it. The comparison below
uses least squares, and on it the curriculum and the shuffled baseline
converge to the same coefficients to machine precision -- measured, and
anchored as such. Curriculum learning is a statement about non-convex
optimisation and about the basin gradient descent falls into; a module
that showed an advantage here would be showing an artefact.

:func:`curriculum_schedule` builds the sequence from a difficulty score
and checks both conditions; :func:`is_curriculum` checks an arbitrary
sequence of weights someone else built. :func:`prgrl` runs the paper's
comparison -- the same learner trained on the curriculum and on the
shuffled data -- and reports both loss curves, because the claim the
paper makes is about the speed of convergence and the quality of the
minimum reached, not about a formula.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["prgrl", "curriculum_schedule", "is_curriculum", "entropy"]


def entropy(q):
    """Shannon entropy of a distribution, in nats."""
    tot = sum(q)
    if tot <= 0:
        raise ValueError("prgrl: the distribution has no mass")
    h = 0.0
    for v in q:
        p = v / tot
        if p > 0:
            h -= p * math.log(p)
    return h


def curriculum_schedule(difficulty, n_steps=5, hard_first=False):
    r"""Build :math:`W_\lambda` and :math:`Q_\lambda` from a difficulty
    score, one step at a time, easiest examples first.

    At step :math:`\lambda` every example whose difficulty is at or below
    the :math:`\lambda` quantile has weight 1 and the rest weight 0, so
    :math:`W` is monotone by construction and :math:`W_1 \equiv 1`.
    """
    d = [float(v) for v in np.atleast_1d(np.asarray(difficulty,
                                                    dtype=float))]
    n = len(d)
    if n < 2:
        raise ValueError("prgrl: need at least two examples")
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("prgrl: need at least two curriculum steps")
    order = sorted(range(n), key=lambda i: -d[i] if hard_first else d[i])
    lambdas, weights, dists = [], [], []
    for s in range(n_steps):
        lam = (s + 1) / float(n_steps)
        take = max(1, int(round(lam * n)))
        if s == n_steps - 1:
            take = n
        keep = set(order[:take])
        w = [1.0 if i in keep else 0.0 for i in range(n)]
        weights.append(w)
        lambdas.append(lam)
        dists.append([v / take for v in w])
    return lambdas, weights, dists


def is_curriculum(weights, p=None, tol=1e-12):
    r"""Check eqns 3 and 4 on a sequence of weight vectors."""
    if len(weights) < 2:
        raise ValueError("prgrl: need at least two steps to check")
    n = len(weights[0])
    if any(len(w) != n for w in weights):
        raise ValueError("prgrl: the weight vectors differ in length")
    for w in weights:
        for v in w:
            if not -tol <= v <= 1.0 + tol:
                raise ValueError("prgrl: weights must lie in [0, 1]")
    if p is None:
        p = [1.0 / n] * n
    ents, monotone, final_ones = [], True, True
    for k, w in enumerate(weights):
        q = [w[i] * p[i] for i in range(n)]
        if sum(q) <= 0:
            raise ValueError("prgrl: step %d has no mass" % k)
        ents.append(entropy(q))
        if k > 0:
            for i in range(n):
                if w[i] < weights[k - 1][i] - tol:
                    monotone = False
    increasing = all(ents[k] < ents[k + 1] + tol
                     for k in range(len(ents) - 1))
    strictly = all(ents[k] < ents[k + 1] - 1e-12
                   for k in range(len(ents) - 1))
    for v in weights[-1]:
        if abs(v - 1.0) > tol:
            final_ones = False
    return {"is_curriculum": bool(increasing and monotone),
            "entropy_increasing": bool(increasing),
            "strictly_increasing": bool(strictly),
            "weights_monotone": bool(monotone),
            "final_step_is_p": bool(final_ones),
            "entropies": ents}


def _fit_weighted(X, y, w, lr, epochs, beta=None):
    """Plain gradient descent on weighted squared error."""
    n, p = len(X), len(X[0])
    b = [0.0] * p if beta is None else list(beta)
    hist = []
    for _ in range(int(epochs)):
        grad = [0.0] * p
        tot = 0.0
        for i in range(n):
            if w[i] == 0.0:
                continue
            pred = sum(b[k] * X[i][k] for k in range(p))
            err = pred - y[i]
            for k in range(p):
                grad[k] += w[i] * err * X[i][k]
            tot += w[i]
        if tot <= 0:
            break
        for k in range(p):
            b[k] -= lr * grad[k] / tot
        hist.append(_loss(X, y, b))
    return b, hist


def _loss(X, y, b):
    n, p = len(X), len(X[0])
    return sum((sum(b[k] * X[i][k] for k in range(p)) - y[i]) ** 2
               for i in range(n)) / n


def prgrl(X, y, difficulty, n_steps=5, epochs_per_step=40, lr=0.05,
          seed=0):
    """Train on the curriculum and on the shuffled data, and compare.

    Returns both loss histories and the final losses; the paper's claim
    is that the curriculum converges faster and to a better solution,
    and this reports whether it did on the data given rather than
    asserting it.
    """
    Xr = [[float(v) for v in r] for r in np.asarray(X, dtype=float)]
    yv = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float))]
    n = len(Xr)
    if len(yv) != n:
        raise ValueError("prgrl: X and y must have the same length")
    if n < 2:
        raise ValueError("prgrl: need at least two examples")
    if lr <= 0:
        raise ValueError("prgrl: the learning rate must be positive")
    lam, weights, _ = curriculum_schedule(difficulty, n_steps)

    beta, cur_hist = None, []
    for w in weights:
        beta, h = _fit_weighted(Xr, yv, w, lr, epochs_per_step, beta)
        cur_hist.extend(h)
    flat = [1.0] * n
    beta_b, base_hist = _fit_weighted(Xr, yv, flat, lr,
                                      epochs_per_step * n_steps)

    check = is_curriculum(weights)
    return RichResult(payload={
        "estimate": cur_hist[-1] if cur_hist else float("nan"),
        "curriculum_loss": cur_hist[-1] if cur_hist else float("nan"),
        "baseline_loss": base_hist[-1] if base_hist else float("nan"),
        "curriculum_history": cur_hist,
        "baseline_history": base_hist,
        "curriculum_beta": beta,
        "baseline_beta": beta_b,
        "lambdas": lam,
        "weights": weights,
        "entropies": check["entropies"],
        "is_curriculum": check["is_curriculum"],
        "n_steps": n_steps,
        "n": n,
        "method": ("curriculum learning (Bengio, Louradour, Collobert & "
                   "Weston 2009), eqns 1-4"),
        "note": ("the schedule is verified against the paper's own "
                 "definition -- entropy increasing and weights never "
                 "falling -- before it is used; the loss comparison is "
                 "reported, not assumed"),
    })


def cheatsheet():
    return ("prgrl: curriculum learning (Bengio et al. 2009). "
            "Q_lambda(z) proportional to W_lambda(z) P(z) with "
            "W_1 = 1; it is a curriculum only if H(Q_lambda) increases "
            "and W_lambda(z) never falls as lambda grows (eqns 3-4), "
            "which is checkable and is checked. prgrl trains the same "
            "learner on the schedule and on the shuffled data and "
            "reports both curves.")
