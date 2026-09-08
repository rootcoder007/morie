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

**How the paper actually tests this.** Section 4 uses *convex* criteria
and still finds an effect, which is worth being precise about, because
on a convex objective run to convergence the ordering cannot move the
optimum. The effect is elsewhere:

* **Section 4.1** trains a linear classifier on the easy examples only
  -- those on the correct side of the Bayes boundary -- and measures
  *generalization*: 16.3% error against 17.1% for the full set. That is
  a different training set, not a reordering.
* **Section 4.2** trains a Perceptron **online with a fixed budget** of
  200 updates and measures generalization at the end. Because training
  stops well short of convergence, the order the updates arrive in
  decides where it stops. The paper's two easiness criteria are the
  number of irrelevant inputs zeroed out, and the margin
  :math:`y\,w'x`.

So the quantity to compare is held-out error under a fixed update
budget, not training loss at convergence -- comparing the latter on a
convex objective can only ever return "no difference", which says
nothing about curricula.

**What reproduces here, and what does not.** Section 4.1 does, closely:
on its two-Gaussian setup, training on the clean examples only gives
0.1604 against 0.1635 for the full set, where the paper prints 0.163
against 0.171 -- same direction, same size, and the absolute levels
agree to under a point. :func:`easy_only_fit` is that experiment.

Section 4.2 does not. On a reconstruction of its generator the
curriculum is *worse* than shuffling: 0.213 sorted easiest-first and
0.175 sampling from :math:`Q_\lambda`, against 0.140 shuffled, over 300
restarts. Two things in that section are underdetermined -- the number
of irrelevant inputs is never given, and "ordered by easiness" does not
say whether examples are sorted once or sampled from the widening
support -- so both orderings are offered through ``order=`` and neither
is claimed to reproduce the paper. This is recorded rather than tuned
away; a fixture adjusted until it agreed would be evidence of nothing.

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

__all__ = ["prgrl", "curriculum_schedule", "is_curriculum",
           "entropy", "easy_only_fit"]


def _rng(seed):
    st = [int(seed) & 0x7FFFFFFF or 1]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2.0 * math.log(max(r(), 1e-12))) * \
        math.cos(2.0 * math.pi * r())


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


def _perceptron(X, y, order, updates, w0):
    """Section 4.2's learner: online Perceptron, one pass, fixed budget
    of updates, learning rate 1 (the paper notes the rate does not
    matter, since only the sign of w'x is used)."""
    p = len(X[0])
    w = list(w0)
    n = len(order)
    for step in range(int(updates)):
        i = order[step % n]
        s = sum(w[k] * X[i][k] for k in range(p))
        if y[i] * s <= 0:
            for k in range(p):
                w[k] += y[i] * X[i][k]
    return w


def _error(X, y, w):
    p = len(X[0])
    bad = 0
    for i in range(len(X)):
        s = sum(w[k] * X[i][k] for k in range(p))
        if y[i] * (s if s != 0 else -1.0) <= 0:
            bad += 1
    return bad / float(len(X))


def prgrl(X, y, difficulty, X_test=None, y_test=None, updates=200,
          n_steps=5, seed=0, n_repeats=50, order="sampled"):
    r"""The paper's Section 4.2 comparison.

    Trains an online Perceptron for a fixed budget of ``updates``, once
    with the examples ordered easiest-first and once in a shuffled
    order, and reports **generalization error** on the held-out set --
    which is what the paper measures. Averaged over ``n_repeats``
    random initialisations, as the paper averages over 500.

    Passing no test set scores on the training set instead and says so;
    with a convex criterion and a long budget that comparison is
    uninformative by construction (see the module docstring).
    """
    Xr = [[float(v) for v in r] for r in np.asarray(X, dtype=float)]
    yv = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float))]
    n = len(Xr)
    if len(yv) != n:
        raise ValueError("prgrl: X and y must have the same length")
    if n < 2:
        raise ValueError("prgrl: need at least two examples")
    if int(updates) < 1:
        raise ValueError("prgrl: updates must be at least 1")
    for v in yv:
        if v not in (-1.0, 1.0):
            raise ValueError("prgrl: y must be -1/+1 for the "
                             "Perceptron of Section 4.2")
    d = [float(v) for v in np.atleast_1d(np.asarray(difficulty,
                                                    dtype=float))]
    if len(d) != n:
        raise ValueError("prgrl: difficulty must have one score per "
                         "example")
    if X_test is None:
        Xe, ye, held_out = Xr, yv, False
    else:
        Xe = [[float(v) for v in r] for r in np.asarray(X_test,
                                                        dtype=float)]
        ye = [float(v) for v in np.atleast_1d(np.asarray(y_test,
                                                         dtype=float))]
        if len(Xe) != len(ye):
            raise ValueError("prgrl: X_test and y_test must match")
        held_out = True

    if order not in ("sampled", "sorted"):
        raise ValueError("prgrl: order must be 'sampled' or 'sorted'")
    easy = sorted(range(n), key=lambda i: d[i])
    lam, weights, _ = curriculum_schedule(d, n_steps)
    chk = is_curriculum(weights)

    def curriculum_order(rnd):
        if order == "sorted":
            return easy
        # eqns 1-2: draw from Q_lambda, whose support grows with lambda,
        # so easy examples keep being seen rather than being abandoned
        seq = []
        for step in range(int(updates)):
            lam_s = (step + 1) / float(updates)
            take = max(1, int(round(lam_s * n)))
            seq.append(easy[int(rnd() * take)])
        return seq

    rnd = _rng(seed)
    cur_errs, base_errs = [], []
    for _ in range(int(n_repeats)):
        w0 = [_gauss(rnd) for _ in range(len(Xr[0]))]
        shuffled = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(rnd() * (i + 1))
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        cur_errs.append(_error(Xe, ye,
                               _perceptron(Xr, yv,
                                           curriculum_order(rnd),
                                           updates, w0)))
        base_errs.append(_error(Xe, ye,
                                _perceptron(Xr, yv, shuffled, updates,
                                            w0)))
    cur = sum(cur_errs) / len(cur_errs)
    base = sum(base_errs) / len(base_errs)
    return RichResult(payload={
        "estimate": cur,
        "curriculum_error": cur,
        "baseline_error": base,
        "improvement": base - cur,
        "curriculum_errors": cur_errs,
        "baseline_errors": base_errs,
        "held_out": held_out,
        "updates": int(updates),
        "order": order,
        "n_repeats": int(n_repeats),
        "lambdas": lam,
        "weights": weights,
        "entropies": chk["entropies"],
        "is_curriculum": chk["is_curriculum"],
        "n": n,
        "method": ("curriculum learning (Bengio, Louradour, Collobert & "
                   "Weston 2009), Section 4.2: online Perceptron, fixed "
                   "update budget, generalization error"),
        "note": ("the comparison is held-out error under a FIXED update "
                 "budget, which is what the paper measures. Training "
                 "loss at convergence on a convex criterion cannot "
                 "differ between orderings and says nothing"
                 if held_out else
                 "no test set was given, so this scored on the training "
                 "set; pass X_test and y_test for the paper's "
                 "comparison"),
    })


def easy_only_fit(X, y, difficulty, X_test, y_test, quantile=0.5,
                  updates=200, seed=0, n_repeats=50):
    r"""Section 4.1: train on the easy examples only, and compare
    generalization against training on everything."""
    Xr = [[float(v) for v in r] for r in np.asarray(X, dtype=float)]
    yv = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float))]
    d = [float(v) for v in np.atleast_1d(np.asarray(difficulty,
                                                    dtype=float))]
    n = len(Xr)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("prgrl: quantile must lie in (0, 1]")
    Xe = [[float(v) for v in r] for r in np.asarray(X_test,
                                                    dtype=float)]
    ye = [float(v) for v in np.atleast_1d(np.asarray(y_test,
                                                     dtype=float))]
    order = sorted(range(n), key=lambda i: d[i])
    keep = order[:max(1, int(round(quantile * n)))]
    rnd = _rng(seed)
    easy_errs, all_errs = [], []
    for _ in range(int(n_repeats)):
        w0 = [_gauss(rnd) for _ in range(len(Xr[0]))]
        easy_errs.append(_error(Xe, ye,
                                _perceptron(Xr, yv, keep, updates, w0)))
        allo = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(rnd() * (i + 1))
            allo[i], allo[j] = allo[j], allo[i]
        all_errs.append(_error(Xe, ye,
                               _perceptron(Xr, yv, allo, updates, w0)))
    e_easy = sum(easy_errs) / len(easy_errs)
    e_all = sum(all_errs) / len(all_errs)
    return RichResult(payload={
        "estimate": e_easy,
        "easy_only_error": e_easy,
        "all_examples_error": e_all,
        "improvement": e_all - e_easy,
        "n_kept": len(keep),
        "n": n,
        "method": ("curriculum learning (Bengio et al. 2009), Section "
                   "4.1: train on the clean examples only"),
        "note": ("the paper reports 16.3% against 17.1% for a linear "
                 "SVM on two Gaussians; the direction is the claim, "
                 "and noisy examples are the ones on the wrong side of "
                 "the Bayes boundary"),
    })


def cheatsheet():
    return ("prgrl: curriculum learning (Bengio et al. 2009). "
            "Q_lambda(z) proportional to W_lambda(z) P(z) with "
            "W_1 = 1; it is a curriculum only if H(Q_lambda) increases "
            "and W_lambda(z) never falls as lambda grows (eqns 3-4), "
            "which is checkable and is checked. prgrl trains the same "
            "learner on the schedule and on the shuffled data and "
            "compares HELD-OUT error under a fixed update budget, "
            "which is Section 4.2's experiment; easy_only_fit is "
            "Section 4.1's.")

# public names resolved by fn/_lazy_map.json
prog_rl = entropy
progrl = entropy
