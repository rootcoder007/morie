# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model, identify, estimate, refute: the four-step causal workflow.

Sharma A, Kiciman E (2020), *DoWhy: an end-to-end library for causal
inference*, arXiv:2011.04216; the identification step follows Pearl J
(2009), *Causality*, 2nd ed., Sec 3.3 (back-door criterion) and
Shpitser I, VanderWeele T, Robins JM (2010) on adjustment.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["model_identify_estimate_refute", "backdoor_sets",
           "is_backdoor_admissible"]

_METHOD = "Model-identify-estimate-refute causal workflow"


def _parents(adj, j):
    return set(np.flatnonzero(adj[:, j]).tolist())


def _descendants(adj, start):
    """All nodes reachable from ``start`` by directed edges."""
    seen, stack = set(), [start]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]).tolist():
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _reachable(adj, x, Z):
    """Bayes-ball reachability: nodes d-connected to x given Z.

    Koller and Friedman (2009) Alg 3.1. A state is a (node, direction)
    pair where direction records how the node was entered: "up" means
    arrived from a child, travelling against an arrow, and "down" means
    arrived from a parent. The asymmetry is the whole algorithm --
    conditioning blocks chains and forks but OPENS colliders, so a
    collider passes the ball on only when it, or one of its
    descendants, is in Z.
    """
    A = np.asarray(adj, dtype=bool)
    Z = set(int(z) for z in Z)
    anc = set()
    stack = list(Z)
    while stack:
        u = stack.pop()
        if u in anc:
            continue
        anc.add(u)
        stack.extend(np.flatnonzero(A[:, u]).tolist())

    seen = set()
    reach = set()
    frontier = [(int(x), "up")]
    while frontier:
        u, d = frontier.pop()
        if (u, d) in seen:
            continue
        seen.add((u, d))
        if u not in Z:
            reach.add(u)
        if d == "up" and u not in Z:
            for pa in np.flatnonzero(A[:, u]).tolist():
                frontier.append((pa, "up"))
            for ch in np.flatnonzero(A[u]).tolist():
                frontier.append((ch, "down"))
        elif d == "down":
            if u not in Z:
                for ch in np.flatnonzero(A[u]).tolist():
                    frontier.append((ch, "down"))
            if u in anc:
                for pa in np.flatnonzero(A[:, u]).tolist():
                    frontier.append((pa, "up"))
    reach.discard(int(x))
    return reach


def _d_connected(adj, x, y, Z):
    """Is x d-connected to y given Z?"""
    return int(y) in _reachable(adj, x, Z)


def is_backdoor_admissible(adj, treatment, outcome, Z):
    """Does ``Z`` satisfy Pearl's back-door criterion?

    Two conditions: no member of ``Z`` is a descendant of the
    treatment, and ``Z`` blocks every path from treatment to outcome
    that starts with an arrow *into* the treatment.
    """
    A = np.asarray(adj, dtype=bool)
    Z = set(int(z) for z in Z)
    t, y = int(treatment), int(outcome)
    if t in Z or y in Z:
        return False
    desc = _descendants(A, t)
    if Z & desc:
        return False
    # delete arrows out of the treatment; d-separation in that graph is
    # exactly "every back-door path is blocked"
    B = A.copy()
    B[t, :] = False
    return not _d_connected(B, t, y, Z)


def backdoor_sets(adj, treatment, outcome, candidates=None, max_size=None):
    """Enumerate admissible adjustment sets, smallest first.

    The parents of the treatment are admissible whenever the graph is
    causally sufficient, and are returned first when they qualify. The
    empty set is checked too -- if it is admissible there is no
    confounding to adjust for, and adjusting anyway costs precision.
    """
    A = np.asarray(adj, dtype=bool)
    n = A.shape[0]
    t, y = int(treatment), int(outcome)
    desc = _descendants(A, t)
    pool = ([i for i in range(n) if i not in (t, y) and i not in desc]
            if candidates is None
            else [int(c) for c in candidates])
    lim = len(pool) if max_size is None else int(max_size)
    found = []
    from itertools import combinations
    for k in range(0, min(lim, len(pool)) + 1):
        for combo in combinations(pool, k):
            if is_backdoor_admissible(A, t, y, combo):
                found.append(tuple(sorted(combo)))
        if found and k >= 1:
            break        # smallest sufficient sets are enough
    return found


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def model_identify_estimate_refute(dag, data, treatment, outcome,
                                   estimator="backdoor", adjustment=None,
                                   n_refute=100, seed=0, alpha=0.05):
    r"""Run the four steps and report where the answer came from.

    The value of the workflow is not the number at step three. It is
    that steps two and four are separated from it: identification says
    *which* quantity the data can speak to, estimation produces it, and
    refutation asks whether it survives perturbations that ought not to
    change it.

    **Model.** ``dag`` is an adjacency matrix with ``dag[i, j]`` true
    when :math:`i \to j`. It is an assumption, supplied by the analyst,
    and nothing here tests it.

    **Identify.** The back-door criterion is checked properly by
    d-separation rather than by the common shortcut of "adjust for
    everything measured". That shortcut is not conservative: adjusting
    for a collider, or for a mediator, *creates* bias where there was
    none. ``adjusted_for_collider`` and ``adjusted_for_mediator`` flag
    both cases, and the estimate is still returned so the damage can be
    seen rather than merely warned about.

    **Estimate.** Regression adjustment on the identified set.

    **Refute.** Three checks, each with a known correct answer:

    * *Placebo treatment* -- replace the treatment with a permutation
      of itself. The effect should collapse to zero. If it does not,
      the estimate is being driven by something other than treatment
      variation.
    * *Random common cause* -- add an independent covariate to the
      adjustment set. The estimate should not move; a large shift means
      it is unstable to irrelevant conditioning.
    * *Subset* -- refit on random 80 per cent subsets. The spread
      should be comparable to the reported standard error.

    A refutation that passes is not proof of anything. It rules out
    specific failures, and the ones it rules out are the ones listed.

    Parameters
    ----------
    dag : array-like, shape (p, p), boolean
        Adjacency matrix over the columns of ``data``.
    data : array-like, shape (n, p)
        Observations, one column per graph node.
    treatment, outcome : int
        Column indices.
    estimator : {"backdoor"}
    adjustment : sequence of int, optional
        Force a particular adjustment set instead of identifying one.
    n_refute : int
        Replications for each refutation.
    seed : int
    alpha : float

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci_lower``/``ci_upper``,
        ``identified``, ``adjustment_set``, ``all_backdoor_sets``,
        ``placebo_effect``, ``random_cause_effect``,
        ``subset_sd``, ``refutations_passed``.

    References
    ----------
    Pearl J (2009) *Causality*, 2nd ed., Sec 3.3.
    Sharma A, Kiciman E (2020) arXiv:2011.04216.
    """
    A = np.asarray(dag, dtype=bool)
    D = np.atleast_2d(np.asarray(data, dtype=float))
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"dag must be square; got shape {A.shape}.")
    p = A.shape[0]
    if D.shape[1] != p:
        if D.shape[0] == p:
            D = D.T
        else:
            raise ValueError(
                f"data has {D.shape[1]} columns but the dag has {p} nodes."
            )
    n = D.shape[0]
    t, y = int(treatment), int(outcome)
    if not 0 <= t < p or not 0 <= y < p:
        raise ValueError(f"treatment and outcome must lie in 0 .. {p - 1}.")
    if t == y:
        raise ValueError("treatment and outcome must differ.")
    if estimator != "backdoor":
        raise ValueError('only estimator="backdoor" is implemented.')
    if np.any(A.diagonal()):
        raise ValueError("the dag has a self-loop.")
    if n < p + 3:
        raise ValueError(f"need more rows than nodes; got {n} rows, {p} nodes.")

    # ---- identify -------------------------------------------------
    sets = backdoor_sets(A, t, y)
    if adjustment is not None:
        Z = tuple(sorted(int(z) for z in adjustment))
        identified = is_backdoor_admissible(A, t, y, Z)
    elif sets:
        Z = sets[0]
        identified = True
    else:
        Z = tuple()
        identified = False

    desc_t = _descendants(A, t)
    # a node with two or more parents is a collider; conditioning on it
    # opens the path between them rather than blocking it
    colliders = [z for z in Z if len(_parents(A, z)) >= 2]
    mediators = [z for z in Z if z in desc_t]

    def fit(dat, tcol, zcols):
        X = np.column_stack([np.ones(dat.shape[0]), dat[:, tcol]]
                            + [dat[:, c] for c in zcols])
        b = _ols(X, dat[:, y])
        resid = dat[:, y] - X @ b
        dof = max(dat.shape[0] - X.shape[1], 1)
        s2 = float(resid @ resid) / dof
        XtXi = np.linalg.pinv(X.T @ X)
        return float(b[1]), float(math.sqrt(max(s2 * XtXi[1, 1], 0.0)))

    eff, se = fit(D, t, list(Z))

    # ---- refute ---------------------------------------------------
    rng = np.random.default_rng(seed)
    placebo = np.empty(n_refute)
    common = np.empty(n_refute)
    subset = np.empty(n_refute)
    for i in range(n_refute):
        Dp = D.copy()
        Dp[:, t] = rng.permutation(D[:, t])
        placebo[i] = fit(Dp, t, list(Z))[0]

        Dc = np.column_stack([D, rng.normal(size=n)])
        common[i] = fit(Dc, t, list(Z) + [p])[0]

        idx = rng.choice(n, size=max(int(0.8 * n), p + 3), replace=False)
        subset[i] = fit(D[idx], t, list(Z))[0]

    placebo_mean = float(np.mean(placebo))
    placebo_sd = float(np.std(placebo, ddof=1))
    common_mean = float(np.mean(common))
    subset_sd = float(np.std(subset, ddof=1))

    pass_placebo = abs(placebo_mean) < 2.0 * max(placebo_sd, 1e-12)
    pass_common = abs(common_mean - eff) < max(0.1 * abs(eff), 2 * se)
    pass_subset = subset_sd < 3.0 * max(se, 1e-12)
    passed = int(pass_placebo) + int(pass_common) + int(pass_subset)

    zc = _z(1 - alpha / 2)
    out = RichResult(
        title="Model, identify, estimate, refute",
        summary_lines=[
            ("Effect", eff),
            ("SE", se),
            ("Identified", identified),
            ("Adjustment set", list(Z)),
            ("Refutations passed", f"{passed} of 3"),
        ],
        tables=[{
            "title": "Refutation",
            "headers": ["Check", "Expected", "Observed", "Passed"],
            "rows": [
                ["Placebo treatment", 0.0, placebo_mean, pass_placebo],
                ["Random common cause", eff, common_mean, pass_common],
                ["80% subsets (sd)", se, subset_sd, pass_subset],
            ],
        }],
        payload={
            "estimate": eff,
            "se": se,
            "ci_lower": eff - zc * se,
            "ci_upper": eff + zc * se,
            "identified": identified,
            "adjustment_set": list(Z),
            "all_backdoor_sets": sets,
            "n_backdoor_sets": len(sets),
            "adjusted_for_collider": colliders,
            "adjusted_for_mediator": mediators,
            "placebo_effect": placebo_mean,
            "placebo_sd": placebo_sd,
            "random_cause_effect": common_mean,
            "subset_sd": subset_sd,
            "passed_placebo": pass_placebo,
            "passed_random_cause": pass_common,
            "passed_subset": pass_subset,
            "refutations_passed": passed,
            "n": n,
            "method": _METHOD,
        },
        interpretation=(
            f"Adjusting for {list(Z)} identifies the effect as {eff:.4f}; "
            f"{passed} of 3 refutations passed."
            if identified else
            "No admissible adjustment set exists in this graph, so the "
            "effect is NOT identified and the number below is a regression "
            "coefficient, not a causal effect."
        ),
    )
    if not identified:
        out.warnings.append(
            "The back-door criterion is not satisfied by any subset of the "
            "measured variables. Adjusting anyway does not make the estimate "
            "causal; it makes it a different biased number."
        )
    if mediators:
        out.warnings.append(
            f"Node(s) {mediators} in the adjustment set are descendants of "
            "the treatment. Conditioning on a mediator removes part of the "
            "effect being measured, so this estimate is attenuated by "
            "construction."
        )
    if colliders:
        out.warnings.append(
            f"Node(s) {colliders} in the adjustment set are colliders. "
            "Conditioning on a collider CREATES an association between its "
            "parents where none existed, so adjusting here adds bias rather "
            "than removing it."
        )
    if not pass_placebo:
        out.warnings.append(
            f"The placebo treatment returned {placebo_mean:.4g} rather than "
            "zero. Something other than treatment variation is driving the "
            "estimate."
        )
    return out


def _z(q):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return (
        "midor: four-step causal workflow -- back-door identification by "
        "d-separation, regression estimation, then placebo, random-common-"
        "cause and subset refutations"
    )
