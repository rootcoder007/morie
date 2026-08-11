"""Discrete Bayesian network inference by variable elimination."""

from . import _array_core as np  # noqa: F401  (kept for house-style parity)

from ._richresult import RichResult

__all__ = ["baynet", "bayes_network"]


def _factor_from_cpt(child, parents, table, card):
    """CPT -> factor over (parents..., child), row-major nested lists."""
    scope = list(parents) + [child]
    dims = [card[v] for v in scope]
    flat = []

    def _walk(t, depth):
        if depth == len(dims):
            flat.append(float(t))
            return
        if len(t) != dims[depth]:
            raise ValueError(f"CPT for {child!r} has wrong extent at axis {depth}")
        for item in t:
            _walk(item, depth + 1)

    _walk(table, 0)
    return scope, flat


def _f_index(scope, dims, assign):
    idx = 0
    for v, d in zip(scope, dims):
        idx = idx * d + assign[v]
    return idx


def _multiply(f1, f2, card):
    s1, t1 = f1
    s2, t2 = f2
    scope = list(s1) + [v for v in s2 if v not in s1]
    dims = [card[v] for v in scope]
    total = 1
    for d in dims:
        total *= d
    out = [0.0] * total
    d1 = [card[v] for v in s1]
    d2 = [card[v] for v in s2]
    assign = {}
    for flat in range(total):
        rem = flat
        for v, d in zip(reversed(scope), reversed(dims)):
            assign[v] = rem % d
            rem //= d
        out[flat] = t1[_f_index(s1, d1, assign)] * t2[_f_index(s2, d2, assign)]
    return scope, out


def _marginalize(f, var, card):
    scope, table = f
    if var not in scope:
        return f
    new_scope = [v for v in scope if v != var]
    dims = [card[v] for v in scope]
    new_dims = [card[v] for v in new_scope]
    total = 1
    for d in new_dims:
        total *= d
    out = [0.0] * total
    assign = {}
    old_total = 1
    for d in dims:
        old_total *= d
    for flat in range(old_total):
        rem = flat
        for v, d in zip(reversed(scope), reversed(dims)):
            assign[v] = rem % d
            rem //= d
        out[_f_index(new_scope, new_dims, assign)] += table[flat]
    return new_scope, out


def _reduce(f, evidence, card):
    scope, table = f
    hits = [v for v in scope if v in evidence]
    if not hits:
        return f
    new_scope = [v for v in scope if v not in evidence]
    dims = [card[v] for v in scope]
    new_dims = [card[v] for v in new_scope]
    total = 1
    for d in new_dims:
        total *= d
    out = [0.0] * total
    assign = {}
    old_total = 1
    for d in dims:
        old_total *= d
    for flat in range(old_total):
        rem = flat
        ok = True
        for v, d in zip(reversed(scope), reversed(dims)):
            assign[v] = rem % d
            rem //= d
        for v in hits:
            if assign[v] != evidence[v]:
                ok = False
                break
        if ok:
            out[_f_index(new_scope, new_dims, assign)] += table[flat]
    return new_scope, out


def baynet(graph, cpts, evidence=None, query=None):
    """
    Posterior P(query | evidence) in a discrete Bayesian network by
    variable elimination (sum-product with factors).

    The joint factorizes as P(X) = prod_v P(v | parents(v)). Each CPT
    becomes a factor; evidence rows are selected out; the hidden
    non-query variables are summed out one at a time (the elimination
    of z multiplies every factor whose scope contains z and
    marginalizes z from the product); the surviving factors are
    multiplied and normalized. Elimination order: lexicographic over
    the hidden variables -- deterministic, identical in the R arm, and
    order affects only cost, never the result (VE Theorem in the
    source).

    Sources
    -------
    Zhang, N. L. & Poole, D. (1994). A simple approach to Bayesian
    network computations. *Proc. 10th Canadian Conference on AI*,
    171-178 (the VE algorithm; sum out one variable at a time from the
    product of relevant factors)
    (fetched-wave3/zhang-poole-1994-simple-approach-bn.pdf).
    Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*,
    Morgan Kaufmann (network semantics: the CPT factorization).
    Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models*,
    MIT Press, Ch. 9 (sum-product variable elimination; cited for the
    standard treatment).

    Parameters
    ----------
    graph : dict
        node -> list of parent names (ordered as in the CPT axes).
    cpts : dict
        node -> nested list, axes (parent_1, ..., parent_k, node).
    evidence : dict, optional
        node -> observed state index (0-based).
    query : str
        The query node.

    Returns
    -------
    RichResult
        Keys: posterior (list over query states), states (cardinality),
        estimate (argmax state), normalizer (probability of evidence
        when evidence is given).
    """
    if query is None:
        raise ValueError("`query` is required")
    evidence = dict(evidence or {})
    nodes = sorted(graph)
    card = {}
    for v in nodes:
        t = cpts[v]
        for _ in graph[v]:
            t = t[0]
        card[v] = len(t)
    if query not in card:
        raise ValueError(f"unknown query node {query!r}")
    for v, s in evidence.items():
        if v not in card:
            raise ValueError(f"unknown evidence node {v!r}")
        if not (0 <= int(s) < card[v]):
            raise ValueError(f"evidence state out of range for {v!r}")
        evidence[v] = int(s)
    factors = []
    for v in nodes:
        scope, flat = _factor_from_cpt(v, graph[v], cpts[v], card)
        factors.append(_reduce((scope, flat), evidence, card))
    hidden = [v for v in nodes if v != query and v not in evidence]
    for z in hidden:
        touch = [f for f in factors if z in f[0]]
        keep = [f for f in factors if z not in f[0]]
        if not touch:
            continue
        prod = touch[0]
        for f in touch[1:]:
            prod = _multiply(prod, f, card)
        keep.append(_marginalize(prod, z, card))
        factors = keep
    result = ([], [1.0])
    for f in factors:
        result = _multiply(result, f, card)
    scope, table = result
    if scope == []:
        raise ValueError("query eliminated; check inputs")
    if scope != [query]:
        # sum out any stray dimensions (cannot happen with valid input)
        for v in [v for v in scope if v != query]:
            scope, table = _marginalize((scope, table), v, card)
        scope, table = (scope, table)
    norm = sum(table)
    if norm <= 0:
        raise ValueError("zero-probability evidence")
    post = [t / norm for t in table]
    est = 0
    for i in range(1, len(post)):
        if post[i] > post[est] + 1e-15:
            est = i
    return RichResult(payload={
        "posterior": post, "states": card[query], "estimate": int(est),
        "normalizer": float(norm), "query": query,
        "method": "Variable elimination (Zhang-Poole 1994), lexicographic order",
    })


# long descriptive alias (stub-era name)
bayes_network = baynet


def cheatsheet():
    return "baynet: discrete BN posterior via sum-product variable elimination"
