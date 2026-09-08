# morie.fn -- function file (rootcoder007/morie)
"""Back-door criterion for identifying a causal effect by adjustment."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["backdoor_criterion"]


def _parse(dag):
    """Accept {node: [children]} or an edge list, return parents/children."""
    children, parents = {}, {}
    if isinstance(dag, dict):
        items = [(u, v) for u, vs in dag.items() for v in vs]
        nodes = set(dag) | {v for _, vs in dag.items() for v in vs}
    else:
        items = [(u, v) for u, v in dag]
        nodes = {n for e in items for n in e}
    for n in nodes:
        children.setdefault(n, set())
        parents.setdefault(n, set())
    for u, v in items:
        children[u].add(v)
        parents[v].add(u)
    return children, parents, nodes


def _descendants(node, children):
    seen, stack = set(), [node]
    while stack:
        cur = stack.pop()
        for c in children.get(cur, ()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _has_cycle(children, nodes):
    colour = dict.fromkeys(nodes, 0)

    def visit(n):
        colour[n] = 1
        for c in children.get(n, ()):
            if colour[c] == 1 or (colour[c] == 0 and visit(c)):
                return True
        colour[n] = 2
        return False

    return any(colour[n] == 0 and visit(n) for n in nodes)


def _paths(x, y, children, parents):
    """Every simple undirected path from x to y, as node lists with edge dirs."""
    out = []
    stack = [(x, [x], [])]
    while stack:
        cur, path, dirs = stack.pop()
        if cur == y:
            out.append((path, dirs))
            continue
        for nxt in children.get(cur, ()):
            if nxt not in path:
                stack.append((nxt, path + [nxt], dirs + ["->"]))
        for nxt in parents.get(cur, ()):
            if nxt not in path:
                stack.append((nxt, path + [nxt], dirs + ["<-"]))
    return out


def _blocked(path, dirs, Z, children):
    """True when conditioning on Z blocks this path (d-separation)."""
    for i in range(1, len(path) - 1):
        node = path[i]
        incoming, outgoing = dirs[i - 1], dirs[i]
        is_collider = incoming == "->" and outgoing == "<-"
        if is_collider:
            # A collider blocks unless it, or any descendant, is in Z.
            if node not in Z and not (_descendants(node, children) & Z):
                return True
        else:
            # Chain or fork: blocked exactly when the middle node is in Z.
            if node in Z:
                return True
    return False


def backdoor_criterion(dag, X, Y, Z=()):
    r"""Check whether Z satisfies the back-door criterion for (X, Y).

    Pearl's criterion has two parts:

    1. No node in Z is a descendant of X.
    2. Z blocks every back-door path from X to Y -- every path that
       leaves X through an arrow *into* X.

    When both hold, the causal effect is identified by adjustment:

    .. math::

        P(Y = y \mid do(X = x)) = \sum_z P(Y = y \mid X = x, Z = z)\,P(Z = z)

    which is what :func:`morie.fn.bdrj.backdoor_adjustment_formula`
    computes. This function is the check that formula assumes and cannot
    perform on its own: the adjustment is arithmetic over data, while
    the criterion is a claim about the graph.

    Blocking is d-separation, and the collider rule is what makes it
    more than "condition on everything". A path is blocked at a chain or
    fork exactly when the middle node is in Z, but at a collider the
    logic reverses: the collider blocks the path *unless* it or one of
    its descendants is in Z. Adjusting for a collider therefore *opens*
    a path that was closed, which is why adding covariates can create
    confounding rather than remove it. Both failure modes are reported
    by name.

    Parameters
    ----------
    dag : dict or iterable of pairs
        The graph, either ``{node: [children]}`` or an edge list of
        ``(parent, child)``. Must be acyclic.
    X, Y : hashable
        Treatment and outcome nodes.
    Z : iterable, optional
        Candidate adjustment set. The empty set is a valid candidate and
        is checked like any other.

    Returns
    -------
    RichResult
        keys: ``satisfied``, ``descendant_violations`` (members of Z
        below X), ``open_paths`` (unblocked back-door paths, as readable
        strings), ``backdoor_paths`` (all of them), ``n_backdoor``,
        ``reason``, ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*,
    2nd edn. Cambridge University Press. Definition 3.3.1, the back-door
    criterion.
    """
    children, parents, nodes = _parse(dag)
    Zs = set(Z)
    for name, node in (("X", X), ("Y", Y)):
        if node not in nodes:
            raise ValueError(f"{name} = {node!r} is not a node of the graph.")
    missing = Zs - nodes
    if missing:
        raise ValueError(f"Z contains nodes not in the graph: {sorted(map(str, missing))}.")
    if X in Zs or Y in Zs:
        raise ValueError("Z must not contain X or Y.")
    if _has_cycle(children, nodes):
        raise ValueError("dag contains a cycle; the back-door criterion is defined for DAGs.")

    desc_x = _descendants(X, children)
    bad_desc = sorted(map(str, Zs & desc_x))

    # Back-door paths are those leaving X via an arrow into X.
    back = [(p, d) for p, d in _paths(X, Y, children, parents) if d and d[0] == "<-"]
    open_paths = []
    for p, d in back:
        if not _blocked(p, d, Zs, children):
            open_paths.append(" ".join(a + " " + b for a, b in zip(d, map(str, p[1:]))).join([str(p[0]) + " ", ""]))

    ok = not bad_desc and not open_paths
    if ok:
        reason = "Z satisfies the back-door criterion; the effect is identified by adjustment."
    elif bad_desc:
        reason = f"Z contains descendants of X: {bad_desc}. Adjusting for them blocks part of the effect itself."
    else:
        reason = f"{len(open_paths)} back-door path(s) remain open, so confounding is not removed."

    return RichResult(
        title="Back-door criterion",
        payload={
            "satisfied": bool(ok),
            "descendant_violations": bad_desc,
            "open_paths": open_paths,
            "backdoor_paths": [" ".join([str(p[0])] + [f"{dd} {nn}" for dd, nn in zip(d, map(str, p[1:]))]) for p, d in back],
            "n_backdoor": len(back),
            "adjustment_set": sorted(map(str, Zs)),
            "reason": reason,
            "method": "Back-door criterion (Pearl 2009, Def. 3.3.1) via d-separation",
        },
    )


def cheatsheet():
    return "bdcrt: back-door criterion check for an adjustment set"
