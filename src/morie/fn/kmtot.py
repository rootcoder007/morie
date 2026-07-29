# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tree of Thoughts: beam search over a tree of partial reasoning
steps."""

from ._richresult import RichResult

__all__ = ["kamath_tree_of_thoughts"]


def kamath_tree_of_thoughts(problem, branch_factor, max_depth, model,
                            beam=1):
    """Search a tree of thought-nodes, each scored by the model's own
    evaluation, keeping the best ``beam`` states at every depth.

    ``model(state, branch_factor) -> [(thought, score), ...]`` is the
    caller's proposer-and-evaluator; it must return at most
    ``branch_factor`` children, and that is enforced -- a model that
    quietly returns twenty children turns the search exponential and
    the "branch factor" into a lie. Path scores are cumulative sums of
    the node scores, and ties keep the earlier-generated path so the
    search is reproducible.

    ``beam = 1`` is greedy depth-first-by-score; a larger beam is the
    BFS variant of the paper. A node returning no children is a dead
    end and is dropped, with the count reported.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, tree-of-thoughts
    (Yao et al. 2023).

    Examples
    --------
    >>> def m(state, b):
    ...     return [(state + str(i), float(i)) for i in range(1, b + 1)]
    >>> out = kamath_tree_of_thoughts("", 2, 2, m)
    >>> out["best_state"], out["estimate"]
    ('22', 4.0)
    >>> out["best_path"]
    ['2', '22']
    >>> wide = kamath_tree_of_thoughts("", 2, 2, m, beam=2)
    >>> wide["n_expanded"]
    3
    """
    b = int(branch_factor)
    depth = int(max_depth)
    beam = int(beam)
    if b < 1:
        raise ValueError(f"branch_factor must be at least 1; got {b}.")
    if depth < 1:
        raise ValueError(f"max_depth must be at least 1; got {depth}.")
    if beam < 1:
        raise ValueError(f"beam must be at least 1; got {beam}.")
    if not callable(model):
        raise ValueError(
            "model must be callable (state, branch_factor) -> "
            "[(thought, score), ...].")

    # (state, cumulative score, path)
    frontier = [(problem, 0.0, [])]
    expanded, dead_ends = 0, 0
    for _ in range(depth):
        children = []
        for state, score, path in frontier:
            out = model(state, b)
            expanded += 1
            try:
                kids = [tuple(c) for c in out]
            except TypeError:
                raise ValueError(
                    "model must return a sequence of (thought, score) "
                    "pairs.") from None
            if len(kids) > b:
                raise ValueError(
                    f"the model returned {len(kids)} children for a "
                    f"branch factor of {b}.")
            if not kids:
                dead_ends += 1
                continue
            for c in kids:
                if len(c) != 2:
                    raise ValueError(
                        "each child must be a (thought, score) pair.")
                thought, s = c
                try:
                    s = float(s)
                except (TypeError, ValueError):
                    raise ValueError(
                        "a child's score is not numeric.") from None
                children.append((thought, score + s, path + [thought]))
        if not children:
            break
        children.sort(key=lambda t: -t[1])
        frontier = children[:beam]
    if not frontier or not frontier[0][2]:
        raise ValueError(
            "the search produced no complete thought at all; every "
            "branch was a dead end.")
    best_state, best_score, best_path = frontier[0]
    return RichResult(payload={
        "best_state": best_state, "best_path": best_path,
        "best_score": best_score,
        "frontier": [(s, sc) for s, sc, _ in frontier],
        "n_expanded": expanded, "n_dead_ends": dead_ends,
        "depth": len(best_path), "beam": beam,
        "branch_factor": b,
        "estimate": best_score, "n": len(frontier),
        "method": "Tree-of-thoughts beam search over scored thoughts"})


def cheatsheet():
    return "kmtot: beam search over model-scored thoughts; branch factor enforced"
