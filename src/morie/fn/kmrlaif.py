# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RLAIF: a reward model fitted to AI feedback via Bradley-Terry."""

import numpy as np

from ._richresult import RichResult
from .kmrmloss import kamath_reward_model_training_loss

__all__ = ["kamath_rlaif_objective"]


def _pairs(ai_preferences):
    out = []
    for p in ai_preferences:
        if isinstance(p, dict):
            if "winner" not in p or "loser" not in p:
                raise ValueError(
                    "a dict preference needs 'winner' and 'loser' keys.")
            out.append((p["winner"], p["loser"]))
        else:
            pair = tuple(p)
            if len(pair) != 2:
                raise ValueError(
                    f"each preference must be (winner, loser); got {p!r}.")
            out.append(pair)
    return out


def _strongly_connected(n, edges):
    """Kosaraju on the 'winner -> loser' digraph: the Bradley-Terry MLE
    exists iff it is strongly connected."""
    fwd = [[] for _ in range(n)]
    bwd = [[] for _ in range(n)]
    for a, b in edges:
        fwd[a].append(b)
        bwd[b].append(a)

    def reach(adj):
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    return len(reach(fwd)) == n and len(reach(bwd)) == n


def kamath_rlaif_objective(ai_preferences, max_iter=1000, tol=1e-12):
    """preferences_AI = AI_judge(y_w, y_l, principle); r_phi fits them
    with Bradley-Terry.

    The preferences are AI-generated -- that is the only thing RLAIF
    changes -- and the fit is the ordinary BT maximum likelihood,
    solved by the minorise-maximise iteration (Zermelo's algorithm),
    which needs no gradient step size and converges monotonically.

    If the "beats" digraph is not strongly connected the MLE does not
    exist (some item is unbeaten, so its strength diverges); that is
    raised rather than returned as a large finite number. The fitted
    scores' training loss is reported via ``morie.fn.kmrmloss``.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 5, RLAIF (Bai et al.
    2022; Lee et al. 2023).

    Examples
    --------
    >>> out = kamath_rlaif_objective([(0, 1), (1, 0)])
    >>> [round(v, 9) for v in out["strengths"]]
    [0.5, 0.5]
    >>> out2 = kamath_rlaif_objective([(0, 1), (0, 1), (1, 0)])
    >>> abs(out2["strengths"][0] - 2 / 3) < 1e-9
    True
    >>> import math
    >>> abs(out2["scores"][0] - out2["scores"][1] - math.log(2)) < 1e-9
    True
    """
    prefs = _pairs(ai_preferences)
    if not prefs:
        raise ValueError("no AI preferences supplied.")
    items = sorted({i for pair in prefs for i in pair}, key=repr)
    index = {it: k for k, it in enumerate(items)}
    n = len(items)
    if n < 2:
        raise ValueError(
            "preferences over a single item carry no information.")
    edges = [(index[w], index[l]) for w, l in prefs]
    for w, l in edges:
        if w == l:
            raise ValueError("an item cannot be preferred to itself.")
    if not _strongly_connected(n, edges):
        raise ValueError(
            "the preference graph is not strongly connected, so the "
            "Bradley-Terry maximum likelihood does not exist (some "
            "item is never beaten and its strength diverges).")

    wins = np.zeros(n)
    counts = np.zeros((n, n))
    for w, l in edges:
        wins[w] += 1
        counts[w, l] += 1
        counts[l, w] += 1

    p = np.full(n, 1.0 / n)
    for _ in range(int(max_iter)):
        denom = np.array([
            np.sum(counts[i] / (p[i] + p)) - counts[i, i] / (2 * p[i])
            for i in range(n)])
        if np.any(denom <= 0):
            raise ValueError("the BT iteration hit a zero denominator.")
        new = wins / denom
        new = new / new.sum()
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    scores = np.log(p)
    sw = np.array([scores[w] for w, _ in edges])
    sl = np.array([scores[l] for _, l in edges])
    loss = kamath_reward_model_training_loss(sw, sl)
    return RichResult(payload={
        "items": items,
        "strengths": [float(v) for v in p],
        "scores": [float(v) for v in scores],
        "loss": float(loss["estimate"]),
        "accuracy": float(loss["accuracy"]),
        "n_preferences": len(edges),
        "estimate": float(loss["estimate"]),
        "n": n,
        "method": "RLAIF Bradley-Terry reward fit (loss via kmrmloss)"})


def cheatsheet():
    return "kmrlaif: BT MLE over AI preferences; non-connected graph refused"
