# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero's MCTS with a neural prior.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815,
whose search is stated to be "identical to AlphaGo Zero" (Silver et al.,
*Nature* 550, 354-359), and Schrittwieser, J. et al. (2020),
arXiv:1911.08265 (FETCHED), appendix B, which prints all three phases
explicitly.  Each simulation

  select   descends by argmax_a [ Q(s,a) + U(s,a) ], with
           U(s,a) = c_puct P(s,a) sqrt(sum_b N(s,b)) / (1 + N(s,a))
           -- the c2 -> infinity limit of MuZero's rule, since
           log((sum_b N + c2 + 1)/c2) -> 0;
  expand   evaluates (p, v) = f_theta(s) once at the new leaf, sets
           P(s,a) = p_a and N = W = Q = 0;
  backup   walks the path back to the root updating N <- N + 1,
           W <- W + G, Q <- W / N, negating the value at each ply in a
           two-player game.

After ``num_sim`` simulations the search policy is the normalised root
visit count, pi(a) proportional to N(root, a).

DETERMINISM.  The simulation budget is fixed (``num_sim``), never a
wall-clock budget; ties in the selection break to the lowest action
index; and the root Dirichlet noise, when wanted, must be passed in as
``root_noise`` rather than drawn.  Nothing consults a clock or a seed.

CONTRACT.  States are identified by a *scalar id*.  ``step(s, a)``
returns the id of the successor, ``net(s)`` returns ``(p, v)``.  When no
``step`` is given the search is one ply deep, which is the root search
AlphaZero performs before its first descent.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_search"]


def alphazero_search(state, net, num_sim, step=None, c_puct=1.25,
                     max_depth=None, terminal=None, alternate=True,
                     root_noise=None, eps=0.25):
    """Run ``num_sim`` PUCT simulations from ``state``.

    Returns
    -------
    RichResult with payload:
        estimate : index of the most-visited root action
        pi       : normalised root visit counts
        n, q, p  : root edge statistics
        value    : the root value estimate W(root)/N(root)
        n_nodes  : number of distinct states expanded
    """
    if max_depth is None:
        max_depth = 1 if step is None else 64
    ids = []
    P = []
    N = []
    W = []
    V = []

    def expand(s):
        out = net(s)
        if isinstance(out, tuple) and len(out) == 2:
            p, v = k.vec(out[0]), float(out[1])
        else:
            p, v = k.vec(out), 0.0
        tot = 0.0
        for x in p:
            tot += x
        if tot > 0.0:
            p = [x / tot for x in p]
        ids.append(s)
        P.append(p)
        N.append([0.0] * len(p))
        W.append([0.0] * len(p))
        V.append(v)
        return len(ids) - 1

    def find(s):
        for i in range(len(ids)):
            if ids[i] == s:
                return i
        return -1

    root = expand(state)
    if root_noise is not None:
        et = k.vec(root_noise)
        tot = 0.0
        for x in et:
            tot += x
        if tot > 0.0:
            et = [x / tot for x in et]
        e = float(eps)
        P[root] = [(1.0 - e) * P[root][i] + e * et[i] for i in range(len(P[root]))]

    for _ in range(int(num_sim)):
        node = root
        path = []
        depth = 0
        while True:
            if terminal is not None and terminal(ids[node]):
                v = V[node]
                break
            if depth >= max_depth:
                v = V[node]
                break
            tot = 0.0
            for x in N[node]:
                tot += x
            rt = math.sqrt(tot) if tot > 0.0 else 0.0
            best = 0
            bestscore = None
            for a in range(len(P[node])):
                q = W[node][a] / N[node][a] if N[node][a] > 0.0 else 0.0
                sc = q + c_puct * P[node][a] * rt / (1.0 + N[node][a])
                if bestscore is None or sc > bestscore:
                    bestscore = sc
                    best = a
            path.append((node, best))
            s2 = step(ids[node], best) if step is not None else None
            if s2 is None:
                v = V[node]
                break
            idx = find(s2)
            if idx < 0:
                idx = expand(s2)
                v = V[idx]
                break
            node = idx
            depth += 1
        acc = v
        for i in range(len(path) - 1, -1, -1):
            if alternate:
                acc = -acc
            nd, a = path[i]
            N[nd][a] = N[nd][a] + 1.0
            W[nd][a] = W[nd][a] + acc

    tot = 0.0
    for x in N[root]:
        tot += x
    pi = [x / tot if tot > 0.0 else 0.0 for x in N[root]]
    q = [W[root][a] / N[root][a] if N[root][a] > 0.0 else 0.0
         for a in range(len(N[root]))]
    best = 0
    for a in range(1, len(pi)):
        if pi[a] > pi[best]:
            best = a
    wsum = 0.0
    for x in W[root]:
        wsum += x
    return RichResult(
        title="AlphaZero MCTS",
        summary_lines=[("simulations", int(num_sim)), ("best action", best)],
        payload={
            "estimate": float(best),
            "action": best,
            "pi": pi,
            "n": N[root],
            "q": q,
            "p": P[root],
            "value": wsum / tot if tot > 0.0 else 0.0,
            "n_nodes": len(ids),
            "method": "AlphaZero MCTS with a neural prior (PUCT selection)",
        },
    )


def cheatsheet():
    return "alpz: AlphaZero MCTS + neural prior"
