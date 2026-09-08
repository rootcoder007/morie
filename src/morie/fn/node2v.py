# morie.fn -- function file (rootcoder007/morie)
r"""node2vec: the neighbourhood definition is the model.

Skip-gram over a corpus embeds a word so it predicts nearby words. The
network analogue treats a graph as a document and a sampled node
sequence as a sentence, maximising

.. math:: \max_f \sum_{u \in V} \log \Pr(N_S(u) \mid f(u))

under conditional independence of the neighbours and a softmax
parametrised by a dot product of features. Everything then hinges on
what :math:`N_S(u)` means -- and the paper's point is that **there is
no winning sampling strategy** across networks and tasks, which is
exactly what earlier methods fixed in advance.

**Two notions of similarity, pulling opposite ways.** Breadth-first
sampling stays local: it characterises a node by its immediate
structural role, and because the sampled nodes repeat, it estimates the
one-hop distribution with low variance. Depth-first roams: it reaches
distant parts of the graph and reveals community structure, at the cost
of higher variance and of sampled nodes that may be barely related to
the source. Real networks show a mixture of both, so the sampler should
interpolate rather than choose.

**The second-order walk that interpolates.** Having just traversed
:math:`(t,v)`, the unnormalised transition to :math:`x` is
:math:`\pi_{vx} = \alpha_{pq}(t,x)\cdot w_{vx}` with

.. math:: \alpha_{pq}(t,x) = \begin{cases}
          1/p & d_{tx} = 0\\ 1 & d_{tx} = 1\\
          1/q & d_{tx} = 2\end{cases},

where :math:`d_{tx}` is the shortest-path distance from the *previous*
node. So :math:`p` prices returning, and :math:`q` prices leaving the
neighbourhood: large :math:`q` keeps the walk local (BFS-like), small
:math:`q` pushes it outward (DFS-like). The walk is second-order
because :math:`\alpha` depends on where the walk came from -- a
first-order walk cannot express this, and the anchor checks the bias
by measuring return frequency against :math:`p`.

References
----------
Grover, A. & Leskovec, J. (2016) "node2vec: Scalable Feature Learning
for Networks", *Proceedings of the 22nd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining (KDD '16)*, 855-864,
doi:10.1145/2939672.2939754, arXiv:1607.00653. Sec. 2 (the
document/sentence analogy and the observation that no single sampling
strategy wins across networks and tasks). Sec. 3 (the maximum
likelihood objective of eq. (1) under conditional independence and a
softmax over the dot product; Sec. 3.1 on BFS giving a low-variance
microscopic view versus DFS giving a macroscopic community view, and
that real networks mix both; Sec. 3.2.2's second-order walk with
alpha_pq keyed on the shortest-path distance d_tx from the previous
node).

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013)
"Distributed Representations of Words and Phrases and their
Compositionality", *NIPS 2013*, 3111-3119, arXiv:1310.4546. Skip-gram
with negative sampling.

Perozzi, B., Al-Rfou, R. & Skiena, S. (2014) "DeepWalk: Online
Learning of Social Representations", *KDD '14*, 701-710,
doi:10.1145/2623330.2623732. The uniform random walk node2vec
generalises.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["alpha_pq", "transition_probabilities", "walk",
           "generate_walks", "skipgram_pairs"]

_EPS = 1e-12


def alpha_pq(d_tx, p, q):
    r"""Sec. 3.2.2: :math:`1/p`, :math:`1`, :math:`1/q` for
    :math:`d_{tx}` = 0, 1, 2."""
    d = int(d_tx)
    if float(p) <= 0.0 or float(q) <= 0.0:
        raise ValueError("node2v: p and q must be positive")
    if d == 0:
        return 1.0 / float(p)
    if d == 1:
        return 1.0
    if d == 2:
        return 1.0 / float(q)
    raise ValueError("node2v: d_tx must be 0, 1 or 2 for a "
                     "second-order walk, got %d" % d)


def _dist(adj, t, x):
    if t == x:
        return 0
    if x in adj.get(t, ()):
        return 1
    return 2


def transition_probabilities(adj, t, v, p, q, weights=None):
    r""":math:`\pi_{vx}/Z` over the neighbours of :math:`v`.

    ``t`` is the node the walk came from; passing ``None`` gives the
    first-order (uniform or weight-proportional) step used at the
    start of a walk.
    """
    nb = sorted(adj.get(v, ()))
    if not nb:
        raise ValueError("node2v: node %r has no neighbours" % (v,))
    pi = []
    for x in nb:
        w = 1.0 if weights is None else float(weights.get((v, x), 1.0))
        a = 1.0 if t is None else alpha_pq(_dist(adj, t, x), p, q)
        pi.append(a * w)
    Z = sum(pi)
    return {"nodes": nb, "probabilities": [v_ / Z for v_ in pi],
            "unnormalized": pi, "Z": Z}


def walk(adj, start, length, p=1.0, q=1.0, rng=None, weights=None):
    r"""One second-order random walk."""
    r = np.random.default_rng(0) if rng is None else rng
    path = [start]
    prev = None
    for _ in range(int(length) - 1):
        tp = transition_probabilities(adj, prev, path[-1], p, q,
                                      weights)
        u = float(r.uniform())
        acc, nxt = 0.0, tp["nodes"][-1]
        for i in range(len(tp["nodes"])):
            acc += tp["probabilities"][i]
            if u <= acc:
                nxt = tp["nodes"][i]
                break
        prev = path[-1]
        path.append(nxt)
    return path


def generate_walks(adj, num_walks=10, length=10, p=1.0, q=1.0, seed=0,
                   weights=None):
    r"""``num_walks`` walks from every node."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(num_walks)):
        for v in sorted(adj):
            out.append(walk(adj, v, length, p, q, rng, weights))
    return RichResult(payload={
        "estimate": out, "walks": out, "p": float(p), "q": float(q),
        "n_walks": len(out), "length": int(length),
        "method": "second-order biased random walk; Grover & Leskovec "
                  "(2016) Sec. 3.2.2",
        "note": "large q keeps the walk local (BFS-like), small q "
                "pushes it outward (DFS-like); p prices returning",
    })


def skipgram_pairs(walks, window=2):
    r"""(centre, context) pairs within the window -- the corpus the
    objective of eq. (1) is maximised over."""
    w = int(window)
    if w < 1:
        raise ValueError("node2v: the window must be at least 1")
    pairs = []
    for path in walks:
        for i in range(len(path)):
            for j in range(max(0, i - w), min(len(path), i + w + 1)):
                if i != j:
                    pairs.append((path[i], path[j]))
    return pairs


def cheatsheet():
    return ("node2v: graph as document, walk as sentence, skip-gram on "
            "top. The point is that NO sampling strategy wins "
            "everywhere: BFS gives a low-variance local structural "
            "view, DFS a macroscopic community view, and real networks "
            "mix both. A SECOND-ORDER walk interpolates -- having come "
            "from t, the bias to x is 1/p if returning, 1 if x "
            "neighbours t, 1/q otherwise. Large q stays local, small q "
            "roams. A first-order walk cannot express this.")


# compact alias per ledger/NAMING.md
node2vec = generate_walks
