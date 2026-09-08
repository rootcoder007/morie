# morie.fn -- function file (rootcoder007/morie)
r"""Message passing neural networks: one framework, eight models.

The paper's first contribution is not an architecture but a
reformulation: at least eight published models for graphs turn out to
be the same algorithm with different choices of three functions. The
forward pass has a **message passing phase** of :math:`T` steps,

.. math:: m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw}),
          \qquad h_v^{t+1} = U_t(h_v^t, m_v^{t+1}),

and a **readout phase** producing a graph-level output
:math:`\hat y = R(\{h_v^T \mid v \in G\})`.

**Two invariances are structural, not incidental.** The message sum is
over the neighbour set, so it is permutation-invariant; the readout
must be invariant to node relabelling for the graph-level prediction to
be well defined at all. Break either and the model gives different
answers for the same molecule under a different atom ordering -- which
the anchor checks by permuting.

**Why the sum, and where edges enter.** Messages depend on the edge
feature :math:`e_{vw}`, which is how bond type is carried in the
chemistry setting the paper targets; a model that ignores edge features
cannot distinguish a single from a double bond between the same atoms.
The paper's own variant uses an edge-network message
:math:`A(e_{vw})h_w^t` and a GRU update with weights tied across time
steps.

**Readout choices matter for what can be expressed.** A plain sum over
node states is invariant but loses which node contributed what; the
set2set readout is order-invariant by construction and more expressive.
Both are offered here, alongside the gated readout
:math:`\sum_v \sigma(i(h_v^T, h_v^0)) \odot j(h_v^T)`, whose gate
:math:`i` sees the *initial* state too, so the readout can tell a node
that changed from one that did not.

References
----------
Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G. E.
(2017) "Neural Message Passing for Quantum Chemistry", *Proceedings of
the 34th International Conference on Machine Learning (ICML 2017)*,
PMLR 70, 1263-1272, arXiv:1704.01212. Sec. 2 (the MPNN framework:
message passing over T steps with message functions M_t and vertex
update functions U_t, eq. (1); the readout R and the requirement that
it be invariant to node permutation; edge features e_vw), and Sec. 3
(the edge-network message function, the GRU update with weights tied
across time steps, and the set2set readout).

Li, Y., Tarlow, D., Brockschmidt, M. & Zemel, R. (2016) "Gated Graph
Sequence Neural Networks", *ICLR 2016*, arXiv:1511.05493. The GRU
update reused as U_t.

Vinyals, O., Bengio, S. & Kudlur, M. (2016) "Order Matters: Sequence
to sequence for sets", *ICLR 2016*, arXiv:1511.06391. The set2set
readout.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["message", "update_gru", "message_passing", "readout",
           "is_permutation_invariant"]

_EPS = 1e-12
_READOUTS = ("sum", "mean", "gated")


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def message(h_v, h_w, e_vw, A=None):
    r"""Eq. (1)'s :math:`M_t`.

    With ``A`` supplied this is the edge network :math:`A(e_{vw})h_w`,
    so the bond type changes the message -- without edge features a
    single and a double bond between the same atoms are
    indistinguishable.
    """
    hw = [float(v) for v in k.vec(h_w)]
    if A is None:
        e = float(e_vw) if not isinstance(e_vw, (list, tuple)) \
            else float(k.vec(e_vw)[0])
        return [e * v for v in hw]
    M = A(e_vw)
    return [sum(M[o][j] * hw[j] for j in range(len(hw)))
            for o in range(len(M))]


def update_gru(h, m, Wz, Uz, Wr, Ur, Wh, Uh):
    r""":math:`U_t` as a GRU, with weights tied across steps."""
    n = len(h)

    def lin(W, U, a, b):
        return [sum(W[o][j] * a[j] for j in range(len(a)))
                + sum(U[o][j] * b[j] for j in range(len(b)))
                for o in range(n)]

    z = [_sig(v) for v in lin(Wz, Uz, m, h)]
    r = [_sig(v) for v in lin(Wr, Ur, m, h)]
    hh = [math.tanh(v) for v in
          lin(Wh, Uh, m, [r[i] * h[i] for i in range(n)])]
    return [(1.0 - z[i]) * h[i] + z[i] * hh[i] for i in range(n)]


def message_passing(H0, adj, edge_features, T=3, A=None,
                    update=None):
    r"""T rounds of eq. (1).

    ``update=None`` uses :math:`h^{t+1}_v = h^t_v + m^{t+1}_v`, which
    keeps the framework visible without committing to a GRU.
    """
    H = [[float(v) for v in r] for r in k.mat(H0)]
    if int(T) < 1:
        raise ValueError("mpfn: T must be at least 1")
    for _ in range(int(T)):
        new = []
        for v in range(len(H)):
            nb = sorted(adj.get(v, ()))
            m = [0.0] * len(H[v])
            for w in nb:
                e = edge_features.get((v, w),
                                      edge_features.get((w, v), 1.0))
                mm = message(H[v], H[w], e, A)
                m = [m[i] + mm[i] for i in range(len(m))]
            new.append(update(H[v], m) if update is not None
                       else [H[v][i] + m[i] for i in range(len(m))])
        H = new
    return H


def readout(H, how="sum", H0=None, i_fn=None, j_fn=None):
    r""":math:`R`, which must be invariant to node relabelling.

    ``gated`` implements :math:`\sum_v \sigma(i(h_v^T,h_v^0))\odot
    j(h_v^T)`, whose gate sees the initial state and so can tell a node
    that changed from one that did not.
    """
    if how not in _READOUTS:
        raise ValueError("mpfn: readout must be one of %s, got %r"
                         % (", ".join(_READOUTS), how))
    rows = [[float(v) for v in r] for r in k.mat(H)]
    d = len(rows[0])
    if how == "sum":
        return [sum(rows[v][f] for v in range(len(rows)))
                for f in range(d)]
    if how == "mean":
        return [sum(rows[v][f] for v in range(len(rows))) / len(rows)
                for f in range(d)]
    if H0 is None or i_fn is None or j_fn is None:
        raise ValueError("mpfn: the gated readout needs H0, i_fn and "
                         "j_fn")
    acc = [0.0] * d
    for v in range(len(rows)):
        g = i_fn(rows[v], H0[v])
        jv = j_fn(rows[v])
        acc = [acc[f] + _sig(g[f]) * jv[f] for f in range(d)]
    return acc


def is_permutation_invariant(H, adj, edge_features, perm, T=3,
                             how="sum", tol=1e-9):
    r"""Relabel the nodes and check the readout is unchanged.

    A graph-level prediction that moves under relabelling is not
    well defined.
    """
    base = readout(message_passing(H, adj, edge_features, T), how)
    n = len(H)
    inv = [0] * n
    for i in range(n):
        inv[perm[i]] = i
    Hp = [H[inv[i]] for i in range(n)]
    adjp = {perm[v]: sorted(perm[w] for w in adj.get(v, ()))
            for v in adj}
    efp = {}
    for (a, b), e in edge_features.items():
        efp[(perm[a], perm[b])] = e
    other = readout(message_passing(Hp, adjp, efp, T), how)
    dev = max(abs(base[f] - other[f]) for f in range(len(base)))
    return {"invariant": dev < float(tol), "max_deviation": dev,
            "readout": base}


def cheatsheet():
    return ("mpfn: at least EIGHT published graph models are the same "
            "algorithm with different M_t, U_t and R. Message phase: "
            "m_v = sum_{w in N(v)} M_t(h_v, h_w, e_vw), then "
            "h_v <- U_t(h_v, m_v); readout R over the final states. "
            "The sum makes messages permutation-invariant and the "
            "READOUT MUST BE TOO, or the graph prediction changes when "
            "atoms are renumbered. Edge features carry bond type -- "
            "without them a single and a double bond between the same "
            "atoms are identical.")


# compact alias per ledger/NAMING.md
messagepassing = message_passing
