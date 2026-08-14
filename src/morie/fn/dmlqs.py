# morie.fn -- function file (rootcoder007/morie)
r"""D-MPNN: messages on directed bonds, not atoms.

Two families of molecular property model had both worked: neural
networks on computed fingerprints or expert descriptors, and graph
convolutions that learn a representation from the molecular graph.
D-MPNN is the second kind with one change to the message passing
phase, plus one addition.

**Messages live on directed bonds.** A generic MPNN passes messages
between *atoms*; D-MPNN passes them along *directed edges*. The
motivation is stated exactly: to prevent **totters** -- messages
travelling a path :math:`v_1 v_2 \dots v_n` where
:math:`v_i = v_{i+2}` for some :math:`i`. A message that goes
:math:`A \to B` and immediately comes back :math:`B \to A` carries
:math:`A`'s own information back to :math:`A` dressed as news, and
such excursions are likely to introduce noise into the representation.

The update therefore excludes the reverse edge:

.. math:: m_{vw}^{t+1} = \sum_{u \in N(v)\setminus\{w\}} h_{uv}^{t},
          \qquad h_{vw}^{t+1} = \tau\big(h_{vw}^0 + W m_{vw}^{t+1}\big),

and it is that single exclusion :math:`\setminus\{w\}` that does the
work. The anchor removes it and counts the tottering paths that
reappear.

**Atom representations come at the end**, by summing the incoming bond
messages, so the graph-level readout is unchanged.

**And the addition: computed features alongside the learned ones.**
The learned molecular representation is concatenated with
molecule-level descriptors before the property head -- the paper's
second improvement, and an acknowledgement that expert features and
learned ones are not in competition.

References
----------
Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H.,
Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A.,
Settels, V., Jaakkola, T., Jensen, K. & Barzilay, R. (2019)
"Analyzing Learned Molecular Representations for Property
Prediction", *Journal of Chemical Information and Modeling* 59(8),
3370-3388, doi:10.1021/acs.jcim.9b00237, arXiv:1904.01561. The
Methods: building on the MPNN framework of Gilmer et al. by adopting a
message-passing paradigm based on updating representations of DIRECTED
BONDS rather than atoms; the motivation of preventing TOTTERS, that is
avoiding messages being passed along any path v_1 v_2 ... v_n where
v_i = v_{i+2}, since such excursions are likely to introduce noise
into the graph representation; and the further improvement of
combining computed molecule-level features with the learned
representation.

Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G. E.
(2017) "Neural Message Passing for Quantum Chemistry", *ICML 2017*,
PMLR 70, 1263-1272, arXiv:1704.01212. The framework; implemented in
:mod:`mpfn`.

Dai, H., Dai, B. & Song, L. (2016) "Discriminative Embeddings of
Latent Variable Models for Structured Data", *ICML 2016*, PMLR 48,
2702-2711, arXiv:1603.05629. structure2vec, the directed variant this
paper renames D-MPNN.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["directed_edges", "count_totters", "dmpnn_message_pass",
           "atom_readout", "concat_descriptors"]

_EPS = 1e-12


def directed_edges(adj):
    r"""Both directions of every bond, as separate objects."""
    out = []
    for v in sorted(adj):
        for w in sorted(set(adj[v]) - {v}):
            out.append((v, w))
    return sorted(set(out))


def count_totters(adj, length=3, exclude_reverse=True):
    r"""Paths with :math:`v_i = v_{i+2}` -- the excursions to prevent.

    ``exclude_reverse=False`` counts what a generic MPNN allows, which
    is what makes the comparison meaningful rather than definitional.
    """
    L = int(length)
    if L < 3:
        raise ValueError("dmlqs: a totter needs at least 3 steps")
    paths, tot = 0, 0

    def walk(path):
        nonlocal paths, tot
        if len(path) == L:
            paths += 1
            if any(path[i] == path[i + 2]
                   for i in range(len(path) - 2)):
                tot += 1
            return
        v = path[-1]
        for w in sorted(set(adj.get(v, ())) - {v}):
            if exclude_reverse and len(path) >= 2 and w == path[-2]:
                continue
            walk(path + [w])

    for s in sorted(adj):
        walk([s])
    return {"paths": paths, "totters": tot,
            "fraction": tot / float(paths) if paths else 0.0,
            "excluded_reverse": bool(exclude_reverse)}


def dmpnn_message_pass(h0, adj, T=3, W=None, activation="relu",
                       exclude_reverse=True):
    r"""Update directed-bond hidden states, excluding the reverse edge.

    ``exclude_reverse=False`` reinstates the totters, so the effect of
    the exclusion can be measured.
    """
    H = {k_: [float(v) for v in k.vec(h0[k_])] for k_ in h0}
    d = len(next(iter(H.values())))

    def act(x):
        if activation == "relu":
            return max(0.0, x)
        if activation == "tanh":
            return math.tanh(x)
        raise ValueError("dmlqs: activation must be relu or tanh, "
                         "got %r" % (activation,))

    H0 = {k_: list(H[k_]) for k_ in H}
    for _ in range(int(T)):
        new = {}
        for (v, w) in H:
            m = [0.0] * d
            for u in sorted(set(adj.get(v, ())) - {v}):
                if exclude_reverse and u == w:
                    continue
                if (u, v) in H:
                    for a in range(d):
                        m[a] += H[(u, v)][a]
            if W is None:
                new[(v, w)] = [act(H0[(v, w)][a] + m[a])
                               for a in range(d)]
            else:
                Wm = [sum(W[o][a] * m[a] for a in range(d))
                      for o in range(d)]
                new[(v, w)] = [act(H0[(v, w)][a] + Wm[a])
                               for a in range(d)]
        H = new
    return {"edge_states": H, "T": int(T),
            "excluded_reverse": bool(exclude_reverse),
            "note": "the exclusion of the reverse edge IS the "
                    "anti-tottering mechanism"}


def atom_readout(edge_states, adj, n):
    r"""Atom representations by summing incoming bond messages."""
    N = int(n)
    d = len(next(iter(edge_states.values())))
    out = []
    for v in range(N):
        acc = [0.0] * d
        for u in sorted(set(adj.get(v, ())) - {v}):
            if (u, v) in edge_states:
                for a in range(d):
                    acc[a] += edge_states[(u, v)][a]
        out.append(acc)
    return out


def concat_descriptors(learned, descriptors):
    r"""Concatenate expert molecule-level features with the learned
    representation.

    The paper's second improvement -- the two kinds of feature are not
    in competition.
    """
    a = [float(v) for v in k.vec(learned)]
    b = [float(v) for v in k.vec(descriptors)]
    return RichResult(payload={
        "estimate": a + b, "representation": a + b,
        "learned_dim": len(a), "descriptor_dim": len(b),
        "method": "D-MPNN representation with computed features; "
                  "Yang et al. (2019)",
    })


def cheatsheet():
    return ("dmlqs: pass messages along DIRECTED BONDS, not atoms. The "
            "stated reason is TOTTERS -- paths v1 v2 ... vn with "
            "v_i = v_{i+2}, where a message goes A -> B and comes "
            "straight back carrying A's own information as news, "
            "adding noise. The mechanism is one exclusion: "
            "m_vw = sum over u in N(v) EXCLUDING w. Atom "
            "representations are formed at the end from incoming bond "
            "messages, and computed molecule-level descriptors are "
            "concatenated with the learned representation -- expert "
            "features and learned ones are not rivals.")


# compact alias per ledger/NAMING.md
directedmpnn = dmpnn_message_pass
