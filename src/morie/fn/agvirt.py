# morie.fn -- function file (rootcoder007/morie)
"""Virtual loss for parallel Monte-Carlo tree search."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["virtloss", "alphazero_virtual_loss"]


def virtloss(W, N, pending, nvl=1):
    """Node values with virtual losses charged to in-flight simulations.

    In tree parallelisation several threads descend from the root at the
    same time and would otherwise take the same path.  A thread entering a
    node is charged a virtual loss straight away, so the node's value
    drops and the next thread only follows it if it is still better than
    its siblings; the charge is cancelled when that thread backs its real
    result up.  Counting a virtual loss as n_vl extra visits each scoring
    the worst outcome gives

        N'(s,a) = N(s,a) + n_vl * pending(s,a)
        W'(s,a) = W(s,a) - n_vl * pending(s,a)
        Q'(s,a) = W'(s,a) / N'(s,a),   0 when N' = 0.

    Parameters
    ----------
    W : array-like
        Accumulated action values (total, not mean).
    N : array-like
        Visit counts.
    pending : array-like
        Number of threads currently inside each child.
    nvl : int
        Virtual losses charged per in-flight thread.

    Returns
    -------
    RichResult
        ``Q``, ``N``, ``W``, ``Qclean``, ``k``, ``nvl``.

    References
    ----------
    Chaslot, G. M. J-B., Winands, M. H. M. and van den Herik, H. J.
    (2008), "Parallel Monte-Carlo tree search", Computers and Games 2008,
    Lecture Notes in Computer Science 5131, 60-71, Sect. 3.3, read from
    the authors' own PDF at dke.maastrichtuniversity.nl.  They describe
    virtual loss qualitatively -- one loss assigned when a thread visits
    a node in the selection phase and removed when that thread
    back-propagates -- and attribute it to Coulom.  The arithmetic above
    is the counter form of exactly that rule; the paper states no
    equation, so nothing beyond the described bookkeeping is claimed.
    """
    W = C.vec(W)
    N = C.vec(N)
    P = C.vec(pending)
    k = len(W)
    if len(N) != k or len(P) != k:
        raise ValueError("W, N and pending must have the same length")
    if any(v < 0.0 for v in N) or any(v < 0.0 for v in P):
        raise ValueError("counts must be non-negative")
    nvl = float(nvl)
    Nv = [N[i] + nvl * P[i] for i in range(k)]
    Wv = [W[i] - nvl * P[i] for i in range(k)]
    Q = [0.0 if Nv[i] == 0.0 else Wv[i] / Nv[i] for i in range(k)]
    Qc = [0.0 if N[i] == 0.0 else W[i] / N[i] for i in range(k)]
    return RichResult(payload={
        "Q": Q, "N": Nv, "W": Wv, "Qclean": Qc, "k": k, "nvl": nvl,
        "method": "Virtual loss in parallel MCTS (Chaslot et al. 2008 Sect. 3.3)"})


alphazero_virtual_loss = virtloss


def cheatsheet():
    return "agvirt: Virtual loss for parallel Monte-Carlo tree search."
