# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mixture-of-experts top-k routing with auxiliary load-balance loss.

Router and combination: Lepikhin, D., Lee, H., Xu, Y., Chen, D.,
Firat, O., Huang, Y., Krikun, M., Shazeer, N. and Chen, Z. (2021),
"GShard: Scaling Giant Models with Conditional Computation and
Automatic Sharding", ICLR 2021, arXiv:2006.16668 (Section 2.1,
Algorithm 1): gate values are the softmax of a linear router,
G(x) = softmax(W_g x); each token is dispatched to its top-k experts
and the output is the gate-weighted sum of the selected experts,
y = sum_{i in topk} G(x)_i E_i(x), with the selected gates
renormalised to sum to one.

Auxiliary loss: Fedus, W., Zoph, B. and Shazeer, N. (2022), "Switch
Transformers: Scaling to Trillion Parameter Models with Simple and
Efficient Sparsity", JMLR 23(120), arXiv:2101.03961, Eqs 4-6:

    aux = alpha . N . sum_i f_i P_i

where f_i is the fraction of tokens whose ARGMAX expert is i (Eq 5),
P_i the mean router probability mass on expert i over the batch
(Eq 6), N the number of experts, alpha ~ 1e-2. Under perfectly
uniform routing sum_i f_i P_i = 1/N, so aux = alpha -- that limiting
case is the test anchor.

Experts here are explicit linear maps (list of (d_in x d_out)
matrices), keeping the reference implementation deterministic.

Sources: fetched-wave3/lepikhin-etal-2021-gshard-arxiv2006.16668.pdf
(Sec 2.1); fetched-wave3/fedus-etal-2022-switch-transformer-
arxiv2101.03961.pdf (Eqs 1, 4-6).
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["moetop", "moe_topk_routing"]


def moetop(x, W_g, experts, k=2, alpha=0.01):
    """MoE top-k routing (GShard Sec 2.1; Switch Eqs 4-6 aux loss).

    Parameters
    ----------
    x : array-like, shape (T, d_in)
        Token batch.
    W_g : array-like, shape (d_in, N)
        Router weights; logits are x @ W_g.
    experts : sequence of array-like, each (d_in, d_out)
        Linear expert maps E_i(x) = x @ experts[i].
    k : int
        Experts per token. Default 2 (GShard).
    alpha : float
        Aux-loss coefficient. Default 1e-2 (Switch).

    Returns
    -------
    result : RichResult
        Keys: output (T x d_out), gates (T x N softmax), topk_indices,
        topk_gates (renormalised), aux_loss, f (Eq 5), P (Eq 6),
        estimate, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    Wg = np.atleast_2d(np.asarray(W_g, dtype=float))
    T, din = X.shape
    if Wg.shape[0] != din:
        raise ValueError(f"moetop: W_g rows {Wg.shape[0]} != token width {din}")
    N = Wg.shape[1]
    if not experts or len(experts) != N:
        raise ValueError(
            f"moetop: need one expert per router column ({N}), got {len(experts) if experts else 0}")
    k = int(k)
    if not 1 <= k <= N:
        raise ValueError(f"moetop: k must lie in 1..{N}, got {k}")
    Es = []
    dout = None
    for i, E in enumerate(experts):
        Em = np.atleast_2d(np.asarray(E, dtype=float))
        if Em.shape[0] != din:
            raise ValueError(f"moetop: expert {i} rows {Em.shape[0]} != {din}")
        if dout is None:
            dout = Em.shape[1]
        elif Em.shape[1] != dout:
            raise ValueError("moetop: experts disagree on output width")
        Es.append(Em)
    logits = X @ Wg
    gates = []
    for row in logits:
        r = [float(v) for v in row]
        m = max(r)
        e = [math.exp(v - m) for v in r]
        z = sum(e)
        gates.append([v / z for v in e])
    out = [[0.0] * dout for _ in range(T)]
    top_idx = []
    top_gate = []
    argmax_count = [0] * N
    for t in range(T):
        order = sorted(range(N), key=lambda i: (-gates[t][i], i))
        sel = order[:k]
        gsel = [gates[t][i] for i in sel]
        zs = sum(gsel)
        gnorm = [g / zs for g in gsel]
        top_idx.append(sel)
        top_gate.append(gnorm)
        argmax_count[order[0]] += 1
        xt = X[t]
        for g, i in zip(gnorm, sel):
            yi = xt @ Es[i]
            for c in range(dout):
                out[t][c] += g * float(yi[c])
    f = [c / T for c in argmax_count]
    P = [sum(gates[t][i] for t in range(T)) / T for i in range(N)]
    aux = float(alpha) * N * sum(fi * pi for fi, pi in zip(f, P))
    return RichResult(payload={
        "output": out,
        "gates": gates,
        "topk_indices": top_idx,
        "topk_gates": top_gate,
        "aux_loss": aux,
        "f": f,
        "P": P,
        "n_experts": N,
        "k": k,
        "estimate": float(out[0][0]),
        "n": int(T),
        "method": "MoE top-k routing + Switch aux load-balance loss (GShard Sec 2.1; Switch Eqs 4-6)",
    })


def moe_topk_routing(y=None, x=None, W_g=None, experts=None, k=2, alpha=0.01):
    """Back-compatible wrapper over :func:`moetop` (old stub name)."""
    if x is None or W_g is None or experts is None:
        raise ValueError("moe_topk_routing: x, W_g and experts are required")
    return moetop(x, W_g, experts, k=k, alpha=alpha)


def cheatsheet():
    return "moetop: MoE top-k routing + aux loss (GShard arXiv:2006.16668; Switch arXiv:2101.03961 Eqs 4-6)"

# public names resolved by fn/_lazy_map.json
moetopkrouting = moetop
