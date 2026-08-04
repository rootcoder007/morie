# morie.fn -- slice s04 (rootcoder007/morie)
"""Backpropagation via chain rule for multi-layer networks.

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 379-425], Chapter 10, Section
10.8 and Section 10.8.1, equations (10.12) to (10.17), pp. 411-413, and
the hand computation of Illustrative Example 10.1, pp. 413-417.

Section 10.8.1 gives the algorithm literally:

  Step  9. delta_ij   = (y_ij - yhat_ij) g^(l)'(z_ij)
  Step 10. psi_ik     = g^(h)'(z_ik) sum_{j=1..L} delta_ij w_jk^(l)
  Step 11. w_jk^(l)(t+1) = w_jk^(l)(t) + eta delta_ij V_ik      (10.13)
  Step 12. w_kp^(h)(t+1) = w_kp^(h)(t) + eta psi_ik x_ip        (10.17)

and Step 8 accumulates the loss E = (1/(2 n L)) sum_ij (yhat_ij-y_ij)^2.
Equation (10.16) is the same psi written as the chain rule product; the
sum over j is "because each hidden neuron is connected to all the output
units".

Example 10.1 runs the four-pattern data set through it by hand and
prints V, yhat, delta, psi, both updated weight vectors and E = 0.03519.
Those printed numbers are the anchor: they are values the book computed,
not values this code produced.

The increments returned here are the book's Delta w divided by eta, so
that w_new = w_old + eta * gradient reproduces (10.13) and (10.17).
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["backpropagation_chain_rule"]


def _dact(name, a):
    """g'(z) written in terms of the activation a = g(z)."""
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "linear":
        return 1.0
    if name == "relu":
        return 1.0 if a > 0.0 else 0.0
    if name == "tanh":
        return 1.0 - a * a
    raise ValueError("backpropagation_chain_rule: unknown activation %r" % (name,))


def backpropagation_chain_rule(layers, activations, loss_grad, act_fun="sigmoid"):
    """Layer gradients of a feedforward network by the chain rule.

    Parameters
    ----------
    layers : sequence of matrices
        Weights W_1 .. W_L.  W_l has one row per unit of layer l and
        one column per unit of layer l-1 plus a leading bias column, so
        z_l = W_l [1, a_{l-1}].
    activations : sequence of matrices
        The forward pass a_0, a_1, ..., a_L, one row per pattern.
        a_0 is the input (no bias column; it is prepended internally).
    loss_grad : array-like
        n-by-units_L matrix of dE/dyhat, which Step 9 takes to be
        (y - yhat).
    act_fun : str or sequence of str
        Activation of each layer; a single name applies to all.

    Returns
    -------
    estimate  : gradients[0][0][0], the first weight increment
    gradients : one increment matrix per layer, shaped like layers
    deltas    : the per-layer errors, deltas[-1] = delta, deltas[0] = psi
    """
    W = [k.mat(w) for w in layers]
    A = [k.mat(a) for a in activations]
    Gd = k.mat(loss_grad)
    L = len(W)
    if L == 0:
        raise ValueError("backpropagation_chain_rule: no layers supplied")
    if len(A) != L + 1:
        raise ValueError("backpropagation_chain_rule: need L+1 activation blocks for L layers")
    n = len(A[0])
    if n == 0:
        raise ValueError("backpropagation_chain_rule: no patterns supplied")
    for a in A:
        if len(a) != n:
            raise ValueError("backpropagation_chain_rule: activation blocks disagree on the pattern count")
    if len(Gd) != n or len(Gd[0]) != len(A[L][0]):
        raise ValueError("backpropagation_chain_rule: loss_grad does not match the output layer")
    fns = [act_fun] * L if isinstance(act_fun, str) else list(act_fun)
    if len(fns) != L:
        raise ValueError("backpropagation_chain_rule: one activation name per layer is required")
    for l in range(L):
        if len(W[l]) != len(A[l + 1][0]):
            raise ValueError("backpropagation_chain_rule: layer %d has the wrong number of rows" % l)
        if len(W[l][0]) != len(A[l][0]) + 1:
            raise ValueError("backpropagation_chain_rule: layer %d has the wrong number of columns" % l)
    # Step 9, then Step 10 backwards through the layers.
    deltas = [None] * L
    d = [[Gd[i][j] * _dact(fns[L - 1], A[L][i][j]) for j in range(len(A[L][0]))]
         for i in range(n)]
    deltas[L - 1] = d
    for l in range(L - 2, -1, -1):
        u = len(A[l + 1][0])
        nxt = deltas[l + 1]
        new = []
        for i in range(n):
            row = []
            for kk in range(u):
                s = 0.0
                for j in range(len(W[l + 1])):
                    s += nxt[i][j] * W[l + 1][j][kk + 1]
                row.append(s * _dact(fns[l], A[l + 1][i][kk]))
            new.append(row)
        deltas[l] = new
    # Steps 11 and 12: the increment is sum_i delta_i outer [1, a_{l-1,i}].
    grads = []
    for l in range(L):
        rows = len(W[l])
        cols = len(W[l][0])
        G = [[0.0] * cols for _ in range(rows)]
        for i in range(n):
            prev = [1.0] + list(A[l][i])
            for j in range(rows):
                dj = deltas[l][i][j]
                for c in range(cols):
                    G[j][c] += dj * prev[c]
        grads.append(G)
    # Step 8's accumulated loss, with the book's 1/(2 n L) scaling.
    outs = len(A[L][0])
    E = 0.0
    for i in range(n):
        for j in range(outs):
            E += Gd[i][j] * Gd[i][j]
    E = E / (2.0 * n * outs)
    return RichResult(
        title="Backpropagation gradients",
        summary_lines=[("layers", L), ("patterns", n)],
        payload={
            "estimate": grads[0][0][0],
            "gradients": grads,
            "deltas": deltas,
            "loss": E,
            "n": n,
            "method": "delta/psi recursion of Chapter 10 Sect. 10.8.1 steps 9-12, eqs. (10.12)-(10.17)",
        },
    )


def cheatsheet():
    return "bprop: Backpropagation via chain rule for multi-layer networks"
