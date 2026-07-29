# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reverse-mode autodiff: one backward sweep of the chain rule over a graph."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_reverse_mode_autodiff"]

_METHOD = "Reverse-mode automatic differentiation"


def geron_reverse_mode_autodiff(graph, loss_grad=1.0, output=None):
    r"""Accumulate adjoints from the loss back to every input.

    .. math::
        \frac{\partial L}{\partial x}
          = \sum_{y \in \mathrm{children}(x)}
            \frac{\partial L}{\partial y}\,\frac{\partial y}{\partial x}

    One backward sweep gives the gradient with respect to *every* node,
    at roughly the cost of one forward pass -- which is the entire reason
    neural networks are trained this way and not by forward-mode (one
    sweep per input) or by finite differences (two evaluations per input,
    see :mod:`morie.fn.grnud`).

    ``graph`` maps each node to its parents and the local partials
    :math:`\partial y/\partial x` recorded during the forward pass.  A
    node's adjoint may only be read once every child has contributed, so
    the sweep runs in reverse topological order; a cycle is not a
    computation graph and raises.

    Parameters
    ----------
    graph : mapping
        ``{node: {parent: local_partial}}``. Nodes with no entry (or an
        empty one) are leaves.
    loss_grad : float, optional
        Seed adjoint of the output node, normally 1.
    output : hashable, optional
        Output node; inferred as the unique node that is nobody's parent.

    Returns
    -------
    RichResult
        Payload keys ``gradients`` (node -> dL/dnode), ``leaf_gradients``,
        ``order`` (reverse topological order), ``output``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Appendix A, Reverse-mode autodiff section.

    Examples
    --------
    ``L = a * b`` with ``a = 3``, ``b = 4``: the recorded partials are
    ``dL/da = b = 4`` and ``dL/db = a = 3``.

    >>> g = {"L": {"a": 4.0, "b": 3.0}}
    >>> r = geron_reverse_mode_autodiff(g)
    >>> r["gradients"]["a"], r["gradients"]["b"]
    (4.0, 3.0)

    A node feeding two paths accumulates both -- the sum in the formula,
    and the thing a naive implementation drops.  ``L = u + v``,
    ``u = 2x``, ``v = 3x`` gives ``dL/dx = 5``:

    >>> g2 = {"L": {"u": 1.0, "v": 1.0}, "u": {"x": 2.0}, "v": {"x": 3.0}}
    >>> geron_reverse_mode_autodiff(g2)["gradients"]["x"]
    5.0
    """
    if not isinstance(graph, dict):
        try:
            graph = dict(graph)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("graph must be a mapping node -> {parent: partial}.") from exc
    if not graph:
        raise ValueError("graph is empty; there is nothing to differentiate.")

    parents = {}
    nodes = set()
    for node, ps in graph.items():
        if not isinstance(ps, dict):
            raise ValueError(f"graph[{node!r}] must be a dict of parent -> partial.")
        parents[node] = {p: float(v) for p, v in ps.items()}
        for v in parents[node].values():
            if not np.isfinite(v):
                raise ValueError(f"graph[{node!r}] holds a non-finite partial derivative.")
        nodes.add(node)
        nodes.update(ps)
    for n in nodes:
        parents.setdefault(n, {})

    children_count = {n: 0 for n in nodes}
    for node, ps in parents.items():
        for p in ps:
            children_count[p] += 1
    sinks = [n for n in nodes if children_count[n] == 0]
    if output is None:
        if len(sinks) != 1:
            raise ValueError(
                f"cannot infer the output node: {len(sinks)} nodes have no children "
                f"({sorted(map(str, sinks))}); pass output=."
            )
        output = sinks[0]
    elif output not in nodes:
        raise ValueError(f"output node {output!r} is not in the graph.")

    # Reverse topological order from the output, Kahn-style on the sub-DAG.
    order = []
    seen = set()
    temp = set()

    def visit(n):
        if n in seen:
            return
        if n in temp:
            raise ValueError(f"graph has a cycle through {n!r}; this is not a DAG.")
        temp.add(n)
        for p in parents[n]:
            visit(p)
        temp.discard(n)
        seen.add(n)
        order.append(n)

    visit(output)
    order.reverse()          # output first, leaves last

    loss_grad = float(loss_grad)
    if not np.isfinite(loss_grad):
        raise ValueError(f"loss_grad must be finite, got {loss_grad}.")
    grads = {n: 0.0 for n in order}
    grads[output] = loss_grad
    for n in order:
        for p, local in parents[n].items():
            if p in grads:
                grads[p] += grads[n] * local

    leaves = {n: g for n, g in grads.items() if not parents[n]}
    return RichResult(
        title="Reverse-mode autodiff",
        summary_lines=[("Nodes", len(order)), ("Output", str(output))],
        payload={
            "gradients": grads,
            "leaf_gradients": leaves,
            "order": [str(n) for n in order],
            "output": output,
            "estimate": leaves,
            "n": len(order),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrad: dL/dx = sum over children of (dL/dy)(dy/dx), one reverse sweep; fan-out ACCUMULATES"
