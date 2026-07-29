# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reverse-mode autodiff via chain rule (what PyTorch autograd computes)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_autograd_chain_rule"]

_METHOD = "Reverse-mode automatic differentiation (chain rule)"


def geron_autograd_chain_rule(graph, grad_output):
    r"""Propagate a gradient backwards through a chain of local Jacobians.

    For a composition :math:`y = f_L \circ \dots \circ f_1(x)` reverse
    mode walks the chain from the output end,

    .. math::
        \frac{\partial L}{\partial x} =
        \frac{\partial L}{\partial y}\,
        J_L\, J_{L-1} \cdots J_1,

    each step being one vector-Jacobian product.  Doing it in this order
    -- rather than left to right -- is the whole trick: every
    intermediate is a *vector*, so one backward sweep costs the same as
    one forward sweep no matter how many inputs there are.

    Parameters
    ----------
    graph : sequence
        Nodes in **forward** order, ``graph[0]`` applied first. Each node
        is either a 2-D array of shape ``(out_dim, in_dim)`` -- the local
        Jacobian -- or a callable implementing the vector-Jacobian
        product, mapping an upstream gradient of length ``out_dim`` to
        one of length ``in_dim``.
    grad_output : array-like
        Seed gradient :math:`\partial L / \partial y`, length equal to the
        output dimension of the last node.

    Returns
    -------
    RichResult
        Payload keys ``grad_input``, ``intermediate_grads`` (one per
        node, output end first), ``depth``, ``grad_norm``, ``estimate``
        (L2 norm of the input gradient), ``n``, ``method``.

    References
    ----------
    Géron Ch 10, Autograd / Automatic Differentiation section.

    Examples
    --------
    ``x -> u`` with Jacobian ``diag(2, 3)``, then ``u -> y`` summing the
    two coordinates. Seeding with ``dL/dy = 1`` recovers ``[2, 3]``:

    >>> J1 = [[2.0, 0.0], [0.0, 3.0]]
    >>> J2 = [[1.0, 1.0]]
    >>> r = geron_autograd_chain_rule([J1, J2], [1.0])
    >>> r["grad_input"]
    [2.0, 3.0]
    >>> r["depth"]
    2

    A callable node works the same way:

    >>> r2 = geron_autograd_chain_rule([J1, lambda g: np.array([g[0], g[0]])],
    ...                                [1.0])
    >>> r2["grad_input"]
    [2.0, 3.0]
    """
    nodes = list(graph)
    if not nodes:
        raise ValueError("graph is empty; nothing to differentiate.")
    g = np.asarray(grad_output, dtype=float).ravel()
    if g.size == 0:
        raise ValueError("grad_output is empty.")
    if not np.all(np.isfinite(g)):
        raise ValueError("grad_output contains non-finite values.")

    intermediates = [g.tolist()]
    for k in range(len(nodes) - 1, -1, -1):
        node = nodes[k]
        if callable(node):
            out = np.asarray(node(g), dtype=float).ravel()
            if out.size == 0:
                raise ValueError(f"node {k} returned an empty gradient.")
        else:
            J = np.asarray(node, dtype=float)
            if J.ndim != 2:
                raise ValueError(
                    f"node {k} must be a 2-D Jacobian (out_dim, in_dim) or a "
                    f"callable, got ndim={J.ndim}."
                )
            if not np.all(np.isfinite(J)):
                raise ValueError(f"node {k} Jacobian contains non-finite values.")
            if J.shape[0] != g.size:
                raise ValueError(
                    f"node {k} Jacobian has out_dim={J.shape[0]} but the upstream "
                    f"gradient has length {g.size}."
                )
            out = g @ J
        if not np.all(np.isfinite(out)):
            raise ValueError(f"node {k} produced a non-finite gradient.")
        g = out
        intermediates.append(g.tolist())

    return RichResult(
        title="Reverse-mode autodiff",
        summary_lines=[("Depth", len(nodes)), ("‖dL/dx‖", float(np.linalg.norm(g)))],
        payload={
            "grad_input": g.tolist(),
            "intermediate_grads": intermediates,
            "depth": len(nodes),
            "grad_norm": float(np.linalg.norm(g)),
            "estimate": float(np.linalg.norm(g)),
            "n": int(g.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "graut: reverse-mode autodiff -- chain of vector-Jacobian products, output end first"
