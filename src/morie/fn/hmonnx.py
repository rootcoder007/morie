# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trace and validate an ONNX export graph."""

import json

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_onnx_export"]

_SHAPE_PRESERVING = {"relu", "tanh", "sigmoid", "softmax", "dropout", "identity", "erf", "gelu"}


def geron_onnx_export(model, args, file=None):
    """
    Export a model to the ONNX graph format for cross-platform inference.

    Formula: torch.onnx.export(model, args, file)

    morie.fn is numpy-only, so no protobuf is emitted and none is
    pretended: what is written (when ``file`` is given) is the traced
    graph as JSON. The part that matters is done for real -- torch's
    exporter works by TRACING the model on a concrete example input, and
    tracing is exactly what fails silently in practice. This function
    performs that trace: it pushes ``args``' shape through every layer,
    resolves each node's input and output shapes, and REFUSES on the
    first mismatch or unsupported op instead of writing a graph that
    cannot run.

    The tracing caveat is the same one torch has: a shape resolved from
    one example is baked in, so control flow that depends on the data is
    not captured by any of this.

    ``model`` is a sequence of layer mappings: ``{"op": "Gemm",
    "in_features": n, "out_features": m}``, ``{"op": "Flatten"}``, or any
    shape-preserving op (Relu, Tanh, Sigmoid, Softmax, Dropout, Gelu).

    Parameters
    ----------
    model : sequence of mappings
        Layer specifications in execution order.
    args : array-like
        A concrete example input, batch dimension included.
    file : str, optional
        Path for the traced graph (JSON, not protobuf).

    Returns
    -------
    result : RichResult
        Keys: nodes, input_shape, output_shape, n_parameters, graph,
        file, is_protobuf, estimate, n, method.

    Examples
    --------
    >>> m = [{"op": "Gemm", "in_features": 3, "out_features": 2}, {"op": "Relu"}]
    >>> r = geron_onnx_export(m, np.zeros((1, 3)))
    >>> r["input_shape"], r["output_shape"], len(r["nodes"])
    ((1, 3), (1, 2), 2)
    >>> int(r["n_parameters"]), bool(r["is_protobuf"])
    (8, False)

    A shape the graph cannot accept is caught at export time, which is
    the entire point of tracing on a concrete input:

    >>> geron_onnx_export(m, np.zeros((1, 5)))
    Traceback (most recent call last):
        ...
    ValueError: geron_onnx_export: node 0 (Gemm) expects 3 input features but received 5

    References
    ----------
    Geron Appendix B
    """
    layers = list(model)
    if not layers:
        raise ValueError("geron_onnx_export: model has no layers to trace")
    a = np.asarray(args)
    if a.ndim < 1:
        raise ValueError("geron_onnx_export: args must be an array with at least one dimension (include the batch)")
    shape = tuple(int(v) for v in a.shape)
    in_shape = shape

    nodes = []
    params = 0
    for i, spec in enumerate(layers):
        if not hasattr(spec, "get"):
            raise ValueError(f"geron_onnx_export: layer {i} must be a mapping with an 'op' key, got {type(spec).__name__}")
        op = str(spec.get("op", "")).strip()
        if not op:
            raise ValueError(f"geron_onnx_export: layer {i} has no 'op'")
        low = op.lower()
        before = shape
        if low in ("gemm", "linear", "matmul"):
            nin = spec.get("in_features")
            nout = spec.get("out_features")
            if nin is None or nout is None:
                raise ValueError(f"geron_onnx_export: node {i} ({op}) needs in_features and out_features")
            nin, nout = int(nin), int(nout)
            if shape[-1] != nin:
                raise ValueError(
                    f"geron_onnx_export: node {i} ({op}) expects {nin} input features but received {shape[-1]}"
                )
            shape = shape[:-1] + (nout,)
            params += nin * nout + (nout if spec.get("bias", True) else 0)
        elif low == "flatten":
            shape = (shape[0], int(np.prod(shape[1:]))) if len(shape) > 1 else shape
        elif low in _SHAPE_PRESERVING:
            pass
        else:
            raise ValueError(
                f"geron_onnx_export: node {i} has unsupported op {op!r}; "
                f"supported: Gemm/Linear, Flatten, {sorted(_SHAPE_PRESERVING)}"
            )
        nodes.append({"index": i, "op": op, "input_shape": before, "output_shape": shape})

    graph = {
        "ir_format": "traced-graph-json",
        "input": {"name": "input", "shape": list(in_shape)},
        "output": {"name": "output", "shape": list(shape)},
        "nodes": [{"index": n["index"], "op": n["op"], "input_shape": list(n["input_shape"]),
                   "output_shape": list(n["output_shape"])} for n in nodes],
        "n_parameters": int(params),
    }
    if file is not None:
        with open(file, "w", encoding="utf-8") as fh:
            json.dump(graph, fh, indent=2)

    return RichResult(
        title="ONNX export trace",
        summary_lines=[("Nodes", len(nodes)), ("Input", in_shape), ("Output", shape)],
        warnings=[
            "No protobuf is written: morie.fn is numpy-only, so the graph is emitted as JSON. "
            "Shapes are traced from one concrete input, so data-dependent control flow is not captured."
        ],
        interpretation="Tracing on a real input is what turns a shape bug into an export error instead of a runtime one.",
        payload={
            "nodes": nodes,
            "graph": graph,
            "input_shape": in_shape,
            "output_shape": shape,
            "n_parameters": int(params),
            "file": file,
            "is_protobuf": False,
            "estimate": shape,
            "n": len(nodes),
            "method": "Shape-tracing ONNX export plan validated against a concrete example input",
        },
    )


def cheatsheet():
    return "hmonnx: ONNX export trace and shape validation"
