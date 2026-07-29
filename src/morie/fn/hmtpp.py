# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tensor parallelism: split individual tensors across devices."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_tensor_parallelism"]


def geron_tensor_parallelism(model, n_devices=2, x=None, scheme="column"):
    """
    Tensor parallelism: split individual tensors across devices.

    Formula: shard W into W_1 ... W_N across devices; all-reduce output

    The sharded computation is actually performed and checked against the
    unsharded one, because the whole claim of tensor parallelism is that
    it is mathematically identical, only distributed:

    * **column parallel** -- ``W`` is split by columns,
      ``y_i = x W_i``, and the outputs are *concatenated*. No
      communication is needed for the forward pass.
    * **row parallel** -- ``W`` is split by rows and ``x`` by columns to
      match, ``y = sum_i x_i W_i``, so the partial products must be
      **all-reduced** (summed) across devices. That sum is the
      communication, and its volume is reported.

    Megatron-LM chains the two (column-parallel then row-parallel) so that
    only one all-reduce per transformer block is needed. Both schemes are
    exact: `max_diff` against the single-device result is returned.

    Parameters
    ----------
    model : array-like or sequence of array-like
        A weight matrix, or a list of them (each sharded independently).
    n_devices : int, default 2
        Devices N (>= 1). The sharded axis must be divisible by N.
    x : array-like, optional
        Input activations (n, d_in); defaults to ones.
    scheme : {"column", "row"}, default "column"
        Sharding scheme.

    Returns
    -------
    result : RichResult
        Keys: output, reference, max_diff, shards, params_per_device,
        comm_elements, all_reduce, estimate, n, method.

    Examples
    --------
    Column parallel: shards are concatenated and nothing is communicated.

    >>> W = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    >>> r = geron_tensor_parallelism(W, 2, x=[[1.0, 1.0]])
    >>> r["output"].tolist()
    [[6.0, 8.0, 10.0, 12.0]]
    >>> round(float(r["max_diff"]), 12)
    0.0
    >>> [s.shape for s in r["shards"][0]]
    [(2, 2), (2, 2)]
    >>> int(r["comm_elements"])
    0

    Row parallel: each device holds half the rows, and the partial sums
    must be all-reduced -- 4 elements per device here.

    >>> r2 = geron_tensor_parallelism(W, 2, x=[[1.0, 1.0]], scheme="row")
    >>> r2["output"].tolist()
    [[6.0, 8.0, 10.0, 12.0]]
    >>> bool(r2["all_reduce"])
    True
    >>> int(r2["comm_elements"])
    8

    References
    ----------
    Géron Ch 17
    """
    try:
        arr = np.asarray(model, dtype=float)
    except (TypeError, ValueError):
        arr = None
    if arr is not None and arr.ndim == 2:
        mats = [arr]
    elif arr is not None and arr.ndim == 3:
        mats = list(arr)
    elif isinstance(model, (list, tuple)) and model:
        mats = [np.asarray(w, dtype=float) for w in model]
    else:
        raise ValueError("geron_tensor_parallelism: model must be a 2-D weight matrix or a list of them")
    for i, w in enumerate(mats):
        if w.ndim != 2 or w.size == 0:
            raise ValueError(f"geron_tensor_parallelism: weight {i} must be a non-empty 2-D matrix, got shape {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError(f"geron_tensor_parallelism: weight {i} contains non-finite values")
    N = int(n_devices)
    if N < 1:
        raise ValueError(f"geron_tensor_parallelism: n_devices must be >= 1, got {N}")
    sch = str(scheme).lower()
    if sch not in ("column", "row"):
        raise ValueError(f"geron_tensor_parallelism: scheme must be 'column' or 'row', got {scheme!r}")

    a = np.ones((1, mats[0].shape[0])) if x is None else np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] != mats[0].shape[0]:
        raise ValueError(
            f"geron_tensor_parallelism: x has width {a.shape[1]} but the first weight expects {mats[0].shape[0]}"
        )
    for i in range(1, len(mats)):
        if mats[i - 1].shape[1] != mats[i].shape[0]:
            raise ValueError(
                f"geron_tensor_parallelism: weight {i - 1} outputs {mats[i - 1].shape[1]} but weight {i} "
                f"expects {mats[i].shape[0]}"
            )

    ref = a
    for w in mats:
        ref = ref @ w

    shards_all = []
    comm = 0
    h = a
    for w in mats:
        axis = 1 if sch == "column" else 0
        if w.shape[axis] % N:
            raise ValueError(
                f"geron_tensor_parallelism: {sch}-parallel needs the {'output' if axis else 'input'} "
                f"dimension {w.shape[axis]} divisible by {N} devices"
            )
        shards = np.split(w, N, axis=axis)
        shards_all.append(shards)
        if sch == "column":
            h = np.hstack([h @ s for s in shards])
        else:
            x_sh = np.split(h, N, axis=1)
            partials = [xi @ si for xi, si in zip(x_sh, shards)]
            h = sum(partials)
            comm += N * int(np.prod(partials[0].shape))  # every device contributes its partial sum

    diff = float(np.max(np.abs(h - ref)))
    per_dev = int(sum(s[0].size for s in shards_all))

    return RichResult(
        title="Tensor parallelism",
        summary_lines=[
            ("Devices", N),
            ("Scheme", sch),
            ("Parameters per device", per_dev),
            ("All-reduce elements", comm),
            ("Max difference vs single device", diff),
        ],
        interpretation=(
            "Tensor parallelism splits one layer, so it needs fast interconnect within a step; the "
            "column-then-row pairing is what keeps a transformer block down to a single all-reduce."
        ),
        payload={
            "output": h,
            "reference": ref,
            "max_diff": diff,
            "shards": shards_all,
            "params_per_device": per_dev,
            "params_total": int(sum(w.size for w in mats)),
            "comm_elements": int(comm),
            "all_reduce": bool(sch == "row" and N > 1),
            "scheme": sch,
            "estimate": diff,
            "n": int(a.shape[0]),
            "method": f"{sch}-parallel sharding across {N} devices, verified against the unsharded computation",
        },
    )


def cheatsheet():
    return "hmtpp: Tensor parallelism: split individual tensors across devices"
