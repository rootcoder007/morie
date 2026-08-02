# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model parallelism: split model weights across devices."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_model_parallelism"]


def _weights(model):
    """Parameter count per layer, from ints, arrays, dicts or a mapping."""
    if hasattr(model, "items") and not isinstance(model, (list, tuple)):
        names = list(model.keys())
        raw = [model[k] for k in names]
    else:
        raw = list(model)
        names = [f"layer{i}" for i in range(len(raw))]
    out = []
    for i, item in enumerate(raw):
        if hasattr(item, "get") and not isinstance(item, (list, tuple, np.ndarray)):
            if "params" not in item:
                raise ValueError(f"geron_model_parallelism: layer {i} mapping has no 'params' entry")
            out.append(float(item["params"]))
        elif np.ndim(item) == 0:
            out.append(float(item))
        else:
            out.append(float(np.asarray(item).size))
    return names, np.asarray(out, dtype=float)


def _partition(w, k):
    """Contiguous k-way split minimising the largest device load (binary search)."""

    def fits(cap):
        used, load = 1, 0.0
        for v in w:
            if v > cap:
                return False
            if load + v > cap:
                used += 1
                load = v
                if used > k:
                    return False
            else:
                load += v
        return True

    lo, hi = float(w.max()), float(w.sum())
    for _ in range(200):
        if hi - lo <= 1e-9 * max(1.0, hi):
            break
        mid = 0.5 * (lo + hi)
        if fits(mid):
            hi = mid
        else:
            lo = mid
    cap = hi
    assign = np.zeros(w.size, dtype=int)
    dev, load = 0, 0.0
    n = w.size
    for i, v in enumerate(w):
        # Open a new device when this one is full, or when the layers left
        # only just cover the devices left -- otherwise a device sits unused.
        if dev < k - 1 and (load + v > cap or (n - i) <= (k - 1 - dev)):
            dev += 1
            load = 0.0
        assign[i] = dev
        load += v
    return assign


def geron_model_parallelism(model, n_devices):
    """
    Model parallelism: split model weights across devices.

    Formula: layer or tensor partition over N devices

    Layers are cut CONTIGUOUSLY, because a split that interleaves layers
    across devices sends activations back and forth on every boundary
    crossed. The partition minimises the largest device load, found by
    binary search on the capacity -- the classic split-array problem, and
    optimal, unlike a greedy fill.

    The honest result is usually disappointing, which is why the
    utilisation is reported: with sequential layers only ONE device runs
    at a time, so an even split still leaves N-1 devices idle. That is
    why Geron reaches for data parallelism first and for pipeline
    parallelism (see :func:`~morie.fn.hmppp.geron_pipeline_parallelism`)
    when a model genuinely will not fit.

    Parameters
    ----------
    model : sequence or mapping
        Per-layer parameter counts, weight arrays, or mappings carrying a
        ``"params"`` entry.
    n_devices : int
        Devices (>= 1, at most one per layer).

    Returns
    -------
    result : RichResult
        Keys: assignment, device_loads, max_load, imbalance,
        naive_utilisation, cut_points, estimate, n, method.

    Examples
    --------
    Layers of 10, 20, 30, 40 over two devices: the balanced contiguous
    cut is (10+20+30) against 40, so the busiest device holds 60.

    >>> r = geron_model_parallelism([10, 20, 30, 40], 2)
    >>> [int(a) for a in r["assignment"]], [float(v) for v in r["device_loads"]]
    ([0, 0, 0, 1], [60.0, 40.0])
    >>> float(r["max_load"])
    60.0

    Four devices take a layer each, and the imbalance is what the layer
    sizes force:

    >>> r4 = geron_model_parallelism([10, 20, 30, 40], 4)
    >>> [int(a) for a in r4["assignment"]]
    [0, 1, 2, 3]
    >>> round(float(r4["imbalance"]), 6)
    0.6

    Sequential layers keep only one device busy:

    >>> round(float(r["naive_utilisation"]), 6)
    0.5

    References
    ----------
    Geron Ch 17
    """
    names, w = _weights(model)
    if w.size == 0:
        raise ValueError("geron_model_parallelism: model has no layers")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("geron_model_parallelism: layer sizes must be finite and non-negative")
    k = int(n_devices)
    if k < 1:
        raise ValueError(f"geron_model_parallelism: n_devices must be >= 1, got {n_devices!r}")
    if k > w.size:
        raise ValueError(f"geron_model_parallelism: {k} devices for {w.size} layers; a layer cannot be split here")

    assign = _partition(w, k)
    loads = np.array([w[assign == d].sum() for d in range(k)], dtype=float)
    mean = float(loads.mean())
    imbalance = float((loads.max() - mean) / mean) if mean > 0 else 0.0
    cuts = [int(i) for i in range(1, w.size) if assign[i] != assign[i - 1]]

    return RichResult(
        title="Model parallelism partition",
        summary_lines=[("Devices", k), ("Max load", float(loads.max())), ("Imbalance", imbalance)],
        interpretation="Sequential layers idle every device but one; pipeline the microbatches to fix that.",
        payload={
            "assignment": assign,
            "layer_names": names,
            "layer_sizes": w,
            "device_loads": loads,
            "max_load": float(loads.max()),
            "imbalance": imbalance,
            "naive_utilisation": 1.0 / k,
            "cut_points": cuts,
            "n_transfers": len(cuts),
            "estimate": assign,
            "n": int(w.size),
            "method": "Optimal contiguous layer partition minimising the maximum device load",
        },
    )


def cheatsheet():
    return "hmmpp: Model parallelism, balanced contiguous layer partition"
