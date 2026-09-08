# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PyTorch-style tensor construction, backed by numpy."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_pytorch_tensor"]

_DTYPES = {
    "float64": np.float64, "double": np.float64,
    "float32": np.float32, "float": np.float32,
    "float16": np.float16, "half": np.float16,
    "bfloat16": np.float32,  # numpy has no bfloat16; stored as float32
    "int64": np.int64, "long": np.int64,
    "int32": np.int32, "int": np.int32,
    "int16": np.int16, "short": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}
_DEVICES = ("cpu", "cuda", "mps")


def geron_pytorch_tensor(x, device="cpu", dtype=None):
    """
    PyTorch tensor: an n-d array on CPU, GPU or MPS.

    Formula: t = torch.tensor(x, device, dtype)

    No torch call is made -- morie.fn is numpy-only -- and none is
    faked. What is real here is everything that does not need the
    library: the dtype name is resolved to a concrete numpy dtype by the
    same table torch uses, the device string is validated, and the
    array's shape, element size, strides and total bytes are reported for
    the tensor that would result.

    Two of torch's habits are worth knowing and are surfaced rather than
    hidden. ``torch.tensor`` defaults floats to float32, NOT to numpy's
    float64, so a numpy array round-tripped through torch silently loses
    precision; ``dtype_changed`` says whether that happened. And bfloat16
    has no numpy equivalent: it is stored here as float32, with the same
    exponent range but 16 more mantissa bits than the real thing, so
    rounding behaviour will NOT match a bfloat16 device.

    Parameters
    ----------
    x : array-like
        Data.
    device : {"cpu", "cuda", "mps"}, default "cpu"
        Target device; only "cpu" can actually execute here.
    dtype : str, optional
        Torch dtype name; defaults to float32 for floating input, as
        ``torch.tensor`` does, and to the natural integer type otherwise.

    Returns
    -------
    result : RichResult
        Keys: tensor, dtype, device, shape, nbytes, itemsize, strides,
        dtype_changed, on_device, estimate, n, method.

    Examples
    --------
    A 2x2 float32 tensor is 16 bytes:

    >>> r = geron_pytorch_tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> r["dtype"], r["shape"], int(r["nbytes"])
    ('float32', (2, 2), 16)

    Float input defaults to float32, not numpy's float64:

    >>> bool(r["dtype_changed"])
    True

    An explicit dtype is honoured:

    >>> geron_pytorch_tensor([[1.0, 2.0]], dtype="float64")["nbytes"]
    16

    A device that does not exist is an error:

    >>> geron_pytorch_tensor([1.0], device="tpu")
    Traceback (most recent call last):
        ...
    ValueError: geron_pytorch_tensor: device must be one of ('cpu', 'cuda', 'mps'), got 'tpu'

    References
    ----------
    Geron Ch 10
    """
    dev = str(device).split(":")[0].lower()
    if dev not in _DEVICES:
        raise ValueError(f"geron_pytorch_tensor: device must be one of {_DEVICES}, got {device!r}")
    a = np.asarray(x)
    if a.size == 0:
        raise ValueError("geron_pytorch_tensor: x is empty")
    src = a.dtype

    if dtype is None:
        target = np.float32 if np.issubdtype(src, np.floating) else src.type
        name = "float32" if np.issubdtype(src, np.floating) else np.dtype(src).name
    else:
        name = str(dtype).replace("torch.", "").lower()
        if name not in _DTYPES:
            raise ValueError(f"geron_pytorch_tensor: unknown dtype {dtype!r}; known: {sorted(_DTYPES)}")
        target = _DTYPES[name]
    t = np.ascontiguousarray(a.astype(target))

    return RichResult(
        title="Tensor",
        summary_lines=[("dtype", name), ("device", dev), ("shape", tuple(int(v) for v in t.shape))],
        warnings=(
            ["bfloat16 has no numpy equivalent and is stored as float32; rounding will not match a real bfloat16 device"]
            if name == "bfloat16"
            else []
        )
        + ([f"device {dev!r} is a plan only: morie.fn is numpy-only and executes on the CPU"] if dev != "cpu" else []),
        interpretation="torch.tensor defaults floats to float32; a numpy float64 array loses precision on the way in.",
        payload={
            "tensor": t,
            "dtype": name,
            "numpy_dtype": np.dtype(target).name,
            "device": dev,
            "shape": tuple(int(v) for v in t.shape),
            "ndim": int(t.ndim),
            "itemsize": int(t.itemsize),
            "nbytes": int(t.nbytes),
            "strides": tuple(int(v) for v in t.strides),
            "dtype_changed": bool(np.dtype(target) != src),
            "on_device": dev == "cpu",
            "estimate": t,
            "n": int(t.size),
            "method": "Tensor construction with torch dtype/device semantics, numpy-backed",
        },
    )


def cheatsheet():
    return "hmpttn: PyTorch-style tensor construction (numpy-backed)"
