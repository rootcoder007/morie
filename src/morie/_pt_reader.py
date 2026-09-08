"""Native reader for PyTorch zip-format checkpoints (``torch.save``).

The format (torch/serialization.py, zip-file serialization used since
torch 1.6): the ``.pt`` file is a zip archive holding

* ``<name>/data.pkl`` -- a pickle of the object graph in which every
  tensor's storage is externalized through a persistent id
  ``('storage', <StorageClass>, key, location, numel)``;
* ``<name>/data/<key>`` -- the raw little-endian buffer of each storage;
* ``<name>/version``.

Tensors are rebuilt by ``torch._utils._rebuild_tensor_v2(storage,
storage_offset, size, stride, requires_grad, backward_hooks)``.

This module unpickles that graph with the same allowlist posture as
``torch.load(weights_only=True)``: only the storage classes, the tensor
rebuild helpers, and basic containers resolve; anything else raises.
No torch import anywhere.
"""

from __future__ import annotations

import pickle
import struct
import zipfile

# Storage class name -> (struct format char, bytes per element).
# bfloat16 has no struct code; handled explicitly in _decode.
_DTYPES = {
    "FloatStorage": ("f", 4),
    "DoubleStorage": ("d", 8),
    "HalfStorage": ("e", 2),
    "BFloat16Storage": ("bf16", 2),
    "LongStorage": ("q", 8),
    "IntStorage": ("i", 4),
    "ShortStorage": ("h", 2),
    "CharStorage": ("b", 1),
    "ByteStorage": ("B", 1),
    "BoolStorage": ("?", 1),
}


class _StorageType:
    def __init__(self, name):
        self.name = name


class _Storage:
    def __init__(self, raw, dtype_name):
        self.raw = raw
        self.dtype_name = dtype_name


def _decode(storage, offset, count):
    fmt, width = _DTYPES[storage.dtype_name]
    start = offset * width
    buf = storage.raw[start:start + count * width]
    if fmt == "bf16":
        # bfloat16 is the top half of an IEEE-754 float32
        out = []
        for i in range(count):
            (hi,) = struct.unpack_from("<H", buf, 2 * i)
            (v,) = struct.unpack("<f", struct.pack("<I", hi << 16))
            out.append(v)
        return out
    vals = struct.unpack_from("<%d%s" % (count, fmt), buf, 0)
    return [float(v) for v in vals]


class PtArray(list):
    """Flat row-major float carrier with the array surface the GGUF
    writer uses (ndim/shape/size/flatten/astype/tobytes)."""

    def __init__(self, values, shape):
        super().__init__(values)
        self.shape = tuple(int(s) for s in shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return len(self)

    def flatten(self):
        return PtArray(list(self), (len(self),))

    def astype(self, dtype):
        del dtype  # carrier is float64; the target matters in tobytes
        return PtArray(list(self), self.shape)

    def tobytes(self, fmt="f"):
        return struct.pack("<%d%s" % (len(self), fmt), *self)


class PtTensor:
    """Just enough tensor surface for ``t.float().cpu().numpy()``."""

    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(s) for s in size)
        self.stride = tuple(int(s) for s in stride)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        n = 1
        for s in self.size:
            n *= s
        # fast path: C-contiguous
        contig = []
        acc = 1
        for s in reversed(self.size):
            contig.insert(0, acc)
            acc *= s
        if tuple(contig) == self.stride or n <= 1:
            return PtArray(_decode(self.storage, self.offset, n),
                           self.size)
        # strided gather
        flat_max = self.offset
        for dim, st in zip(self.size, self.stride):
            flat_max += (dim - 1) * st
        base = _decode(self.storage, 0, flat_max + 1)
        out = []

        def rec(idx, pos):
            if len(idx) == len(self.size):
                out.append(base[pos])
                return
            d = len(idx)
            for i in range(self.size[d]):
                rec(idx + [i], pos + i * self.stride[d])

        rec([], self.offset)
        return PtArray(out, self.size)


def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                       requires_grad=False, backward_hooks=None,
                       metadata=None):
    del requires_grad, backward_hooks, metadata
    return PtTensor(storage, storage_offset, size, stride)


class _Unpickler(pickle.Unpickler):
    """weights_only-equivalent allowlist unpickler."""

    _ALLOWED = {
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
        ("collections", "OrderedDict"): dict,
    }

    def __init__(self, fh, zf, prefix):
        super().__init__(fh)
        self._zf = zf
        self._prefix = prefix

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return self._ALLOWED[(module, name)]
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(
            "refusing to unpickle %s.%s: only tensors and containers "
            "load (the weights_only contract)" % (module, name))

    def persistent_load(self, pid):
        kind = pid[0]
        if kind != "storage":
            raise pickle.UnpicklingError(
                "unknown persistent id %r" % (kind,))
        storage_type, key = pid[1], pid[2]
        name = storage_type.name if isinstance(storage_type,
                                               _StorageType) \
            else getattr(storage_type, "__name__", str(storage_type))
        if name not in _DTYPES:
            raise pickle.UnpicklingError(
                "unsupported storage type %r" % name)
        raw = self._zf.read("%s/data/%s" % (self._prefix, key))
        return _Storage(raw, name)


def load_checkpoint(path):
    """Load a ``torch.save`` zip checkpoint without torch.

    Returns the pickled object graph with every tensor as a
    :class:`PtTensor`. Raises ``pickle.UnpicklingError`` on anything
    outside the tensor/container allowlist -- the same refusal
    ``torch.load(weights_only=True)`` makes, for the same reason.
    """
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith("/data.pkl")]
        if not pkl:
            raise ValueError("%s is not a torch zip checkpoint "
                             "(no data.pkl)" % path)
        prefix = pkl[0][:-len("/data.pkl")]
        with zf.open(pkl[0]) as fh:
            return _Unpickler(fh, zf, prefix).load()
