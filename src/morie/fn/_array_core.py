"""morie array core: numpy-free primitives with numpy-compatible syntax.

De-numpy campaign phase 2 foundation.  This module implements, in pure
Python, the primitives that cover the bulk of morie.fn's numpy usage
(the l14 inventory: 321,553 call sites, top-40 attrs = 96.3%).  Modules
switch by replacing `import numpy as np` with
`from morie.fn import _array_core as np` — call sites keep their `np.`
spelling.  C kernels in morie_core will back these entry points later;
this file is the always-available fallback and the reference semantics.

Scope notes:
- marr is a thin list-of-floats array (1-D and 2-D) with the ndarray
  surface morie.fn actually uses: arithmetic, comparisons, sum/mean/std,
  indexing, shape, tolist.
- linalg covers solve/inv/lstsq/norm via Gaussian elimination and normal
  equations (LAPACK-free).
- random.default_rng is the existing native Philox generator when the
  compiled core is present; here a Python fallback with the same API
  subset (normal, uniform, integers) built on SplitMix64 -- NOT the gr*
  LCG (banned) and clearly labeled non-Philox until the C hook lands.
"""

from __future__ import annotations

import builtins as _bi
import math as _math

pi = _math.pi
e = _math.e
inf = float("inf")
nan = float("nan")


def _num(v):
    """Coerce one element, preserving int and complex.

    numpy keeps an integer dtype for integer data, and that matters: an
    integer from arange() is usable as a slice index, a float is not.
    Complex is preserved for the same reason -- silently taking float()
    of it would raise, or worse, drop the imaginary part.
    """
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, complex)):
        return v
    return float(v)


class marr:
    """Minimal array: nested lists of floats, 1-D or 2-D."""

    __slots__ = ("data", "shape", "_is_mask", "_is_index", "_aif_keep")

    def __init__(self, data):
        if isinstance(data, marr):
            self.data = [row[:] for row in data.data] \
                if isinstance(data.data[0], list) else data.data[:]
            self.shape = data.shape
            return
        if isinstance(data, (int, float, complex)):
            data = [_num(data)]
        if hasattr(data, "tolist") and not isinstance(data, marr):
            fshape = tuple(int(v) for v in getattr(data, "shape", ()) or ())
            data = data.tolist()                 # foreign arrays (numpy)
            if isinstance(data, (int, float, complex)):
                data = [_num(data)]
            if len(fshape) == 2 and fshape[0] == 0:
                self.data = []
                self.shape = fshape
                return
        data = list(data)
        if data and hasattr(data[0], "tolist") \
                and not isinstance(data[0], marr):
            data = [r.tolist() if hasattr(r, "tolist") else r for r in data]
        if data and isinstance(data[0], (list, tuple, marr)):
            rows = [list(map(float, (r.data if isinstance(r, marr) else r)))
                    for r in data]
            ncol = len(rows[0])
            if any(len(r) != ncol for r in rows):
                raise ValueError("ragged rows")
            self.data = rows
            self.shape = (len(rows), ncol)
        else:
            self.data = [_num(v) for v in data]
            self.shape = (len(self.data),)

    # -- helpers -----------------------------------------------------
    def _flat(self):
        if len(self.shape) == 1:
            return self.data
        return [v for row in self.data for v in row]

    def _map(self, fn):
        if len(self.shape) == 1:
            return marr([fn(v) for v in self.data])
        return marr([[fn(v) for v in row] for row in self.data])

    def _zip(self, other, fn):
        o = asarray(other)
        if o.shape == (1,) and len(o.shape) == 1:
            return self._map(lambda v: fn(v, o.data[0]))
        if self.shape == (1,) and len(self.shape) == 1:
            s = self.data[0]
            return o._map(lambda v: fn(s, v))
        if o.shape == self.shape:
            if len(self.shape) == 1:
                return marr([fn(a, b) for a, b in zip(self.data, o.data)])
            return marr([[fn(a, b) for a, b in zip(r1, r2)]
                         for r1, r2 in zip(self.data, o.data)])
        # 2-D broadcasting: (n,m) with (n,1), (1,m), (n,) rows or (m,) cols
        a2, b2 = _b2(self), _b2(o)
        n = _bi.max(a2.shape[0], b2.shape[0])
        m = _bi.max(a2.shape[1], b2.shape[1])
        for arr in (a2, b2):
            if arr.shape[0] not in (1, n) or arr.shape[1] not in (1, m):
                raise ValueError("shape mismatch %s vs %s"
                                 % (self.shape, o.shape))
        out = [[fn(a2.data[i if a2.shape[0] > 1 else 0]
                   [j if a2.shape[1] > 1 else 0],
                   b2.data[i if b2.shape[0] > 1 else 0]
                   [j if b2.shape[1] > 1 else 0])
                for j in range(m)] for i in range(n)]
        return marr(out)

    # -- python protocol ---------------------------------------------
    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        if len(self.shape) == 1:
            return iter(self.data)
        return iter([marr(row) for row in self.data])

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 0:
                return self
            if len(idx) == 1:
                return self[idx[0]]
            if Ellipsis in idx:
                rest = tuple(v for v in idx if v is not Ellipsis)
                if len(rest) == 1:
                    if len(self.shape) == 1:
                        return self[rest[0]]
                    return self[(slice(None), rest[0])]
                idx = rest
            i, j = idx
            if j is None:                       # x[:, None] -> column
                base = self._flat() if i == slice(None) else None
                if base is None:
                    raise ValueError("unsupported index")
                return marr([[v] for v in base])
            if i is None:                       # x[None, :] -> row
                return marr([self._flat()])
            if len(self.shape) == 2:
                if isinstance(i, (marr, list)) and \
                        isinstance(j, (marr, list)):
                    # np.ix_-style outer gather
                    iv = [int(v) for v in
                          (i._flat() if isinstance(i, marr) else i)]
                    jv = [int(v) for v in
                          (j._flat() if isinstance(j, marr) else j)]
                    return marr([[self.data[r2][c2] for c2 in jv]
                                 for r2 in iv])
                if isinstance(i, (marr, list)) or (
                        hasattr(i, "tolist") and not isinstance(i, slice)):
                    # array-of-rows selector: mask or integer indices
                    iv = i._flat() if isinstance(i, marr) else (
                        i.tolist() if hasattr(i, "tolist") else list(i))
                    if getattr(i, "_is_mask", False) or (
                            not getattr(i, "_is_index", False)
                            and len(iv) == self.shape[0]
                            and _pyall(isinstance(v, bool) or v in (0.0, 1.0)
                                       for v in iv)
                            and (getattr(i, "_is_mask", False)
                                 or _pyall(isinstance(v, bool)
                                           for v in iv))):
                        rows = [self.data[k] for k, m in enumerate(iv) if m]
                    else:
                        rows = [self.data[int(v)] for v in iv]
                    if isinstance(j, slice):
                        return marr([r[j] for r in rows])
                    return marr([r[int(j)] for r in rows])
                if isinstance(i, slice) and isinstance(j, (marr, list)):
                    # column-array selector: x[:, idx]
                    jv = [int(v) for v in
                          (j._flat() if isinstance(j, marr) else j)]
                    return marr([[r[c] for c in jv]
                                 for r in self.data[i]])
                if isinstance(i, slice) or isinstance(j, slice):
                    rows = self.data[i] if isinstance(i, slice) \
                        else [self.data[i]]
                    picked = [(r[j] if isinstance(j, slice) else [r[j]])
                              for r in rows]
                    if isinstance(i, slice) and isinstance(j, slice):
                        return marr(picked)
                    if isinstance(i, slice):
                        return marr([row[0] for row in picked])
                    return marr(picked[0])
                if isinstance(j, (marr, list)) and isinstance(i, int):
                    # x[row, mask_or_index]
                    row = self.data[i]
                    jv = j._flat() if isinstance(j, marr) else list(j)
                    if getattr(j, "_is_mask", False) or (
                            not getattr(j, "_is_index", False)
                            and len(jv) == len(row)
                            and _pyall(v in (0.0, 1.0) for v in jv)):
                        return marr([v for v, m2 in zip(row, jv) if m2])
                    return marr([row[int(v)] for v in jv])
                return self.data[i][j]
            raise ValueError("unsupported index for 1-D")
        if isinstance(idx, marr):
            if len(idx.shape) == 2:
                same_shape = idx.shape == self.shape
                is_mask2 = getattr(idx, "_is_mask", False) or (
                    same_shape and not getattr(idx, "_is_index", False)
                    and _pyall(v in (0.0, 1.0) for v in idx._flat()))
                if is_mask2 and same_shape:
                    # 2-D boolean mask: numpy returns the selected
                    # elements as a flat array, row-major
                    return marr([self.data[r][c]
                                 for r in range(self.shape[0])
                                 for c in range(self.shape[1])
                                 if idx.data[r][c]])
                # 2-D fancy index: gather preserving the index shape
                return marr([[self.data[int(v)] for v in row]
                             for row in idx.data])
            vals = idx._flat()
            is_mask = getattr(idx, "_is_mask", False) or (
                not getattr(idx, "_is_index", False)
                and idx.shape == (self.shape[0],)
                and _pyall(v in (0.0, 1.0) for v in vals))
            if is_mask:
                keep = [k for k, m in enumerate(vals) if m != 0]
            else:
                keep = [int(v) for v in vals]   # fancy integer indexing
            out = marr([self.data[k] for k in keep])
            if getattr(self, "_is_index", False):
                out._is_index = True
            return out
        if hasattr(idx, "dtype") and hasattr(idx, "tolist"):
            # foreign (numpy) index array
            vals = idx.tolist()
            if getattr(idx.dtype, "kind", "") == "b":
                keep = [k for k, m in enumerate(vals) if m]
            else:
                keep = [int(v) for v in vals]
            out = marr([self.data[k] for k in keep])
            if getattr(self, "_is_index", False):
                out._is_index = True
            return out
        if isinstance(idx, (list, tuple)):
            if idx and _pyall(isinstance(b, bool) for b in idx) \
                    and len(idx) == len(self.data):
                return marr([self.data[k]
                             for k, b in enumerate(idx) if b])
            return marr([self.data[int(k)] for k in idx])
        if isinstance(idx, float):
            # indices produced by where()/argmax() flow back in as
            # whole-valued floats (marr is float-backed); numpy would
            # hand back int64 here
            if idx != int(idx):
                raise TypeError("non-integer array index %r" % idx)
            idx = int(idx)
        out = self.data[idx]
        if isinstance(out, list):
            return marr(out)
        return out

    @property
    def dtype(self):
        # the list-backed core is always float64; masks surface as
        # bool through __array__, but dtype reports the storage type
        return float64

    @property
    def size(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def ndim(self):
        return len(self.shape)

    def ravel(self):
        return marr(self._flat())

    def squeeze(self, axis=None):
        del axis
        if len(self.shape) == 2:
            if self.shape[0] == 1:
                return marr(self.data[0][:])
            if self.shape[1] == 1:
                return marr([row[0] for row in self.data])
        if self.shape == (1,):
            return self.data[0]
        return marr(self)

    def copy(self):
        out = marr([])
        out.data = [row[:] for row in self.data] \
            if len(self.shape) == 2 else self.data[:]
        out.shape = self.shape
        return out

    def astype(self, dtype=None, copy=True):
        del dtype, copy
        return marr(self)

    def flatten(self):
        return marr(self._flat())

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        f = self._flat()
        if len(shape) == 1 and (shape[0] == -1 or shape[0] == len(f)):
            return marr(f)
        if len(shape) == 2:
            n, m = shape
            if n == -1:
                n = len(f) // m
            if m == -1:
                m = len(f) // n
            if n * m != len(f):
                raise ValueError("cannot reshape")
            return marr([f[i * m:(i + 1) * m] for i in range(n)])
        raise ValueError("unsupported reshape %r" % (shape,))

    def __int__(self):
        # without this, int() falls back to the buffer protocol and
        # tries to parse the raw float64 bytes as a literal
        return int(self.__float__())

    def __index__(self):
        v = self.__float__()
        if v != int(v):
            raise TypeError("only integer-valued marr can index")
        return int(v)

    def __float__(self):
        f = self._flat()
        if len(f) != 1:
            raise ValueError("only single-element marr converts to float")
        return f[0]

    def __or__(self, o):
        out = self._zip(o, lambda a, b: 1.0 if (a != 0 or b != 0)
                        else 0.0)
        if getattr(self, "_is_mask", False) \
                or getattr(o, "_is_mask", False):
            out._is_mask = True
        return out

    def __and__(self, o):
        out = self._zip(o, lambda a, b: 1.0 if (a != 0 and b != 0)
                        else 0.0)
        if getattr(self, "_is_mask", False) \
                or getattr(o, "_is_mask", False):
            out._is_mask = True
        return out

    def __invert__(self):
        out = self._map(lambda v: 0.0 if v != 0 else 1.0)
        if getattr(self, "_is_mask", False):
            out._is_mask = True
        return out

    def __bool__(self):
        f = self._flat()
        if len(f) != 1:
            raise ValueError("truth value of multi-element marr is "
                             "ambiguous")
        return f[0] != 0

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple) and len(idx) == 0:
            v = asarray(value)
            f = v._flat() if isinstance(v, marr) else [float(v)]
            if len(self.shape) == 2:
                m2 = self.shape[1]
                if len(f) == 1:
                    f = f * (self.shape[0] * m2)
                for r2 in range(self.shape[0]):
                    self.data[r2] = [float(x)
                                     for x in f[r2 * m2:(r2 + 1) * m2]]
            else:
                if len(f) == 1:
                    f = f * len(self.data)
                self.data[:] = [float(x) for x in f]
            return
        if isinstance(idx, tuple) and len(idx) == 1:
            self[idx[0]] = value
            return
        if isinstance(idx, tuple) and len(self.shape) == 2:
            i, j = idx
            if isinstance(i, slice) and isinstance(j, (marr, list)):
                # x[:, mask_or_idx] = v
                jv = j._flat() if isinstance(j, marr) else list(j)
                if getattr(j, "_is_mask", False) or (
                        not getattr(j, "_is_index", False)
                        and len(jv) == self.shape[1]
                        and _pyall(v2 in (0.0, 1.0) for v2 in jv)):
                    cols = [c for c, m2 in enumerate(jv) if m2]
                else:
                    cols = [int(v2) for v2 in jv]
                va = asarray(value)
                vf = va._flat() if isinstance(va, marr) else [float(va)]
                for r2 in range(*i.indices(self.shape[0])):
                    for ci, c in enumerate(cols):
                        self.data[r2][c] = float(
                            vf[0] if len(vf) == 1 else vf[ci % len(vf)])
                return
            if isinstance(i, (marr, list, tuple)) \
                    and isinstance(j, (marr, list, tuple)):
                iv = [int(v) for v in
                      (i._flat() if isinstance(i, marr) else i)]
                jv = [int(v) for v in
                      (j._flat() if isinstance(j, marr) else j)]
                va = asarray(value)
                vf = va._flat() if isinstance(va, marr) else [float(va)]
                if len(vf) == 1:
                    vf = vf * len(iv)
                for r2, c2, v2 in zip(iv, jv, vf):
                    self.data[r2][c2] = float(v2)
                return
            if isinstance(i, (marr, list, tuple)) \
                    and not isinstance(i, slice):
                flags = [bool(v) for v in (
                    i._flat() if isinstance(i, marr) else i)]
                if len(flags) == self.shape[0]:
                    rows = [r for r, fl in enumerate(flags) if fl]
                else:
                    rows = [int(v) for v in (
                        i._flat() if isinstance(i, marr) else i)]
                cols = (range(*j.indices(self.shape[1]))
                        if isinstance(j, slice) else [int(j)])
                v = asarray(value)
                vals = list(v._flat())
                scalar = len(vals) == 1
                for ri, r in enumerate(rows):
                    for ci, c in enumerate(cols):
                        self.data[r][c] = vals[0] if scalar else \
                            vals[(ri * len(cols) + ci) % len(vals)]
                return
            if isinstance(i, slice) or isinstance(j, slice):
                rows = range(*i.indices(self.shape[0])) \
                    if isinstance(i, slice) else [i]
                cols = range(*j.indices(self.shape[1])) \
                    if isinstance(j, slice) else [j]
                v = asarray(value)
                flat = list(v._flat())
                if len(flat) == len(rows) * len(cols) and len(flat) > 1 \
                        and (len(v.shape) == 1
                             or v.shape == (len(rows), len(cols))):
                    for ri, r in enumerate(rows):
                        for ci, c in enumerate(cols):
                            self.data[r][c] = flat[ri * len(cols) + ci]
                    return
                v2 = _b2(v)
                for ri, r in enumerate(rows):
                    for ci, c in enumerate(cols):
                        self.data[r][c] = v2.data[
                            ri if v2.shape[0] > 1 else 0][
                            ci if v2.shape[1] > 1 else 0]
                return
            self.data[i][j] = float(value)
        elif isinstance(idx, slice):
            vals = asarray(value)._flat()
            rng = range(*idx.indices(self.shape[0]))
            if len(vals) == 1:
                vals = vals * len(rng)
            for k, r in zip(rng, vals):
                self.data[k] = r
        elif isinstance(idx, (list, tuple, marr)) or (
                hasattr(idx, "dtype") and hasattr(idx, "tolist")):
            if isinstance(idx, marr):
                raw = list(idx._flat())
            elif hasattr(idx, "dtype"):
                raw = list(idx.tolist())
            else:
                raw = list(idx)
            is_mask = getattr(idx, "_is_mask", False) or (
                getattr(getattr(idx, "dtype", None), "kind", "")
                == "b") or (
                raw and len(raw) == len(self.data)
                and _bi.all(isinstance(b, bool) for b in raw))
            if is_mask:
                ids = [k for k, b in enumerate(raw) if b]
            else:
                ids = [int(v) for v in raw]
            vals = list(asarray(value)._flat()) \
                if not isinstance(value, (int, float)) else None
            for k, r in enumerate(ids):
                self.data[r] = float(value) if vals is None \
                    else float(vals[k if len(vals) > 1 else 0])
        else:
            if len(self.shape) == 2 and isinstance(idx, int):
                v = asarray(value)
                f = v._flat()
                if len(f) == 1:
                    f = f * self.shape[1]
                if len(f) != self.shape[1]:
                    raise ValueError("row assignment length mismatch")
                self.data[idx] = [float(x) for x in f]
                return
            self.data[idx] = float(value)

    @property
    def __array_interface__(self):
        """numpy consumes this BEFORE the PEP-3118 buffer, so tagged
        masks surface as bool and index arrays as int64 — the raw
        buffer stays float64 for the compiled kernels."""
        import array as _pa
        import ctypes as _ct
        f = self._flat()
        if getattr(self, "_is_mask", False):
            buf = bytes(bytearray(1 if v else 0 for v in f))
            typestr = "|b1"
        elif getattr(self, "_is_index", False):
            buf = _pa.array("q", [int(v) for v in f]).tobytes()
            typestr = "<i8"
        else:
            buf = _pa.array("d", [float(v) for v in f]).tobytes()
            typestr = "<f8"
        self._aif_keep = buf
        addr = _ct.cast(_ct.c_char_p(buf), _ct.c_void_p).value
        return {"version": 3, "shape": self.shape,
                "typestr": typestr, "data": (addr, True)}

    def __buffer__(self, flags):
        """PEP 688 buffer protocol (Python >= 3.12): expose the flat
        float64 data so nanobind kernels and memoryview consumers get
        the array without numpy. Snapshot semantics: the exported
        buffer is a copy, matching the immutable-input contract of the
        compiled kernels. Tagged masks/index arrays refuse the buffer
        (numpy >= 2.5 prefers it over __array_interface__, which would
        surface them as float64 and break real-numpy indexing)."""
        del flags
        if getattr(self, "_is_mask", False) or \
                getattr(self, "_is_index", False):
            raise BufferError("tagged mask/index arrays export via "
                              "__array_interface__")
        if len(getattr(self, "shape", (0,))) == 2:
            # a flat buffer would lose the 2-D shape; numpy falls back
            # to __array_interface__, which carries it
            raise BufferError("2-D arrays export via "
                              "__array_interface__")
        import array as _pa
        buf = _pa.array("d", [float(v) for v in self._flat()])
        return memoryview(buf)

    def __array__(self, dtype=None, copy=None):
        """numpy interop for mixed test environments: masks surface as
        bool arrays so real-numpy indexing works. Never imports numpy
        itself — only cooperates when the caller already has it."""
        del copy
        import sys as _sys
        _np = _sys.modules.get("numpy")
        if _np is None:
            raise TypeError("numpy not loaded")
        if dtype is None and getattr(self, "_is_mask", False):
            dtype = bool
        elif dtype is None and getattr(self, "_is_index", False):
            dtype = "int64"
        return _np.asarray(self.tolist(), dtype=dtype)

    def tolist(self):
        return [row[:] for row in self.data] \
            if len(self.shape) == 2 else self.data[:]

    def __repr__(self):
        return "marr(%r)" % (self.tolist(),)

    # -- arithmetic ---------------------------------------------------
    def __add__(self, o):
        return self._zip(o, lambda a, b: a + b)
    __radd__ = __add__

    def __sub__(self, o):
        return self._zip(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._zip(o, lambda a, b: b - a)

    def __mul__(self, o):
        return self._zip(o, lambda a, b: a * b)
    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._zip(o, lambda a, b: a / b)

    def __rtruediv__(self, o):
        return self._zip(o, lambda a, b: b / a)

    def __pow__(self, o):
        return self._zip(o, lambda a, b: a ** b)

    def __rpow__(self, o):
        return self._zip(o, lambda a, b: b ** a)

    def __mod__(self, o):
        return self._zip(o, lambda a, b: a % b)

    def __rmod__(self, o):
        return self._zip(o, lambda a, b: b % a)

    def __floordiv__(self, o):
        return self._zip(o, lambda a, b: a // b)

    def __rfloordiv__(self, o):
        return self._zip(o, lambda a, b: b // a)

    def __neg__(self):
        return self._map(lambda v: -v)

    def __matmul__(self, o):
        return matmul(self, o)

    # -- comparisons (return 0/1 masks, tagged for indexing) ----------
    def _tag_mask(self, out):
        out._is_mask = True
        return out

    def __lt__(self, o):
        return self._tag_mask(
            self._zip(o, lambda a, b: 1.0 if a < b else 0.0))

    def __le__(self, o):
        return self._tag_mask(
            self._zip(o, lambda a, b: 1.0 if a <= b else 0.0))

    def __gt__(self, o):
        return self._tag_mask(
            self._zip(o, lambda a, b: 1.0 if a > b else 0.0))

    def __ge__(self, o):
        return self._tag_mask(
            self._zip(o, lambda a, b: 1.0 if a >= b else 0.0))

    def __eq__(self, o):
        try:
            return self._tag_mask(
                self._zip(o, lambda a, b: 1.0 if a == b else 0.0))
        except (TypeError, ValueError):
            return NotImplemented

    def __ne__(self, o):
        try:
            return self._tag_mask(
                self._zip(o, lambda a, b: 1.0 if a != b else 0.0))
        except (TypeError, ValueError):
            return NotImplemented

    __hash__ = None

    def __abs__(self):
        return self._map(lambda v: v if v >= 0 else -v)

    def __class_getitem__(cls, item):
        # typing-generic support: NDArray[np.float64] etc. resolve to
        # the class itself (PEP 560); dtype detail carries no runtime
        # meaning for the list-backed core.
        del item
        return cls

    def clip(self, a_min=None, a_max=None, lower=None, upper=None):
        lo = a_min if a_min is not None else lower
        hi = a_max if a_max is not None else upper

        def one(v):
            if lo is not None and v < lo:
                return float(lo)
            if hi is not None and v > hi:
                return float(hi)
            return v
        return self._map(one)

    def argmax(self):
        f = self._flat()
        return f.index(_bi.max(f))

    # -- reductions ----------------------------------------------------
    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        del dtype, out
        if axis is None:
            v = float(_math.fsum(self._flat()))
            return marr([v]) if keepdims else v
        if len(self.shape) != 2:
            # numpy: axis 0 / -1 on a 1-D array is the full reduction
            v = float(_math.fsum(self._flat()))
            return marr([v]) if keepdims else v
        if axis == 0:
            out = [_math.fsum(self.data[i][j]
                              for i in range(self.shape[0]))
                   for j in range(self.shape[1])]
            return marr([out]) if keepdims else marr(out)
        out = [_math.fsum(row) for row in self.data]
        return marr([[v] for v in out]) if keepdims else marr(out)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        del dtype, out
        if axis is None or len(self.shape) != 2:
            f = self._flat()
            v = float(_math.fsum(f) / len(f))
            return marr([v]) if keepdims else v
        s = self.sum(axis=axis)
        d = self.shape[0] if axis == 0 else self.shape[1]
        out = s / float(d)
        if keepdims:
            return marr([out.tolist()]) if axis == 0 else \
                marr([[v] for v in out._flat()])
        return out

    def var(self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=False):
        del dtype, out
        if keepdims and axis is not None and len(self.shape) == 2:
            v = self.var(axis=axis, ddof=ddof)
            return marr([v.tolist()]) if axis in (0, -2) else \
                marr([[x] for x in v._flat()])
        if axis is not None and len(self.shape) == 2:
            m = self.mean(axis=axis)
            if axis == 0:
                return marr([_math.fsum(
                    (self.data[i][j] - m.data[j]) ** 2
                    for i in range(self.shape[0]))
                    / (self.shape[0] - ddof)
                    for j in range(self.shape[1])])
            return marr([_math.fsum((v - m.data[i]) ** 2
                                    for v in row)
                         / (self.shape[1] - ddof)
                         for i, row in enumerate(self.data)])
        f = self._flat()
        m = _math.fsum(f) / len(f)
        return float(_math.fsum((v - m) ** 2 for v in f) / (len(f) - ddof))

    def std(self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=False):
        del dtype, out
        if keepdims and axis is not None and len(self.shape) == 2:
            v = self.std(axis=axis, ddof=ddof)
            return marr([v.tolist()]) if axis in (0, -2) else \
                marr([[x] for x in v._flat()])
        v = self.var(axis=axis, ddof=ddof)
        if isinstance(v, marr):
            return marr([_math.sqrt(u) for u in v._flat()])
        return _math.sqrt(v)

    def max(self, axis=None, out=None, keepdims=False, initial=None):
        del out
        if initial is not None and axis is None:
            f = self._flat()
            return _bi.max([float(initial)] + f)
        if axis is not None and len(self.shape) == 2:
            if axis in (0, -2):
                out2 = [_bi.max(r[c] for r in self.data)
                        for c in range(self.shape[1])]
                return marr([out2]) if keepdims else marr(out2)
            out2 = [_bi.max(r) for r in self.data]
            return marr([[v] for v in out2]) if keepdims \
                else marr(out2)
        return float(_bi.max(self._flat()))

    def min(self, axis=None, out=None, keepdims=False, initial=None):
        del out
        if initial is not None and axis is None:
            f = self._flat()
            return _bi.min([float(initial)] + f)
        if axis is not None and len(self.shape) == 2:
            if axis in (0, -2):
                out2 = [_bi.min(r[c] for r in self.data)
                        for c in range(self.shape[1])]
                return marr([out2]) if keepdims else marr(out2)
            out2 = [_bi.min(r) for r in self.data]
            return marr([[v] for v in out2]) if keepdims \
                else marr(out2)
        return float(_bi.min(self._flat()))

    def all(self, axis=None, out=None, keepdims=False):
        del out, keepdims
        if axis is not None and len(self.shape) == 2:
            if axis in (0, -2):
                return self._tag_mask(marr(
                    [1.0 if _pyall(r[c] for r in self.data) else 0.0
                     for c in range(self.shape[1])]))
            return self._tag_mask(marr(
                [1.0 if _pyall(r) else 0.0 for r in self.data]))
        return _bi.all(v != 0 for v in self._flat())

    def any(self, axis=None, out=None, keepdims=False):
        del out, keepdims
        if axis is not None and len(self.shape) == 2:
            if axis in (0, -2):
                return self._tag_mask(marr(
                    [1.0 if _pyany(r[c] for r in self.data) else 0.0
                     for c in range(self.shape[1])]))
            return self._tag_mask(marr(
                [1.0 if _pyany(r) else 0.0 for r in self.data]))
        return _bi.any(v != 0 for v in self._flat())

    @property
    def T(self):
        if len(self.shape) == 1:
            return marr(self.data)
        return marr([[self.data[i][j] for i in range(self.shape[0])]
                     for j in range(self.shape[1])])


def _b2(a):
    """View as 2-D for broadcasting: 1-D (m,) becomes a (1, m) row."""
    if len(a.shape) == 2:
        return a
    return marr([a.data])


ndarray = marr
class _DTypeF64:
    """Storage dtype marker: equal to float, numpy.float64 (by name),
    and the string "float64" so dtype comparisons written against any
    of the three conventions hold."""

    def __eq__(self, other):
        if other in (float, "float64"):
            return True
        return getattr(other, "__name__", "") == "float64"

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash("float64")

    def __repr__(self):
        return "dtype('float64')"

    name = "float64"
    kind = "f"
    str = "<f8"
    itemsize = 8

    @property
    def dtype(self):
        """numpy dtype-protocol hook: real-numpy calls that receive
        this marker (astype/zeros with dtype=float64) resolve through
        this attribute. Only cooperates when numpy is already loaded —
        the core itself never imports it."""
        import sys as _sys
        _np = _sys.modules.get("numpy")
        if _np is not None:
            return _np.dtype("float64")
        return float

    def __call__(self, v):
        return float(v)


float64 = _DTypeF64()


# ------------------------------------------------------------ construction

class _ObjDType:
    kind = "O"
    name = "object"

    def __eq__(self, other):
        return other is object or other == "object" \
            or getattr(other, "kind", "") == "O"

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    def __repr__(self):
        return "dtype('O')"


class oarr(list):
    """Object-mode array: numpy's dtype=object surface for string /
    mixed data. List-backed; comparisons yield tagged masks so the
    usual mask-indexing pipelines work."""

    def tolist(self):
        return list(self)

    def _map(self, fn):
        out = [fn(v) for v in self]
        try:
            return marr([float(v) for v in out])
        except (TypeError, ValueError):
            return oarr(out)

    @property
    def shape(self):
        return (len(self),)

    @property
    def size(self):
        return len(self)

    def _flat(self):
        return list(self)

    @property
    def ndim(self):
        return 1

    @property
    def dtype(self):
        return _ObjDType()

    def copy(self):
        return oarr(self)

    def ravel(self):
        return oarr(self)

    def flatten(self):
        return oarr(self)

    def astype(self, dtype=None):
        if dtype is None or _is_object_like(None, dtype):
            return oarr(self)
        return marr([float(v) for v in self])

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if shape in ((-1,), (len(self),)):
            return oarr(self)
        raise ValueError("oarr reshape supports 1-D only")

    def __eq__(self, other):
        out = marr([1.0 if v == other else 0.0 for v in self])
        out._is_mask = True
        return out

    def __ne__(self, other):
        out = marr([1.0 if v != other else 0.0 for v in self])
        out._is_mask = True
        return out

    __hash__ = None

    def __getitem__(self, key):
        if isinstance(key, marr):
            vals = key._flat()
            if getattr(key, "_is_mask", False):
                return oarr(v for v, m in zip(self, vals) if m)
            return oarr(list(self)[int(v)] for v in vals)
        if hasattr(key, "dtype") and hasattr(key, "tolist"):
            vals = key.tolist()
            if getattr(key.dtype, "kind", "") == "b":
                return oarr(v for v, m in zip(self, vals) if m)
            return oarr(list(self)[int(v)] for v in vals)
        out = list.__getitem__(self, key)
        return oarr(out) if isinstance(out, list) else out


def _is_object_like(x, dtype):
    if dtype is not None and (dtype is object
                              or getattr(dtype, "__name__", "")
                              == "object" or dtype == "object"):
        return True
    xdt = getattr(x, "dtype", None)
    if xdt is not None and (xdt is object
                            or getattr(xdt, "kind", "") in ("O", "U", "S")
                            or str(getattr(xdt, "name", xdt)) == "object"):
        return True
    if isinstance(x, (list, tuple)) and x and any(
            isinstance(v, str) for v in x):
        return True
    return False


def asarray(x, dtype=None):
    if isinstance(x, oarr) and dtype is None:
        return x
    if hasattr(x, "columns") and hasattr(x, "to_numpy") \
            and not isinstance(x, marr):
        # frame-like (native or real pandas): take its array form
        x = x.to_numpy()
    if _is_object_like(x, dtype):
        return oarr(x.tolist() if hasattr(x, "tolist") else x)
    if isinstance(x, marr):
        return x
    try:
        return marr(x)
    except (TypeError, ValueError):
        # non-numeric payload (strings via an untyped container):
        # numpy would build an object array here
        return oarr(x.tolist() if hasattr(x, "tolist") else x)


def array(x, dtype=None):
    if _is_object_like(x, dtype):
        return oarr(x.tolist() if hasattr(x, "tolist") else x)
    del dtype
    return marr(x)


def atleast_1d(x):
    return asarray(x)


def atleast_2d(x):
    a = asarray(x)
    return a if len(a.shape) == 2 else marr([a.data])


def arange(start, stop=None, step=1, dtype=None):
    """Like numpy.arange, including its dtype rule.

    numpy yields an INTEGER array when start, stop and step are all
    integers, and that is load-bearing: a float cannot be used as a slice
    index, so returning floats unconditionally broke every caller that
    wrote x[: n - m] for m in arange(...).
    """
    if stop is None:
        start, stop = 0, start
    exact = (dtype is None
             and all(isinstance(v, int) and not isinstance(v, bool)
                     for v in (start, stop, step)))
    if dtype is not None:
        exact = dtype in (int, "int", "int64", "int32", "i8", "i4")
    n = _bi.max(0, int(_math.ceil((stop - start) / step - 1e-12)))
    if exact:
        return marr([int(start) + i * int(step) for i in range(n)])
    return marr([float(start) + i * float(step) for i in range(n)])


def zeros(n, dtype=None):
    if isinstance(n, tuple) and len(n) == 1:
        n = n[0]
    if isinstance(n, tuple):
        out = marr([[0.0] * n[1] for _ in range(n[0])]) \
            if n[0] else marr([])
        if not n[0]:
            out.shape = (0, int(n[1]))
        return out
    return marr([0.0] * int(n))


def ones(n, dtype=None):
    if isinstance(n, tuple) and len(n) == 1:
        n = n[0]
    if isinstance(n, tuple):
        return marr([[1.0] * n[1] for _ in range(n[0])])
    return marr([1.0] * int(n))


def full(n, v, dtype=None):
    del dtype
    if isinstance(n, (tuple, list)):
        if len(n) == 2:
            return marr([[float(v)] * int(n[1])
                         for _ in range(int(n[0]))])
        n = n[0]
    return marr([float(v)] * int(n))


def linspace(a, b, n):
    n = int(n)
    if n == 1:
        return marr([float(a)])
    step = (b - a) / (n - 1)
    return marr([a + i * step for i in range(n)])


def eye(n, m=None, dtype=None):
    del dtype
    m = int(n) if m is None else int(m)
    return marr([[1.0 if i == j else 0.0 for j in range(m)]
                 for i in range(int(n))])


def diag(x):
    a = asarray(x)
    if len(a.shape) == 1:
        n = a.shape[0]
        return marr([[a.data[i] if i == j else 0.0 for j in range(n)]
                     for i in range(n)])
    return marr([a.data[i][i] for i in range(_bi.min(a.shape))])


def column_stack(cols):
    cs = [asarray(c) for c in cols]
    n = cs[0].shape[0]
    out = [[] for _ in range(n)]
    for c in cs:
        if c.shape[0] != n:
            raise ValueError("column_stack needs equal-length inputs")
        if len(c.shape) == 1:
            for i in range(n):
                out[i].append(c.data[i])
        else:
            for i in range(n):
                out[i].extend(c.data[i])
    return marr(out)


def concatenate(parts, axis=0):
    arrs = [asarray(p) for p in parts]
    if axis in (0, None) and len(arrs[0].shape) == 1:
        out = []
        for a in arrs:
            out.extend(a._flat())
        return marr(out)
    if axis in (0, None):
        return vstack(arrs)
    if axis in (1, -1):
        return hstack(arrs)
    if axis == -2:
        # -2 on rank-2 operands is the row axis
        return vstack(arrs)
    raise ValueError("concatenate: unsupported axis %r" % (axis,))


# ------------------------------------------------------------- elementwise

def _uf(fn):
    def wrapped(x):
        if isinstance(x, ndlist):
            return ndlist(wrapped(marr(b)) for b in x)
        if isinstance(x, carr) or (
                isinstance(x, (list, tuple))
                and x and isinstance(x[0], complex)):
            vals = [fn(v) for v in x]
            if any(isinstance(v, complex) for v in vals):
                return carr(vals)
            return marr([float(v) for v in vals])
        a = asarray(x)
        if a.shape == (1,) and not isinstance(x, (list, tuple, marr)):
            return fn(a.data[0])
        return a._map(fn)
    return wrapped


sqrt = _uf(_math.sqrt)
exp = _uf(_math.exp)
log = _uf(_math.log)
log1p = _uf(_math.log1p)
abs = _uf(_bi.abs)  # noqa: A001  # builtin: complex -> magnitude


def clip(x, lo, hi):
    # numpy allows None for an open bound
    def one(v):
        if lo is not None and v < lo:
            return float(lo)
        if hi is not None and v > hi:
            return float(hi)
        return v
    a = asarray(x)
    if isinstance(a, marr):
        return a._map(one)
    return one(float(a))


def maximum(x, y):
    return asarray(x)._zip(y, lambda a, b: a if a >= b else b)


def minimum(x, y):
    return asarray(x)._zip(y, lambda a, b: a if a <= b else b)


def where(cond, a=None, b=None):
    c = asarray(cond)
    if a is None:                       # np.where(mask) -> (indices,)
        if len(c.shape) == 2:
            rows, cols = [], []
            for r in range(c.shape[0]):
                for cc in range(c.shape[1]):
                    if c.data[r][cc]:
                        rows.append(float(r))
                        cols.append(float(cc))
            ri, ci = marr(rows), marr(cols)
            ri._is_index = ci._is_index = True
            return (ri, ci)
        out = marr([float(i) for i, v in enumerate(c._flat())
                    if v != 0])
        out._is_index = True
        return (out,)
    if len(c.shape) == 2:
        # full broadcasting over the condition's shape
        ab = _b2(asarray(a))
        bb2 = _b2(asarray(b))

        def pick(src, r, cc):
            row = src.data[r if src.shape[0] > 1 else 0]
            return row[cc if src.shape[1] > 1 else 0]
        return marr([[pick(ab, r, cc) if c.data[r][cc]
                      else pick(bb2, r, cc)
                      for cc in range(c.shape[1])]
                     for r in range(c.shape[0])])
    aa, bb = asarray(a), asarray(b)
    if isinstance(aa, oarr) or isinstance(bb, oarr) or \
            isinstance(a, str) or isinstance(b, str):
        # object branches (e.g. np.where(p < .05, "*", ""))
        av = None if isinstance(a, str) else (
            list(aa) if isinstance(aa, oarr) else None)
        bv = None if isinstance(b, str) else (
            list(bb) if isinstance(bb, oarr) else None)
        n = c.shape[0]
        return oarr([(av[i] if av is not None else a)
                     if c.data[i] != 0 else
                     (bv[i] if bv is not None else b)
                     for i in range(n)])
    if aa.shape == (1,):
        aa = full(c.shape[0], aa.data[0])
    if bb.shape == (1,):
        bb = full(c.shape[0], bb.data[0])
    return marr([aa.data[i] if c.data[i] != 0 else bb.data[i]
                 for i in range(c.shape[0])])


def isfinite(x):
    if isinstance(x, (int, float)):
        return _math.isfinite(float(x))
    a = asarray(x)
    if not isinstance(a, marr):
        a = marr([float(v) for v in a])
    return a._map(lambda v: 1.0 if _math.isfinite(v) else 0.0)


def dot(a, b):
    aa, bb = asarray(a), asarray(b)
    if len(aa.shape) == 1 and len(bb.shape) == 1:
        if aa.shape != bb.shape:
            raise ValueError("shape mismatch")
        return float(_math.fsum(x * y for x, y in zip(aa.data, bb.data)))
    return matmul(aa, bb)


def matmul(a, b):
    a_arr = asarray(a)
    a_was_1d = len(a_arr.shape) == 1
    aa = atleast_2d(a_arr)
    b_arr = asarray(b)
    b_was_1d = len(b_arr.shape) == 1
    bb = marr([[v] for v in b_arr.data]) if b_was_1d else b_arr
    n, k = aa.shape
    k2, m = bb.shape
    if k != k2:
        raise ValueError("shape mismatch")
    out = [[_math.fsum(aa.data[i][t] * bb.data[t][j] for t in range(k))
            for j in range(m)] for i in range(n)]
    if b_was_1d and a_was_1d:
        return out[0][0]
    if b_was_1d:
        return marr([row[0] for row in out])
    if a_was_1d:
        return marr(out[0])
    return marr(out)


# --------------------------------------------------------------- reductions

def sum(x, axis=None, dtype=None, keepdims=False):  # noqa: A001
    del dtype
    if isinstance(x, ndlist):
        return ndlist(marr(b).sum(axis=axis, keepdims=keepdims)
                      for b in x)
    return asarray(x).sum(axis=axis, keepdims=keepdims)


def mean(x, axis=None, dtype=None, keepdims=False):
    del dtype
    return asarray(x).mean(axis=axis, keepdims=keepdims)


def std(x, axis=None, ddof=0, dtype=None, keepdims=False):
    del dtype, keepdims
    return asarray(x).std(axis=axis, ddof=ddof)


def var(x, axis=None, ddof=0, dtype=None, keepdims=False):
    del dtype, keepdims
    return asarray(x).var(axis=axis, ddof=ddof)


def max(x, axis=None, keepdims=False):  # noqa: A001
    if isinstance(x, ndlist):
        return ndlist(marr(b).max(axis=axis, keepdims=keepdims)
                      for b in x)
    del keepdims
    return asarray(x).max(axis=axis)


def min(x, axis=None, keepdims=False):  # noqa: A001
    del keepdims
    return asarray(x).min(axis=axis)


def _axis_reduce(x, axis, red):
    ax = asarray(x)
    if len(ax.shape) == 1:
        return red(ax._flat())
    a = atleast_2d(ax)
    if axis == 0:
        return marr([red(a.data[i][j] for i in range(a.shape[0]))
                     for j in range(a.shape[1])])
    return marr([red(row) for row in a.data])


def all(x, axis=None):  # noqa: A001
    if axis is None:
        return asarray(x).all()
    return _axis_reduce(x, axis,
                        lambda it: 1.0 if _bi.all(v != 0 for v in it)
                        else 0.0)


def any(x, axis=None):  # noqa: A001
    if axis is None:
        return asarray(x).any()
    return _axis_reduce(x, axis,
                        lambda it: 1.0 if _bi.any(v != 0 for v in it)
                        else 0.0)


def sort(x):
    return marr(sorted(asarray(x)._flat()))


def unique(x, return_inverse=False, return_counts=False,
           return_index=False):
    a = asarray(x)
    if isinstance(a, oarr):
        vals = list(a)
        uniq = sorted(set(vals), key=str)
    else:
        vals = a._flat()
        uniq = sorted(set(vals))
    if not (return_inverse or return_counts or return_index):
        return oarr(uniq) if isinstance(a, oarr) else marr(uniq)
    pos = {v: i for i, v in enumerate(uniq)}
    out = [oarr(uniq) if isinstance(a, oarr) else marr(uniq)]
    if return_index:
        first = {}
        for i, v in enumerate(vals):
            if v not in first:
                first[v] = i
        ix = marr([float(first[v]) for v in uniq])
        ix._is_index = True
        out.append(ix)
    if return_inverse:
        inv = marr([float(pos[v]) for v in vals])
        inv._is_index = True
        out.append(inv)
    if return_counts:
        cnt = {}
        for v in vals:
            cnt[v] = cnt.get(v, 0) + 1
        out.append(marr([float(cnt[v]) for v in uniq]))
    return tuple(out)



def allclose(a, b, atol=1e-8, rtol=1e-5):
    aa, bb = asarray(a)._flat(), asarray(b)._flat()
    if len(aa) != len(bb):
        return False
    return _pyall(_math.isclose(x, y, rel_tol=rtol, abs_tol=atol)
                  for x, y in zip(aa, bb))


_pyall = _bi.all
_pyany = _bi.any
_pysum = _bi.sum
_pymax = _bi.max


# ------------------------------------------------------------------ linalg

class _Linalg:
    @staticmethod
    def solve(a, b):
        aa = atleast_2d(a)
        bvec = asarray(b)
        n = aa.shape[0]
        if aa.shape[1] != n or bvec.shape[0] != n:
            raise ValueError("shape mismatch")
        two_d = len(bvec.shape) == 2
        bb = bvec.tolist() if two_d else [[v] for v in bvec.data]
        m = [row[:] + brow[:] for row, brow in zip(aa.tolist(), bb)]
        ncol = len(m[0])
        for col in range(n):
            piv = _pymax(range(col, n), key=lambda r: _bi.abs(m[r][col]))
            if _bi.abs(m[piv][col]) < 1e-300:
                raise linalg.LinAlgError("singular matrix")
            m[col], m[piv] = m[piv], m[col]
            pv = m[col][col]
            m[col] = [v / pv for v in m[col]]
            for r in range(n):
                if r != col and m[r][col] != 0:
                    f = m[r][col]
                    m[r] = [m[r][j] - f * m[col][j] for j in range(ncol)]
        sol = [row[n:] for row in m]
        if two_d:
            return marr(sol)
        return marr([row[0] for row in sol])

    @staticmethod
    def inv(a):
        aa = atleast_2d(a)
        return _Linalg.solve(aa, eye(aa.shape[0]))

    @staticmethod
    def norm(x, ord=None, axis=None):  # noqa: A002
        a = asarray(x)
        if axis is not None and len(a.shape) == 2:
            rows = a.data if axis in (1, -1) else \
                [[a.data[i][j] for i in range(a.shape[0])]
                 for j in range(a.shape[1])]
            return marr([_Linalg.norm(marr(r), ord=ord)
                         for r in rows])
        f = a._flat()
        if ord in (None, 2, "fro"):
            return _math.sqrt(_math.fsum(v * v for v in f))
        if ord == 1:
            return _math.fsum(_bi.abs(v) for v in f)
        if ord == _math.inf:
            return _bi.max(_bi.abs(v) for v in f)
        if ord == -_math.inf:
            return _bi.min(_bi.abs(v) for v in f)
        return _math.fsum(_bi.abs(v) ** ord for v in f) ** (1.0 / ord)

    @staticmethod
    def qr(a, mode="reduced"):
        """Householder QR; returns (Q, R) with Q (m,k), R (k,n), k=min(m,n)."""
        A = asarray(a)
        m_, n_ = A.shape
        R = [row[:] for row in A.data]
        Q = [[1.0 if i == j else 0.0 for j in range(m_)]
             for i in range(m_)]
        for k in range(_bi.min(m_ - 1, n_)):
            # Householder vector for column k
            x = [R[i][k] for i in range(k, m_)]
            normx = _math.sqrt(_math.fsum(v * v for v in x))
            if normx == 0.0:
                continue
            alpha = -normx if x[0] >= 0 else normx
            v = list(x)
            v[0] -= alpha
            vnorm2 = _math.fsum(u * u for u in v)
            if vnorm2 == 0.0:
                continue
            # R = H R
            for j in range(k, n_):
                dot = _math.fsum(v[i] * R[k + i][j]
                                 for i in range(len(v)))
                c = 2.0 * dot / vnorm2
                for i in range(len(v)):
                    R[k + i][j] -= c * v[i]
            # Q = Q H
            for i in range(m_):
                dot = _math.fsum(Q[i][k + t] * v[t]
                                 for t in range(len(v)))
                c = 2.0 * dot / vnorm2
                for t in range(len(v)):
                    Q[i][k + t] -= c * v[t]
        kk = _bi.min(m_, n_)
        if mode == "complete":
            return marr(Q), marr(R)
        Qr = [[Q[i][j] for j in range(kk)] for i in range(m_)]
        Rr = [[R[i][j] for j in range(n_)] for i in range(kk)]
        return marr(Qr), marr(Rr)


    @staticmethod
    def eigvals(a):
        from ._sci_core import eigvals as _ev
        return _ev(a)

    @staticmethod
    def eig(a):
        """General eigen-decomposition: values via Faddeev-LeVerrier +
        Durand-Kerner, vectors via inverse iteration. Real output when
        the spectrum is real."""
        from ._sci_core import eigvals as _ev
        A = atleast_2d(a)
        n = A.shape[0]
        vals = _ev(A).tolist()
        real_ok = _bi.all(_bi.abs(v.imag) < 1e-9 for v in vals)
        vecs = []
        for lam in vals:
            lam_use = lam.real if real_ok else lam
            M = [[A.data[i][j] - ((lam_use + 1e-10) if i == j
                                  else 0.0)
                  for j in range(n)] for i in range(n)]
            v = [1.0] * n
            for _ in range(60):
                try:
                    v = list(_Linalg.solve(marr(M), marr(v))._flat())
                except Exception:
                    break
                nrm = _math.sqrt(_math.fsum(u * u for u in v)) \
                    or 1.0
                v = [u / nrm for u in v]
            vecs.append(v)
        w = marr([v.real for v in vals]) if real_ok else carr(vals)
        V = marr([[vecs[j][i] for j in range(n)] for i in range(n)])
        return w, V


linalg = _Linalg()


# ------------------------------------------------------------------ random


def _pack_choice(vals):
    """marr for numeric draws, oarr for anything else -- numpy's choice
    keeps the pool's dtype (strings stay strings)."""
    try:
        return marr([float(v) for v in vals])
    except (TypeError, ValueError):
        return oarr(list(vals))


class _SplitMix64:
    """Python fallback RNG (SplitMix64 -> floats).

    ponytail: placeholder until the native Philox hook from morie_core is
    wired in; deterministic, well-distributed, NOT the banned gr* LCG.
    """

    def __init__(self, seed):
        if isinstance(seed, Philox):
            seed = seed.key
        self.state = (seed or 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

    def _next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def _fill(self, one, size):
        """size = None -> scalar; int -> 1-D marr; (n, m) -> 2-D marr."""
        if size is None:
            return one()
        if isinstance(size, (tuple, list)):
            if len(size) == 1:
                return marr([float(one()) for _ in range(int(size[0]))])
            if len(size) == 2:
                n, m = int(size[0]), int(size[1])
                return marr([[float(one()) for _ in range(m)]
                             for _ in range(n)])
            raise ValueError("size ndim > 2 unsupported")
        return marr([float(one()) for _ in range(int(size))])

    def uniform(self, low=0.0, high=1.0, size=None):
        def one():
            return low + (high - low) * (self._next() >> 11) / (1 << 53)
        return self._fill(one, size)

    def standard_normal(self, size=None):
        return self.normal(0.0, 1.0, size)

    def random(self, size=None):
        return self.uniform(0.0, 1.0, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        # numpy broadcasts an array-valued loc/scale against size:
        # loc=[0, 6] with size=(40, 2) shifts column j by loc[j].
        if hasattr(loc, "_flat") or isinstance(loc, (list, tuple)) or \
                hasattr(scale, "_flat") or isinstance(scale, (list, tuple)):
            def _vals(v):
                if hasattr(v, "_flat"):
                    return [float(x) for x in v._flat()]
                if isinstance(v, (list, tuple)):
                    return [float(x) for x in v]
                return [float(v)]
            lv, sv = _vals(loc), _vals(scale)
            z = self.normal(0.0, 1.0, size)
            if len(getattr(z, "shape", (0,))) == 2:
                nr, nc = z.shape
                return marr([[lv[j % len(lv)] + sv[j % len(sv)]
                              * z.data[i][j] for j in range(nc)]
                             for i in range(nr)])
            zf = list(z._flat()) if hasattr(z, "_flat") else [float(z)]
            out = [lv[i % len(lv)] + sv[i % len(sv)] * zf[i]
                   for i in range(len(zf))]
            return marr(out) if size is not None else out[0]

        def one():
            u1 = _pymax(self.uniform(), 1e-300)
            u2 = self.uniform()
            return loc + scale * _math.sqrt(-2 * _math.log(u1)) \
                * _math.cos(2 * _math.pi * u2)
        return self._fill(one, size)

    def _u(self):
        return (self._next() >> 11) / (1 << 53)

    def poisson(self, lam=1.0, size=None):
        # numpy draws one variate per element of an array-valued lam
        # (broadcast against size; here: size defaults to lam.shape).
        if hasattr(lam, "_flat") or isinstance(lam, (list, tuple)):
            lams = list(lam._flat()) if hasattr(lam, "_flat") else \
                [float(v) for v in lam]
            return marr([float(self.poisson(L)) for L in lams])

        def one():
            L = float(lam)
            if L < 30.0:                      # Knuth product method
                target = _math.exp(-L)
                k, p = 0, 1.0
                while True:
                    p *= _pymax(self._u(), 1e-300)
                    if p <= target:
                        return float(k)
                    k += 1
            # normal approximation + correction for large lambda
            while True:
                v = L + _math.sqrt(L) * self.normal()
                if v >= 0:
                    return float(int(v + 0.5))
        return self._fill(one, size)

    def exponential(self, scale=1.0, size=None):
        def one():
            return -float(scale) * _math.log(_pymax(self._u(), 1e-300))
        return self._fill(one, size)

    def standard_gamma(self, shape, size=None):
        return self.gamma(shape, 1.0, size)

    def gamma(self, shape, scale=1.0, size=None):
        a = float(shape)

        def one():
            # Marsaglia-Tsang (2000); boost for a < 1
            aa = a if a >= 1.0 else a + 1.0
            d = aa - 1.0 / 3.0
            c = 1.0 / _math.sqrt(9.0 * d)
            while True:
                x = self.normal()
                v = (1.0 + c * x) ** 3
                if v <= 0:
                    continue
                u = _pymax(self._u(), 1e-300)
                if _math.log(u) < 0.5 * x * x + d - d * v \
                        + d * _math.log(v):
                    g = d * v
                    if a < 1.0:
                        g *= _pymax(self._u(), 1e-300) ** (1.0 / a)
                    return g * float(scale)
        return self._fill(one, size)

    def beta(self, a, b, size=None):
        def one():
            x = self.gamma(a)
            y = self.gamma(b)
            return x / (x + y)
        return self._fill(one, size)

    def binomial(self, n, p, size=None):
        # numpy broadcasts an array-valued p (or n) against size:
        # one draw per element when size is omitted.
        if hasattr(p, "_flat") or isinstance(p, (list, tuple)) or \
                hasattr(n, "_flat") or isinstance(n, (list, tuple)):
            def _vals(v):
                if hasattr(v, "_flat"):
                    return [float(x) for x in v._flat()]
                if isinstance(v, (list, tuple)):
                    return [float(x) for x in v]
                return None
            pv, nv = _vals(p), _vals(n)
            m = len(pv) if pv is not None else len(nv)
            if size is not None and int(size) != m:
                raise ValueError("size does not match the length of p/n")
            return marr([float(self.binomial(
                int(nv[i % len(nv)]) if nv is not None else int(n),
                pv[i % len(pv)] if pv is not None else float(p)))
                for i in range(m)])

        def one():
            nn, pp = int(n), float(p)
            if nn * _bi.min(pp, 1.0 - pp) < 30.0:
                return float(_bi.sum(1 for _ in range(nn)
                                     if self._u() < pp))
            while True:                       # normal approx, clipped
                v = nn * pp + _math.sqrt(nn * pp * (1 - pp)) \
                    * self.normal()
                k = int(v + 0.5)
                if 0 <= k <= nn:
                    return float(k)
        return self._fill(one, size)

    def chisquare(self, df, size=None):
        return self.gamma(float(df) / 2.0, 2.0, size)

    def geometric(self, p, size=None):
        def one():
            return float(int(_math.log(_pymax(self._u(), 1e-300))
                             / _math.log(1.0 - float(p))) + 1)
        return self._fill(one, size)

    def shuffle(self, seq):
        # Fisher-Yates in place on a plain list
        if isinstance(seq, marr):
            data = seq.data
        else:
            data = seq
        for i in range(len(data) - 1, 0, -1):
            j = self._next() % (i + 1)
            data[i], data[j] = data[j], data[i]

    def permutation(self, n):
        if isinstance(n, int):
            out = list(range(n))
        else:
            out = list(asarray(n)._flat())
        self.shuffle(out)
        return marr([float(v) for v in out])

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        def one():
            return _math.exp(self.normal(float(mean), float(sigma)))
        return self._fill(one, size)

    def _gamma_variate(self, shape):
        # Marsaglia & Tsang (2000) squeeze method; shape < 1 boosted
        # via Gamma(a+1) * U^(1/a)
        a = float(shape)
        if a < 1.0:
            u = self.random()
            while u <= 0.0:
                u = self.random()
            return self._gamma_variate(a + 1.0) * u ** (1.0 / a)
        d = a - 1.0 / 3.0
        c = 1.0 / _math.sqrt(9.0 * d)
        while True:
            x = self.normal()
            v = (1.0 + c * x) ** 3
            if v <= 0.0:
                continue
            u = self.random()
            if u < 1.0 - 0.0331 * x ** 4:
                return d * v
            if u > 0.0 and _math.log(u) < 0.5 * x * x + d * (
                    1.0 - v + _math.log(v)):
                return d * v

    def gamma(self, shape, scale=1.0, size=None):
        def one():
            return self._gamma_variate(shape) * float(scale)
        return self._fill(one, size)

    def dirichlet(self, alpha, size=None):
        al = [float(v) for v in (alpha._flat()
                                 if isinstance(alpha, marr) else alpha)]

        def draw():
            g = [self._gamma_variate(a) for a in al]
            t = _pysum(g)
            return [v / t for v in g]
        if size is None:
            return marr(draw())
        return marr([draw() for _ in range(int(size))])

    def beta(self, a, b, size=None):
        def one():
            x = self._gamma_variate(a)
            y = self._gamma_variate(b)
            return x / (x + y)
        return self._fill(one, size)

    def choice(self, a, size=None, replace=True, p=None):
        if p is not None:
            pool = list(range(int(a))) if isinstance(a, int) \
                else list(asarray(a)._flat())
            w = [float(v) for v in (p._flat() if isinstance(p, marr)
                                    else list(p))]
            if len(w) != len(pool):
                raise ValueError("p must have the same size as a")
            tot = _pysum(w)
            if tot <= 0 or _pyany(v < 0 for v in w):
                raise ValueError("probabilities must be non-negative "
                                 "and sum to a positive value")
            if size is None:
                k, flat_scalar = 1, True
            else:
                if isinstance(size, (tuple, list)):
                    k = 1
                    for d in size:
                        k *= int(d)
                else:
                    k = int(size)
                flat_scalar = False
            out = []
            wl, pl = list(w), list(pool)
            for _ in range(k):
                t = _pysum(wl)
                u = self.random() * t
                c = 0.0
                pick = len(wl) - 1
                for i2, wv in enumerate(wl):
                    c += wv
                    if u <= c:
                        pick = i2
                        break
                out.append(pl[pick])
                if not replace:
                    del pl[pick], wl[pick]
            if flat_scalar:
                return out[0]
            return _pack_choice(out)
        if isinstance(a, int):
            pool = list(range(int(a)))
        elif isinstance(a, (list, tuple)):
            pool = list(a)          # keep the element type; numpy
        else:                       # returns int64 for an int pool
            pool = list(asarray(a)._flat())
        if size is None:
            return pool[self._next() % len(pool)]
        if isinstance(size, (tuple, list)):
            size = 1 if not size else int(
                size[0]) * (int(size[1]) if len(size) > 1 else 1)
        k = int(size)
        if replace:
            return _pack_choice([pool[self._next() % len(pool)]
                                 for _ in range(k)])
        if k > len(pool):
            raise ValueError("cannot sample more than population without "
                             "replacement")
        idx = list(range(len(pool)))
        self.shuffle(idx)
        return _pack_choice([pool[i] for i in idx[:k]])

    def integers(self, low, high=None, size=None):
        if high is None:
            low, high = 0, low

        def one():
            return low + self._next() % (high - low)
        return self._fill(one, size)


class _Random:
    @staticmethod
    def default_rng(seed=None):
        return _SplitMix64(seed if seed is not None else 0)


random = _Random()


# ------------------------------------------- extended primitives (sweep 1)

def isin(x, values):
    vs = set(asarray(values)._flat())
    return asarray(x)._map(lambda v: 1.0 if v in vs else 0.0)


def isclose(a, b, rtol=1e-5, atol=1e-8):
    return asarray(a)._zip(
        b, lambda x, y: 1.0 if _math.isclose(x, y, rel_tol=rtol,
                                             abs_tol=atol) else 0.0)


def diff(x, n=1, axis=-1):
    a = asarray(x)
    if len(a.shape) == 2:
        rows = a.data if axis in (1, -1) else \
            [[a.data[i][j] for i in range(a.shape[0])]
             for j in range(a.shape[1])]
        out = [list(diff(marr(r), n=n)._flat()) for r in rows]
        if axis in (1, -1):
            return marr(out)
        return marr([[out[j][i] for j in range(len(out))]
                     for i in range(len(out[0]))])
    f = list(a._flat())
    for _ in range(int(n)):
        f = [f[i + 1] - f[i] for i in range(len(f) - 1)]
    return marr(f)


def trace(a):
    aa = atleast_2d(a)
    return float(_math.fsum(aa.data[i][i]
                            for i in range(_bi.min(aa.shape))))


def logaddexp(a, b):
    def f(x, y):
        hi, lo = (x, y) if x >= y else (y, x)
        return hi + _math.log1p(_math.exp(lo - hi))
    return asarray(a)._zip(b, f) if isinstance(a, (marr, list, tuple)) \
        or isinstance(b, (marr, list, tuple)) else f(float(a), float(b))


def tile(x, reps):
    a = asarray(x)
    if isinstance(reps, (tuple, list)):
        reps = [int(r) for r in reps]
        if len(reps) == 1:
            reps = reps[0]
        elif len(reps) == 2:
            m, k = reps
            if len(a.shape) == 1:
                row = list(a._flat()) * k
                return marr([list(row) for _ in range(m)])
            if len(a.shape) == 2:
                rows = [list(r) * k for r in a.data]
                return marr([list(r) for _ in range(m) for r in rows])
            raise ValueError("tile: unsupported input rank")
        else:
            raise ValueError("tile: reps rank > 2 unsupported")
    if len(a.shape) == 2:
        return marr(a.data * int(reps))
    return marr(a._flat() * int(reps))


def take_along_axis(a, idx, axis=-1):
    aa = atleast_2d(asarray(a))
    ia = atleast_2d(asarray(idx))
    if axis not in (-1, 1):
        raise ValueError("take_along_axis: axis -1 only (rank-2 core)")
    out = marr([[aa.data[r][int(ia.data[r][c])]
                 for c in range(ia.shape[1])]
                for r in range(ia.shape[0])])
    return out if len(asarray(a).shape) == 2 else marr(out.data[0])


def put_along_axis(a, idx, values, axis=-1):
    if axis not in (-1, 1):
        raise ValueError("put_along_axis: axis -1 only (rank-2 core)")
    aa = a if isinstance(a, marr) else asarray(a)
    ia = atleast_2d(asarray(idx))
    va = atleast_2d(asarray(values))
    for r in range(ia.shape[0]):
        for c in range(ia.shape[1]):
            v = va.data[r][c if va.shape[1] > 1 else 0]
            if len(aa.shape) == 2:
                aa.data[r][int(ia.data[r][c])] = float(v)
            else:
                aa.data[int(ia.data[r][c])] = float(v)


def broadcast_to(x, shape):
    a = asarray(x)
    shape = tuple(int(v) for v in shape)
    if len(shape) == 2 and len(a.shape) == 1:
        return marr([a.data[:] for _ in range(shape[0])])
    if len(shape) == 3 and len(a.shape) == 2:
        # rank-3 broadcast surfaces as a nested list (rank-2 core);
        # einsum and the module loops consume nested lists directly
        return ndlist([[row[:] for row in a.data]
                       for _ in range(shape[0])])
    if shape == a.shape:
        return marr(a)
    raise ValueError("broadcast_to: unsupported %r -> %r"
                     % (a.shape, shape))


def expand_dims(x, axis):
    a = asarray(x)
    if len(a.shape) == 1:
        if axis in (0, -2):
            return marr([a.data[:]])
        return marr([[v] for v in a.data])
    raise ValueError("expand_dims: rank-2 core")


def squeeze(x, axis=None):
    del axis
    a = asarray(x)
    if len(a.shape) == 2:
        if a.shape[0] == 1:
            return marr(a.data[0][:])
        if a.shape[1] == 1:
            return marr([row[0] for row in a.data])
    return a


def repeat(x, reps, axis=None):
    if isinstance(reps, (list, tuple, marr)) or (
            hasattr(reps, "tolist") and not isinstance(reps, (int, float))):
        rl = [int(v) for v in
              (reps._flat() if isinstance(reps, marr) else
               (reps.tolist() if hasattr(reps, "tolist") else reps))]
        a2 = asarray(x)
        f2 = a2._flat() if isinstance(a2, marr) else list(a2)
        out2 = marr([v for v, r2 in zip(f2, rl) for _ in range(r2)])
        if isinstance(a2, marr) and getattr(a2, "_is_index", False):
            out2._is_index = True
        return out2
    if isinstance(x, list) and x and isinstance(x[0], (list, marr)) \
            and _nested_shape(x) and len(_nested_shape(x)) == 3:
        # rank-3 nested-list block: repeat whole blocks along axis 0
        if axis == 0:
            return ndlist(b.tolist() if isinstance(b, marr) else
                          [r[:] for r in b] for b in x
                          for _ in range(int(reps)))
        raise ValueError("repeat: rank-3 supports axis=0 only")
    a = asarray(x)
    if axis == 0 and len(a.shape) == 2:
        return marr([row[:] for row in a.data
                     for _ in range(int(reps))])
    if axis in (1, -1) and len(a.shape) == 2:
        return marr([[v for v in row for _ in range(int(reps))]
                     for row in a.data])
    return marr([v for v in a._flat() for _ in range(int(reps))])


def _lu_slogdet(m):
    n = len(m)
    m = [row[:] for row in m]
    sign = 1.0
    logdet = 0.0
    for col in range(n):
        piv = _pymax(range(col, n), key=lambda r: _bi.abs(m[r][col]))
        if _bi.abs(m[piv][col]) < 1e-300:
            return 0.0, -inf
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
            sign = -sign
        pv = m[col][col]
        if pv < 0:
            sign = -sign
        logdet += _math.log(_bi.abs(pv))
        for r in range(col + 1, n):
            f = m[r][col] / pv
            for j in range(col, n):
                m[r][j] -= f * m[col][j]
    return sign, logdet


class _LinalgExt:
    @staticmethod
    def slogdet(a):
        return _lu_slogdet(atleast_2d(a).tolist())

    @staticmethod
    def eigvalsh(a):
        # cyclic Jacobi for symmetric matrices; fine for the small
        # covariance matrices morie.fn passes here
        m = [row[:] for row in atleast_2d(a).tolist()]
        n = len(m)
        for _sweep in range(100):
            off = _math.sqrt(_math.fsum(m[i][j] ** 2 for i in range(n)
                                        for j in range(n) if i != j))
            if off < 1e-14:
                break
            for p in range(n - 1):
                for q in range(p + 1, n):
                    if _bi.abs(m[p][q]) < 1e-300:
                        continue
                    theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                    t = (1.0 if theta >= 0 else -1.0) / (
                        _bi.abs(theta) + _math.sqrt(theta * theta + 1.0))
                    c = 1.0 / _math.sqrt(t * t + 1.0)
                    s = t * c
                    for k in range(n):
                        mkp, mkq = m[k][p], m[k][q]
                        m[k][p] = c * mkp - s * mkq
                        m[k][q] = s * mkp + c * mkq
                    for k in range(n):
                        mpk, mqk = m[p][k], m[q][k]
                        m[p][k] = c * mpk - s * mqk
                        m[q][k] = s * mpk + c * mqk
        return marr(sorted(m[i][i] for i in range(n)))

    @staticmethod
    def cond(a):
        # Frobenius-norm condition estimate; morie.fn uses this only as a
        # near-singularity guard
        aa = atleast_2d(a)
        try:
            ai = _Linalg.inv(aa)
        except ValueError:
            return inf
        fro = lambda m: _math.sqrt(_math.fsum(v * v for v in m._flat()))  # noqa: E731
        return fro(aa) * fro(ai)


def _matrix_rank(a, tol=None):
    """Rank via row-reduction with partial pivoting (float tolerance)."""
    m = [row[:] for row in atleast_2d(a).tolist()]
    nrow = len(m)
    ncol = len(m[0])
    if tol is None:
        scale = _pymax((_pymax(_bi.abs(v) for v in row) for row in m),
                       default=0.0)
        tol = 1e-12 * _pymax(scale, 1.0)
    rank = 0
    row = 0
    for col in range(ncol):
        piv = None
        best = tol
        for r in range(row, nrow):
            if _bi.abs(m[r][col]) > best:
                best = _bi.abs(m[r][col])
                piv = r
        if piv is None:
            continue
        m[row], m[piv] = m[piv], m[row]
        pv = m[row][col]
        for r in range(nrow):
            if r != row and _bi.abs(m[r][col]) > 0:
                fct = m[r][col] / pv
                m[r] = [m[r][j] - fct * m[row][j] for j in range(ncol)]
        rank += 1
        row += 1
        if row == nrow:
            break
    return rank


def _jacobi_eigh(a):
    """Symmetric eigendecomposition (values, vectors) via cyclic Jacobi."""
    m = [row[:] for row in atleast_2d(a).tolist()]
    n = len(m)
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(100):
        off = _math.sqrt(_math.fsum(m[i][j] ** 2 for i in range(n)
                                    for j in range(n) if i != j))
        if off < 1e-14:
            break
        for p_ in range(n - 1):
            for q_ in range(p_ + 1, n):
                if _bi.abs(m[p_][q_]) < 1e-300:
                    continue
                theta = (m[q_][q_] - m[p_][p_]) / (2.0 * m[p_][q_])
                t = (1.0 if theta >= 0 else -1.0) / (
                    _bi.abs(theta) + _math.sqrt(theta * theta + 1.0))
                c = 1.0 / _math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp, mkq = m[k][p_], m[k][q_]
                    m[k][p_] = c * mkp - s * mkq
                    m[k][q_] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p_][k], m[q_][k]
                    m[p_][k] = c * mpk - s * mqk
                    m[q_][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p_], v[k][q_]
                    v[k][p_] = c * vkp - s * vkq
                    v[k][q_] = s * vkp + c * vkq
    vals = [m[i][i] for i in range(n)]
    return vals, v


def _pinv_extended(a, rcond=1e-15):
    """Moore-Penrose pseudo-inverse together with the singular values
    used to build it.

    ``A+ = V diag(s+) U^T`` where ``s+_i = 1/s_i`` for
    ``s_i > rcond * max(s)`` and ``0`` otherwise.  This is
    numpy.linalg.pinv (whose default ``rcond`` is also 1e-15) with the
    singular values returned alongside, i.e. the contract of
    ``statsmodels.tools.tools.pinv_extended`` (0.14.6), which the
    OLS/GLM fit path needs in order to report the rank.

    Going through the SVD keeps the pseudo-inverse accurate for
    rank-deficient and ill-conditioned input; forming ``A^T A`` first
    would square the condition number (Golub & Van Loan 2013, *Matrix
    Computations* 4th ed., sec. 5.5.2).
    """
    aa = atleast_2d(a)
    m_, n_ = aa.shape
    u, sv, vt = _svd(aa)
    svals = list(sv._flat())
    cutoff = rcond * (_bi.max(svals) if svals else 0.0)
    inv_s = [1.0 / v if v > cutoff else 0.0 for v in svals]
    k = len(svals)
    # V diag(s+) U^T  ->  (n, m)
    out = [[_math.fsum(vt.data[c][i] * inv_s[c] * u.data[j][c]
                       for c in range(k))
            for j in range(m_)] for i in range(n_)]
    return marr(out), svals


def _pinv(a, rcond=1e-15):
    """Moore-Penrose pseudo-inverse via the SVD.

    See :func:`_pinv_extended`; this drops the singular values.
    """
    return _pinv_extended(a, rcond)[0]


_LinalgExt.pinv = staticmethod(_pinv)
_LinalgExt.matrix_rank = staticmethod(_matrix_rank)
linalg.matrix_rank = _matrix_rank
linalg.pinv = _pinv
linalg.pinv_extended = _pinv_extended
linalg.slogdet = _LinalgExt.slogdet
linalg.eigvalsh = _LinalgExt.eigvalsh
linalg.cond = _LinalgExt.cond


def prod(x):
    out = 1.0
    for v in asarray(x)._flat():
        out *= v
    return float(out)


def outer(a, b):
    fa, fb = asarray(a)._flat(), asarray(b)._flat()
    return marr([[x * y for y in fb] for x in fa])


def interp(x, xp, fp, left=None, right=None):
    xs, ys = asarray(xp)._flat(), asarray(fp)._flat()
    lo = ys[0] if left is None else float(left)
    hi = ys[-1] if right is None else float(right)

    def one(v):
        if v < xs[0]:
            return lo
        if v > xs[-1]:
            return hi
        if v == xs[0]:
            return ys[0]
        if v == xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= v <= xs[i + 1]:
                w = (v - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + w * (ys[i + 1] - ys[i])
        return ys[-1]
    a = asarray(x)
    if not isinstance(x, (list, tuple, marr)):
        return one(float(x))
    return a._map(one)


def trapezoid(y, x=None, dx=1.0, axis=None):
    ya = asarray(y)
    if len(ya.shape) == 2:
        if axis == 0:
            ya = ya.T
        return marr([trapezoid(row, x=x, dx=dx) for row in ya.data])
    fy = ya._flat()
    if x is None:
        fx = [i * dx for i in range(len(fy))]
    else:
        fx = asarray(x)._flat()
    return float(_math.fsum((fx[i + 1] - fx[i]) * (fy[i + 1] + fy[i]) / 2.0
                            for i in range(len(fy) - 1)))


def sliding_window_view(x, window):
    """1-D sliding windows as a 2-D array (numpy.lib.stride_tricks subset)."""
    f = asarray(x)._flat()
    w = int(window)
    if w < 1 or w > len(f):
        raise ValueError("invalid window length")
    return marr([f[i:i + w] for i in range(len(f) - w + 1)])


class _DTypeNarrow:
    """float32/float16 marker: equal to float for comparisons but
    carrying its own name so binary readers (frombuffer) pick the
    right struct format."""

    def __init__(self, name):
        self.__name__ = name
        self.name = name
        self.kind = "f"

    def __call__(self, v):
        return float(v)

    def __eq__(self, other):
        return other is float or other == self.name \
            or getattr(other, "__name__", "") in (self.name, "float")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "dtype(%r)" % self.name


float32 = _DTypeNarrow("float32")
int64 = int
int32 = int


def zeros_like(x, dtype=None):
    a = asarray(x)
    if len(a.shape) == 2:
        return marr([[0.0] * a.shape[1] for _ in range(a.shape[0])])
    return marr([0.0] * a.shape[0])


def ones_like(x, dtype=None):
    a = asarray(x)
    if len(a.shape) == 2:
        return marr([[1.0] * a.shape[1] for _ in range(a.shape[0])])
    return marr([1.0] * a.shape[0])


tanh = _uf(_math.tanh)
sinh = _uf(_math.sinh)
cosh = _uf(_math.cosh)
sin = _uf(_math.sin)
cos = _uf(_math.cos)
tan = _uf(_math.tan)
arctan = _uf(_math.atan)
arcsin = _uf(_math.asin)
arccos = _uf(_math.acos)
sign = _uf(lambda v: 0.0 if v == 0 else (1.0 if v > 0 else -1.0))
floor = _uf(_math.floor)
ceil = _uf(_math.ceil)
round = _uf(lambda v: float(_bi.round(v)))  # noqa: A001
log2 = _uf(_math.log2)
log10 = _uf(_math.log10)
expm1 = _uf(_math.expm1)
isnan = _uf(lambda v: 1.0 if v != v else 0.0)


def vectorize(fn):
    def wrapped(x, *args, **kw):
        if isinstance(x, (list, tuple, marr)):
            return asarray(x)._map(lambda v: float(fn(v, *args, **kw)))
        return fn(x, *args, **kw)
    return wrapped


def cumsum(x):
    out = []
    total = 0.0
    for v in asarray(x)._flat():
        total += v
        out.append(total)
    return marr(out)


def argmax(x, axis=None):
    a = asarray(x)
    if axis is None or len(a.shape) == 1:
        f = a._flat()
        return f.index(_bi.max(f))
    if axis == 0:
        return marr([float(_bi.max(range(a.shape[0]),
                                   key=lambda i: a.data[i][j]))
                     for j in range(a.shape[1])])
    return marr([float(_bi.max(range(a.shape[1]),
                               key=lambda j: row[j]))
                 for row in a.data])


def argmin(x, axis=None):
    a = asarray(x)
    if axis is None or len(a.shape) == 1:
        f = a._flat()
        return f.index(_bi.min(f))
    if axis == 0:
        return marr([float(_bi.min(range(a.shape[0]),
                                   key=lambda i: a.data[i][j]))
                     for j in range(a.shape[1])])
    return marr([float(_bi.min(range(a.shape[1]),
                               key=lambda j: row[j]))
                 for row in a.data])


def argsort(x, axis=None, kind=None):
    del kind
    a = asarray(x)
    if axis in (-1, 1) and len(a.shape) == 2:
        out = marr([[float(i) for i in
                     sorted(range(len(row)), key=lambda k: row[k])]
                    for row in a.data])
        out._is_index = True
        return out
    f = a._flat()
    return marr([float(i) for i in
                 sorted(range(len(f)), key=lambda k: f[k])])


float16 = _DTypeNarrow("float16")


class _FInfo:
    """numpy.finfo subset for float64 (all our dtypes alias float)."""

    def __init__(self, dtype=None):
        del dtype
        import sys as _sys
        fi = _sys.float_info
        self.eps = fi.epsilon
        self.max = fi.max
        self.min = -fi.max
        self.tiny = fi.min
        self.smallest_normal = fi.min
        self.resolution = 1e-15
        self.bits = 64


def finfo(dtype=None):
    return _FInfo(dtype)


int16 = _DTypeNarrow("int16")
int8 = _DTypeNarrow("int8")
uint8 = _DTypeNarrow("uint8")

bool_ = bool


def median(x):
    f = sorted(asarray(x)._flat())
    n = len(f)
    if n == 0:
        raise ValueError("median of empty array")
    mid = n // 2
    if n % 2:
        return f[mid]
    return 0.5 * (f[mid - 1] + f[mid])


def percentile(x, q, axis=None):
    """Linear-interpolation percentile (numpy default method)."""
    a = asarray(x)
    if axis is not None and len(a.shape) == 2:
        if axis in (0, -2):
            cols = [[a.data[r][c] for r in range(a.shape[0])]
                    for c in range(a.shape[1])]
            per_col = [percentile(col, q) for col in cols]
            if isinstance(q, (list, tuple, marr)):
                return marr([[float(pc[i2]) for pc in per_col]
                             for i2 in range(len(list(q)))])
            return marr([float(v) for v in per_col])
        rows_p = [percentile(row, q) for row in a.data]
        if isinstance(q, (list, tuple, marr)):
            return marr([[float(rp[i2]) for rp in rows_p]
                         for i2 in range(len(list(q)))])
        return marr([float(v) for v in rows_p])
    f = sorted(a._flat())
    n = len(f)

    def one(qq):
        if not 0 <= qq <= 100:
            raise ValueError("q in [0, 100]")
        pos = qq / 100.0 * (n - 1)
        lo = int(_math.floor(pos))
        hi = int(_math.ceil(pos))
        if lo == hi:
            return f[lo]
        return f[lo] + (pos - lo) * (f[hi] - f[lo])
    if isinstance(q, (list, tuple)):
        return marr([one(float(v)) for v in q])
    return one(float(q))


def quantile(x, q):
    if isinstance(q, (list, tuple, marr)):
        qf = q._flat() if isinstance(q, marr) else q
        return percentile(x, [100.0 * float(v) for v in qf])
    return percentile(x, 100.0 * float(q))


def empty(n, dtype=None):
    return zeros(n)


def subtract(a, b):
    if not isinstance(a, (list, tuple, marr)) \
            and not isinstance(b, (list, tuple, marr)):
        return float(a) - float(b)
    return asarray(a)._zip(b, lambda x, y: x - y)


def add(a, b):
    if not isinstance(a, (list, tuple, marr)) \
            and not isinstance(b, (list, tuple, marr)):
        return float(a) + float(b)
    return asarray(a)._zip(b, lambda x, y: x + y)


def multiply(a, b):
    if not isinstance(a, (list, tuple, marr)) \
            and not isinstance(b, (list, tuple, marr)):
        return float(a) * float(b)
    return asarray(a)._zip(b, lambda x, y: x * y)


def divide(a, b):
    if not isinstance(a, (list, tuple, marr)) \
            and not isinstance(b, (list, tuple, marr)):
        return float(a) / float(b)
    return asarray(a)._zip(b, lambda x, y: x / y)


def fill_diagonal(a, val):
    m = atleast_2d(a)
    for i in range(_bi.min(m.shape)):
        m.data[i][i] = float(val)
    # in-place on the caller's 2-D marr (same list objects)


def size(x):
    a = asarray(x)
    n = 1
    for d in a.shape:
        n *= d
    return n


def count_nonzero(x):
    return int(_bi.sum(1 for v in asarray(x)._flat() if v != 0))


def shape(x):
    if isinstance(x, marr):
        return x.shape
    return asarray(x).shape


def ndim(x):
    return asarray(x).ndim


class errstate:
    """No-op numpy.errstate stand-in (Python floats already raise/inf)."""

    def __init__(self, **kw):
        del kw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _LinAlgError(ValueError):
    pass


linalg.LinAlgError = _LinAlgError


# ------------------------------------------- batch 2: gap-scan closure

def corrcoef(x, y=None, rowvar=True):
    if y is None:
        a = atleast_2d(x)
        if not rowvar:
            a = a.T
        rows = [marr(r) for r in a.data]
    else:
        rows = [asarray(x), asarray(y)]
    n = len(rows)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            xi, xj = rows[i], rows[j]
            xd = xi - xi.mean()
            yd = xj - xj.mean()
            den = _math.sqrt(dot(xd, xd) * dot(yd, yd))
            out[i][j] = dot(xd, yd) / den if den else nan
    return marr(out)


def cov(x, y=None, ddof=1):
    if y is None:
        a = atleast_2d(x)
        rows = [marr(r) for r in a.data]
    else:
        rows = [asarray(x), asarray(y)]
    n = len(rows)
    m = rows[0].shape[0]
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            xd = rows[i] - rows[i].mean()
            yd = rows[j] - rows[j].mean()
            out[i][j] = dot(xd, yd) / (m - ddof)
    return marr(out)




class ndlist(list):
    """Thin rank>=3 container: nested lists with .shape/.tolist and
    elementwise scalar arithmetic. The rank-2 core stays marr; this
    only carries higher-rank results between module-level loops."""

    @property
    def shape(self):
        return _nested_shape(self)

    @property
    def ndim(self):
        return len(self.shape)

    def tolist(self):
        def conv(v):
            if isinstance(v, marr):
                return v.tolist()
            if isinstance(v, list):
                return [conv(x) for x in v]
            return v
        return [conv(v) for v in self]

    def _ew(self, other, fn):
        if isinstance(other, ndlist):
            # blockwise: zip leading axis, marr broadcasting handles
            # the rest ((q,1) vs (q,k) etc.)
            return ndlist(marr(a)._zip(marr(b), fn)
                          for a, b in zip(self, other))
        if isinstance(other, (marr, list)) and not isinstance(
                other, ndlist) and isinstance(other, marr):
            return ndlist(marr(v)._zip(other, fn) for v in self)

        def walk(v):
            if isinstance(v, marr):
                return v._map(lambda x: fn(x, float(other)))
            if isinstance(v, list):
                return [walk(x) for x in v]
            return fn(float(v), float(other))
        return ndlist(walk(v) for v in self)

    def __truediv__(self, o):
        return self._ew(o, lambda a, b: a / b)

    def __mul__(self, o):
        return self._ew(o, lambda a, b: a * b)
    __rmul__ = __mul__

    def __add__(self, o):
        return self._ew(o, lambda a, b: a + b)

    def __sub__(self, o):
        return self._ew(o, lambda a, b: a - b)


def _nested_shape(x):
    sh = []
    v = x
    while isinstance(v, (list, tuple)) or isinstance(v, marr):
        if isinstance(v, marr):
            return tuple(sh) + v.shape
        sh.append(len(v))
        if not len(v):
            break
        v = v[0]
    return tuple(sh)


def _nested_get(x, idx):
    v = x.tolist() if isinstance(x, marr) else x
    for i in idx:
        v = v[i]
    return float(v)


def einsum(spec, *ops):
    """General Einstein summation over nested-list / marr operands of
    any rank (explicit and implicit forms; ellipsis expanded against
    the operand rank). Index extents are validated across operands.
    Rank <= 2 results return marr; higher ranks return nested lists
    (the native core is rank-2)."""
    import itertools as _it
    spec = spec.replace(" ", "")
    if "->" in spec:
        lhs, out_labels = spec.split("->")
    else:
        lhs, out_labels = spec, None
    in_specs = lhs.split(",")
    if len(in_specs) != len(ops):
        raise ValueError("einsum: operand count mismatch")
    shapes = [_nested_shape(op) for op in ops]

    # expand ellipsis against actual ranks
    if "..." in spec:
        free = [c for c in "zyxwvu" if c not in spec]
        ell_rank = 0
        for sp, sh in zip(in_specs, shapes):
            if "..." in sp:
                ell_rank = _bi.max(ell_rank,
                                   len(sh) - (len(sp) - 3))
        ell = "".join(free[:ell_rank])
        in_specs = [sp.replace("...", ell) for sp in in_specs]
        if out_labels is not None:
            out_labels = out_labels.replace("...", ell)

    dims = {}
    for sp, sh in zip(in_specs, shapes):
        if len(sp) != len(sh):
            raise ValueError("einsum: spec %r vs shape %r" % (sp, sh))
        for c, d in zip(sp, sh):
            if c in dims and dims[c] != d:
                raise ValueError("einsum: size mismatch for %r" % c)
            dims[c] = d
    if out_labels is None:
        counts = {}
        for sp in in_specs:
            for c in sp:
                counts[c] = counts.get(c, 0) + 1
        out_labels = "".join(sorted(c for c, n in counts.items()
                                    if n == 1))
    sum_labels = sorted(set("".join(in_specs)) - set(out_labels))

    raw = [op.tolist() if isinstance(op, marr) else op for op in ops]

    def cell(out_idx):
        env = dict(zip(out_labels, out_idx))
        total = 0.0
        for combo in _it.product(*(range(dims[c]) for c in sum_labels)):
            env.update(zip(sum_labels, combo))
            prod = 1.0
            for sp, op in zip(in_specs, raw):
                prod *= _nested_get(op, [env[c] for c in sp])
            total += prod
        return total

    if not out_labels:
        return cell(())

    def build(labels, prefix):
        if not labels:
            return cell(tuple(prefix))
        return [build(labels[1:], prefix + [i])
                for i in range(dims[labels[0]])]

    out = build(list(out_labels), [])
    if len(out_labels) <= 2:
        return marr(out)
    return ndlist(out)



def block(rows):
    """numpy.block for the 2-D nested-list case: each inner list is a
    row of blocks joined left-to-right, rows stacked top-to-bottom."""
    out = []
    for row in rows:
        mats = [atleast_2d(asarray(b)) for b in row]
        h = mats[0].shape[0]
        for m in mats:
            if m.shape[0] != h:
                raise ValueError("block row heights differ")
        for i in range(h):
            out.append([v for m in mats for v in m.data[i]])
    return marr(out)


def vstack(parts):
    rows = []
    for p in parts:
        a = asarray(p)
        if len(a.shape) == 1:
            rows.append(a.data[:])
        else:
            rows.extend(r[:] for r in a.data)
    return marr(rows)


def hstack(parts):
    arrs = [asarray(p) for p in parts]
    if len(arrs[0].shape) == 1:
        return concatenate(parts)
    rows = [[] for _ in range(arrs[0].shape[0])]
    for a in arrs:
        a2 = atleast_2d(a)
        for i, r in enumerate(a2.data):
            rows[i].extend(r)
    return marr(rows)


def stack(parts, axis=0):
    arrs = [asarray(p) for p in parts]
    if arrs and len(arrs[0].shape) == 2:
        if axis == 0:
            # rank-3 result surfaces as a nested list (rank-2 core)
            return ndlist([[row[:] for row in a2.data]
                           for a2 in arrs])
        raise ValueError("stack: 2-D parts support axis=0 only")
    rows = [a2._flat() for a2 in arrs]
    if axis in (0, None):
        return marr(rows)
    if axis in (1, -1):
        return marr([[rows[j][i] for j in range(len(rows))]
                     for i in range(len(rows[0]))])
    raise ValueError("stack: unsupported axis %r" % (axis,))


dstack = None  # rarely used; assigned below if needed


def searchsorted(a, v, side="left"):
    import bisect
    f = asarray(a)._flat()
    fn_ = bisect.bisect_left if side == "left" else bisect.bisect_right

    def one(x):
        return float(fn_(f, x))
    if isinstance(v, (list, tuple, marr)):
        return asarray(v)._map(one)
    return int(one(float(v)))


def flatnonzero(x):
    return marr([float(i) for i, v in enumerate(asarray(x)._flat())
                 if v != 0])


def nonzero(x):
    return (flatnonzero(x),)


def triu_indices(n, k=0):
    ii, jj = [], []
    for i in range(n):
        for j in range(i + k, n):
            if j >= 0:
                ii.append(float(i))
                jj.append(float(j))
    return marr(ii), marr(jj)


def tril_indices(n, k=0):
    ii, jj = [], []
    for i in range(n):
        for j in range(0, _bi.min(i + k + 1, n)):
            ii.append(float(i))
            jj.append(float(j))
    return marr(ii), marr(jj)


def diag_indices(n):
    idx = marr([float(i) for i in range(n)])
    return idx, marr(idx.data[:])


def triu(a, k=0):
    m = atleast_2d(a)
    return marr([[m.data[i][j] if j >= i + k else 0.0
                  for j in range(m.shape[1])] for i in range(m.shape[0])])


def tril(a, k=0):
    m = atleast_2d(a)
    return marr([[m.data[i][j] if j <= i + k else 0.0
                  for j in range(m.shape[1])] for i in range(m.shape[0])])


def convolve(a, v, mode="full"):
    x = asarray(a)._flat()
    y = asarray(v)._flat()
    n, m = len(x), len(y)
    full = [0.0] * (n + m - 1)
    for i in range(n):
        for j in range(m):
            full[i + j] += x[i] * y[j]
    if mode == "full":
        return marr(full)
    if mode == "same":
        start = (m - 1) // 2
        return marr(full[start:start + n])
    if mode == "valid":
        lo, hi = _bi.min(n, m) - 1, _bi.max(n, m)
        return marr(full[lo:hi])
    raise ValueError("bad mode")


def histogram(x, bins=10, range=None):  # noqa: A002
    f = asarray(x)._flat()
    lo = _bi.min(f) if range is None else range[0]
    hi = _bi.max(f) if range is None else range[1]
    if isinstance(bins, (list, tuple, marr)):
        edges = asarray(bins)._flat()
    else:
        step = (hi - lo) / bins
        edges = [lo + i * step for i in _bi.range(bins + 1)] \
            if hasattr(_bi, "range") else [lo + i * step
                                           for i in list(__import__("builtins").range(bins + 1))]
    counts = [0.0] * (len(edges) - 1)
    for v in f:
        if v < edges[0] or v > edges[-1]:
            continue
        for b in range_(len(edges) - 1):
            if edges[b] <= v < edges[b + 1] or (b == len(edges) - 2
                                                and v == edges[-1]):
                counts[b] += 1.0
                break
    return marr(counts), marr(edges)


def range_(n):
    import builtins
    return builtins.range(n)


def bincount(x, minlength=0):
    f = [int(v) for v in asarray(x)._flat()]
    n = _bi.max([minlength - 1] + f) + 1 if f or minlength else 0
    out = [0.0] * n
    for v in f:
        out[v] += 1.0
    return marr(out)


def meshgrid(x, y):
    fx, fy = asarray(x)._flat(), asarray(y)._flat()
    gx = marr([fx[:] for _ in fy])
    gy = marr([[v] * len(fx) for v in fy])
    return gx, gy


def average(x, weights=None):
    a = asarray(x)
    if weights is None:
        return a.mean()
    w = asarray(weights)
    return float(dot(w, a) / w.sum())


def array_equal(a, b):
    fa, fb = asarray(a)._flat(), asarray(b)._flat()
    return len(fa) == len(fb) and fa == fb


def take(x, idx):
    f = asarray(x)._flat()
    return marr([f[int(i)] for i in asarray(idx)._flat()])


def append(x, v):
    f = asarray(x)._flat()[:]
    f.extend(asarray(v)._flat())
    return marr(f)


def delete(x, idx, axis=None):
    a = asarray(x)
    if axis is not None and len(a.shape) == 2:
        drop = {int(i) for i in (asarray(idx)._flat()
                                 if isinstance(idx, (list, tuple, marr))
                                 else [idx])}
        if axis in (0, -2):
            return marr([r[:] for i, r in enumerate(a.data)
                         if i not in drop])
        return marr([[v for j, v in enumerate(r) if j not in drop]
                     for r in a.data])
    f = a._flat()
    drop = {int(i) for i in (asarray(idx)._flat()
                             if isinstance(idx, (list, tuple, marr))
                             else [idx])}
    return marr([v for i, v in enumerate(f) if i not in drop])


def insert(x, pos, v):
    f = asarray(x)._flat()[:]
    f[int(pos):int(pos)] = asarray(v)._flat()
    return marr(f)


def flip(x):
    return marr(asarray(x)._flat()[::-1])


def _nan_filter(x):
    return [v for v in asarray(x)._flat() if v == v]


def _keepdims_wrap(out, axis, keepdims):
    if not keepdims:
        return out
    if isinstance(out, marr):
        return marr([out.tolist()]) if axis in (0, -2) else \
            marr([[v] for v in out._flat()])
    return marr([float(out)])


def _nan_axis(x, axis, red):
    """Apply a nan-skipping reduction along an axis of a 2-D array."""
    a = atleast_2d(asarray(x))
    if axis in (0, -2):
        cols = [[a.data[r][c] for r in range(a.shape[0])
                 if a.data[r][c] == a.data[r][c]]
                for c in range(a.shape[1])]
        return marr([red(col) for col in cols])
    rows = [[v for v in row if v == v] for row in a.data]
    return marr([red(r) for r in rows])


def _median_of(v):
    if not v:
        return nan
    sv = sorted(v)
    n = len(sv)
    return sv[n // 2] if n % 2 else 0.5 * (sv[n // 2 - 1] + sv[n // 2])


def nanmean(x, axis=None, keepdims=False):
    if axis is not None and len(asarray(x).shape) == 2:
        return _keepdims_wrap(
            _nan_axis(x, axis,
                      lambda v: _math.fsum(v) / len(v) if v else nan),
            axis, keepdims)
    f = _nan_filter(x)
    return _keepdims_wrap(float(_math.fsum(f) / len(f)), axis, keepdims)


def nansum(x, axis=None, keepdims=False):
    if axis is not None and len(asarray(x).shape) == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _math.fsum(v)),
                              axis, keepdims)
    return float(_math.fsum(_nan_filter(x)))


def nanstd(x, axis=None, ddof=0, keepdims=False):
    f = _nan_filter(x)
    m = _math.fsum(f) / len(f)
    return _math.sqrt(_math.fsum((v - m) ** 2 for v in f) / (len(f) - ddof))


def nanvar(x, axis=None, ddof=0, keepdims=False):
    return nanstd(x, ddof=ddof) ** 2


def nanmax(x, axis=None, keepdims=False):
    if axis is not None and len(asarray(x).shape) == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _bi.max(v) if v else nan),
                              axis, keepdims)
    return float(_bi.max(_nan_filter(x)))


def nanmin(x, axis=None, keepdims=False):
    if axis is not None and len(asarray(x).shape) == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _bi.min(v) if v else nan),
                              axis, keepdims)
    return float(_bi.min(_nan_filter(x)))


def nanmedian(x, axis=None, keepdims=False):
    if axis is not None and len(asarray(x).shape) == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _median_of(v)),
                              axis, keepdims)
    return median(_nan_filter(x))


def nanargmax(x):
    f = asarray(x)._flat()
    best, bi_ = None, -1
    for i, v in enumerate(f):
        if v == v and (best is None or v > best):
            best, bi_ = v, i
    return bi_


def nanpercentile(x, q):
    return percentile(_nan_filter(x), q)


def nanquantile(x, q):
    return quantile(_nan_filter(x), q)


def vander(x, n=None, increasing=False):
    f = asarray(x)._flat()
    n = len(f) if n is None else int(n)
    powers = list(range_(n)) if increasing else list(range_(n))[::-1]
    return marr([[v ** p for p in powers] for v in f])


def polyfit(x, y, deg):
    v = vander(x, deg + 1)
    beta, *_r = linalg.lstsq(v, asarray(y))
    return beta


def polyval(p, x):
    c = asarray(p)._flat()

    def one(v):
        out = 0.0
        for coef in c:
            out = out * v + coef
        return out
    if isinstance(x, (list, tuple, marr)):
        return asarray(x)._map(one)
    return one(float(x))


def polyder(p):
    c = asarray(p)._flat()
    n = len(c) - 1
    return marr([c[i] * (n - i) for i in range_(n)])


def polymul(a, b):
    return convolve(a, b, mode="full")


def poly(roots):
    out = [1.0]
    for r in asarray(roots)._flat():
        out = convolve(out, [1.0, -r]).tolist()
    return marr(out)


def kron(a, b):
    aa, bb = atleast_2d(a), atleast_2d(b)
    out = []
    for i in range_(aa.shape[0]):
        for k in range_(bb.shape[0]):
            row = []
            for j in range_(aa.shape[1]):
                for m in range_(bb.shape[1]):
                    row.append(aa.data[i][j] * bb.data[k][m])
            out.append(row)
    return marr(out)


def ix_(rows, cols):
    r = marr([float(v) for v in asarray(rows)._flat()])
    c = marr([float(v) for v in asarray(cols)._flat()])
    r._is_index = True
    c._is_index = True
    return (r, c)


def cumprod(x):
    out = []
    total = 1.0
    for v in asarray(x)._flat():
        total *= v
        out.append(total)
    return marr(out)


def ediff1d(x):
    return diff(x)


def setdiff1d(a, b):
    bs = set(asarray(b)._flat()) if not isinstance(b, (int, float)) \
        else {float(b)}
    seen = set()
    out = []
    for v in sorted(asarray(a)._flat()):
        if v not in bs and v not in seen:
            seen.add(v)
            out.append(v)
    return marr(out)


def union1d(a, b):
    vals = set(asarray(a)._flat()) | set(asarray(b)._flat())
    return marr(sorted(vals))


def intersect1d(a, b):
    vals = set(asarray(a)._flat()) & set(asarray(b)._flat())
    return marr(sorted(vals))


def in1d(a, b):
    bs = set(asarray(b)._flat())
    return marr([1.0 if v in bs else 0.0 for v in asarray(a)._flat()])



def real(x):
    """Real part.  marr holds complex values, so this must actually take
    the real part rather than pass the value through."""
    return asarray(x)._map(lambda v: v.real if isinstance(v, complex)
                           else float(v))


def imag(x):
    """Imaginary part.  Returned a hard zero before, which was a silent
    wrong answer for every complex input."""
    return asarray(x)._map(lambda v: v.imag if isinstance(v, complex)
                           else 0.0)


def angle(x):
    """Phase angle in radians, atan2(imag, real).

    The previous version tested `v >= 0`, which raises on a complex value
    and, for real input, only ever returned 0 or pi.
    """
    def _ang(v):
        if isinstance(v, complex):
            return _math.atan2(v.imag, v.real)
        return 0.0 if v >= 0 else _math.pi
    return asarray(x)._map(_ang)


def conjugate(x):
    """Complex conjugate.  Returned its argument unchanged before, so
    every conjugate-multiply in a spectrum was wrong."""
    return asarray(x)._map(lambda v: v.conjugate()
                           if isinstance(v, complex) else v)


conj = conjugate


def isreal(x):
    """Elementwise: is this element real-valued?  numpy.isreal returns a
    boolean array and is True for a complex whose imaginary part is 0."""
    return asarray(x)._map(
        lambda v: 1.0 if not isinstance(v, complex) or v.imag == 0 else 0.0)


def iscomplex(x):
    """Elementwise: does this element have a non-zero imaginary part?"""
    return asarray(x)._map(
        lambda v: 1.0 if isinstance(v, complex) and v.imag != 0 else 0.0)


def isrealobj(x):
    """Whole-array: does the container hold no complex element at all?
    Unlike isreal this looks at storage, so a complex 0j counts."""
    return not any(isinstance(v, complex) for v in asarray(x)._flat())


def iscomplexobj(x):
    return not isrealobj(x)


class _RClass:
    """numpy.r_ : concatenate the arguments into one 1-D array.

    Only the concatenation behaviour is provided -- the slice/step-string
    forms of numpy.r_ are not, and asking for one raises rather than
    silently returning something else.
    """

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        out = []
        for item in key:
            if isinstance(item, str):
                raise ValueError(
                    "r_ string directives (%r) are not supported" % item)
            if isinstance(item, slice):
                raise ValueError("r_ slice syntax is not supported; pass "
                                 "arange(...) explicitly")
            if isinstance(item, (int, float, complex)):
                out.append(item)
            else:
                out.extend(list(asarray(item)._flat()))
        return marr(out)


r_ = _RClass()


def square(x):
    return asarray(x)._map(lambda v: v * v)


def exp2(x):
    return asarray(x)._map(lambda v: 2.0 ** v)


def hypot(a, b):
    return asarray(a)._zip(b, _math.hypot)


def rint(x):
    return asarray(x)._map(lambda v: float(_bi.round(v)))


def geomspace(a, b, n):
    la, lb = _math.log(a), _math.log(b)
    return marr([_math.exp(la + i * (lb - la) / (n - 1))
                 for i in range_(int(n))])


def split(x, k):
    f = asarray(x)._flat()
    step = len(f) // int(k)
    return [marr(f[i * step:(i + 1) * step]) for i in range_(int(k))]


def empty_like(x, dtype=None):
    return zeros_like(x)


def spacing(x):
    import sys as _s
    return _s.float_info.epsilon * _bi.max(_bi_abs(float(x)), 1.0)


integer = int
number = float
floating = float


def issubdtype(a, b):
    del a, b
    return True     # all our dtypes are float; callers gate float paths


def array_str(x):
    return repr(asarray(x))


def select(conds, choices, default=0.0):
    n = asarray(conds[0]).shape[0]
    out = [float(default)] * n
    for c, ch in zip(conds, choices):
        cf = asarray(c)._flat()
        chf = asarray(ch)._flat() if isinstance(ch, (list, tuple, marr)) \
            else [float(ch)] * n
        for i in range_(n):
            if cf[i] != 0 and out[i] == float(default):
                out[i] = chf[i]
    return marr(out)


def lexsort(keys):
    arrs = [asarray(k)._flat() for k in keys]
    n = len(arrs[0])
    order = sorted(range_(n), key=lambda i: tuple(a[i]
                                                 for a in reversed(arrs)))
    return marr([float(i) for i in order])


def cross(a, b):
    x, y = asarray(a)._flat(), asarray(b)._flat()
    return marr([x[1] * y[2] - x[2] * y[1],
                 x[2] * y[0] - x[0] * y[2],
                 x[0] * y[1] - x[1] * y[0]])


def partition(x, k, axis=None):
    del k  # sorted output satisfies the partition contract
    a = asarray(x)
    if axis in (-1, 1) and len(a.shape) == 2:
        return marr([sorted(row) for row in a.data])
    if axis in (0, -2) and len(a.shape) == 2:
        cols = [sorted(a.data[r][c] for r in range(a.shape[0]))
                for c in range(a.shape[1])]
        return marr([[cols[c][r] for c in range(a.shape[1])]
                     for r in range(a.shape[0])])
    return sort(a)


def _eigh(a):
    vals, vecs = _jacobi_eigh(a)
    order = sorted(range_(len(vals)), key=lambda i: vals[i])
    w = marr([vals[i] for i in order])
    v = marr([[vecs[r][i] for i in order] for r in range_(len(vals))])
    return w, v


def _det(a):
    sign, logdet = _lu_slogdet(atleast_2d(a).tolist())
    if logdet == -inf:
        return 0.0
    return sign * _math.exp(logdet)


def _cholesky(a):
    m = atleast_2d(a).tolist()
    n = len(m)
    low = [[0.0] * n for _ in range_(n)]
    for i in range_(n):
        for j in range_(i + 1):
            s = _math.fsum(low[i][k] * low[j][k] for k in range_(j))
            if i == j:
                val = m[i][i] - s
                if val <= 0:
                    raise linalg.LinAlgError("matrix not positive definite")
                low[i][j] = _math.sqrt(val)
            else:
                low[i][j] = (m[i][j] - s) / low[j][j]
    return marr(low)


def _svd(a, full_matrices=False, compute_uv=True):
    """SVD. C core: one-sided Jacobi (Demmel & Veselic 1992), high
    relative accuracy in every singular value. Fallback: eigh of
    A^T A (accurate only above ~sqrt(eps)*s_max)."""
    del full_matrices
    aa = atleast_2d(a)
    if _HAS_CORE and hasattr(_CK, "jacobi_svd"):
        import array as _pa
        m0, n0 = aa.shape
        transposed = m0 < n0
        w0 = aa.T if transposed else aa
        m, n = w0.shape
        flat = _pa.array("d", [v for row in w0.data for v in row])
        u_b, s_b, v_b = _CK.jacobi_svd(flat, m, n)
        sv = _pa.array("d"); sv.frombytes(s_b)
        svals = list(sv)
        if not compute_uv:
            return marr(svals)
        ub = _pa.array("d"); ub.frombytes(u_b)
        vb = _pa.array("d"); vb.frombytes(v_b)
        u = marr([[ub[r * n + c] / svals[c] if svals[c] > 1e-300 else 0.0
                   for c in range(n)] for r in range(m)])
        vt = marr([[vb[r * n + c] for r in range(n)] for c in range(n)])
        if transposed:
            # A = (U S V^T)^T of the transpose: swap the factors
            return vt.T, marr(svals), u.T
        return u, marr(svals), vt
    ata = matmul(aa.T, aa)
    w, v = _eigh(ata)
    order = sorted(range_(w.shape[0]), key=lambda i: -w.data[i])
    svals = [_math.sqrt(_bi.max(w.data[i], 0.0)) for i in order]
    if not compute_uv:
        return marr(svals)
    vt = marr([[v.data[r][i] for r in range_(v.shape[0])] for i in order])
    us = []
    for k, i in enumerate(order):
        col = matmul(aa, marr([v.data[r][i] for r in range_(v.shape[0])]))
        s = svals[k]
        us.append([c / s if s > 1e-300 else 0.0 for c in col.data])
    u = marr([[us[k][r] for k in range_(len(order))]
              for r in range_(aa.shape[0])])
    return u, marr(svals), vt


linalg.eigh = _eigh
linalg.det = _det
linalg.cholesky = _cholesky
def _lstsq(a, b, rcond=None):
    """Minimum-norm least-squares solution via the SVD.

    ``rcond`` follows numpy.linalg.lstsq: ``None`` means
    ``eps * max(M, N)``; a negative value means machine precision alone
    (the LAPACK ``dgelsd`` convention that numpy's legacy ``rcond=-1``
    forwards, and what statsmodels passes from ``_MinimalWLS``);
    otherwise the ratio is used as given.  Singular values at or below
    ``rcond * max(s)`` are treated as zero.
    """
    aa = atleast_2d(asarray(a))
    bv = asarray(b)._flat()
    n, k = aa.shape
    u, sv, vt = _svd(aa)
    svl = list(sv._flat())
    eps = 2.220446049250313e-16
    if rcond is None:
        ratio = eps * _bi.max(n, k)
    elif rcond < 0:
        ratio = eps
    else:
        ratio = float(rcond)
    cut = (_bi.max(svl) if svl else 0.0) * ratio
    uy = [_math.fsum(u.data[r][c] * bv[r] for r in range(n))
          for c in range(len(svl))]
    z = [uy[c] / svl[c] if svl[c] > cut else 0.0
         for c in range(len(svl))]
    x = marr([_math.fsum(vt.data[c][j] * z[c]
                         for c in range(len(svl)))
              for j in range(k)])
    resid = marr([])
    rank = _pysum(1 for v in svl if v > cut)
    return x, resid, rank, sv


linalg.lstsq = _lstsq
linalg.svd = _svd


# --------------------------------------------------------------- fft

class carr:
    """Minimal 1-D complex array for FFT results."""

    def __init__(self, data):
        self.data = [complex(v) for v in data]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return carr(self.data[i])
        return self.data[i]

    def __buffer__(self, flags):
        """PEP 688 buffer protocol (Python >= 3.12): expose the flat
        float64 data so nanobind kernels and memoryview consumers get
        the array without numpy. Snapshot semantics: the exported
        buffer is a copy, matching the immutable-input contract of the
        compiled kernels. Tagged masks/index arrays refuse the buffer
        (numpy >= 2.5 prefers it over __array_interface__, which would
        surface them as float64 and break real-numpy indexing)."""
        del flags
        if getattr(self, "_is_mask", False) or \
                getattr(self, "_is_index", False):
            raise BufferError("tagged mask/index arrays export via "
                              "__array_interface__")
        import array as _pa
        buf = _pa.array("d", [float(v) for v in self._flat()])
        return memoryview(buf)

    def __array__(self, dtype=None, copy=None):
        """numpy interop for mixed test environments: masks surface as
        bool arrays so real-numpy indexing works. Never imports numpy
        itself — only cooperates when the caller already has it."""
        del copy
        import sys as _sys
        _np = _sys.modules.get("numpy")
        if _np is None:
            raise TypeError("numpy not loaded")
        if dtype is None and getattr(self, "_is_mask", False):
            dtype = bool
        elif dtype is None and getattr(self, "_is_index", False):
            dtype = "int64"
        return _np.asarray(self.tolist(), dtype=dtype)

    def tolist(self):
        return self.data[:]

    @property
    def real(self):
        return marr([v.real for v in self.data])

    @property
    def imag(self):
        return marr([v.imag for v in self.data])

    def conj(self):
        return carr([v.conjugate() for v in self.data])

    conjugate = conj

    @property
    def shape(self):
        return (len(self.data),)

    def __abs__(self):
        return marr([_bi.abs(v) for v in self.data])

    def _binop(self, other, fn):
        if isinstance(other, carr):
            return carr([fn(a, b) for a, b in zip(self.data, other.data)])
        if isinstance(other, marr):
            return carr([fn(a, b) for a, b in
                         zip(self.data, other._flat())])
        if isinstance(other, (list, tuple)):
            return carr([fn(a, b) for a, b in zip(self.data, other)])
        return carr([fn(a, other) for a in self.data])

    def __mul__(self, o):
        return self._binop(o, lambda a, b: a * b)

    __rmul__ = __mul__

    def __add__(self, o):
        return self._binop(o, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, o):
        return self._binop(o, lambda a, b: a - b)

    def __truediv__(self, o):
        return self._binop(o, lambda a, b: a / b)


def _fft_pow2(a, invert):
    n = len(a)
    if n == 1:
        return a[:]
    # iterative Cooley-Tukey, bit-reversal
    j = 0
    a = a[:]
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    ln = 2
    while ln <= n:
        ang = (2.0 if invert else -2.0) * _math.pi / ln
        wl = complex(_math.cos(ang), _math.sin(ang))
        for i in range(0, n, ln):
            w = complex(1.0)
            for k in range(ln // 2):
                u = a[i + k]
                v = a[i + k + ln // 2] * w
                a[i + k] = u + v
                a[i + k + ln // 2] = u - v
                w *= wl
        ln <<= 1
    return a


def _fft_bluestein(a, invert):
    n = len(a)
    sign = 1.0 if invert else -1.0
    # chirp
    w = [complex(_math.cos(sign * _math.pi * (k * k % (2 * n)) / n),
                 _math.sin(sign * _math.pi * (k * k % (2 * n)) / n))
         for k in range(n)]
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    fa = [a[k] * w[k] for k in range(n)] + [0j] * (m - n)
    fb = [0j] * m
    for k in range(n):
        fb[k] = w[k].conjugate()
        if k:
            fb[m - k] = w[k].conjugate()
    fa = _fft_pow2(fa, False)
    fb = _fft_pow2(fb, False)
    fc = [x * y for x, y in zip(fa, fb)]
    fc = _fft_pow2(fc, True)
    fc = [v / m for v in fc]
    return [fc[k] * w[k] for k in range(n)]


def _fft_any(a, invert):
    n = len(a)
    if n == 0:
        return []
    if n & (n - 1) == 0:
        return _fft_pow2(a, invert)
    return _fft_bluestein(a, invert)


def _tocomplex(x):
    if isinstance(x, carr):
        return x.data[:]
    if hasattr(x, "_flat"):
        return [complex(v) for v in x._flat()]
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [complex(v) for v in x]


class _FFT:
    @staticmethod
    def fft(x, n=None, axis=-1):
        # 1-D only: axis is accepted for numpy call-compatibility
        # and must select the single existing axis.
        if axis not in (-1, 0):
            raise ValueError("only 1-D transforms are "
                             "supported; axis=%r" % (axis,))
        a = _tocomplex(x)
        if n is not None:
            a = a[:n] + [0j] * _bi.max(0, n - len(a))
        return carr(_fft_any(a, False))

    @staticmethod
    def ifft(x, n=None, axis=-1):
        # 1-D only: axis is accepted for numpy call-compatibility
        # and must select the single existing axis.
        if axis not in (-1, 0):
            raise ValueError("only 1-D transforms are "
                             "supported; axis=%r" % (axis,))
        a = _tocomplex(x)
        if n is not None:
            a = a[:n] + [0j] * _bi.max(0, n - len(a))
        out = _fft_any(a, True)
        return carr([v / len(a) for v in out])

    @staticmethod
    def rfft(x, n=None, axis=-1):
        # 1-D only: axis is accepted for numpy call-compatibility
        # and must select the single existing axis.
        if axis not in (-1, 0):
            raise ValueError("only 1-D transforms are "
                             "supported; axis=%r" % (axis,))
        a = _tocomplex(x)
        if n is not None:
            a = a[:n] + [0j] * _bi.max(0, n - len(a))
        full = _fft_any(a, False)
        return carr(full[:len(a) // 2 + 1])

    @staticmethod
    def irfft(x, n=None, axis=-1):
        # 1-D only: axis is accepted for numpy call-compatibility
        # and must select the single existing axis.
        if axis not in (-1, 0):
            raise ValueError("only 1-D transforms are "
                             "supported; axis=%r" % (axis,))
        half = _tocomplex(x)
        m = len(half)
        if n is None:
            n = 2 * (m - 1)
        full = half[:]
        for k in range(1, m - 1 if n % 2 == 0 else m):
            idx = n - k
            if idx >= len(full):
                full += [0j] * (idx - len(full) + 1)
            full[idx] = half[k].conjugate()
        full = full[:n] + [0j] * _bi.max(0, n - len(full))
        out = _fft_any(full, True)
        return marr([(v / n).real for v in out])

    @staticmethod
    def fftfreq(n, d=1.0):
        out = []
        half = (n - 1) // 2 + 1
        for k in range(half):
            out.append(k / (n * d))
        for k in range(-(n // 2), 0):
            out.append(k / (n * d))
        return marr(out)

    @staticmethod
    def rfftfreq(n, d=1.0):
        return marr([k / (n * d) for k in range(n // 2 + 1)])

    @staticmethod
    def fftshift(x):
        a = _tocomplex(x) if isinstance(x, carr) else None
        if a is not None:
            n = len(a)
            return carr(a[(n + 1) // 2:] + a[:(n + 1) // 2])
        v = list(asarray(x)._flat())
        n = len(v)
        return marr(v[(n + 1) // 2:] + v[:(n + 1) // 2])

    @staticmethod
    def ifftshift(x):
        v = list(asarray(x)._flat())
        n = len(v)
        return marr(v[n // 2:] + v[:n // 2])


fft = _FFT()


def _tag_predicate(fn):
    def wrapped(*a, **k):
        out = fn(*a, **k)
        if isinstance(out, marr):
            out._is_mask = True
        return out
    wrapped.__name__ = fn.__name__
    return wrapped


for _pname in ("isnan", "isfinite", "isinf", "isin", "isclose"):
    if _pname in globals():
        globals()[_pname] = _tag_predicate(globals()[_pname])
del _pname


# --------------------------------------------------------------- final tail

def full_like(a, fill_value, dtype=None):
    del dtype
    x = asarray(a)
    if len(x.shape) == 2:
        return marr([[float(fill_value)] * x.shape[1]
                     for _ in range(x.shape[0])])
    return marr([float(fill_value)] * x.shape[0])


def ascontiguousarray(a, dtype=None):
    del dtype
    return asarray(a).copy()


def array_split(a, sections, axis=0):
    x = asarray(a)
    n = x.shape[0]
    k = int(sections)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out = []
    pos = 0
    if len(x.shape) == 2 and axis in (1, -1):
        m = x.shape[1]
        sizes = [m // k + (1 if i < m % k else 0) for i in range(k)]
        for s in sizes:
            out.append(marr([row[pos:pos + s] for row in x.data]))
            pos += s
        return out
    for s in sizes:
        out.append(marr(x.data[pos:pos + s]))
        pos += s
    return out


def logspace(start, stop, num=50, base=10.0):
    step = (stop - start) / (num - 1) if num > 1 else 0.0
    return marr([base ** (start + i * step) for i in range(num)])


def hanning(n):
    if n == 1:
        return marr([1.0])
    return marr([0.5 - 0.5 * _math.cos(2.0 * _math.pi * i / (n - 1))
                 for i in range(n)])


def hamming(n):
    if n == 1:
        return marr([1.0])
    return marr([0.54 - 0.46 * _math.cos(2.0 * _math.pi * i / (n - 1))
                 for i in range(n)])


def blackman(n):
    if n == 1:
        return marr([1.0])
    return marr([0.42 - 0.5 * _math.cos(2.0 * _math.pi * i / (n - 1))
                 + 0.08 * _math.cos(4.0 * _math.pi * i / (n - 1))
                 for i in range(n)])


def bartlett(n):
    if n == 1:
        return marr([1.0])
    return marr([1.0 - _bi.abs(2.0 * i / (n - 1) - 1.0)
                 for i in range(n)])


def _bessel_i0(x):
    total = 1.0
    term = 1.0
    for k in range(1, 60):
        term *= (x / 2.0) ** 2 / (k * k)
        total += term
        if term < 1e-18 * total:
            break
    return total


def kaiser(n, beta):
    if n == 1:
        return marr([1.0])
    d = _bessel_i0(beta)
    return marr([_bessel_i0(beta * _math.sqrt(
        1.0 - (2.0 * i / (n - 1) - 1.0) ** 2)) / d
        for i in range(n)])


def correlate(a, v, mode="valid"):
    av = list(asarray(a)._flat())
    vv = list(asarray(v)._flat())
    # np.correlate: sum a[k+j] * conj(v[j])
    full = []
    n, m = len(av), len(vv)
    for lag in range(-(m - 1), n):
        s = 0.0
        for j in range(m):
            k = lag + j
            if 0 <= k < n:
                s += av[k] * vv[j]
        full.append(s)
    if mode == "full":
        return marr(full)
    if mode == "same":
        start = (m - 1) // 2
        return marr(full[start:start + n])
    lo = m - 1
    return marr(full[lo:len(full) - (m - 1)])


def unwrap(p, discont=None):
    v = list(asarray(p)._flat())
    d = discont if discont is not None else _math.pi
    out = [v[0]]
    offset = 0.0
    for i in range(1, len(v)):
        diff = v[i] - v[i - 1]
        if diff > d:
            offset -= 2.0 * _math.pi * _math.ceil(
                (diff - d) / (2.0 * _math.pi))
        elif diff < -d:
            offset += 2.0 * _math.pi * _math.ceil(
                (-diff - d) / (2.0 * _math.pi))
        out.append(v[i] + offset)
    return marr(out)


def roll(a, shift, axis=None):
    x = asarray(a)
    if len(x.shape) == 2 and axis is not None:
        if axis == 0:
            s = int(shift) % x.shape[0]
            return marr(x.data[-s:] + x.data[:-s])
        s = int(shift) % x.shape[1]
        return marr([row[-s:] + row[:-s] for row in x.data])
    f = list(x._flat())
    s = int(shift) % len(f)
    return marr(f[-s:] + f[:-s])


def power(a, b):
    return asarray(a)._zip(b, lambda x, y: x ** y)


def gradient(f, *varargs):
    v = list(asarray(f)._flat())
    dx = float(varargs[0]) if varargs and isinstance(
        varargs[0], (int, float)) else 1.0
    xs = (list(asarray(varargs[0])._flat())
          if varargs and not isinstance(varargs[0], (int, float))
          else None)
    n = len(v)
    out = []
    for i in range(n):
        if i == 0:
            h = (xs[1] - xs[0]) if xs else dx
            out.append((v[1] - v[0]) / h)
        elif i == n - 1:
            h = (xs[-1] - xs[-2]) if xs else dx
            out.append((v[-1] - v[-2]) / h)
        else:
            h2 = (xs[i + 1] - xs[i - 1]) if xs else 2.0 * dx
            out.append((v[i + 1] - v[i - 1]) / h2)
    return marr(out)


def arctanh(x):
    return _map_unary(x, _math.atanh)


def arctan2(y, x):
    ya = asarray(y)
    return ya._zip(x, lambda a, b: _math.atan2(a, b))


def degrees(x):
    return _map_unary(x, _math.degrees)


def deg2rad(x):
    return _map_unary(x, _math.radians)


rad2deg = degrees
radians = deg2rad


def _map_unary(x, fn):
    if isinstance(x, (int, float)):
        return fn(float(x))
    return asarray(x)._map(fn)


def unravel_index(indices, shape):
    if isinstance(indices, (int, float)):
        idx = int(indices)
        return (idx // shape[1], idx % shape[1]) \
            if len(shape) == 2 else (idx,)
    out_r, out_c = [], []
    for v in asarray(indices)._flat():
        out_r.append(float(int(v) // shape[1]))
        out_c.append(float(int(v) % shape[1]))
    return marr(out_r), marr(out_c)


def trapz(y, x=None, dx=1.0):
    return trapezoid(y, x=x, dx=dx)


def ptp(a, axis=None):
    x = asarray(a)
    if axis is None or len(x.shape) == 1:
        f = x._flat()
        return _bi.max(f) - _bi.min(f)
    if axis == 0:
        return marr([_bi.max(x.data[i][j] for i in range(x.shape[0]))
                     - _bi.min(x.data[i][j]
                               for i in range(x.shape[0]))
                     for j in range(x.shape[1])])
    return marr([_bi.max(row) - _bi.min(row) for row in x.data])


def pad(a, pad_width, mode="constant", constant_values=0.0):
    v = list(asarray(a)._flat())
    if isinstance(pad_width, int):
        lo = hi = pad_width
    else:
        lo, hi = pad_width
    if mode == "constant":
        c = float(constant_values)
        return marr([c] * lo + v + [c] * hi)
    if mode == "edge":
        return marr([v[0]] * lo + v + [v[-1]] * hi)
    if mode == "reflect":
        n = len(v)

        def refl(i):
            # reflect without repeating the edge, numpy 'reflect'
            if n == 1:
                return 0
            period = 2 * (n - 1)
            i %= period
            return i if i < n else period - i
        left = [v[refl(-i)] for i in range(lo, 0, -1)]
        right = [v[refl(n - 1 + i)] for i in range(1, hi + 1)]
        return marr(left + v + right)
    if mode == "wrap":
        return marr(v[-lo:] + v + v[:hi])
    raise ValueError("unsupported pad mode %r" % mode)


def packbits(a):
    bits = [1 if v != 0 else 0 for v in asarray(a)._flat()]
    while len(bits) % 8:
        bits.append(0)
    out = []
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        out.append(float(byte))
    return marr(out)


def unpackbits(a):
    out = []
    for v in asarray(a)._flat():
        byte = int(v) & 0xFF
        for k in range(7, -1, -1):
            out.append(float((byte >> k) & 1))
    return marr(out)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    big = 1.7976931348623157e308

    def one(v):
        if v != v:
            return float(nan)
        if v == _math.inf:
            return float(posinf) if posinf is not None else big
        if v == -_math.inf:
            return float(neginf) if neginf is not None else -big
        return v
    return _map_unary(x, one)


def isscalar(x):
    return isinstance(x, (int, float, complex, bool, str))


def isinf(x):
    out = _map_unary(x, lambda v: 1.0 if v in (_math.inf, -_math.inf)
                     else 0.0)
    if isinstance(out, marr):
        out._is_mask = True
    return out


def digitize(x, bins, right=False):
    import bisect
    bv = list(asarray(bins)._flat())
    f = bisect.bisect_left if right else bisect.bisect_right

    def one(v):
        return float(f(bv, v))
    return _map_unary(x, one)


def histogram2d(x, y, bins=10, range=None):  # noqa: A002
    xv = list(asarray(x)._flat())
    yv = list(asarray(y)._flat())
    if isinstance(bins, int):
        bx = by = bins
    else:
        bx, by = bins
    if range is not None:
        (xlo, xhi), (ylo, yhi) = range
    else:
        xlo, xhi = _bi.min(xv), _bi.max(xv)
        ylo, yhi = _bi.min(yv), _bi.max(yv)
    H = [[0.0] * by for _ in _pyrange(bx)]
    for a, b in zip(xv, yv):
        i = int((a - xlo) / (xhi - xlo) * bx) if xhi > xlo else 0
        j = int((b - ylo) / (yhi - ylo) * by) if yhi > ylo else 0
        i = _bi.max(0, _bi.min(i, bx - 1))
        j = _bi.max(0, _bi.min(j, by - 1))
        H[i][j] += 1.0
    xe = [xlo + (xhi - xlo) * k / bx for k in _pyrange(bx + 1)]
    ye = [ylo + (yhi - ylo) * k / by for k in _pyrange(by + 1)]
    return marr(H), marr(xe), marr(ye)


_pyrange = __import__("builtins").range


def ndindex(*shape):
    import itertools
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return itertools.product(*(
        _pyrange(int(s)) for s in shape))


def roots(coeffs):
    """Polynomial roots via Durand-Kerner (complex output)."""
    c = [float(v) for v in asarray(coeffs)._flat()]
    while c and c[0] == 0.0:
        c = c[1:]
    n = len(c) - 1
    if n < 1:
        return carr([])
    c = [v / c[0] for v in c]
    rs = [complex(0.4, 0.9) ** k for k in _pyrange(n)]
    for _ in _pyrange(500):
        new = []
        for i in _pyrange(n):
            num = complex(1.0)
            for j in _pyrange(n):
                if j != i:
                    num *= (rs[i] - rs[j])
            pv = complex(0.0)
            for cf in c:
                pv = pv * rs[i] + cf
            new.append(rs[i] - pv / num if num != 0 else rs[i])
        if _bi.max(_bi.abs(a - b) for a, b in zip(new, rs)) < 1e-13:
            rs = new
            break
        rs = new
    return carr(rs)


newaxis = None
complex128 = complex
NDArray = marr          # typing shim for `from numpy.typing import`


class datetime64:
    """Thin ISO-date wrapper for the single call site using it."""

    def __init__(self, value):
        import datetime as _dt
        if isinstance(value, str):
            self._d = _dt.datetime.fromisoformat(value)
        else:
            self._d = value

    def __repr__(self):
        return "datetime64(%r)" % self._d.isoformat()

    def item(self):
        return self._d


# --------------------------------------------------------------- random tail

class RandomState(_SplitMix64):
    """numpy legacy RandomState API on the native stream."""

    def __init__(self, seed=None):
        super().__init__(seed if seed is not None else 0)

    def rand(self, *shape):
        if not shape:
            return self.uniform()
        if len(shape) == 1:
            return self.uniform(0.0, 1.0, shape[0])
        return self.uniform(0.0, 1.0, shape)

    def randn(self, *shape):
        if not shape:
            return self.normal()
        if len(shape) == 1:
            return self.normal(0.0, 1.0, shape[0])
        return self.normal(0.0, 1.0, shape)

    def randint(self, low, high=None, size=None):
        return self.integers(low, high, size)

    def rand_seed(self, s):
        self.__init__(s)

    seed = rand_seed


Generator = _SplitMix64


class Philox:
    """Seed container accepted by default_rng (native SplitMix stream;
    the bit-exact R-parity Philox lives in morie_core — see
    reference_morie_native_rng)."""

    def __init__(self, key=0):
        self.key = int(key)


class _RandomNS(_Random):
    RandomState = RandomState
    Generator = Generator
    Philox = Philox

    def __init__(self):
        self._global = RandomState(0)

    def seed(self, s=None):
        self._global = RandomState(s if s is not None else 0)

    def rand(self, *shape):
        return self._global.rand(*shape)

    def randn(self, *shape):
        return self._global.randn(*shape)

    def randint(self, low, high=None, size=None):
        return self._global.randint(low, high, size)

    def random(self, size=None):
        return self._global.random(size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._global.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._global.normal(loc, scale, size)

    def poisson(self, lam=1.0, size=None):
        return self._global.poisson(lam, size)

    def binomial(self, n, p, size=None):
        return self._global.binomial(n, p, size)

    def exponential(self, scale=1.0, size=None):
        return self._global.exponential(scale, size)

    def choice(self, a, size=None, replace=True, p=None):
        return self._global.choice(a, size, replace, p)

    def shuffle(self, seq):
        return self._global.shuffle(seq)

    def permutation(self, n):
        return self._global.permutation(n)

    @staticmethod
    def default_rng(seed=None):
        if isinstance(seed, Philox):
            seed = seed.key
        return _SplitMix64(seed if seed is not None else 0)


random = _RandomNS()


def frombuffer(buf, dtype="float64", count=-1):
    import struct
    fmt_map = {"float64": ("d", 8), "float32": ("f", 4),
               "float16": ("e", 2),
               "int64": ("q", 8), "int32": ("i", 4),
               "int8": ("b", 1), "int16": ("h", 2),
               "uint8": ("B", 1), "uint64": ("Q", 8)}
    key = dtype if isinstance(dtype, str) else getattr(
        dtype, "__name__", "float64")
    fmt, size = fmt_map.get(key, ("d", 8))
    n = len(buf) // size if count in (-1, None) else int(count)
    vals = struct.unpack("<%d%s" % (n, fmt), bytes(buf[:n * size]))
    return marr([float(v) for v in vals])


# --------------------------------------------------------------- C dispatch

# Compiled morie._core kernels (nanobind, see libmorie/linalg_core.hpp)
# take over the hot paths when the extension is importable; the pure-
# Python implementations above remain the reference arm and the
# fallback on source-only installs.
try:
    from morie import _core as _CK
    _HAS_CORE = hasattr(_CK, "matmul")
except Exception:                       # pragma: no cover - env-specific
    _CK = None
    _HAS_CORE = False

if _HAS_CORE:
    import array as _pyarray

    _py_matmul = matmul

    def _buf(flat):
        out = _pyarray.array("d")
        out.fromlist([float(v) for v in flat])
        return out

    def _unbuf(bts):
        out = _pyarray.array("d")
        out.frombytes(bts)
        return list(out)

    def matmul(a, b):  # noqa: F811
        A = asarray(a)
        B = asarray(b)
        if len(A.shape) != 2 or len(B.shape) != 2:
            return _py_matmul(a, b)
        n, m = A.shape
        m2, p = B.shape
        if m != m2:
            return _py_matmul(a, b)     # let the reference arm raise
        flat = _CK.matmul(_buf(A._flat()), _buf(B._flat()), n, m, p)
        vals = _unbuf(flat)
        return marr([vals[i * p:(i + 1) * p] for i in range(n)])

    _py_solve = _Linalg.solve

    def _c_solve(a, b):
        A = atleast_2d(a)
        n = A.shape[0]
        bv = asarray(b)
        b_was_1d = len(bv.shape) == 1
        k = 1 if b_was_1d else bv.shape[1]
        try:
            flat = _CK.solve(_buf(A._flat()), _buf(bv._flat()), n, k)
        except Exception:
            raise linalg.LinAlgError("singular matrix") from None
        vals = _unbuf(flat)
        if b_was_1d:
            return marr(vals)
        # 2-D right-hand side keeps 2-D shape even for k == 1
        # (inv of a 1x1 matrix must stay a matrix)
        return marr([vals[i * k:(i + 1) * k] for i in range(n)])

    _Linalg.solve = staticmethod(_c_solve)

    _py_fft_any = _fft_any

    def _fft_any(a, invert):  # noqa: F811
        n = len(a)
        if n == 0:
            return []
        re, im = _CK.fft(_buf(v.real for v in a),
                         _buf(v.imag for v in a), invert)
        rev, imv = _unbuf(re), _unbuf(im)
        out = [complex(rev[i], imv[i]) for i in range(n)]
        if invert and n & (n - 1) == 0:
            # compiled fft returns unscaled inverse for pow2 to match
            # the pure-Python contract used by fft.ifft
            return out
        return out
    # rebind the FFT namespace onto the dispatched transform
    _FFT_dispatch_note = "compiled"


def _tag_index(fn):
    def wrapped(*a, **k):
        out = fn(*a, **k)
        if isinstance(out, marr):
            out._is_index = True
        elif isinstance(out, tuple):
            for o in out:
                if isinstance(o, marr):
                    o._is_index = True
        return out
    wrapped.__name__ = fn.__name__
    return wrapped


for _iname in ("argsort", "where", "nonzero", "flatnonzero",
               "searchsorted", "digitize", "argmax", "argmin"):
    if _iname in globals():
        globals()[_iname] = _tag_index(globals()[_iname])
del _iname
