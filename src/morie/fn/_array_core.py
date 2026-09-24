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
- random.default_rng is a SplitMix64 stream (see _SplitMix64); the
  compiled core is present; here a Python fallback with the same API
  subset (normal, uniform, integers) built on SplitMix64 -- NOT the gr*
  LCG (banned) and clearly labeled non-Philox until the C hook lands.
"""

from __future__ import annotations

import builtins as _bi
import cmath as _cmath
import math as _math
import struct as _struct
import warnings as _warnings

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



# --------------------------------------------------------------- dtypes
# Struct format and width for every dtype the core can store or emit.
# tobytes() has to reproduce numpy's byte layout exactly: the GGUF writer
# packs an array and the loader reads it straight back.
_DTYPE_FMT = {
    "float64": ("d", 8), "float32": ("f", 4), "float16": ("e", 2),
    "int64": ("q", 8), "int32": ("i", 4), "int16": ("h", 2),
    "int8": ("b", 1),
    "uint64": ("Q", 8), "uint32": ("I", 4), "uint16": ("H", 2),
    "uint8": ("B", 1),
}


def _dtype_name(dtype):
    """Canonical numpy-style name for whatever dtype spelling arrived."""
    if dtype is None:
        return "float64"
    if dtype is int:
        return "int64"
    if dtype is float:
        return "float64"
    if dtype is bool:
        return "bool"
    if isinstance(dtype, str):
        return dtype
    return (getattr(dtype, "name", None)
            or getattr(dtype, "__name__", None) or "float64")


def _dtype_cast(v, name):
    """The value a numpy array of *name* would actually hold.

    Unknown names pass through untouched, so astype() to a dtype the core
    does not model stays the no-op it has always been.
    """
    if name not in _DTYPE_FMT:
        return v
    if isinstance(v, complex):
        v = v.real
    fmt, size = _DTYPE_FMT[name]
    if fmt == "d":
        return float(v)
    if fmt in ("f", "e"):
        # round-trip through the narrow width: float32/float16 arrays do
        # not keep the extra mantissa bits, and a later tobytes() would
        # silently drop them anyway.
        return _struct.unpack("<" + fmt, _struct.pack("<" + fmt,
                                                      float(v)))[0]
    iv = int(v)                       # numpy truncates toward zero
    bits = size * 8
    iv &= (1 << bits) - 1
    if fmt.islower() and iv >= (1 << (bits - 1)):
        iv -= 1 << bits               # signed wrap-around
    return iv


def _is_int_typed(a):
    dt = getattr(a, "_dt", None)
    return isinstance(dt, str) and (dt.startswith("int") or dt.startswith("uint"))


def _mask_positions(sel, axis_len):
    """Positions selected by ``sel`` if it is a boolean mask, else None.

    numpy tells a mask from an integer index array by dtype. This core is
    float-backed, so it uses the same convention the bare-index path has
    always used: an explicit ``_is_mask`` tag (set by comparisons), or a
    selector of exactly the axis length whose entries are all True/False
    -- Python bools, not the floats 0.0 and 1.0, which stay integer
    indices. ``_is_index`` always wins, so an index array is never
    mistaken for a mask.
    """
    if getattr(sel, "_is_index", False) or _is_int_typed(sel):
        return None
    vals = sel._flat() if isinstance(sel, marr) else (
        sel.tolist() if hasattr(sel, "tolist") and
        not isinstance(sel, (list, tuple)) else list(sel))
    if getattr(sel, "_is_mask", False):
        return [k for k, m in enumerate(vals) if m]
    if getattr(getattr(sel, "dtype", None), "kind", "") == "b":
        return [k for k, m in enumerate(vals) if m]
    if len(vals) == axis_len and vals and \
            _pyall(isinstance(v, bool) for v in vals):
        return [k for k, m in enumerate(vals) if m]
    return None


def _store(arr, value):
    """One stored element: int for an int-typed array when the value is
    integral; a non-integral value stored into an int-typed array turns
    the array float (numpy would truncate to int64 -- the modules were
    written against the float-backed core and never expected 2.5 -> 2,
    so silent truncation is the one numpy behaviour this core refuses)."""
    if isinstance(value, marr):
        f = value._flat()
        if len(f) == 1:
            value = f[0]
    if isinstance(value, complex):
        # complex assignment keeps the value (numpy would need a complex
        # dtype; the float-backed core carries complex elements as-is)
        return value
    if _is_int_typed(arr):
        v = float(value)
        if v.is_integer():
            return int(v)
        arr._dt = None
        return v
    return float(value)


def _carry(src, out):
    """Copy the mask / int tags from src onto a derived array."""
    if isinstance(out, marr):
        if getattr(src, "_is_mask", False):
            out._is_mask = True
        elif getattr(src, "_dt", None) is not None:
            out._dt = src._dt
            if _is_int_typed(src):
                f = out._flat()
                if _bi.all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in f):
                    _typed(out, int)
    return out


def _empty2d(nrow, ncol):
    """A (nrow, ncol) marr with a zero dimension; the constructor cannot
    infer ncol from empty data, so the shape is set directly."""
    out = marr([])
    out.data = [[] for _ in range(nrow)] if ncol == 0 else []
    out.shape = (nrow, ncol)
    return out


def _ix(v):
    """An index value: integral floats (what iterating a marr yields)
    are accepted where numpy accepts its integer scalars."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _slice_bound(v):
    """A slice bound: numpy accepts its integer scalars and 0-d arrays;
    here those arrive as integral floats or one-element marrs."""
    if v is None or isinstance(v, int):
        return v
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        raise TypeError("slice indices must be integers or None or have "
                        "an __index__ method")
    if isinstance(v, marr):
        f = v._flat()
        if len(f) == 1:
            return _slice_bound(f[0])
    if hasattr(v, "__index__"):
        return v.__index__()
    return v


def _norm_slice(s):
    if s.start is None and s.stop is None and s.step is None:
        return s
    return slice(_slice_bound(s.start), _slice_bound(s.stop),
                 _slice_bound(s.step))


class marr:
    """Minimal array: nested lists of floats, 1-D or 2-D."""

    __slots__ = ("data", "shape", "_is_mask", "_is_index", "_aif_keep",
                 "_dt", "_ix_outer")

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
            rows = [list(map(_num, (r.data if isinstance(r, marr) else r)))
                    for r in data]
            ncol = len(rows[0])
            if _bi.any(len(r) != ncol for r in rows):
                raise ValueError("ragged rows")
            self.data = rows
            self.shape = (len(rows), ncol)
            first = data[0].data[0] if isinstance(data[0], marr) else (
                data[0][0] if len(data[0]) else None)
            if isinstance(first, (int, bool)):    # gate: scan only if the
                raw = [v for r in data              # first element could be
                       for v in (r.data if isinstance(r, marr) else r)]
                if _bi.all(isinstance(v, int) and not isinstance(v, bool)
                           for v in raw):
                    self._dt = "int64"
                elif _bi.all(isinstance(v, bool) for v in raw):
                    self._is_mask = True
        else:
            self.data = [_num(v) for v in data]
            self.shape = (len(self.data),)
            if self.data and isinstance(data[0], (int, bool)) and \
                    _bi.all(isinstance(v, int) and not isinstance(v, bool)
                            for v in data):
                # numpy: int64. Needed so a 0/1-valued INDEX array such
                # as np.array([0]) is not read as a boolean mask when it
                # happens to have the axis length (grdetr's gt[cols]).
                self._dt = "int64"
            elif self.data and isinstance(data[0], bool) and \
                    _bi.all(isinstance(v, bool) for v in data):
                self._is_mask = True

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

    def diagonal(self, offset=0, axis1=0, axis2=1):
        del axis1, axis2
        return diag(self, k=offset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = _norm_slice(idx)
        if isinstance(idx, tuple):
            if any(isinstance(v, slice) for v in idx):
                idx = tuple(_norm_slice(v) if isinstance(v, slice) else v
                            for v in idx)
            if any(isinstance(v, tuple) for v in idx):
                # numpy reads x[:, (0, 2)] as x[:, [0, 2]]; an
                # itertools.combinations tuple is what callers pass here
                idx = tuple(list(v) if isinstance(v, tuple) else v
                            for v in idx)
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
            if len(idx) == 3:
                r3 = _newaxis_rank3(self, idx)
                if r3 is not None:
                    return r3
                gen = _newaxis_general(self, idx)
                if gen is not None:
                    return gen
                raise ValueError(
                    "unsupported 3-element index %r; the rank-2 core "
                    "supports new axes among slices and integer indices, "
                    "as in x[:, None, :] or x[:, None, None]" % (idx,))
            if len(idx) > 3 and None in idx:
                gen = _newaxis_general(self, idx)
                if gen is not None:
                    return gen
            i, j = idx
            if len(self.shape) == 2 and (i is None or j is None) and                     (i == slice(None) or j == slice(None)):
                # numpy: on a 2-D array x[:, None] is x[:, None, :] and
                # x[None, :] is x[None, :, :] (rank 3). Flattening here
                # made the pairwise idiom x[:, None] - x[None, :] return
                # an (nk, nk) matrix and the trailing-axis sum a vector.
                full = slice(None)
                return _newaxis_rank3(
                    self, (full, None, full) if j is None else (None, full, full))
            if j is None:                       # x[:, None] -> column
                if len(self.shape) == 2 and isinstance(i, slice):
                    return _newaxis_rank3(self[i], (slice(None), None,
                                                    slice(None)))
                if isinstance(i, slice):
                    return marr([[v] for v in self._flat()[i]])
                if isinstance(i, (int, float)):
                    return marr([self._flat()[int(i)]])
                raise ValueError("unsupported index")
            if i is None:                       # x[None, :] -> row
                return marr([self._flat()])
            if len(self.shape) == 2:
                if isinstance(i, (marr, list)) and \
                        isinstance(j, (marr, list)):
                    iv = [int(v) for v in
                          (i._flat() if isinstance(i, marr) else i)]
                    jv = [int(v) for v in
                          (j._flat() if isinstance(j, marr) else j)]
                    ix_pair = getattr(i, "_ix_outer", False) and \
                        getattr(j, "_ix_outer", False)
                    if not ix_pair and len(iv) == len(jv):
                        # numpy: paired integer arrays gather pointwise,
                        # x[[0, 1], [2, 3]] -> [x[0, 2], x[1, 3]]. The
                        # outer gather is only what np.ix_ asks for.
                        return marr([self.data[r2][c2]
                                     for r2, c2 in zip(iv, jv)])
                    return marr([[self.data[r2][c2] for c2 in jv]
                                 for r2 in iv])
                if isinstance(i, (marr, list)) or (
                        hasattr(i, "tolist") and not isinstance(i, slice)):
                    # array-of-rows selector: mask or integer indices
                    iv = i._flat() if isinstance(i, marr) else (
                        i.tolist() if hasattr(i, "tolist") else list(i))
                    pos = _mask_positions(i, self.shape[0])
                    if pos is not None:
                        rows = [self.data[k] for k in pos]
                    else:
                        rows = [self.data[int(v)] for v in iv]
                    if isinstance(j, slice):
                        return marr([r[j] for r in rows])
                    return marr([r[int(j)] for r in rows])
                if isinstance(i, slice) and isinstance(j, (marr, list)):
                    # column selector: x[:, idx] or x[:, mask]
                    jv = _mask_positions(j, self.shape[1])
                    if jv is None:
                        jv = [int(v) for v in
                              (j._flat() if isinstance(j, marr) else j)]
                    return marr([[r[c] for c in jv]
                                 for r in self.data[i]])
                if isinstance(i, slice) or isinstance(j, slice):
                    rows = self.data[i] if isinstance(i, slice) \
                        else [self.data[_ix(i)]]
                    picked = [(r[j] if isinstance(j, slice)
                               else [r[_ix(j)]]) for r in rows]
                    if isinstance(i, slice) and isinstance(j, slice):
                        return marr(picked)
                    if isinstance(i, slice):
                        return marr([row[0] for row in picked])
                    return marr(picked[0])
                if isinstance(j, (marr, list)) and isinstance(i, int):
                    # x[row, mask_or_index]
                    row = self.data[i]
                    jv = j._flat() if isinstance(j, marr) else list(j)
                    pos = _mask_positions(j, len(row))
                    if pos is None and not getattr(j, "_is_index", False) \
                            and len(jv) == len(row) \
                            and _pyall(v in (0.0, 1.0) for v in jv):
                        pos = [k for k, m2 in enumerate(jv) if m2]
                    if pos is not None:
                        return marr([row[k] for k in pos])
                    return marr([row[int(v)] for v in jv])
                return self.data[_ix(i)][_ix(j)]
            raise ValueError("unsupported index for 1-D")
        if isinstance(idx, marr):
            if len(idx.shape) == 2:
                same_shape = idx.shape == self.shape
                is_mask2 = getattr(idx, "_is_mask", False) or (
                    same_shape and not getattr(idx, "_is_index", False)
                    and not _is_int_typed(idx)
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
                and not _is_int_typed(idx)
                and idx.shape == (self.shape[0],)
                and _pyall(v in (0.0, 1.0) for v in vals))
            if is_mask:
                keep = [k for k, m in enumerate(vals) if m != 0]
            else:
                keep = [int(v) for v in vals]   # fancy integer indexing
            out = _carry(self, marr([self.data[k] for k in keep]))
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
        if isinstance(out, list) and isinstance(idx, slice):
            return _carry(self, marr(out)) if (out or len(self.shape) != 2) \
                else _empty2d(0, self.shape[1])
        if isinstance(out, list):
            if not out and len(self.shape) == 2:
                # W[:0] is (0, k) in numpy, not (0,): the deflation
                # idiom W[:j].T @ (W[:j] @ v) needs the k at j == 0
                return _empty2d(0, self.shape[1])
            return marr(out)
        return out

    @property
    def dtype(self):
        # the list-backed core is float64 unless astype/frombuffer tagged
        # it; masks surface as bool through __array__.
        name = getattr(self, "_dt", None)
        if name and name != "float64":
            return _DTypeNarrow(name)
        if getattr(self, "_is_mask", False):
            return _DType("bool")
        return float64

    @property
    def real(self):
        """Real part.  numpy exposes this as an attribute, not a method."""
        return real(self)

    @property
    def imag(self):
        """Imaginary part (attribute, matching numpy)."""
        return imag(self)

    def tobytes(self, order="C"):
        """Raw C-order bytes for the array's dtype, numpy's layout."""
        del order
        name = getattr(self, "_dt", None) or "float64"
        fmt = _DTYPE_FMT.get(name, ("d", 8))[0]
        vals = self._flat()
        if fmt in ("d", "f", "e"):
            vals = [float(v.real if isinstance(v, complex) else v)
                    for v in vals]
        else:
            vals = [int(v.real if isinstance(v, complex) else v)
                    for v in vals]
        return _struct.pack("<%d%s" % (len(vals), fmt), *vals)

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
        return _carry(self, marr(self._flat()))

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
        return _carry(self, out)

    def item(self, *idx):
        """numpy.ndarray.item: the single element as a Python scalar."""
        flat = self._flat()
        if idx:
            k = idx[0] if len(idx) == 1 else idx[0] * self.shape[1] + idx[1]
            return flat[int(k)]
        if len(flat) != 1:
            raise ValueError("can only convert an array of size 1 to a "
                             "Python scalar")
        return flat[0]

    def round(self, decimals=0):
        return round(self, decimals)

    def view(self, dtype=None):
        del dtype
        return self

    def conj(self):
        return self.copy()

    conjugate = conj

    def astype(self, dtype=None, copy=True):
        del copy
        name = _dtype_name(dtype)
        if _is_bool_dtype(dtype) or name == "bool":
            out = self._map(lambda v: 1.0 if v != 0 else 0.0)
            out._is_mask = True
            return out
        if name not in _DTYPE_FMT:
            return marr(self)
        out = self._map(lambda v: _dtype_cast(v, name))
        out._dt = None if name == "float64" else name
        return out

    def flatten(self):
        return _carry(self, marr(self._flat()))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        f = self._flat()
        if len(shape) == 1 and (shape[0] == -1 or shape[0] == len(f)):
            return marr(f)
        if len(shape) == 2:
            n, m = shape
            if n == -1:
                n = len(f) // m if m else 0
            if m == -1:
                m = len(f) // n if n else 0
            if n * m != len(f):
                raise ValueError("cannot reshape")
            if n == 0 or m == 0:
                return _empty2d(n, m)
            return _carry(self, marr([f[i * m:(i + 1) * m]
                                      for i in range(n)]))
        if len(shape) >= 3:
            dims = [int(d) for d in shape]
            if dims.count(-1) == 1:
                known = 1
                for d in dims:
                    if d != -1:
                        known *= d
                dims[dims.index(-1)] = len(f) // known if known else 0
            total = 1
            for d in dims:
                total *= d
            if total != len(f):
                raise ValueError("cannot reshape %d values into %r"
                                 % (len(f), tuple(dims)))

            def build(vals, ds):
                if len(ds) == 2:
                    return marr([vals[i * ds[1]:(i + 1) * ds[1]]
                                 for i in range(ds[0])])
                step = len(vals) // ds[0] if ds[0] else 0
                return [build(vals[i * step:(i + 1) * step], ds[1:])
                        for i in range(ds[0])]
            return ndlist(build(list(f), dims))
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
        if isinstance(f[0], complex):
            # the same error Python raises for float(1j), not the
            # interpreter's "__float__ returned non-float" protocol message
            raise TypeError("can't convert complex to float")
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

    def __xor__(self, o):
        out = self._zip(o, lambda a, b: 1.0 if ((a != 0) != (b != 0))
                        else 0.0)
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
        idx = _ix(idx)
        if isinstance(idx, slice):
            idx = _norm_slice(idx)
        elif isinstance(idx, tuple) and any(isinstance(v, slice) for v in idx):
            idx = tuple(_norm_slice(v) if isinstance(v, slice) else v for v in idx)
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
            if isinstance(i, (marr, list)) and isinstance(j, (marr, list)) \
                    and not isinstance(i, slice):
                # paired integer index arrays, as returned by
                # diag_indices_from: x[(rows, cols)] = v
                iv = list(i._flat()) if isinstance(i, marr) else list(i)
                jv = list(j._flat()) if isinstance(j, marr) else list(j)
                if getattr(i, "_ix_outer", False) and \
                        getattr(j, "_ix_outer", False):
                    # np.ix_ pair: outer block assignment
                    v = asarray(value)
                    if v.size == 1:
                        block = [[float(v._flat()[0])] * len(jv)
                                 for _ in iv]
                    else:
                        block = atleast_2d(v).tolist()
                    for a, r2 in enumerate(iv):
                        for b, c2 in enumerate(jv):
                            self.data[int(r2)][int(c2)] = float(block[a][b])
                    return
                if len(iv) != len(jv):
                    raise ValueError(
                        "index arrays must be the same length, got %d "
                        "and %d" % (len(iv), len(jv)))
                v = asarray(value)
                vals = list(v._flat()) if isinstance(v, marr) \
                    else [float(value)]
                if len(vals) == 1:
                    vals = vals * len(iv)
                if len(vals) != len(iv):
                    raise ValueError(
                        "cannot assign %d values to %d positions"
                        % (len(vals), len(iv)))
                for r2, c2, val in zip(iv, jv, vals):
                    self.data[int(r2)][int(c2)] = float(val)
                return
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
            if not isinstance(i, slice) \
                    and isinstance(j, (marr, list, tuple)):
                # one row, many columns: W[i, nb] = <vector>. Without
                # this the tuple path fell through to the scalar store
                # and float() rejected the vector right-hand side.
                r2 = int(i)
                raw = [v for v in (j._flat() if isinstance(j, marr) else j)]
                if len(raw) == self.shape[1] \
                        and _bi.all(isinstance(v, bool) for v in raw):
                    cols = [c for c, fl in enumerate(raw) if fl]
                else:
                    cols = [int(v) for v in raw]
                va = asarray(value)
                vals = list(va._flat())
                if len(vals) == 1:
                    vals = vals * len(cols)
                if len(vals) != len(cols):
                    raise ValueError(
                        "cannot assign %d values to %d columns"
                        % (len(vals), len(cols)))
                for c2, v2 in zip(cols, vals):
                    self.data[r2][c2] = float(v2)
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
            self.data[_ix(i)][_ix(j)] = _store(self, value)
        elif isinstance(idx, slice):
            vals = asarray(value)._flat()
            rng = range(*idx.indices(self.shape[0]))
            if len(self.shape) == 2:
                m2 = self.shape[1]
                if len(vals) == 1:
                    vals = vals * (len(rng) * m2)
                elif len(vals) == m2 and len(rng) > 1:
                    vals = vals * len(rng)
                if len(vals) != len(rng) * m2:
                    raise ValueError(
                        "cannot assign %d values to a (%d, %d) slice"
                        % (len(vals), len(rng), m2))
                for r2, k in enumerate(rng):
                    self.data[k] = [float(x)
                                    for x in vals[r2 * m2:(r2 + 1) * m2]]
                return
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
            if is_mask and len(self.shape) == 2 \
                    and len(raw) == self.shape[0] * self.shape[1]:
                # 2-D boolean mask, elementwise (numpy semantics)
                vals = list(asarray(value)._flat()) \
                    if not isinstance(value, (int, float)) else None
                k = 0
                for r in range(self.shape[0]):
                    for c in range(self.shape[1]):
                        if raw[r * self.shape[1] + c]:
                            self.data[r][c] = float(value) \
                                if vals is None \
                                else float(vals[k if len(vals) > 1 else 0])
                            k += 1
                return
            if is_mask:
                ids = [k for k, b in enumerate(raw) if b]
            else:
                ids = [int(v) for v in raw]
            if len(self.shape) == 2:
                # row selection on a 2-D array: assign whole rows,
                # broadcasting a scalar, a length-m row, or a
                # (len(ids), m) / (len(ids), 1) block (numpy
                # semantics). The old code wrote a scalar float into
                # the row slot, silently corrupting the matrix.
                m2 = self.shape[1]
                if isinstance(value, (int, float)):
                    for r in ids:
                        self.data[r] = [float(value)] * m2
                    return
                v2 = _b2(asarray(value))
                if v2.shape[0] not in (1, len(ids)) \
                        or v2.shape[1] not in (1, m2):
                    raise ValueError(
                        "cannot broadcast %s to (%d, %d) selected rows"
                        % (str(v2.shape), len(ids), m2))
                for k, r in enumerate(ids):
                    src_row = v2.data[k if v2.shape[0] > 1 else 0]
                    self.data[r] = [
                        float(src_row[c if v2.shape[1] > 1 else 0])
                        for c in range(m2)]
                return
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
            self.data[idx] = _store(self, value)

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
        return self._zip(o, _ieee_div)

    def __rtruediv__(self, o):
        return self._zip(o, lambda a, b: _ieee_div(b, a))

    def __pow__(self, o):
        return self._zip(o, lambda a, b: a ** b)

    def __rpow__(self, o):
        return self._zip(o, lambda a, b: b ** a)

    def __mod__(self, o):
        return self._zip(o, _ieee_mod)

    def __rmod__(self, o):
        return self._zip(o, lambda a, b: _ieee_mod(b, a))

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
        if type(o).__name__.startswith("Approx"):
            # pytest.approx: hand it plain Python values, as numpy arrays
            # do -- the scalar for approx(x), the list for approx([..]).
            # approx(<marr>) keeps pytest's own reflected comparison.
            if isinstance(getattr(o, "expected", None), marr):
                return NotImplemented
            if type(o).__name__ in ("ApproxScalar", "ApproxDecimal") \
                    and self.size == 1:
                return o == self._flat()[0]
            return o == self.tolist()
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

    def argmax(self, axis=None):
        if axis is not None:
            return argmax(self, axis=axis)
        return _nan_argext(self._flat(), _bi.max)

    def argmin(self, axis=None):
        if axis is not None:
            return argmin(self, axis=axis)
        return _nan_argext(self._flat(), _bi.min)

    # -- reductions ----------------------------------------------------
    def _kd_all(self, v):
        """keepdims form of a full reduction: (1, 1) for 2-D, (1,) for 1-D."""
        return marr([[v]]) if len(self.shape) == 2 else marr([v])

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        del dtype, out
        # numpy: a boolean array sums to an integer count, and callers
        # feed that count straight to range() / indexing
        count = getattr(self, "_is_mask", False) or _is_int_typed(self)

        def fin(v):
            return int(round(v)) if count else float(v)

        def fin_arr(a):
            return _typed(a, int) if count else a
        if axis is None:
            v = fin(_fsum(self._flat()))
            return fin_arr(self._kd_all(v)) if keepdims else v
        if len(self.shape) != 2:
            # numpy: axis 0 / -1 on a 1-D array is the full reduction
            v = fin(_fsum(self._flat()))
            return fin_arr(marr([v])) if keepdims else v
        if axis == 0:
            out = [fin(_fsum(self.data[i][j]
                             for i in range(self.shape[0])))
                   for j in range(self.shape[1])]
            return fin_arr(marr([out])) if keepdims else fin_arr(marr(out))
        out = [fin(_fsum(row)) for row in self.data]
        return fin_arr(marr([[v] for v in out])) if keepdims \
            else fin_arr(marr(out))

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        del dtype, out
        if axis is None or len(self.shape) != 2:
            f = self._flat()
            if not f:
                _warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
                return self._kd_all(_NAN) if keepdims else _NAN
            v = float(_fsum(f) / len(f))
            return self._kd_all(v) if keepdims else v
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
        if keepdims:
            return self._kd_all(self.var(axis=axis, ddof=ddof))
        if axis is not None and len(self.shape) == 2:
            m = self.mean(axis=axis)
            if axis == 0:
                return marr([_fsum(
                    (self.data[i][j] - m.data[j]) ** 2
                    for i in range(self.shape[0]))
                    / (self.shape[0] - ddof)
                    for j in range(self.shape[1])])
            return marr([_fsum((v - m.data[i]) ** 2
                                    for v in row)
                         / (self.shape[1] - ddof)
                         for i, row in enumerate(self.data)])
        f = self._flat()
        if len(f) - ddof <= 0:
            _warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                           stacklevel=2)
            return _NAN
        m = _fsum(f) / len(f)
        return float(_fsum((v - m) ** 2 for v in f) / (len(f) - ddof))

    def std(self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=False):
        del dtype, out
        if keepdims and axis is not None and len(self.shape) == 2:
            v = self.std(axis=axis, ddof=ddof)
            return marr([v.tolist()]) if axis in (0, -2) else \
                marr([[x] for x in v._flat()])
        if keepdims:
            return self._kd_all(self.std(axis=axis, ddof=ddof))
        v = self.var(axis=axis, ddof=ddof)
        if isinstance(v, marr):
            return marr([_math.sqrt(u) for u in v._flat()])
        return _math.sqrt(v)

    def max(self, axis=None, out=None, keepdims=False, initial=None):
        del out
        if keepdims and (axis is None or len(self.shape) != 2):
            return self._kd_all(self.max(axis=axis, initial=initial))
        if initial is not None and axis is None:
            f = self._flat()
            return _nan_ext([float(initial)] + f, _bi.max)
        if axis is not None and len(self.shape) == 2:
            if axis in (0, -2):
                out2 = [_nan_ext((r[c] for r in self.data), _bi.max)
                        for c in range(self.shape[1])]
                return marr([out2]) if keepdims else marr(out2)
            out2 = [_nan_ext(r, _bi.max) for r in self.data]
            return marr([[v] for v in out2]) if keepdims \
                else marr(out2)
        return float(_nan_ext(self._flat(), _bi.max))

    def min(self, axis=None, out=None, keepdims=False, initial=None):
        del out
        if keepdims and (axis is None or len(self.shape) != 2):
            return self._kd_all(self.min(axis=axis, initial=initial))
        if initial is not None and axis is None:
            f = self._flat()
            return _nan_ext([float(initial)] + f, _bi.min)
        if axis is not None and len(self.shape) == 2:
            if axis in (0, -2):
                out2 = [_nan_ext((r[c] for r in self.data), _bi.min)
                        for c in range(self.shape[1])]
                return marr([out2]) if keepdims else marr(out2)
            out2 = [_nan_ext(r, _bi.min) for r in self.data]
            return marr([[v] for v in out2]) if keepdims \
                else marr(out2)
        return float(_nan_ext(self._flat(), _bi.min))

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
        if 0 in self.shape:
            return _empty2d(self.shape[1], self.shape[0])
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

    @property
    def data(self):
        # marr exposes .data; object arrays were the one kind without it
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
    if isinstance(x, (list, tuple)) and x and _bi.any(
            isinstance(v, str) for v in x):
        return True
    return False


def _nested_depth(x, cap=8):
    """How deeply a sequence nests, counting only list/tuple levels."""
    d = 0
    while isinstance(x, (list, tuple)) and x and d < cap:
        d += 1
        x = x[0]
    return d


def _all_bool_payload(x):
    """True when ``x`` is a (possibly nested) sequence of Python bools."""
    if isinstance(x, bool):
        return True
    if isinstance(x, (list, tuple)):
        return bool(x) and _pyall(_all_bool_payload(v) for v in x)
    return False


def asarray(x, dtype=None):
    if isinstance(x, (list, tuple)) and x and _bi.all(
            isinstance(v, (marr, oarr)) and len(getattr(v, "shape", ())) == 2
            for v in x):
        # numpy stacks a list of equal-shaped matrices into one array
        return ndlist([v.tolist() for v in x])
    if isinstance(x, oarr) and dtype is None:
        return x
    if isinstance(x, ndlist):
        # rank >= 3 container: it already holds floats in nested lists.
        # Without this passthrough marr(x) raises on the inner lists and
        # the except branch below flattens it to a 1-D oarr, silently
        # destroying the shape (this is what broke the covariate route
        # of morie.fn.boryis).
        return x
    if hasattr(x, "columns") and hasattr(x, "to_numpy") \
            and not isinstance(x, marr):
        # frame-like (native or real pandas): take its array form
        x = x.to_numpy()
    if _is_object_like(x, dtype):
        return oarr(x.tolist() if hasattr(x, "tolist") else x)
    if isinstance(x, marr):
        return x
    # A rank-3 or deeper nested list has to become an ndlist. marr is the
    # rank-2 core and raises on the inner lists, and the fallback below
    # then makes an object array of ndim 1 -- so every module taking a
    # 3-D table (a conditional probability table, say) saw ndim 1 and
    # rejected its own documented input. ndlist itself was already here;
    # nothing ever built one from a plain nested list.
    if isinstance(x, (list, tuple)) and _nested_depth(x) >= 3:
        return ndlist(x)
    try:
        out = marr(x)
    except (TypeError, ValueError):
        # non-numeric payload (strings via an untyped container):
        # numpy would build an object array here
        return oarr(x.tolist() if hasattr(x, "tolist") else x)
    else:
        # numpy keeps dtype=bool for a list of bools, and indexing
        # depends on it: x[:, mask] must select the True columns, not the
        # columns numbered by True and False. This core is float-backed,
        # so the boolean-ness has to be carried as a tag or it is lost at
        # construction and the mask silently becomes an integer index.
        if dtype is None and _all_bool_payload(x):
            out._is_mask = True
        elif dtype is not None:
            _typed(out, dtype)
        return out


def array(x, dtype=None, copy=True, ndmin=0):
    del copy
    if isinstance(x, (list, tuple)) and x and _bi.all(
            isinstance(v, marr) and len(v.shape) == 2 for v in x):
        return ndlist([v.tolist() for v in x])
    if ndmin >= 2 and _nested_depth(x) < 2 and not isinstance(x, marr):
        return atleast_2d(array(x, dtype))
    if _is_object_like(x, dtype):
        return oarr(x.tolist() if hasattr(x, "tolist") else x)
    if isinstance(x, (list, tuple)) and _nested_depth(x) >= 3:
        return ndlist(x)
    out = marr(x)
    if dtype is not None:
        _typed(out, dtype)
    return out


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
             and _bi.all(isinstance(v, int) and not isinstance(v, bool)
                     for v in (start, stop, step)))
    if dtype is not None:
        exact = dtype in (int, "int", "int64", "int32", "i8", "i4")
    n = _bi.max(0, int(_math.ceil((stop - start) / step - 1e-12)))
    if exact:
        return marr([int(start) + i * int(step) for i in range(n)])
    return marr([float(start) + i * float(step) for i in range(n)])


def _nd_filled(shape, value):
    """Nested lists of `value` with the given shape, for rank >= 3.

    zeros((4, 2, 8)) used to build a (4, 2) marr and DISCARD the last
    dimension, so a rank-3 allocation came back silently the wrong
    shape. Rank 3 and above needs the n-D container.
    """
    def build(dims):
        if len(dims) == 1:
            return [value] * int(dims[0])
        return [build(dims[1:]) for _ in range(int(dims[0]))]
    return ndlist(build(list(shape)))


def _is_int_dtype(dtype):
    name = _dtype_name(dtype) if dtype is not None else ""
    return isinstance(name, str) and (name.startswith("int")
                                      or name.startswith("uint"))


def _is_bool_dtype(dtype):
    return dtype is bool or (isinstance(dtype, str) and dtype.startswith("bool")) \
        or getattr(dtype, "kind", "") == "b"


def _typed(out, dtype):
    """Apply an int / bool dtype request to a freshly built array."""
    if isinstance(out, marr):
        if _is_bool_dtype(dtype):
            out._is_mask = True
            out._dt = None
        elif _is_int_dtype(dtype):
            out._dt = "int64"
            out.data = ([[int(v) for v in r] for r in out.data]
                        if len(out.shape) == 2 else [int(v) for v in out.data])
    return out


def zeros(n, dtype=None):
    return _typed(_zeros(n), dtype)


def _zeros(n, dtype=None):
    if isinstance(n, (tuple, list)) and len(n) == 1:
        n = n[0]
    if isinstance(n, (tuple, list)):
        if len(n) > 2:
            return _nd_filled(n, 0.0)
        out = marr([[0.0] * n[1] for _ in range(n[0])]) \
            if n[0] else marr([])
        if not n[0]:
            out.shape = (0, int(n[1]))
        return out
    return marr([0.0] * int(n))


def ones(n, dtype=None):
    return _typed(_ones(n), dtype)


def _ones(n, dtype=None):
    if isinstance(n, (tuple, list)) and len(n) == 1:
        n = n[0]
    if isinstance(n, (tuple, list)):
        if len(n) > 2:
            return _nd_filled(n, 1.0)
        return marr([[1.0] * n[1] for _ in range(n[0])])
    return marr([1.0] * int(n))


def full(n, v, dtype=None):
    if dtype is None and isinstance(v, int) and not isinstance(v, bool):
        dtype = int                     # numpy: full(3, -1) is int64
    if dtype is None and isinstance(v, bool):
        dtype = bool
    if isinstance(n, (tuple, list)):
        if len(n) == 2:
            out = marr([[float(v)] * int(n[1])
                        for _ in range(int(n[0]))])
            return _typed(out, dtype)
        n = n[0]
    return _typed(marr([float(v)] * int(n)), dtype)


def linspace(a, b, n=50, endpoint=True, retstep=False, dtype=None):
    if dtype is not None and dtype is not float and not _is_float_dtype(dtype):
        out = linspace(a, b, n, endpoint, retstep)
        if retstep:
            return _typed(marr([float(int(v)) for v in out[0]._flat()]), int), out[1]
        return _typed(marr([float(int(v)) for v in out._flat()]), int)
    n = int(n)
    if n <= 0:
        out, step = marr([]), _NAN
    elif n == 1:
        out, step = marr([float(a)]), float(b - a)
    else:
        step = (b - a) / ((n - 1) if endpoint else n)
        out = marr([a + i * step for i in range(n)])
    return (out, step) if retstep else out


def eye(n, m=None, k=0, dtype=None):
    del dtype
    m = int(n) if m is None else int(m)
    k = int(k)
    return marr([[1.0 if j - i == k else 0.0 for j in range(m)]
                 for i in range(int(n))])


def diagonal(a, offset=0, axis1=0, axis2=1):
    del axis1, axis2
    return diag(atleast_2d(asarray(a)), k=offset)


def diag(x, k=0):
    a = asarray(x)
    k = int(k)
    if len(a.shape) == 1:
        n = a.shape[0] + abs(k)
        return marr([[a.data[_bi.min(i, j)] if j - i == k else 0.0
                      for j in range(n)] for i in range(n)])
    r, c = a.shape
    return marr([a.data[i][i + k] for i in range(r)
                 if 0 <= i + k < c])


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
        res = marr(out)
        if _bi.all(getattr(a, "_is_mask", False) for a in arrs):
            res._is_mask = True
        elif _bi.all(_is_int_typed(a) for a in arrs):
            _typed(res, int)
        return res
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
        if isinstance(x, complex):
            return fn(x)
        if isinstance(x, ndlist):
            return ndlist(wrapped(marr(b)) for b in x)
        if isinstance(x, carr) and x.rows is not None:
            rows = [[fn(v) for v in r] for r in x.rows]
            if _bi.any(isinstance(v, complex) for r in rows for v in r):
                return carr(rows)
            return marr([[float(v) for v in r] for r in rows])
        if isinstance(x, carr) or (
                isinstance(x, (list, tuple))
                and x and isinstance(x[0], complex)):
            vals = [fn(v) for v in x]
            if _bi.any(isinstance(v, complex) for v in vals):
                return carr(vals)
            return marr([float(v) for v in vals])
        a = asarray(x)
        if a.shape == (1,) and not isinstance(x, (list, tuple, marr)):
            return fn(a.data[0])
        return a._map(fn)
    return wrapped


def _ieee_exp(v):
    """numpy's exp: overflow saturates to +inf, it does not raise.

    math.exp raises OverflowError instead, which turned an ordinary
    saturating expression -- exp(-exp(x)) in a Kaplan-Meier log-log
    interval -- into a hard failure.
    """
    # Float path first: an isinstance() check on every element roughly
    # doubles the per-element cost, and complex input is the rare case.
    try:
        return _math.exp(v)
    except TypeError:
        return _cmath.exp(v)
    except OverflowError:
        return _INF if v > 0 else 0.0


def _ieee_log(v):
    """numpy's log: log(0) is -inf and log(x < 0) is nan, not an error."""
    try:
        if v > 0:
            return _math.log(v)
        if v == 0:
            return -_INF
        return _NAN
    except TypeError:
        return _cmath.log(v) if v != 0 else complex(-_INF, 0.0)


def _ieee_log1p(v):
    """numpy's log1p: log1p(-1) is -inf and log1p(x < -1) is nan."""
    try:
        if v > -1.0:
            return _math.log1p(v)
        if v == -1.0:
            return -_INF
        return _NAN
    except TypeError:
        return (_cmath.log(1.0 + v) if (1.0 + v) != 0
                else complex(-_INF, 0.0))


def _ieee_sqrt(v):
    """numpy's sqrt on a real negative is nan, not an error."""
    try:
        if v < 0:
            return _NAN
        return _math.sqrt(v)
    except TypeError:
        return _cmath.sqrt(v)


sqrt = _uf(_ieee_sqrt)
exp = _uf(_ieee_exp)
log = _uf(_ieee_log)
log1p = _uf(_ieee_log1p)
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
        return _carry(a, a._map(one))
    if isinstance(x, ndlist) or isinstance(a, ndlist):
        # n-D: recurse and keep the nesting, as numpy.clip does. The
        # scalar store below cannot represent a 3-D right-hand side.
        def walk(v):
            if isinstance(v, (list, tuple, marr)):
                return [walk(u) for u in v]
            return one(float(v))
        return ndlist(walk(list(x)))
    return one(float(a))


class AxisError(ValueError, IndexError):
    """numpy.exceptions.AxisError: an axis that the array does not have."""


def _check_axis(a, axis):
    if axis is None:
        return
    if isinstance(axis, tuple):
        for ax in axis:
            _check_axis(a, ax)
        return
    nd = len(a.shape)
    if not isinstance(axis, int) or axis >= nd or axis < -nd:
        raise AxisError(f"axis {axis} is out of bounds for array of dimension {nd}")


def _nan_ext(it, ext):
    """numpy max/min: a NaN anywhere in the input is the answer.

    builtins.max/min compare pairwise and every comparison with NaN is
    False, so the result depended on where the NaN sat: max([1, nan, 3])
    was 3 and max([nan, 1, 3]) was nan. Same data, permuted, different
    answer, at 2,865 + 633 call sites.
    """
    vals = list(it)
    if not vals:
        raise ValueError(
            "zero-size array to reduction operation which has no identity")
    for v in vals:
        if v != v:
            return _NAN
    return ext(vals)


def _nan_argext(f, ext):
    """numpy argmax/argmin: the first NaN wins, else the first extremum."""
    f = list(f)
    if not f:
        raise ValueError("attempt to get argmax of an empty sequence")
    for i, v in enumerate(f):
        if v != v:
            return i
    return f.index(ext(f))


def _nan_sorted(f, reverse=False):
    """numpy sort: NaN sorts last; Python's sorted() gives no order at all
    with a NaN present (sort([3, nan, 1, 2]) came back unsorted)."""
    out = sorted((v for v in f if v == v), reverse=reverse)
    return out + [v for v in f if v != v]


def _nan_argsorted(f):
    idx = sorted((i for i, v in enumerate(f) if v == v), key=lambda k: f[k])
    return idx + [i for i, v in enumerate(f) if v != v]


def _ieee(fn, even=False):
    """numpy semantics for a math.* function: a domain error is nan, an
    overflow is inf (signed like the input, or +inf for an even function
    such as cosh), and nan/inf inputs pass through instead of raising."""
    def one(v):
        try:
            return fn(v)
        except ValueError:
            return _NAN
        except OverflowError:
            return _INF if (even or v > 0) else -_INF
    return one


def _ieee_log2(v):
    """numpy's log2: log2(0) is -inf and log2(x < 0) is nan."""
    if v > 0:
        return _math.log2(v)
    return -_INF if v == 0 else _NAN


def _ieee_log10(v):
    if v > 0:
        return _math.log10(v)
    return -_INF if v == 0 else _NAN


def _ieee_div(x, y):
    """numpy true_divide: x/0 is +-inf (nan for 0/0), not ZeroDivisionError."""
    try:
        return x / y
    except ZeroDivisionError:
        if x != x or x == 0:
            return _NAN
        same = (x > 0) == (_math.copysign(1.0, y) > 0)
        return _INF if same else -_INF


def _ieee_mod(x, y):
    """numpy mod: x % 0 is nan, not ZeroDivisionError."""
    try:
        return x % y
    except ZeroDivisionError:
        return _NAN


def _fsum(it):
    """math.fsum with numpy's answers for the two cases it raises on:
    inf + -inf is nan, and an intermediate overflow is inf."""
    try:
        return _math.fsum(it)
    except ValueError:
        return _NAN
    except OverflowError:
        return _INF


def _max2(a, b):
    """numpy.maximum: NaN propagates from EITHER operand."""
    if a != a or b != b:
        return _NAN
    return a if a >= b else b


def _min2(a, b):
    """numpy.minimum: NaN propagates from EITHER operand."""
    if a != a or b != b:
        return _NAN
    return a if a <= b else b


def _maximum_fn(x, y):
    # The old lambda was `a if a >= b else b`. Every comparison with NaN is
    # False, so maximum(nan, 0) fell through to `b` and returned 0, while
    # maximum(0, nan) returned nan -- the answer depended on argument ORDER,
    # and an undefined value silently became a number. numpy and R both
    # propagate. splfun computes sqrt(max(K, 0)/pi), so an undefined K came
    # out as L = -r in Python and NA in R: a real cross-language
    # disagreement that simply had not been hit yet.
    return asarray(x)._zip(y, _max2)


def _minimum_fn(x, y):
    return asarray(x)._zip(y, _min2)


class _PairUfunc:
    """numpy.maximum / numpy.minimum with the ufunc methods the modules
    use: accumulate (running max/min along an axis) and reduce."""

    def __init__(self, fn, red):
        self._fn = fn
        self._red = red

    def __call__(self, x, y):
        return self._fn(x, y)

    def accumulate(self, a, axis=0):
        arr = asarray(a)
        if len(arr.shape) == 1:
            out, cur = [], None
            for v in arr.data:
                cur = v if cur is None else self._red(cur, v)
                out.append(cur)
            return marr(out)
        if axis in (0, -2):
            rows, cur = [], None
            for row in arr.data:
                cur = row[:] if cur is None else \
                    [self._red(c, v) for c, v in zip(cur, row)]
                rows.append(cur[:])
            return marr(rows)
        return marr([self.accumulate(marr(row)).data for row in arr.data])

    def reduce(self, a, axis=None):
        arr = asarray(a)
        if axis is None:
            return self._red(arr._flat())
        return arr.max(axis=axis) if self._red is _bi.max \
            else arr.min(axis=axis)


maximum = _PairUfunc(_maximum_fn, _bi.max)
minimum = _PairUfunc(_minimum_fn, _bi.min)


def mod(x, y):
    # Elementwise modulo (Python semantics, sign of the divisor), added
    # together with equal(): hmvilb needs np.mod for its integer check.
    return asarray(x)._zip(y, _ieee_mod)


def equal(x, y):
    # Elementwise equality as 0.0/1.0, matching the mask convention of
    # the other comparators. Added because hmvilb calls np.equal, which
    # did not exist: the 1-D token-id input path was uncallable.
    out = asarray(x)._zip(y, lambda a, b: 1.0 if a == b else 0.0)
    try:
        out._is_mask = True
    except AttributeError:
        pass
    return out


def where(cond, a=None, b=None):
    if a is not None and isinstance(cond, (bool, int, float)) \
            and not isinstance(a, (list, tuple, marr, ndlist)) \
            and not isinstance(b, (list, tuple, marr, ndlist)):
        # all-scalar call: a scalar back (numpy gives a 0-d array, which
        # behaves as one); a kernel evaluated at one point must compare
        # equal to a float
        return _num(a) if cond else _num(b)
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
    if len(c.shape) == 2 or len(asarray(a).shape) == 2 \
            or len(asarray(b).shape) == 2:
        # Broadcast the CONDITION too, not just the two value arrays.
        # It used to iterate over the condition's own shape, so a (1, d)
        # mask against (n, d) values returned a single row -- and callers
        # that then combined it with an (n,) array got that one row
        # spread silently over every sample. fn/copod.py scored all 200
        # points identically because of exactly this.
        cb = _b2(c)
        ab = _b2(asarray(a))
        bb2 = _b2(asarray(b))
        rows = _bi.max(cb.shape[0], ab.shape[0], bb2.shape[0])
        cols = _bi.max(cb.shape[1], ab.shape[1], bb2.shape[1])
        for src, name in ((cb, "cond"), (ab, "a"), (bb2, "b")):
            for got, want, what in ((src.shape[0], rows, "rows"),
                                    (src.shape[1], cols, "columns")):
                if got not in (1, want):
                    raise ValueError(
                        "where: %s has %d %s, cannot broadcast to %d"
                        % (name, got, what, want))

        def pick(src, r, cc):
            row = src.data[r if src.shape[0] > 1 else 0]
            return row[cc if src.shape[1] > 1 else 0]
        return marr([[pick(ab, r, cc) if pick(cb, r, cc)
                      else pick(bb2, r, cc)
                      for cc in range(cols)]
                     for r in range(rows)])
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
    if isinstance(a, carr) or isinstance(b, carr):
        fa = list(a.data) if isinstance(a, carr) else list(asarray(a)._flat())
        fb = list(b.data) if isinstance(b, carr) else list(asarray(b)._flat())
        if len(fa) != len(fb):
            raise ValueError("shape mismatch")
        return _bi.sum(x * y for x, y in zip(fa, fb))
    aa, bb = asarray(a), asarray(b)
    if len(aa.shape) == 1 and len(bb.shape) == 1:
        if aa.shape != bb.shape:
            raise ValueError("shape mismatch")
        if _bi.any(isinstance(v, complex) for v in aa.data) \
                or _bi.any(isinstance(v, complex) for v in bb.data):
            return _bi.sum(x * y for x, y in zip(aa.data, bb.data))
        return float(_fsum(x * y for x, y in zip(aa.data, bb.data)))
    return matmul(aa, bb)


def matmul(a, b):
    a_arr = asarray(a)
    a_was_1d = len(a_arr.shape) == 1
    aa = atleast_2d(a_arr)
    b_arr = asarray(b)
    b_was_1d = len(b_arr.shape) == 1
    if b_was_1d:
        bb = marr([[v] for v in b_arr.data]) if b_arr.size else \
            _empty2d(0, 1)
    else:
        bb = b_arr
    if len(aa.shape) == 2 and aa.shape[0] == 1 and a_was_1d \
            and a_arr.size == 0:
        aa = _empty2d(1, 0)
    n, k = aa.shape
    k2, m = bb.shape
    if n == 0 or m == 0:
        # (0, k) @ (k, m) -> (0, m); numpy returns empty, callers index
        # nothing from it
        return marr([]) if (b_was_1d or a_was_1d) else _empty2d(n, m)
    if k != k2:
        raise ValueError("shape mismatch")
    out = [[_fsum(aa.data[i][t] * bb.data[t][j] for t in range(k))
            for j in range(m)] for i in range(n)]
    if b_was_1d and a_was_1d:
        return out[0][0]
    if b_was_1d:
        return marr([row[0] for row in out])
    if a_was_1d:
        return marr(out[0])
    return marr(out)



def _axis_arg(x, axis):
    """A tuple ``axis`` on a rank-1/2 array: all axes is None, one axis
    is that axis; ndlist keeps the tuple (it reduces several axes)."""
    if not isinstance(axis, tuple) or isinstance(x, ndlist):
        return axis
    nd = len(asarray(x).shape)
    axes = sorted({ax % nd if -nd <= ax < nd else ax for ax in axis})
    if any(ax < 0 or ax >= nd for ax in axes):
        raise AxisError("axis %r is out of bounds for array of dimension %d" % (axis, nd))
    if len(axes) == nd:
        return None
    if len(axes) == 1:
        return axes[0]
    raise AxisError("axis %r is not supported on a rank-%d array" % (axis, nd))


# --------------------------------------------------------------- reductions

def sum(x, axis=None, dtype=None, keepdims=False):  # noqa: A001
    axis = _axis_arg(x, axis)
    _check_axis(asarray(x), axis)
    del dtype
    if isinstance(x, ndlist):
        return x.sum(axis=axis, keepdims=keepdims)
    return asarray(x).sum(axis=axis, keepdims=keepdims)


def mean(x, axis=None, dtype=None, keepdims=False):
    axis = _axis_arg(x, axis)
    _check_axis(asarray(x), axis)
    del dtype
    if isinstance(x, ndlist):
        return x.mean(axis=axis, keepdims=keepdims)
    return asarray(x).mean(axis=axis, keepdims=keepdims)


def std(x, axis=None, ddof=0, dtype=None, keepdims=False):
    axis = _axis_arg(x, axis)
    _check_axis(asarray(x), axis)
    del dtype
    if isinstance(x, ndlist):
        return x.std(axis=axis, ddof=ddof, keepdims=keepdims)
    return asarray(x).std(axis=axis, ddof=ddof, keepdims=keepdims)


def var(x, axis=None, ddof=0, dtype=None, keepdims=False):
    axis = _axis_arg(x, axis)
    _check_axis(asarray(x), axis)
    del dtype
    if isinstance(x, ndlist):
        return x.var(axis=axis, ddof=ddof, keepdims=keepdims)
    return asarray(x).var(axis=axis, ddof=ddof, keepdims=keepdims)


def max(x, axis=None, keepdims=False):  # noqa: A001
    axis = _axis_arg(x, axis)
    _check_axis(asarray(x), axis)
    if isinstance(x, ndlist):
        return x.max(axis=axis, keepdims=keepdims)
    return asarray(x).max(axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):  # noqa: A001
    axis = _axis_arg(x, axis)
    _check_axis(asarray(x), axis)
    if isinstance(x, ndlist):
        return x.min(axis=axis, keepdims=keepdims)
    return asarray(x).min(axis=axis, keepdims=keepdims)


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
        if isinstance(x, ndlist):
            return _bi.all(v != 0 for v in x._flat())
        return asarray(x).all()
    return _axis_reduce(x, axis,
                        lambda it: 1.0 if _bi.all(v != 0 for v in it)
                        else 0.0)


def any(x, axis=None):  # noqa: A001
    if axis is None:
        if isinstance(x, ndlist):
            return _bi.any(v != 0 for v in x._flat())
        return asarray(x).any()
    return _axis_reduce(x, axis,
                        lambda it: 1.0 if _bi.any(v != 0 for v in it)
                        else 0.0)


def sort(x, axis=-1):
    """numpy.sort: NaN last; a 2-D input sorts each row (axis=-1) or each
    column (axis=0), and axis=None flattens."""
    a = asarray(x)
    if len(a.shape) == 2 and axis is not None:
        if axis in (0, -2):
            cols = [_nan_sorted([a.data[i][j] for i in range(a.shape[0])])
                    for j in range(a.shape[1])]
            return marr([[cols[j][i] for j in range(a.shape[1])]
                         for i in range(a.shape[0])])
        return marr([_nan_sorted(row) for row in a.data])
    return marr(_nan_sorted(a._flat()))


def unique(x, return_inverse=False, return_counts=False,
           return_index=False, axis=None):
    a = asarray(x)
    if axis is not None:
        # unique rows (axis=0) or columns (axis=1) of a 2-D array, in
        # lexicographic order, as numpy does
        _check_axis(a, axis)
        if len(a.shape) != 2:
            raise ValueError("unique(axis=) needs a 2-D array in this core")
        rows = a.data if int(axis) == 0 else [list(c) for c in zip(*a.data)]
        keys = [tuple(r) for r in rows]
        uniq = sorted(set(keys))
        pos = {k: i for i, k in enumerate(uniq)}
        u = marr([list(k) for k in uniq]) if int(axis) == 0 \
            else marr([list(c) for c in zip(*[list(k) for k in uniq])])
        if not (return_inverse or return_counts or return_index):
            return u
        out = [u]
        if return_index:
            first = {}
            for i, k in enumerate(keys):
                first.setdefault(k, i)
            ix = marr([float(first[k]) for k in uniq])
            ix._is_index = True
            out.append(ix)
        if return_inverse:
            inv = marr([float(pos[k]) for k in keys])
            inv._is_index = True
            out.append(inv)
        if return_counts:
            cnt = marr([float(keys.count(k)) for k in uniq])
            cnt._dt = "int64"
            out.append(cnt)
        return tuple(out)
    if isinstance(a, oarr):
        vals = list(a)
        uniq = sorted(set(vals), key=str)
    else:
        vals = a._flat()
        # numpy: one NaN, sorted last
        uniq = sorted({v for v in vals if v == v})
        if _bi.any(v != v for v in vals):
            uniq.append(_NAN)
    if not (return_inverse or return_counts or return_index):
        return oarr(uniq) if isinstance(a, oarr) else _carry(a, marr(uniq))
    pos = {v: i for i, v in enumerate(uniq)}
    out = [oarr(uniq) if isinstance(a, oarr) else _carry(a, marr(uniq))]
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
        out.append(_typed(marr([float(cnt[v]) for v in uniq]), int))
    return tuple(out)



def _close_scalar(x, y, rtol, atol, equal_nan):
    """numpy's predicate: |x - y| <= atol + rtol * |y|.

    Asymmetric in y, unlike math.isclose, which this shim used to call.
    Infinities are close only when identical; NaN only under equal_nan.
    """
    if x != x or y != y:
        return bool(equal_nan and x != x and y != y)
    if x in (_INF, _NINF) or y in (_INF, _NINF):
        return x == y
    return _bi.abs(x - y) <= atol + rtol * _bi.abs(y)


def _broadcast_flat(a, b, who):
    """Flatten two operands to a common length, broadcasting a scalar.

    Previously the length check returned False on any mismatch, which
    made every array-vs-scalar comparison silently false.
    """
    xa, xb = asarray(a), asarray(b)
    aa = list(xa._flat())
    bb = list(xb._flat())
    if len(aa) == len(bb):
        return aa, bb
    if len(bb) == 1:
        return aa, bb * len(aa)
    if len(aa) == 1:
        return aa * len(bb), bb
    # numpy row / column broadcasting between a matrix and a vector
    sa, sb = tuple(xa.shape), tuple(xb.shape)
    if len(sa) == 2 and len(sb) == 1 and sb[0] == sa[1]:
        return aa, bb * sa[0]
    if len(sb) == 2 and len(sa) == 1 and sa[0] == sb[1]:
        return aa * sb[0], bb
    if len(sa) == 2 and len(sb) == 2 and sb[1] == 1 and sb[0] == sa[0]:
        return aa, [bb[i] for i in range(sa[0]) for _ in range(sa[1])]
    if len(sa) == 2 and len(sb) == 2 and sa[1] == 1 and sa[0] == sb[0]:
        return [aa[i] for i in range(sb[0]) for _ in range(sb[1])], bb
    raise ValueError(
        "%s: operands could not be broadcast together with %d and %d "
        "values" % (who, len(aa), len(bb)))


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    aa, bb = _broadcast_flat(a, b, "allclose")
    return _pyall(_close_scalar(x, y, rtol, atol, equal_nan)
                  for x, y in zip(aa, bb))


_INF = float("inf")
_NAN = float("nan")
_NINF = float("-inf")
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
    def norm(x, ord=None, axis=None, keepdims=False):  # noqa: A002
        if keepdims:
            out = _Linalg.norm(x, ord=ord, axis=axis)
            a = asarray(x)
            if axis is None:
                if len(a.shape) == 2:
                    return marr([[float(out)]])
                return marr([float(out)])
            if len(a.shape) == 2:
                if axis in (1, -1):
                    return marr([[v] for v in out._flat()])
                return marr([list(out._flat())])
            return out
        if isinstance(x, ndlist) and axis is not None:
            # rank-3: reduce the last axis, giving the (n, n) pairwise
            # matrix. Previously fell through to _flat() and raised
            # "can't multiply sequence by non-int".
            shape = x.shape
            if axis in (-1, len(shape) - 1) and len(shape) == 3:
                return marr([[_Linalg.norm(marr(cell), ord=ord)
                              for cell in row] for row in x.tolist()])
            raise ValueError(
                "norm: rank-%d input supports only axis=-1" % len(shape))
        a = asarray(x)
        if axis is not None and len(a.shape) == 2:
            rows = a.data if axis in (1, -1) else \
                [[a.data[i][j] for i in range(a.shape[0])]
                 for j in range(a.shape[1])]
            return marr([_Linalg.norm(marr(r), ord=ord)
                         for r in rows])
        f = a._flat()
        if ord in (None, 2, "fro"):
            return _math.sqrt(_fsum(v * v for v in f))
        if ord == 1:
            return _fsum(_bi.abs(v) for v in f)
        if ord == _math.inf:
            return _bi.max(_bi.abs(v) for v in f)
        if ord == -_math.inf:
            return _bi.min(_bi.abs(v) for v in f)
        return _fsum(_bi.abs(v) ** ord for v in f) ** (1.0 / ord)

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
            normx = _math.sqrt(_fsum(v * v for v in x))
            if normx == 0.0:
                continue
            alpha = -normx if x[0] >= 0 else normx
            v = list(x)
            v[0] -= alpha
            vnorm2 = _fsum(u * u for u in v)
            if vnorm2 == 0.0:
                continue
            # R = H R
            for j in range(k, n_):
                dot = _fsum(v[i] * R[k + i][j]
                                 for i in range(len(v)))
                c = 2.0 * dot / vnorm2
                for i in range(len(v)):
                    R[k + i][j] -= c * v[i]
            # Q = Q H
            for i in range(m_):
                dot = _fsum(Q[i][k + t] * v[t]
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
                nrm = _math.sqrt(_fsum(u * u for u in v)) \
                    or 1.0
                v = [u / nrm for u in v]
            vecs.append(v)
        w = marr([v.real for v in vals]) if real_ok else carr(vals)
        V = marr([[vecs[j][i] for j in range(n)] for i in range(n)])
        return w, V


linalg = _Linalg()


# ------------------------------------------------------------------ random


class _BitGen:
    """The raw stream behind a generator (numpy's ``.bit_generator``):
    ``random_raw()`` gives the next 64-bit word, ``state`` the position."""

    def __init__(self, gen):
        self._gen = gen

    def random_raw(self, size=None):
        return self._gen._fill(lambda: int(self._gen._next()), size)

    @property
    def state(self):
        return {"bit_generator": "SplitMix64", "state": self._gen.state}


def _pack_choice(vals, size=None):
    """marr for numeric draws, oarr for anything else -- numpy's choice
    keeps the pool's dtype (strings stay strings). A tuple ``size``
    reshapes the draws: choice(size=(n, p)) came back FLAT (n*p,), so
    every genotype / design matrix drawn this way was 1-D."""
    try:
        # _num, not float: numpy's choice keeps the pool's dtype, so an
        # integer pool must come back as integers or "%d" formatting of
        # the result raises.
        out = marr([_num(v) for v in vals])
    except (TypeError, ValueError):
        return oarr(list(vals))
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        return out.reshape(*[int(d) for d in size])
    return out


class _SplitMix64:
    """The generator behind random.default_rng(): a SplitMix64 stream.

    This is the stream every caller gets, with or without the compiled
    core; the R-parity Philox in morie_core is used by the kernels that
    need bit-exact parity with the R arm, not by default_rng(). It is
    deterministic and passes the KS/serial-correlation checks; it is NOT
    the banned gr* LCG.
    """

    def __init__(self, seed):
        if isinstance(seed, _SplitMix64):
            # numpy passes an existing Generator through unchanged
            self.state = seed.state
            return
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
        # _num, not float: one() already returns the type the
        # distribution should produce -- int for integers(), float for
        # normal()/uniform() -- and numpy preserves that.
        if isinstance(size, (tuple, list)):
            if len(size) == 1:
                return marr([_num(one()) for _ in range(int(size[0]))])
            if len(size) == 2:
                n, m = int(size[0]), int(size[1])
                return marr([[_num(one()) for _ in range(m)]
                             for _ in range(n)])
            # rank >= 3: draw in C order into the n-D container; the
            # leading blocks are rank-2 marr so the module loops that
            # index arr[b] get an array, not a list
            def build(dims):
                if len(dims) == 2:
                    return marr([[_num(one()) for _ in range(int(dims[1]))]
                                 for _ in range(int(dims[0]))])
                return [build(dims[1:]) for _ in range(int(dims[0]))]
            return ndlist(build([int(v) for v in size]))
        return marr([_num(one()) for _ in range(int(size))])

    def uniform(self, low=0.0, high=1.0, size=None):
        # Array-valued bounds broadcast, as in numpy.
        if hasattr(low, "_flat") or isinstance(low, (list, tuple)) or \
                hasattr(high, "_flat") or isinstance(high, (list, tuple)):
            def _vals(v):
                if hasattr(v, "_flat"):
                    return [float(x) for x in v._flat()]
                if isinstance(v, (list, tuple)):
                    return [float(x) for x in v]
                return [float(v)]
            lo, hi = _vals(low), _vals(high)
            n = _bi.max(len(lo), len(hi))  # the module max() is the axis reducer
            if size is not None:
                n = int(size[0]) if isinstance(size, (tuple, list)) \
                    else int(size)
            return marr([lo[i % len(lo)] +
                         (hi[i % len(hi)] - lo[i % len(lo)]) *
                         ((self._next() >> 11) / (1 << 53))
                         for i in range(n)])

        def one():
            return low + (high - low) * (self._next() >> 11) / (1 << 53)
        return self._fill(one, size)

    def multivariate_normal(self, mean, cov, size=None):
        """Correlated normal draws: mean + L z, with cov = L L'.

        _SplitMix64 carried eleven univariate distributions and not this
        one, so every caller that wanted correlated normals raised
        AttributeError -- which is what the copula tests were hitting.

        Cholesky is used when cov is positive definite. Sample
        covariance matrices often are not, only positive semi-definite,
        so a symmetric eigendecomposition with negative eigenvalues
        clipped to zero is the fallback; that still gives L L' = cov to
        rounding whenever cov is a valid covariance.
        """
        mu = [float(v) for v in asarray(mean)._flat()]
        d = len(mu)
        c = atleast_2d(asarray(cov, dtype=float))
        if c.shape != (d, d):
            raise ValueError("multivariate_normal: cov is %s but mean has "
                             "%d entries" % (c.shape, d))
        try:
            lo = linalg.cholesky(c)
            L = [[float(lo[i][j]) for j in range(d)] for i in range(d)]
        except Exception:
            w, v = linalg.eigh(c)
            wl = [_bi.max(float(x), 0.0) for x in w._flat()]
            L = [[float(v[i][k]) * _math.sqrt(wl[k]) for k in range(d)]
                 for i in range(d)]
        n = 1 if size is None else int(size)
        if n < 0:
            raise ValueError("multivariate_normal: size must not be "
                             "negative")
        rows = []
        for _ in range(n):
            z = [float(x) for x in
                 atleast_1d(asarray(self.standard_normal(d)))._flat()]
            rows.append([mu[i] + _fsum(L[i][k] * z[k]
                                            for k in range(d))
                         for i in range(d)])
        return marr(rows) if size is not None else marr(rows[0])

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
            if size is None:
                # numpy: with size omitted the result takes the BROADCAST
                # shape of loc and scale. The old code drew a single
                # standard normal and returned out[0], so
                # rng.normal(array_of_n, scale) silently returned ONE float
                # instead of n draws -- simulate_biased_crime_data built a
                # 6000-row frame whose risk_score column had length 1.
                size = _bi.max(len(lv), len(sv))  # the module max() is the axis reducer
            z = self.normal(0.0, 1.0, size)
            if len(getattr(z, "shape", (0,))) == 2:
                nr, nc = z.shape
                return marr([[lv[j % len(lv)] + sv[j % len(sv)]
                              * z.data[i][j] for j in range(nc)]
                             for i in range(nr)])
            zf = list(z._flat()) if hasattr(z, "_flat") else [float(z)]
            out = [lv[i % len(lv)] + sv[i % len(sv)] * zf[i]
                   for i in range(len(zf))]
            return marr(out)

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
        # numpy draws one variate per element of an array-valued scale,
        # which is how a Cox simulation writes rng.exponential(1 / lam).
        # Without this branch float(scale) rejected the whole vector.
        if hasattr(scale, "_flat") or isinstance(scale, (list, tuple)):
            sv = list(scale._flat()) if hasattr(scale, "_flat") else \
                [float(v) for v in scale]
            if size is not None:
                n = int(size[0]) if isinstance(size, (tuple, list)) \
                    else int(size)
                if n != len(sv):
                    raise ValueError(
                        "cannot broadcast a length-%d scale to size %d"
                        % (len(sv), n))
            return marr([-float(v) * _math.log(_pymax(self._u(), 1e-300))
                         for v in sv])

        def one():
            return -float(scale) * _math.log(_pymax(self._u(), 1e-300))
        return self._fill(one, size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        # Inverse-CDF: u ~ U(0,1), h = u - 1/2,
        # x = loc - scale * sign(h) * log(1 - 2|h|).
        def one():
            h = self._u() - 0.5
            s = 1.0 if h > 0 else (-1.0 if h < 0 else 0.0)
            return float(loc) - float(scale) * s * _math.log(
                _pymax(1.0 - 2.0 * abs(h), 1e-300))
        return self._fill(one, size)

    def standard_gamma(self, shape, size=None):
        return self.gamma(shape, 1.0, size)

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

    # ---- the rest of numpy.random.Generator's distribution surface ----
    def standard_exponential(self, size=None):
        return self.exponential(1.0, size)

    def standard_cauchy(self, size=None):
        return self._fill(lambda: _math.tan(_math.pi * (self._u() - 0.5)), size)

    def standard_t(self, df, size=None):
        df = float(df)

        def one():
            z = self.normal(0.0, 1.0)
            g = 2.0 * self._gamma_variate(df / 2.0)
            return z / _math.sqrt(g / df)
        return self._fill(one, size)

    def triangular(self, left, mode, right, size=None):
        left, mode, right = float(left), float(mode), float(right)
        if not left <= mode <= right or left == right:
            raise ValueError("left <= mode <= right and left < right required")
        c = (mode - left) / (right - left)

        def one():
            u = self._u()
            if u < c:
                return left + _math.sqrt(u * (right - left) * (mode - left))
            return right - _math.sqrt((1.0 - u) * (right - left) * (right - mode))
        return self._fill(one, size)

    def weibull(self, a, size=None):
        a = float(a)
        if a <= 0:
            raise ValueError("a must be positive")
        return self._fill(lambda: (-_math.log(1.0 - self._u())) ** (1.0 / a), size)

    def pareto(self, a, size=None):
        a = float(a)
        if a <= 0:
            raise ValueError("a must be positive")
        return self._fill(lambda: (1.0 - self._u()) ** (-1.0 / a) - 1.0, size)

    def power(self, a, size=None):
        a = float(a)
        if a <= 0:
            raise ValueError("a must be positive")
        return self._fill(lambda: self._u() ** (1.0 / a), size)

    def rayleigh(self, scale=1.0, size=None):
        scale = float(scale)
        return self._fill(lambda: scale * _math.sqrt(-2.0 * _math.log(1.0 - self._u())),
                          size)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        loc, scale = float(loc), float(scale)
        return self._fill(lambda: loc - scale * _math.log(-_math.log(1.0 - self._u())),
                          size)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        loc, scale = float(loc), float(scale)

        def one():
            u = self._u()
            return loc + scale * _math.log(u / (1.0 - u))
        return self._fill(one, size)

    def wald(self, mean, scale, size=None):
        mu, lam = float(mean), float(scale)

        def one():
            z = self.normal(0.0, 1.0)
            y = z * z
            x = mu + (mu * mu * y) / (2.0 * lam) - (mu / (2.0 * lam)) * _math.sqrt(
                4.0 * mu * lam * y + mu * mu * y * y)
            return x if self._u() <= mu / (mu + x) else mu * mu / x
        return self._fill(one, size)

    def vonmises(self, mu, kappa, size=None):
        # Best & Fisher (1979) rejection sampler
        mu, kappa = float(mu), float(kappa)
        if kappa < 1e-8:
            return self._fill(lambda: _math.pi * (2.0 * self._u() - 1.0), size)
        tau = 1.0 + _math.sqrt(1.0 + 4.0 * kappa * kappa)
        rho = (tau - _math.sqrt(2.0 * tau)) / (2.0 * kappa)
        r = (1.0 + rho * rho) / (2.0 * rho)

        def one():
            while True:
                u1, u2, u3 = self._u(), self._u(), self._u()
                z = _math.cos(_math.pi * u1)
                f = (1.0 + r * z) / (r + z)
                c = kappa * (r - f)
                if u2 < c * (2.0 - c) or _math.log(c / u2) + 1.0 - c >= 0.0:
                    break
            theta = mu + (_math.acos(f) if u3 > 0.5 else -_math.acos(f))
            return (theta + _math.pi) % (2.0 * _math.pi) - _math.pi
        return self._fill(one, size)

    def f(self, dfnum, dfden, size=None):
        dfnum, dfden = float(dfnum), float(dfden)

        def one():
            return ((2.0 * self._gamma_variate(dfnum / 2.0)) / dfnum) / (
                (2.0 * self._gamma_variate(dfden / 2.0)) / dfden)
        return self._fill(one, size)

    def noncentral_chisquare(self, df, nonc, size=None):
        df, nonc = float(df), float(nonc)

        def one():
            if nonc == 0.0:
                return 2.0 * self._gamma_variate(df / 2.0)
            z = self.normal(_math.sqrt(nonc), 1.0)
            rest = 2.0 * self._gamma_variate((df - 1.0) / 2.0) if df > 1.0 else 0.0
            return z * z + rest
        return self._fill(one, size)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        """(noncentral chi2(dfnum, nonc) / dfnum) / (chi2(dfden) / dfden)."""
        dfnum, dfden, nonc = float(dfnum), float(dfden), float(nonc)
        if dfnum <= 0 or dfden <= 0 or nonc < 0:
            raise ValueError("dfnum > 0, dfden > 0 and nonc >= 0 required")

        def one():
            num = float(self.noncentral_chisquare(dfnum, nonc)) / dfnum
            den = 2.0 * self._gamma_variate(dfden / 2.0) / dfden
            return num / den
        return self._fill(one, size)

    def multivariate_hypergeometric(self, colors, nsample, size=None,
                                    method="marginals"):
        """Counts of each colour in ``nsample`` draws without replacement
        from an urn holding ``colors[k]`` balls of colour k."""
        del method
        colors = [int(c) for c in asarray(colors)._flat()]
        nsample = int(nsample)
        if nsample < 0 or _bi.any(c < 0 for c in colors) \
                or nsample > _bi.sum(colors):
            raise ValueError("nsample must be between 0 and sum(colors)")

        def one():
            left = colors[:]
            total = _bi.sum(left)
            out = [0] * len(left)
            for _ in range(nsample):
                u = self._u() * total
                acc = 0.0
                for k, c in enumerate(left):
                    acc += c
                    if u < acc:
                        out[k] += 1
                        left[k] -= 1
                        total -= 1
                        break
            return out
        if size is None:
            m = marr([float(v) for v in one()])
        else:
            reps = int(size[0]) if isinstance(size, (tuple, list)) else int(size)
            m = marr([[float(v) for v in one()] for _ in range(reps)])
        m._dt = "int64"
        return m

    def permuted(self, x, axis=None, out=None):
        """A shuffled copy: the whole array when axis is None, otherwise
        every 1-D slice along ``axis`` shuffled independently."""
        del out
        a = asarray(x)
        if axis is None or len(a.shape) == 1:
            flat = list(a._flat())
            self.shuffle(flat)
            m = marr(flat)
            if getattr(a, "_dt", None) is not None:
                m._dt = a._dt
            return m if len(a.shape) == 1 else reshape(m, a.shape)
        if len(a.shape) != 2:
            raise ValueError("permuted(axis=) needs a 1-D or 2-D array in this core")
        rows = [row[:] for row in a.data]
        if int(axis) == 1:
            for row in rows:
                self.shuffle(row)
            return marr(rows)
        cols = [list(c) for c in zip(*rows)]
        for c in cols:
            self.shuffle(c)
        return marr([list(r) for r in zip(*cols)])

    def spawn(self, n_children):
        """Independent child generators seeded from this stream."""
        return [type(self)(self._next()) for _ in range(int(n_children))]

    @property
    def bit_generator(self):
        return _BitGen(self)

    def negative_binomial(self, n, p, size=None):
        n, p = float(n), float(p)
        if not 0.0 < p <= 1.0 or n <= 0:
            raise ValueError("n > 0 and 0 < p <= 1 required")

        def one():
            lam = self._gamma_variate(n) * (1.0 - p) / p
            return float(self.poisson(lam)) if lam > 0 else 0.0
        return self._fill(one, size)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        ngood, nbad, nsample = int(ngood), int(nbad), int(nsample)
        if nsample > ngood + nbad or _bi.min(ngood, nbad, nsample) < 0:
            raise ValueError("nsample must be <= ngood + nbad")

        def one():
            good, bad, hits = ngood, nbad, 0
            for _ in range(nsample):
                if self._u() * (good + bad) < good:
                    hits += 1
                    good -= 1
                else:
                    bad -= 1
            return float(hits)
        return self._fill(one, size)

    def multinomial(self, n, pvals, size=None):
        n = int(n)
        pv = [float(v) for v in (pvals.tolist() if hasattr(pvals, "tolist") else pvals)]
        if _bi.abs(_fsum(pv) - 1.0) > 1e-8 or _bi.min(pv) < 0:
            raise ValueError("pvals must be non-negative and sum to 1")

        def one():
            left, rem, out = n, 1.0, []
            for q in pv[:-1]:
                k = int(self.binomial(left, _bi.min(1.0, q / rem))) if rem > 0 and left > 0 else 0
                out.append(float(k))
                left -= k
                rem -= q
            out.append(float(left))
            return out
        if size is None:
            return marr(one())
        return marr([one() for _ in range(int(size))])

    def zipf(self, a, size=None):
        # Devroye (1986) rejection sampler for the Zipf(a) distribution
        a = float(a)
        if a <= 1.0:
            raise ValueError("a must be > 1")
        am1, b = a - 1.0, 2.0 ** (a - 1.0)

        def one():
            while True:
                u, v = 1.0 - self._u(), self._u()
                x = _math.floor(u ** (-1.0 / am1))
                t = (1.0 + 1.0 / x) ** am1
                if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                    return float(x)
        return self._fill(one, size)

    def logseries(self, p, size=None):
        p = float(p)
        if not 0.0 < p < 1.0:
            raise ValueError("0 < p < 1 required")
        r = _math.log1p(-p)

        def one():
            v = self._u()
            if v >= p:
                return 1.0
            u = self._u()
            q = 1.0 - _math.exp(r * u)
            if v <= q * q:
                return float(_math.floor(1.0 + _math.log(v) / _math.log(q)))
            return 1.0 if v >= q else 2.0
        return self._fill(one, size)

    def random_sample(self, size=None):
        return self.random(size)

    def bytes(self, length):
        return _bi.bytes((self._next() >> 56) & 0xFF for _ in range(int(length)))

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
        res = marr([float(v) for v in out])
        if isinstance(n, int) or _is_int_typed(n):
            _typed(res, int)
        return res

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        def one():
            return _math.exp(self.normal(float(mean), float(sigma)))
        return self._fill(one, size)

    def _gamma_variate(self, shape):
        # Marsaglia & Tsang (2000) ACM TOMS 26(3), 363-372, the squeeze
    # method of their section 4; shape < 1 boosted per section 6.
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
            return _pack_choice(out, size)
        if isinstance(a, int):
            pool = list(range(int(a)))
        elif isinstance(a, (list, tuple)):
            pool = list(a)          # keep the element type; numpy
        else:                       # returns int64 for an int pool
            pool = list(asarray(a)._flat())
        if size is None:
            return pool[self._next() % len(pool)]
        shape = size
        if isinstance(size, (tuple, list)):
            size = 1
            for d in shape:
                size *= int(d)
        k = int(size)
        if replace:
            return _pack_choice([pool[self._next() % len(pool)]
                                 for _ in range(k)], shape)
        if k > len(pool):
            raise ValueError("cannot sample more than population without "
                             "replacement")
        idx = list(range(len(pool)))
        self.shuffle(idx)
        return _pack_choice([pool[i] for i in idx[:k]], shape)

    def integers(self, low, high=None, size=None, dtype=None,
                 endpoint=False):
        """Integers in [low, high), or [low, high] with endpoint=True.

        `dtype` is accepted for numpy call-compatibility; the draws are
        Python ints either way, which is what every integer dtype means
        here.
        """
        del dtype
        if high is None:
            low, high = 0, low
        lo, hi = int(low), int(high) + (1 if endpoint else 0)
        if hi <= lo:
            raise ValueError("high must exceed low; got low=%r high=%r"
                             % (low, high))

        def one():
            return lo + self._next() % (hi - lo)
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


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    # same predicate as allclose, so allclose(x, y) == all(isclose(x, y))
    return asarray(a)._zip(
        b, lambda x, y: 1.0 if _close_scalar(x, y, rtol, atol, equal_nan)
        else 0.0)


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
    if len(asarray(a).shape) < 2:
        raise ValueError("diag requires an array of at least two dimensions")
    aa = atleast_2d(a)
    return float(_fsum(aa.data[i][i]
                            for i in range(_bi.min(aa.shape))))


def logaddexp(a, b):
    def f(x, y):
        hi, lo = (x, y) if x >= y else (y, x)
        return hi + _math.log1p(_math.exp(lo - hi))
    return asarray(a)._zip(b, f) if isinstance(a, (marr, list, tuple)) \
        or isinstance(b, (marr, list, tuple)) else f(float(a), float(b))


def tile(x, reps):
    if isinstance(x, (list, tuple, oarr)) and x \
            and any(isinstance(v, str) for v in x):
        r = reps[-1] if isinstance(reps, (list, tuple)) else reps
        return oarr(list(x) * int(r))
    a = asarray(x)
    if not isinstance(reps, (tuple, list)) and len(a.shape) == 2:
        # numpy tiles the LAST axis for a scalar reps: (2, 2) -> (2, 4)
        return _carry(a, marr([list(r) * int(reps) for r in a.data]))
    if isinstance(reps, (tuple, list)):
        reps = [int(r) for r in reps]
        if len(reps) == 1:
            reps = reps[0]
        elif len(reps) == 2:
            m, k = reps
            if len(a.shape) == 1:
                row = list(a._flat()) * k
                return _carry(a, marr([list(row) for _ in range(m)]))
            if len(a.shape) == 2:
                rows = [list(r) * k for r in a.data]
                return _carry(a, marr([list(r) for _ in range(m) for r in rows]))
            raise ValueError("tile: unsupported input rank")
        else:
            raise ValueError("tile: reps rank > 2 unsupported")
    if len(a.shape) == 2:
        return _carry(a, marr(a.data * int(reps)))
    return _carry(a, marr(a._flat() * int(reps)))


def take_along_axis(a, idx, axis=-1):
    if isinstance(a, ndlist):
        # rank-3 (B, L, V) gathered by a rank-3 (B, L, k) index along the
        # last axis: one rank-2 gather per leading block
        if axis not in (-1, 2):
            raise ValueError("take_along_axis: last axis only on rank 3")
        return ndlist(take_along_axis(marr(blk), marr(ib), axis=-1)
                      for blk, ib in zip(a, idx))
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
    if len(shape) == 2 and len(a.shape) == 2:
        n, m = shape
        r, c = a.shape
        if (r in (1, n)) and (c in (1, m)):
            rows = a.data if r == n else [a.data[0]] * n
            return marr([row[:] if c == m else [row[0]] * m
                         for row in rows])
    if len(a.shape) == 1 and a.shape[0] == 1:
        v = a.data[0]
        if len(shape) == 1:
            return marr([v] * shape[0])
        if len(shape) == 2:
            return marr([[v] * shape[1] for _ in range(shape[0])])
    if len(shape) == 2 and len(a.shape) == 1 and a.shape[0] == shape[1]:
        return marr([a.data[:] for _ in range(shape[0])])
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
    a = asarray(x)
    if axis is not None:
        nd = len(a.shape)
        axes = []
        for ax in ((axis,) if isinstance(axis, int) else tuple(axis)):
            ax = int(ax) + nd if int(ax) < 0 else int(ax)
            if ax < 0 or ax >= nd or a.shape[ax] != 1:
                # numpy refuses; silently returning the input hid the
                # caller's mistake
                raise ValueError("cannot select an axis to squeeze out "
                                 "which has size not equal to one")
            axes.append(ax)
        if nd == 2 and len(axes) == 1:
            if axes[0] == 0:
                return marr(a.data[0][:])
            return marr([row[0] for row in a.data])
    if len(a.shape) == 1 and a.shape[0] == 1:
        return float(a.data[0])
    if len(a.shape) == 2:
        if a.shape[0] == 1:
            return marr(a.data[0][:])
        if a.shape[1] == 1:
            return marr([row[0] for row in a.data])
    return a


def repeat(x, reps, axis=None):
    if isinstance(x, (list, tuple, oarr)) and x \
            and any(isinstance(v, str) for v in x):
        # numpy: string / object input stays object dtype
        rl = ([int(v) for v in asarray(reps)._flat()]
              if isinstance(reps, (list, tuple, marr)) else [int(reps)] * len(x))
        return oarr([v for v, r in zip(x, rl) for _ in range(r)])
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
        return _carry(a2, out2) if isinstance(a2, marr) else out2
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
    return _carry(asarray(x), marr([v for v in a._flat() for _ in range(int(reps))]))


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
            off = _math.sqrt(_fsum(m[i][j] ** 2 for i in range(n)
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
    def cond(a, p=None):
        """numpy.linalg.cond: the default (p=None) and p=2 are the ratio
        of the largest to smallest singular value; 'fro' and the 1/inf
        norms multiply the norm of the matrix by the norm of its
        inverse, as numpy does."""
        aa = atleast_2d(a)
        if p is None or p == 2 or p == -2:
            sv = sorted(_svd(aa, compute_uv=False)._flat(), reverse=True)
            if not sv:
                return inf
            big, small = sv[0], sv[-1]
            if p == -2:
                return small / big if big else inf
            return big / small if small else inf
        try:
            ai = _Linalg.inv(aa)
        except ValueError:
            return inf
        if p == "fro":
            fro = lambda m: _math.sqrt(_fsum(v * v for v in m._flat()))  # noqa: E731
            return fro(aa) * fro(ai)
        if p in (1, -1, inf, -inf):
            def nrm(m, which):
                rows = m.data if len(m.shape) == 2 else [m.data]
                if which in (1, -1):          # max/min absolute column sum
                    sums = [_fsum(abs(r[j]) for r in rows) for j in range(len(rows[0]))]
                else:                          # max/min absolute row sum
                    sums = [_fsum(abs(v) for v in r) for r in rows]
                return _bi.max(sums) if which in (1, inf) else _bi.min(sums)
            key = 1 if p in (1, -1) else inf
            want = p in (1, inf)
            return nrm(aa, key if want else key) * nrm(ai, key if want else key) \
                if want else nrm(aa, -key if key == 1 else -inf) * nrm(ai, -key if key == 1 else -inf)
        raise ValueError("invalid norm order %r for cond" % (p,))


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
        off = _math.sqrt(_fsum(m[i][j] ** 2 for i in range(n)
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
    out = [[_fsum(vt.data[c][i] * inv_s[c] * u.data[j][c]
                       for c in range(k))
            for j in range(m_)] for i in range(n_)]
    return marr(out), svals


def _pinv(a, rcond=1e-15):
    """Moore-Penrose pseudo-inverse via the SVD.

    See :func:`_pinv_extended`; this drops the singular values.
    """
    return _pinv_extended(a, rcond)[0]


def ginv(a, tol=None):
    """Moore-Penrose pseudo-inverse using MASS::ginv's cutoff.

    ``MASS::ginv`` (MASS 7.3, Venables & Ripley, *Modern Applied
    Statistics with S*, 4th ed.) keeps the singular values satisfying
    ``d_i > max(tol * d_1, 0)`` with ``tol = sqrt(.Machine$double.eps)``,
    i.e. about 1.49e-8 of the largest singular value, and drops the rest
    of the columns entirely rather than zeroing their reciprocals.

    This is deliberately NOT :func:`_pinv`. numpy's ``pinv`` cuts at
    ``rcond * max(s)`` with ``rcond = 1e-15``, seven orders of magnitude
    tighter, so on a matrix with singular values in between the two the
    answers differ. Both are correct under their own convention; code
    ported from ``MASS::ginv`` must keep MASS's answer, which is why both
    exist. Mirrors the R arm ``MASS_ginv`` in aaa_tail1_core.R.
    """
    aa = atleast_2d(a)
    m_, n_ = aa.shape
    u, sv, vt = _svd(aa)
    svals = list(sv._flat())
    if tol is None:
        tol = _math.sqrt(2.220446049250313e-16)
    cutoff = _bi.max(tol * (svals[0] if svals else 0.0), 0.0)
    keep = [i for i, v in enumerate(svals) if v > cutoff]
    if not keep:
        return marr([[0.0] * m_ for _ in range(n_)])
    out = [[_fsum(vt.data[c][i] * (1.0 / svals[c]) * u.data[j][c]
                       for c in keep)
            for j in range(m_)] for i in range(n_)]
    return marr(out)


_LinalgExt.pinv = staticmethod(_pinv)
_LinalgExt.matrix_rank = staticmethod(_matrix_rank)
linalg.matrix_rank = _matrix_rank
linalg.pinv = _pinv
linalg.pinv_extended = _pinv_extended
linalg.ginv = ginv
_LinalgExt.ginv = staticmethod(ginv)
linalg.slogdet = _LinalgExt.slogdet
linalg.eigvalsh = _LinalgExt.eigvalsh
linalg.cond = _LinalgExt.cond


def prod(x, axis=None, keepdims=False):
    axis = _axis_arg(x, axis)
    _check_axis(asarray(x), axis)
    def _p(vals):
        out = 1.0
        for v in vals:
            out *= v
        return float(out)
    a = asarray(x)
    if axis is not None and len(a.shape) == 2:
        if axis in (0, -2):
            out = marr([_p(a.data[i][j] for i in range(a.shape[0]))
                        for j in range(a.shape[1])])
        else:
            out = marr([_p(row) for row in a.data])
        return _keepdims_wrap(out, axis, keepdims)
    return _keepdims_wrap(_p(a._flat()), axis, keepdims, len(a.shape))


def outer(a, b):
    fa, fb = asarray(a)._flat(), asarray(b)._flat()
    if not fa or not fb:
        return _empty2d(len(fa), len(fb))    # numpy: shape (len(a), len(b))
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
    terms = [(fx[i + 1] - fx[i]) * (fy[i + 1] + fy[i]) / 2.0
             for i in range(len(fy) - 1)]
    if _bi.any(isinstance(v, complex) for v in terms):
        return _bi.sum(terms)
    return float(_fsum(terms))


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
        self.kind = "u" if name.startswith("uint") \
            else "i" if name.startswith("int") else "f"
        self.itemsize = _DTYPE_FMT.get(name, ("d", 8))[1]

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
int16 = int
int8 = int
uint8 = int
uint16 = int
uint32 = int
uint64 = int
int16 = int
int8 = int
uint8 = int
uint16 = int
uint32 = int
uint64 = int


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
sinh = _uf(_ieee(_math.sinh))
cosh = _uf(_ieee(_math.cosh, even=True))
sin = _uf(_ieee(_math.sin))
cos = _uf(_ieee(_math.cos))
tan = _uf(_ieee(_math.tan))
arctan = _uf(_math.atan)
arcsin = _uf(_ieee(_math.asin))
arccos = _uf(_ieee(_math.acos))
sign = _uf(lambda v: v if v != v else
           (0.0 if v == 0 else (1.0 if v > 0 else -1.0)))
floor = _uf(_ieee(_math.floor))
ceil = _uf(_ieee(_math.ceil))
_round0 = _uf(_ieee(lambda v: float(_bi.round(v))))


def round(x, decimals=0):  # noqa: A001
    """numpy.round: round to `decimals` places, half to even.

    The bare _uf wrapper took the array only, so np.round(a, 3) raised
    "wrapped() takes 1 positional argument but 2 were given" -- the
    second argument is part of numpy's signature and callers use it.
    Python's builtin round() is half-to-even like numpy's, so the
    tie-breaking matches; scaling by a power of ten is how numpy
    implements the decimals argument too.
    """
    if decimals == 0:
        return _round0(x)
    f = 10.0 ** decimals
    return _uf(_ieee(lambda v: float(_bi.round(v * f)) / f))(x)


around = round
log2 = _uf(_ieee_log2)
log10 = _uf(_ieee_log10)
expm1 = _uf(_ieee(_math.expm1))
isnan = _uf(lambda v: 1.0 if v != v else 0.0)


def vectorize(fn, otypes=None, excluded=None, signature=None):
    del otypes, excluded, signature

    def wrapped(x, *args, **kw):
        if isinstance(x, (list, tuple, marr)):
            return asarray(x)._map(lambda v: float(fn(v, *args, **kw)))
        return fn(x, *args, **kw)
    return wrapped


def _running(vals, start, op):
    out, total = [], start
    for v in vals:
        total = op(total, v)
        out.append(total)
    return out


def cumsum(x, axis=None):
    a = asarray(x)
    if axis is not None and len(a.shape) == 2:
        if axis in (0, -2):
            cols = [_running([a.data[i][j] for i in range(a.shape[0])], 0.0,
                             lambda t, v: t + v) for j in range(a.shape[1])]
            return marr([[cols[j][i] for j in range(a.shape[1])]
                         for i in range(a.shape[0])])
        return marr([_running(row, 0.0, lambda t, v: t + v) for row in a.data])
    return marr(_running(a._flat(), 0.0, lambda t, v: t + v))


def argmax(x, axis=None):
    _check_axis(asarray(x), axis)
    a = asarray(x)
    if axis is None or len(a.shape) == 1:
        return _nan_argext(a._flat(), _bi.max)
    if axis == 0:
        return marr([float(_nan_argext([a.data[i][j]
                                        for i in range(a.shape[0])], _bi.max))
                     for j in range(a.shape[1])])
    return marr([float(_nan_argext(row, _bi.max)) for row in a.data])


def argmin(x, axis=None):
    _check_axis(asarray(x), axis)
    a = asarray(x)
    if axis is None or len(a.shape) == 1:
        return _nan_argext(a._flat(), _bi.min)
    if axis == 0:
        return marr([float(_nan_argext([a.data[i][j]
                                        for i in range(a.shape[0])], _bi.min))
                     for j in range(a.shape[1])])
    return marr([float(_nan_argext(row, _bi.min)) for row in a.data])


def argsort(x, axis=-1, kind=None):
    """numpy.argsort: NaN last; a 2-D input is sorted along its last axis
    by default, axis=None flattens. Indices come back as an index marr."""
    del kind
    a = asarray(x)
    if len(a.shape) == 2 and axis is not None:
        if axis in (0, -2):
            cols = [_nan_argsorted([a.data[i][j] for i in range(a.shape[0])])
                    for j in range(a.shape[1])]
            out = marr([[float(cols[j][i]) for j in range(a.shape[1])]
                        for i in range(a.shape[0])])
        else:
            out = marr([[float(i) for i in _nan_argsorted(row)]
                        for row in a.data])
        out._is_index = True
        return out
    out = marr([float(i) for i in _nan_argsorted(a._flat())])
    out._is_index = True
    return out


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


def _median_nanaware(vals):
    vals = list(vals)
    if not vals:
        _warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=3)
        return _NAN
    for v in vals:
        if v != v:
            return _NAN
    f = sorted(vals)
    n = len(f)
    mid = n // 2
    return f[mid] if n % 2 else 0.5 * (f[mid - 1] + f[mid])


def median(x, axis=None, keepdims=False):
    """numpy.median: NaN propagates, an empty input is NaN, axis honoured."""
    _check_axis(asarray(x), axis)
    a = asarray(x)
    if axis is not None and len(a.shape) == 2:
        if axis in (0, -2):
            out = marr([_median_nanaware(a.data[i][j] for i in range(a.shape[0]))
                        for j in range(a.shape[1])])
        else:
            out = marr([_median_nanaware(row) for row in a.data])
        return _keepdims_wrap(out, axis, keepdims)
    return _keepdims_wrap(_median_nanaware(a._flat()), axis, keepdims,
                          len(a.shape))


def percentile(x, q, axis=None):
    if isinstance(x, ndlist):
        _q = q
        return _ndlist_reduce(x, axis, False,
                              lambda v: float(percentile(marr(v), _q)))
    """Linear-interpolation percentile (numpy default method).

    A NaN anywhere in the reduced slice makes the answer NaN, as in
    numpy; nanpercentile() is the skipping form.
    """
    a = asarray(x)
    if axis is None and _bi.any(v != v for v in a._flat()):
        return marr([nan] * len(list(q))) if isinstance(q, (list, tuple, marr)) else nan
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
    if isinstance(q, (list, tuple, marr)):
        return marr([one(float(v)) for v in asarray(q)._flat()])
    return one(float(q))


def quantile(x, q, axis=None, method="linear", **kw):
    """numpy.quantile; ``method`` follows the Hyndman-Fan names numpy
    uses (linear, lower, higher, nearest, midpoint, inverted_cdf,
    averaged_inverted_cdf, closest_observation, interpolated_inverted_cdf,
    hazen, weibull, median_unbiased, normal_unbiased)."""
    method = kw.get("interpolation", method)
    if method == "linear":
        if isinstance(q, (list, tuple, marr)):
            qf = q._flat() if isinstance(q, marr) else q
            return percentile(x, [100.0 * float(v) for v in qf], axis=axis)
        return percentile(x, 100.0 * float(q), axis=axis)
    a = asarray(x)
    if axis is not None and len(a.shape) == 2:
        if axis in (0, -2):
            cols = [[a.data[r][c] for r in range(a.shape[0])]
                    for c in range(a.shape[1])]
            return marr([float(quantile(col, q, method=method)) for col in cols]) \
                if not isinstance(q, (list, tuple, marr)) else \
                marr([[float(quantile(col, qq, method=method)) for col in cols]
                      for qq in asarray(q)._flat()])
        return marr([float(quantile(row, q, method=method)) for row in a.data]) \
            if not isinstance(q, (list, tuple, marr)) else \
            marr([[float(quantile(row, qq, method=method)) for row in a.data]
                  for qq in asarray(q)._flat()])
    if isinstance(q, (list, tuple, marr)):
        return marr([float(quantile(a, qq, method=method))
                     for qq in asarray(q)._flat()])
    f = sorted(a._flat())
    n = len(f)
    if n == 0:
        return nan
    if _bi.any(v != v for v in f):
        return nan
    p = float(q)
    if not 0.0 <= p <= 1.0:
        raise ValueError("Quantiles must be in the range [0, 1]")
    # discontinuous methods (H&F 1-3) on the 1-based ordinal g = n p
    if method in ("inverted_cdf", "averaged_inverted_cdf",
                  "closest_observation"):
        g = n * p
        j = int(_math.floor(g))
        frac = g - j
        if method == "inverted_cdf":
            k = j + (1 if frac > 0 else 0)
        elif method == "averaged_inverted_cdf":
            if frac > 0:
                k = j + 1
            else:
                lo_i = _bi.max(j, 1) - 1
                hi_i = _bi.min(j + 1, n) - 1
                return 0.5 * (f[lo_i] + f[hi_i])
        else:  # closest_observation (H&F 3): at g == 0 take the even
            # order statistic, as numpy
            idx = n * p - 1.5
            prev = int(_math.floor(idx))
            gam = idx - prev
            k = prev if (gam == 0 and prev % 2 == 1) else prev + 1
            k += 1
        k = _bi.max(1, _bi.min(k, n))
        return f[k - 1]
    # continuous methods: virtual index (1-based) = alpha + p (n + 1 - alpha - beta)
    ab = {"interpolated_inverted_cdf": (0.0, 1.0), "hazen": (0.5, 0.5),
          "weibull": (0.0, 0.0), "linear": (1.0, 1.0),
          "median_unbiased": (1.0 / 3.0, 1.0 / 3.0),
          "normal_unbiased": (3.0 / 8.0, 3.0 / 8.0)}
    if method in ("lower", "higher", "nearest", "midpoint"):
        alpha = beta = 1.0
    elif method in ab:
        alpha, beta = ab[method]
    else:
        raise ValueError("unknown quantile method %r" % (method,))
    g = alpha + p * (n + 1 - alpha - beta)
    g = _bi.max(1.0, _bi.min(g, float(n)))
    j = int(_math.floor(g))
    frac = g - j
    lo_v = f[j - 1]
    hi_v = f[_bi.min(j + 1, n) - 1]
    if method == "lower":
        return lo_v
    if method == "higher":
        return hi_v if frac > 0 else lo_v
    if method == "midpoint":
        return 0.5 * (lo_v + hi_v) if frac > 0 else lo_v
    if method == "nearest":
        # numpy rounds half to even on the fractional part
        if frac > 0.5 or (frac == 0.5 and j % 2 == 0):
            return hi_v
        return lo_v
    return lo_v + frac * (hi_v - lo_v)


def empty(n, dtype=None):
    return zeros(n, dtype=dtype)


def subtract(a, b):
    if not isinstance(a, (list, tuple, marr)) \
            and not isinstance(b, (list, tuple, marr)):
        return float(a) - float(b)
    return asarray(a)._zip(b, lambda x, y: x - y)


class _AddUfunc:
    """numpy.add as a callable with the two ufunc methods modules use:
    add.at (unbuffered scatter-add, duplicates accumulate) and
    add.accumulate (cumulative sum)."""

    def __call__(self, a, b):
        if not isinstance(a, (list, tuple, marr)) \
                and not isinstance(b, (list, tuple, marr)):
            return float(a) + float(b)
        return asarray(a)._zip(b, lambda x, y: x + y)

    @staticmethod
    def at(a, indices, b=1.0):
        if not isinstance(a, marr):
            raise TypeError("add.at needs an in-place marr target")
        if isinstance(indices, tuple):
            ii = [int(v) for v in asarray(indices[0])._flat()]
            jj = [int(v) for v in asarray(indices[1])._flat()]
            if len(jj) == 1 and len(ii) > 1:
                jj = jj * len(ii)
            if len(ii) == 1 and len(jj) > 1:
                ii = ii * len(jj)
            bv = asarray(b)._flat() if isinstance(b, (list, tuple, marr)) \
                else [float(b)] * len(ii)
            if len(bv) == 1:
                bv = bv * len(ii)
            for i, j, v in zip(ii, jj, bv):
                a.data[i][j] += float(v)
            return None
        ii = [int(v) for v in asarray(indices)._flat()]
        bv = asarray(b)._flat() if isinstance(b, (list, tuple, marr)) \
            else [float(b)] * len(ii)
        if len(bv) == 1:
            bv = bv * len(ii)
        for i, v in zip(ii, bv):
            if len(a.shape) == 2:
                a.data[i] = [x + float(v) for x in a.data[i]]
            else:
                a.data[i] += float(v)
        return None

    @staticmethod
    def accumulate(a, axis=0):
        return cumsum(a, axis=axis)


add = _AddUfunc()


def multiply(a, b):
    if not isinstance(a, (list, tuple, marr)) \
            and not isinstance(b, (list, tuple, marr)):
        return float(a) * float(b)
    return asarray(a)._zip(b, lambda x, y: x * y)


def divide(a, b, out=None, where=None):
    if not isinstance(a, (list, tuple, marr)) \
            and not isinstance(b, (list, tuple, marr)):
        return _ieee_div(float(a), float(b))
    res = asarray(a)._zip(b, _ieee_div)
    if out is None and where is None:
        return res
    return _ufunc_out(res, out, where)


def _ufunc_out(res, out, where):
    """numpy's ``out=`` / ``where=`` contract: elements where the mask
    is false keep the value already in ``out``."""
    if out is None:
        out = zeros_like(res)
    target = out[0] if isinstance(out, tuple) else out
    if not isinstance(target, marr):
        raise TypeError("out= needs a marr target")
    rf = list(res._flat())
    if where is None:
        mask = [True] * len(rf)
    else:
        w = asarray(where)
        mask = [bool(v) for v in w._flat()]
        if len(mask) == 1:
            mask = mask * len(rf)
    if len(target.shape) == 2:
        k = 0
        for r in range(target.shape[0]):
            for c in range(target.shape[1]):
                if mask[k]:
                    target.data[r][c] = rf[k]
                k += 1
    else:
        for k, v in enumerate(rf):
            if mask[k]:
                target.data[k] = v
    return target


def fill_diagonal(a, val, wrap=False):
    del wrap
    m = atleast_2d(a)
    vals = (list(asarray(val)._flat())
            if isinstance(val, (list, tuple, marr)) else [float(val)])
    for i in range(_bi.min(m.shape)):
        m.data[i][i] = float(vals[i % len(vals)])
    # in-place on the caller's 2-D marr (same list objects)


def size(x, axis=None):
    a = asarray(x)
    if axis is not None:
        _check_axis(a, axis)
        return int(a.shape[int(axis)])
    n = 1
    for d in a.shape:
        n *= d
    return n


def count_nonzero(x, axis=None):
    a = asarray(x)
    if axis is None:
        return int(_bi.sum(1 for v in a._flat() if v != 0))
    _check_axis(a, axis)
    out = marr(_reduce_axis(a.tolist(), axis,
                            lambda vs: float(_bi.sum(1 for v in vs if v != 0))))
    out._dt = "int64"
    return out


def shape(x):
    if isinstance(x, marr):
        return x.shape
    return asarray(x).shape


def ndim(x):
    # a Python scalar is 0-d in numpy; this core's asarray() has no 0-d
    # form, so answer the question before building an array
    if isinstance(x, (int, float, complex, bool)):
        return 0
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


def cov(x, y=None, rowvar=True, bias=False, ddof=None):
    if ddof is None:
        ddof = 0 if bias else 1
    if y is None:
        a = atleast_2d(x)
        if not rowvar:
            a = transpose(a)
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




def _newaxis_general(a, idx):
    """Any combination of new axes with slices and integer indices.

    Inserting a unit axis never reorders elements, so the result is the
    base selection reshaped: drop the Nones, index with what is left,
    then put a 1 where each None was. Returns None when the base
    selection is not something this core can index.
    """
    if None not in idx:
        return None
    rest = tuple(v for v in idx if v is not None)
    try:
        base = a[rest] if rest else a
    except (ValueError, TypeError, IndexError):
        return None
    if isinstance(base, (int, float, complex)):
        flat, bshape = [base], []
    elif isinstance(base, marr):
        flat, bshape = list(base._flat()), list(base.shape)
    elif isinstance(base, ndlist):
        flat, bshape = _flatten_nested(base.tolist()), list(base.shape)
    else:
        return None
    dims = iter(bshape)
    out_shape = []
    for v in idx:
        if v is None:
            out_shape.append(1)
        elif isinstance(v, slice):
            nxt = next(dims, None)
            if nxt is None:
                return None
            out_shape.append(nxt)
        # an integer index consumed its axis in the base selection
    out_shape.extend(list(dims))
    total = 1
    for d in out_shape:
        total *= d
    if total != len(flat):
        return None
    if len(out_shape) <= 1:
        return marr(flat)
    if len(out_shape) == 2:
        nc = out_shape[1]
        return marr([flat[i * nc:(i + 1) * nc] for i in range(out_shape[0])])

    def build(vals, ds):
        if len(ds) == 1:
            return list(vals)
        step = 1
        for d in ds[1:]:
            step *= d
        return [build(vals[i * step:(i + 1) * step], ds[1:])
                for i in range(ds[0])]
    return ndlist(build(flat, out_shape))


def _newaxis_rank3(a, idx):
    """x[:, None, :] and friends on a 2-D marr -> rank-3 ndlist.

    Returns None if the index is not one new axis among full slices, so
    the caller can raise rather than guess. This is the pairwise idiom
    `x[:, None, :] - x[None, :, :]`; without it the core raised "too
    many values to unpack" for the 183 modules that write it.
    """
    if len(a.shape) != 2:
        return None
    full = slice(None)
    rows = a.data
    if idx == (full, None, full):
        return ndlist([[r[:]] for r in rows])            # (n, 1, k)
    if idx == (None, full, full):
        return ndlist([[r[:] for r in rows]])            # (1, n, k)
    if idx == (full, full, None):
        return ndlist([[[v] for v in r] for r in rows])  # (n, k, 1)
    # one new axis among slices / integer indices: index without it,
    # then insert the unit axis where numpy puts it
    if len(idx) == 3 and idx.count(None) == 1 and _bi.all(
            v is None or isinstance(v, (int, float, slice)) for v in idx):
        # integer indices consume an axis; the unit axis sits where the
        # None falls among the axes that remain
        kinds = ["u" if v is None else "a" for v in idx
                 if v is None or isinstance(v, slice)]
        pos = kinds.index("u")
        rest = tuple(_ix(v) for v in idx if v is not None)
        sub = a[rest]
        if not isinstance(sub, marr):
            sub = marr([sub])
        if len(sub.shape) == 1:
            return marr([sub.data[:]]) if pos == 0 else marr([[v] for v in sub.data])
        srows = sub.data
        if pos == 0:
            return ndlist([[r[:] for r in srows]])
        if pos == 1:
            return ndlist([[r[:]] for r in srows])
        return ndlist([[[v] for v in r] for r in srows])
    return None


def _sum_all(x):
    if isinstance(x, (list, tuple)):
        return _bi.sum(_sum_all(v) for v in x)
    return float(x)


def _sum_axis(x, axis):
    """Sum a nested list along ``axis``, returning nested lists."""
    return _reduce_axis(x, axis, _sum_all)


def _reduce_axis(x, axis, fn):
    """Reduce a nested list along ``axis`` with ``fn(list_of_floats)``.

    Elements at the same position across the reduced axis are gathered
    into one flat list and handed to ``fn`` (sum, mean, max, min), so
    every reduction shares one traversal instead of one hand-written
    recursion each.
    """
    if axis == 0:
        blocks = [b.tolist() if hasattr(b, "tolist") else b for b in x]
        if not isinstance(blocks[0], (list, tuple)):
            return fn([float(v) for v in blocks])

        def walk(items):
            if isinstance(items[0], (list, tuple)):
                return [walk([it[i] for it in items])
                        for i in range(len(items[0]))]
            return fn([float(v) for v in items])
        return walk(blocks)
    return [_reduce_axis(v, axis - 1, fn) for v in x]


def _reduce_axes(nested, shape, axes, fn):
    """Reduce several axes of a nested list at once."""
    import itertools
    keep = [d for d in range(len(shape)) if d not in axes]

    def get(idx):
        v = nested
        for i in idx:
            v = v[i]
        return v

    def build(prefix):
        if len(prefix) == len(keep):
            vals = []
            for red in itertools.product(*[range(shape[d]) for d in axes]):
                idx = [0] * len(shape)
                for d, i in zip(keep, prefix):
                    idx[d] = i
                for d, i in zip(axes, red):
                    idx[d] = i
                vals.append(get(idx))
            return fn(vals)
        d = keep[len(prefix)]
        return [build(prefix + (i,)) for i in range(shape[d])]
    return build(())


def _ndlist_reduce(self, axis, keepdims, fn):
    nested = self.tolist()
    if axis is None:
        return fn(_flatten_nested(nested))
    shape = self.shape
    if isinstance(axis, tuple):
        axes = sorted({ax % len(shape) for ax in axis})
        if len(axes) == len(shape):
            val = fn(_flatten_nested(nested))
            if not keepdims:
                return val
            red = val
            for _ in shape:
                red = [red]
            return ndlist(red)
        red = _reduce_axes(nested, shape, axes, fn)
        if keepdims:
            for ax in axes:
                red = _expand_axis(red, ax)
        depth = _nested_depth(red)
        if depth >= 3:
            return ndlist(red)
        if depth == 0:
            return red
        return marr(red)
    if axis < 0:
        axis += len(shape)
    if not 0 <= axis < len(shape):
        raise ValueError("axis %r is out of bounds for shape %r"
                         % (axis, shape))
    red = _reduce_axis(nested, axis, fn)
    if keepdims:
        red = _expand_axis(red, axis)
    depth = _nested_depth(red)
    if depth >= 3:
        return ndlist(red)
    if depth == 0:
        return red
    return marr(red)


def _flatten_nested(x):
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            out.extend(_flatten_nested(v))
        else:
            out.append(float(v))
    return out


def _add_nested(a, b):
    if isinstance(a, (list, tuple)):
        return [_add_nested(a[i], b[i]) for i in range(len(a))]
    return float(a) + float(b)


def _expand_axis(x, axis):
    if axis == 0:
        return [x]
    return [_expand_axis(v, axis - 1) for v in x]


def _wrap_block(b):
    """A sub-block of a rank>=3 container as the array type numpy would
    hand back: marr for a matrix or vector, ndlist for rank >= 3."""
    if isinstance(b, (marr, ndlist)):
        return b
    if isinstance(b, list):
        d = _nested_depth(b)
        if d >= 3:
            return ndlist(b)
        if d >= 1:
            return marr(b)
    return b


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
        return [conv(v) for v in self._blocks()]

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Sum over one axis of a rank>=3 container, or over all of it.

        Only what the rank-2 core already offers on marr, lifted to the
        nested case: without it a module that validates a conditional
        probability table with ``table.sum(axis=2)`` cannot be called at
        all once the table really is rank 3.
        """
        del dtype, out
        return _ndlist_reduce(self, axis, keepdims, _sum_all)

    def mean(self, axis=None, dtype=None, keepdims=False):
        del dtype
        return _ndlist_reduce(self, axis, keepdims,
                              lambda v: _sum_all(v) / len(v) if v else nan)

    def max(self, axis=None, keepdims=False):
        return _ndlist_reduce(self, axis, keepdims, _bi.max)

    def var(self, axis=None, ddof=0, keepdims=False):
        def _var(v):
            n = len(v)
            if n - ddof <= 0:
                return nan
            m = _sum_all(v) / n
            return _sum_all([(x - m) ** 2 for x in v]) / (n - ddof)
        return _ndlist_reduce(self, axis, keepdims, _var)

    def std(self, axis=None, ddof=0, keepdims=False):
        out = self.var(axis=axis, ddof=ddof, keepdims=keepdims)
        if isinstance(out, float):
            return out ** 0.5
        return sqrt(out)

    def min(self, axis=None, keepdims=False):
        return _ndlist_reduce(self, axis, keepdims, _bi.min)

    def _blocks(self):
        """The raw sub-lists, bypassing the numpy-style __iter__."""
        return [list.__getitem__(self, i) for i in range(len(self))]

    def __iter__(self):
        """numpy: iterating a rank-n array yields rank-(n-1) arrays."""
        for b in self._blocks():
            yield _wrap_block(b)

    def __getitem__(self, key):
        """x[i, j, k] / x[i, :, k] on a rank>=3 container: integers pick,
        full slices keep the axis; the result collapses to marr / float
        when its rank drops to 2 / 0 (numpy semantics for basic
        indexing). A boolean mask selects the flattened elements it
        marks, as numpy does."""
        if isinstance(key, (ndlist, marr)) and (
                isinstance(key, ndlist) or getattr(key, "_is_mask", False)):
            flat_v = _flatten_nested(self.tolist())
            flat_m = _flatten_nested(key.tolist() if hasattr(key, "tolist") else key)
            if len(flat_m) != len(flat_v):
                raise IndexError("boolean index did not match the array shape")
            return marr([v for v, m in zip(flat_v, flat_m) if m])
        if not isinstance(key, tuple):
            # the raw sub-list, NOT a wrapped copy: __setitem__ and the
            # module-level loops mutate what they index
            return list.__getitem__(self, key)

        def pick(node, keys):
            if not keys:
                return node.tolist() if isinstance(node, marr) else node
            k, rest = keys[0], keys[1:]
            if isinstance(node, marr):
                sub = node[tuple(keys) if len(keys) > 1 else k]
                return sub.tolist() if isinstance(sub, marr) else sub
            if isinstance(k, slice):
                return [pick(v, rest) for v in list(node)[k]]
            return pick(node[int(k)], rest)
        out = pick(self, key)
        depth = _nested_depth(out)
        if depth == 0:
            return float(out)
        if depth <= 2:
            return marr(out)
        return ndlist(out)

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            return list.__setitem__(self, key, value)
        node, keys = self, list(key)
        while keys:
            if isinstance(node, marr):
                node[tuple(keys) if len(keys) > 1 else keys[0]] = value
                return None
            k = keys[0]
            if isinstance(k, slice):
                idxs = list(range(len(node))[k])
                vals = None
                if isinstance(value, (list, tuple, marr, ndlist)) \
                        and len(value) and not isinstance(value, str):
                    va = value if isinstance(value, (marr, ndlist)) else \
                        (ndlist(value) if _nested_depth(value) >= 3 else asarray(value))
                    vshape = tuple(va.shape)
                    n_sub = len([kk for kk in keys[1:] if isinstance(kk, slice)])
                    # numpy aligns trailing dims: a value with one more
                    # dim than the selection below this axis is split
                    # along it
                    if vshape and vshape[0] == len(idxs) and (
                            len(vshape) == 1 + n_sub or len(keys) == 1):
                        vals = [va[j] for j in range(len(idxs))]
                for j, i in enumerate(idxs):
                    v = vals[j] if vals is not None else value
                    if len(keys) == 1:
                        node[i] = float(v)
                    else:
                        sub = node[i]
                        if isinstance(sub, list) and not isinstance(sub, ndlist):
                            sub = ndlist(sub)
                            node[i] = sub
                        if isinstance(sub, marr):
                            sub[tuple(keys[1:]) if len(keys) > 2 else keys[1]] = v
                        else:
                            ndlist.__setitem__(sub, tuple(keys[1:]), v)
                return None
            if len(keys) == 1:
                node[int(k)] = float(value)
                return None
            nxt = node[int(k)]
            if isinstance(nxt, list) and not isinstance(nxt, (ndlist, marr)):
                nxt = ndlist(nxt)
                node[int(k)] = nxt
            node = nxt
            keys = keys[1:]
        return None

    def __neg__(self):
        return self._ew(-1.0, lambda a, b: a * b)

    def _flat(self):
        def walk(v):
            if isinstance(v, marr):
                for x in v._flat():
                    yield x
            elif isinstance(v, list):
                for x in v:
                    for y in walk(x):
                        yield y
            else:
                yield float(v)
        return walk(self)

    def reshape(self, *shape):
        """Row-major reshape, with a single -1 inferred, as numpy does.

        Returns marr for rank 1 or 2 (the rank-2 core) and ndlist above
        that. ndlist had no reshape at all, so the common
        stack(...).reshape(-1, n) idiom could not complete.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        flat = list(self._flat())
        dims = [int(d) for d in shape]
        if dims.count(-1) > 1:
            raise ValueError("can only specify one unknown dimension")
        if -1 in dims:
            known = 1
            for d in dims:
                if d != -1:
                    known *= d
            if known <= 0 or len(flat) % known:
                raise ValueError(
                    "cannot reshape %d values into %s"
                    % (len(flat), tuple(shape)))
            dims[dims.index(-1)] = len(flat) // known
        total = 1
        for d in dims:
            total *= d
        if total != len(flat):
            raise ValueError("cannot reshape %d values into %s"
                             % (len(flat), tuple(dims)))
        if len(dims) == 1:
            return marr(flat)
        if len(dims) == 2:
            nc = dims[1]
            return marr([flat[i * nc:(i + 1) * nc] for i in range(dims[0])])

        def build(vals, ds):
            if len(ds) == 1:
                return list(vals)
            step = 1
            for d in ds[1:]:
                step *= d
            return [build(vals[i * step:(i + 1) * step], ds[1:])
                    for i in range(ds[0])]
        return ndlist(build(flat, dims))

    def _cmp(self, other, fn):
        """Elementwise comparison, as numpy: a nested mask of bools."""
        def walk(v, o):
            if isinstance(v, list):
                return [walk(x, o) for x in v]
            if isinstance(v, marr):
                return [walk(x, o) for x in v._flat()]
            return bool(fn(v, o))
        if isinstance(other, (list, tuple, marr, ndlist)):
            raise ValueError("ndlist comparison supports a scalar operand")
        return ndlist(walk(self._blocks(), other))

    def __gt__(self, other):
        return self._cmp(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._cmp(other, lambda a, b: a >= b)

    def __lt__(self, other):
        return self._cmp(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._cmp(other, lambda a, b: a <= b)

    def _ew(self, other, fn):
        if isinstance(other, ndlist):
            # blockwise: marr broadcasting handles the trailing axes
            # ((q,1) vs (q,k) etc.). The leading axis is broadcast here.
            # This used to be a bare zip(self, other), which TRUNCATES to
            # the shorter operand, so (n,1,k) - (1,n,k) silently returned
            # one block instead of n -- the wrong shape, no error raised.
            a, b = self._blocks(), other._blocks()
            if len(a) != len(b):
                if len(a) == 1:
                    a = a * len(b)
                elif len(b) == 1:
                    b = b * len(a)
                else:
                    raise ValueError(
                        "ndlist: leading axes %d and %d cannot be "
                        "broadcast" % (len(a), len(b)))
            return ndlist(marr(x)._zip(marr(y), fn)
                          for x, y in zip(a, b))
        if isinstance(other, (marr, list)) and not isinstance(
                other, ndlist) and isinstance(other, marr):
            return ndlist(marr(v)._zip(other, fn) for v in self._blocks())

        def walk(v):
            if isinstance(v, marr):
                return v._map(lambda x: fn(x, float(other)))
            if isinstance(v, list):
                return [walk(x) for x in v]
            return fn(float(v), float(other))
        return ndlist(walk(v) for v in self._blocks())

    def __truediv__(self, o):
        return self._ew(o, lambda a, b: a / b)

    def __pow__(self, o):
        # absent entirely, so the `(x[:, None, :] - y[None, :, :]) ** 2`
        # spelling of the pairwise idiom raised TypeError
        return self._ew(o, lambda a, b: a ** b)

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
    row of blocks joined left-to-right, rows stacked top-to-bottom. A
    flat list of 1-D blocks concatenates to a 1-D array, as numpy does."""
    if isinstance(rows, marr):
        return marr(rows)                      # an array is its own block
    if not any(isinstance(r, (list, tuple)) for r in rows):
        parts = [asarray(b) for b in rows]
        if all(len(p_.shape) == 1 for p_ in parts):
            return marr([v for p_ in parts for v in p_.data])
        return hstack(parts)
    if all(isinstance(r, (list, tuple)) and
           all(isinstance(v, (int, float)) for v in r) for r in rows):
        return marr([list(map(float, r)) for r in rows])   # nested scalars
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
        if axis in (2, -1):
            # new trailing axis: out[i][j][p] = parts[p][i][j]. This is
            # the meshgrid -> coordinate-pairs idiom,
            # stack(meshgrid(g, g), -1).reshape(-1, 2), which previously
            # raised before any caller reached its own code.
            nr, nc = arrs[0].shape
            for a2 in arrs:
                if tuple(a2.shape) != (nr, nc):
                    raise ValueError(
                        "stack: all parts must have the same shape, got "
                        "%s and %s" % (tuple(arrs[0].shape),
                                       tuple(a2.shape)))
            return ndlist([[[a2.data[i][j] for a2 in arrs]
                            for j in range(nc)] for i in range(nr)])
        raise ValueError(
            "stack: 2-D parts support axis 0, 2 or -1; got %r" % (axis,))
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
    """numpy.nonzero: one index array per axis."""
    a = asarray(x)
    if len(a.shape) == 2:
        rr, cc = [], []
        for i, row in enumerate(a.data):
            for j, v in enumerate(row):
                if v != 0:
                    rr.append(float(i))
                    cc.append(float(j))
        r, c = marr(rr), marr(cc)
        r._is_index = c._is_index = True
        return (r, c)
    return (flatnonzero(x),)


def triu_indices(n, k=0, m=None):
    m = n if m is None else int(m)
    ii, jj = [], []
    for i in range(n):
        for j in range(_bi.max(i + k, 0), m):
            ii.append(float(i))
            jj.append(float(j))
    return _index_pair(ii, jj)


def tril_indices(n, k=0, m=None):
    m = n if m is None else int(m)
    ii, jj = [], []
    for i in range(n):
        for j in range(0, _bi.min(i + k + 1, m)):
            ii.append(float(i))
            jj.append(float(j))
    return _index_pair(ii, jj)


def _index_pair(ii, jj):
    """Two integer index arrays, tagged as such and int-valued so
    ``tolist()`` gives the ints numpy gives."""
    r, c = _typed(marr(ii), int), _typed(marr(jj), int)
    r._is_index = True
    c._is_index = True
    return r, c


def diag_indices(n):
    idx = marr([float(i) for i in range(n)])
    return idx, marr(idx.data[:])


def diag_indices_from(a):
    """numpy's diag_indices_from; only diag_indices(n) existed, so
    `sigma[np.diag_indices_from(sigma)] = v` raised AttributeError."""
    arr = asarray(a)
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("diag_indices_from: input must be a square "
                         "2-D array, got shape %s" % (tuple(arr.shape),))
    return diag_indices(arr.shape[0])


def _tri_input(a):
    """numpy broadcasts a 1-D input against tri(n, n): the row is
    repeated down an n x n matrix before masking."""
    m = asarray(a)
    if len(m.shape) == 1:
        return marr([m.data[:] for _ in range(m.shape[0])])
    return atleast_2d(m)


def triu(a, k=0):
    m = _tri_input(a)
    return marr([[m.data[i][j] if j >= i + k else 0.0
                  for j in range(m.shape[1])] for i in range(m.shape[0])])


def tril(a, k=0):
    m = _tri_input(a)
    return marr([[m.data[i][j] if j <= i + k else 0.0
                  for j in range(m.shape[1])] for i in range(m.shape[0])])


def _conv_args(a, v):
    aa, vv = asarray(a), asarray(v)
    if len(aa.shape) != 1 or len(vv.shape) != 1:
        raise ValueError("object too deep for desired array")
    if aa.shape[0] == 0:
        raise ValueError("a cannot be empty")
    if vv.shape[0] == 0:
        raise ValueError("v cannot be empty")
    return aa._flat(), vv._flat()


def convolve(a, v, mode="full"):
    x, y = _conv_args(a, v)
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


def histogram(x, bins=10, range=None, density=False, weights=None):  # noqa: A002
    if weights is not None or density:
        counts, edges = histogram(x, bins, range)
        f = list(asarray(x)._flat())
        if weights is not None:
            wv = list(asarray(weights)._flat())
            ed = list(edges._flat())
            wc = [0.0] * (len(ed) - 1)
            for v, w in zip(f, wv):
                if v < ed[0] or v > ed[-1]:
                    continue
                for b in range_(len(ed) - 1):
                    if ed[b] <= v < ed[b + 1] or (b == len(ed) - 2 and v == ed[-1]):
                        wc[b] += float(w)
                        break
            counts = marr(wc)
        if density:
            ed = list(edges._flat())
            tot = _fsum(counts._flat())
            counts = marr([c / (tot * (ed[i + 1] - ed[i])) if tot else 0.0
                           for i, c in enumerate(counts._flat())])
        return counts, edges
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
    return _typed(marr(counts), int), marr(edges)


def range_(n):
    import builtins
    return builtins.range(n)


def bincount(x, weights=None, minlength=0):
    f = [int(v) for v in asarray(x)._flat()]
    n = _bi.max([minlength - 1] + f) + 1 if f or minlength else 0
    out = [0.0] * n
    if weights is None:
        for v in f:
            out[v] += 1.0
    else:
        w = asarray(weights)._flat()
        if len(w) != len(f):
            raise ValueError("weights and x must have the same length")
        for v, wv in zip(f, w):
            out[v] += float(wv)
    return marr(out)


def meshgrid(x, y, indexing="xy"):
    """As numpy.meshgrid, including the ``indexing`` keyword.

    The keyword was missing, so every caller writing the numpy-standard
    ``meshgrid(a, b, indexing="ij")`` got a TypeError -- which is what
    the copula tau quadrature was hitting.

    "xy" (the default, and the previous behaviour) gives arrays of shape
    (len(y), len(x)); "ij" gives (len(x), len(y)), i.e. the transpose.
    """
    if indexing not in ("xy", "ij"):
        raise ValueError("meshgrid: indexing must be 'xy' or 'ij', got %r"
                         % (indexing,))
    fx, fy = asarray(x)._flat(), asarray(y)._flat()
    if indexing == "xy":
        gx = marr([fx[:] for _ in fy])
        gy = marr([[v] * len(fx) for v in fy])
    else:
        gx = marr([[v] * len(fy) for v in fx])
        gy = marr([fy[:] for _ in fx])
    return gx, gy


def average(x, weights=None):
    a = asarray(x)
    if weights is None:
        return a.mean()
    w = asarray(weights)
    return float(dot(w, a) / w.sum())


class _Testing:
    """numpy.testing's two assertions, the only ones this tree uses.

    Left behind by the de-numpy campaign: `np.testing.assert_*` in a
    test raised AttributeError rather than comparing anything.
    """

    @staticmethod
    def assert_allclose(actual, desired, rtol=1e-7, atol=0,
                        equal_nan=True, err_msg="", verbose=True):
        del verbose
        a = list(asarray(actual)._flat())
        d = list(asarray(desired)._flat())
        # numpy broadcasts, so assert_allclose(vec, 5.0) compares every
        # element against 5.0. Demanding equal lengths turned that ordinary
        # idiom into a spurious "shape mismatch: 3 vs 1 values" failure.
        if len(a) != len(d):
            if len(d) == 1:
                d = d * len(a)
            elif len(a) == 1:
                a = a * len(d)
            else:
                raise AssertionError(
                    "shape mismatch: %d vs %d values. %s"
                    % (len(a), len(d), err_msg))
        for i, (x, y) in enumerate(zip(a, d)):
            if equal_nan and x != x and y != y:
                continue
            if not _bi.abs(x - y) <= atol + rtol * _bi.abs(y):
                raise AssertionError(
                    "not close at index %d: %r != %r (rtol=%g, atol=%g). %s"
                    % (i, x, y, rtol, atol, err_msg))

    @staticmethod
    def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
        # numpy: abs(desired - actual) < 1.5 * 10**(-decimal), elementwise
        _Testing.assert_allclose(actual, desired, rtol=0.0, atol=1.5 * 10.0 ** (-decimal),
                                 err_msg=err_msg, verbose=verbose)

    @staticmethod
    def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
        _Testing.assert_allclose(actual, desired, rtol=0.0, atol=1.5 * 10.0 ** (-decimal),
                                 err_msg=err_msg, verbose=verbose)

    @staticmethod
    def assert_approx_equal(actual, desired, significant=7, err_msg="", verbose=True):
        a, d = float(actual), float(desired)
        scale = _bi.max(_bi.abs(a), _bi.abs(d), 1e-300)
        if _bi.abs(a - d) / scale >= 10.0 ** (-(significant - 1)):
            raise AssertionError("not equal to %d significant digits: %r != %r. %s"
                                 % (significant, a, d, err_msg))

    @staticmethod
    def assert_equal(actual, desired, err_msg="", verbose=True):
        if isinstance(actual, (int, float, str, bool)) and isinstance(desired, (int, float, str, bool)):
            if actual != desired and not (actual != actual and desired != desired):
                raise AssertionError("%r != %r. %s" % (actual, desired, err_msg))
            return
        _Testing.assert_array_equal(actual, desired, err_msg=err_msg, verbose=verbose)

    @staticmethod
    def assert_array_less(x, y, err_msg="", verbose=True):
        del verbose
        a = list(asarray(x)._flat())
        d = list(asarray(y)._flat())
        if len(d) == 1:
            d = d * len(a)
        for i, (u, v) in enumerate(zip(a, d)):
            if not u < v:
                raise AssertionError("not less at index %d: %r >= %r. %s" % (i, u, v, err_msg))

    @staticmethod
    def assert_array_equal(actual, desired, err_msg="", verbose=True):
        del verbose
        a = list(asarray(actual)._flat())
        d = list(asarray(desired)._flat())
        if len(d) == 1 and len(a) != 1:
            d = d * len(a)          # scalar expectation broadcasts
        if len(a) != len(d):
            raise AssertionError(
                "shape mismatch: %d vs %d values. %s" % (len(a), len(d),
                                                         err_msg))
        for i, (x, y) in enumerate(zip(a, d)):
            if x != y and not (x != x and y != y):
                raise AssertionError(
                    "arrays differ at index %d: %r != %r. %s"
                    % (i, x, y, err_msg))

    assert_equal = assert_array_equal
    assert_array_almost_equal = staticmethod(
        lambda a, d, decimal=6, **kw: _Testing.assert_allclose(
            a, d, rtol=0, atol=1.5 * 10.0 ** (-decimal)))


testing = _Testing()


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


def flip(x, axis=None):
    """numpy.flip: with no axis every axis is reversed."""
    a = asarray(x)
    if len(a.shape) == 2:
        if axis is None:
            return marr([list(row[::-1]) for row in a.data[::-1]])
        if axis in (0, -2):
            return marr([list(row) for row in a.data[::-1]])
        return marr([list(row[::-1]) for row in a.data])
    return marr(a._flat()[::-1])


def _nan_filter(x):
    return [v for v in asarray(x)._flat() if v == v]


def _keepdims_wrap(out, axis, keepdims, nd=None):
    """Give a reduction numpy's keepdims shape: the reduced axis stays as
    length 1, so a full reduction of a 2-D input is (1, 1), of a 1-D
    input (1,), and an axis reduction of a 2-D input (1, m) or (n, 1)."""
    if not keepdims:
        return out
    if isinstance(out, marr):
        return marr([out.tolist()]) if axis in (0, -2) else \
            marr([[v] for v in out._flat()])
    if nd == 2:
        return marr([[float(out)]])
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
    axis = _axis_arg(x, axis)
    nd = len(asarray(x).shape)
    if axis is not None and nd == 2:
        return _keepdims_wrap(
            _nan_axis(x, axis,
                      lambda v: _fsum(v) / len(v) if v else nan),
            axis, keepdims)
    f = _nan_filter(x)
    if not f:
        _warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
        return _keepdims_wrap(_NAN, axis, keepdims, nd)
    return _keepdims_wrap(float(_fsum(f) / len(f)), axis, keepdims, nd)


def nansum(x, axis=None, keepdims=False):
    axis = _axis_arg(x, axis)
    nd = len(asarray(x).shape)
    if axis is not None and nd == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _fsum(v)),
                              axis, keepdims)
    return _keepdims_wrap(float(_fsum(_nan_filter(x))), axis, keepdims, nd)


def _nanvar_of(v, ddof):
    if len(v) - ddof <= 0:
        return nan
    m = _fsum(v) / len(v)
    return _fsum((u - m) ** 2 for u in v) / (len(v) - ddof)


def nanvar(x, axis=None, ddof=0, keepdims=False):
    """numpy.nanvar; honours axis and keepdims like its siblings (it
    accepted them and returned the flat scalar)."""
    nd = len(asarray(x).shape)
    if axis is not None and nd == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _nanvar_of(v, ddof)),
                              axis, keepdims)
    return _keepdims_wrap(float(_nanvar_of(_nan_filter(x), ddof)), axis,
                          keepdims, nd)


def nanstd(x, axis=None, ddof=0, keepdims=False):
    v = nanvar(x, axis=axis, ddof=ddof, keepdims=keepdims)
    if isinstance(v, marr):
        return v._map(_math.sqrt)
    return _math.sqrt(v)


def nanmax(x, axis=None, keepdims=False):
    axis = _axis_arg(x, axis)
    nd = len(asarray(x).shape)
    if axis is not None and nd == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _bi.max(v) if v else nan),
                              axis, keepdims)
    return _keepdims_wrap(float(_bi.max(_nan_filter(x))), axis, keepdims, nd)


def nanmin(x, axis=None, keepdims=False):
    axis = _axis_arg(x, axis)
    nd = len(asarray(x).shape)
    if axis is not None and nd == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _bi.min(v) if v else nan),
                              axis, keepdims)
    return _keepdims_wrap(float(_bi.min(_nan_filter(x))), axis, keepdims, nd)


def nanmedian(x, axis=None, keepdims=False):
    nd = len(asarray(x).shape)
    if axis is not None and nd == 2:
        return _keepdims_wrap(_nan_axis(x, axis, lambda v: _median_of(v)),
                              axis, keepdims)
    f = _nan_filter(x)
    if not f:
        _warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
        return _keepdims_wrap(nan, axis, keepdims, nd)
    return _keepdims_wrap(median(f), axis, keepdims, nd)


def _nan_arg(f, better):
    best, bi_ = None, -1
    for i, v in enumerate(f):
        if v == v and (best is None or better(v, best)):
            best, bi_ = v, i
    if bi_ < 0:
        # numpy raises; -1 silently indexed the LAST element
        raise ValueError("All-NaN slice encountered")
    return bi_


def _nan_arg_axis(x, axis, better):
    a = asarray(x)
    if axis is None or len(a.shape) == 1:
        return _nan_arg(a._flat(), better)
    _check_axis(a, axis)
    out = marr(_reduce_axis(a.tolist(), axis,
                            lambda vs: float(_nan_arg(vs, better))))
    out._dt = "int64"
    return out


def nanargmax(x, axis=None):
    return _nan_arg_axis(x, axis, lambda v, b: v > b)


def nanargmin(x, axis=None):
    return _nan_arg_axis(x, axis, lambda v, b: v < b)


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
    pa = asarray(p)
    if len(pa.shape) == 2:
        # numpy: 2-D coefficients evaluate one polynomial per column
        # (Horner down axis 0), array-valued coefficients
        if isinstance(x, (list, tuple, marr)):
            raise ValueError("polyval: 2-D coefficients take a scalar x in this core")
        v = float(x)
        out = [0.0] * pa.shape[1]
        for row in pa.data:
            out = [o * v + cc for o, cc in zip(out, row)]
        return marr(out)
    c = pa._flat()

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
    a = asarray(roots)
    if len(a.shape) == 2:
        # numpy: a square matrix gives its characteristic polynomial.
        # Faddeev-LeVerrier builds the coefficients from traces of
        # powers, without an eigen-decomposition.
        n = a.shape[0]
        if a.shape[1] != n:
            raise ValueError("input must be 1d or non-empty square 2d array.")
        A = [row[:] for row in a.data]
        M = [[0.0] * n for _ in range(n)]
        coef = [1.0]
        for k in range(1, n + 1):
            # M_k = A M_{k-1} + c_{n-k+1} I
            AM = [[_fsum(A[i][t] * M[t][j] for t in range(n))
                   for j in range(n)] for i in range(n)]
            for i in range(n):
                AM[i][i] += coef[-1]
            M = AM
            tr = _fsum(_fsum(A[i][t] * M[t][i] for t in range(n))
                       for i in range(n))
            coef.append(-tr / k)
        return marr(coef)
    out = [1.0]
    for r in a._flat():
        out = convolve(out, [1.0, -r]).tolist()
    return marr(out)


def kron(a, b):
    flat = len(asarray(a).shape) == 1 and len(asarray(b).shape) == 1
    aa, bb = atleast_2d(a), atleast_2d(b)
    out = []
    for i in range_(aa.shape[0]):
        for k in range_(bb.shape[0]):
            row = []
            for j in range_(aa.shape[1]):
                for m in range_(bb.shape[1]):
                    row.append(aa.data[i][j] * bb.data[k][m])
            out.append(row)
    # numpy: kron of two vectors is a vector, not a (1, n*m) matrix
    return marr(out[0]) if flat else marr(out)


def ix_(rows, cols):
    """numpy.ix_: an open mesh, (n, 1) and (1, m) index arrays that
    broadcast into a grid. Indexing consumers read them flat and see
    the outer-product tag."""
    r = _typed(marr([[float(v)] for v in asarray(rows)._flat()]), int)
    c = _typed(marr([[float(v) for v in asarray(cols)._flat()]]), int)
    r._is_index = True
    c._is_index = True
    r._ix_outer = True
    c._ix_outer = True
    return (r, c)


def cumprod(x, axis=None):
    a = asarray(x)
    if axis is not None and len(a.shape) == 2:
        if axis in (0, -2):
            cols = [_running([a.data[i][j] for i in range(a.shape[0])], 1.0,
                             lambda t, v: t * v) for j in range(a.shape[1])]
            return marr([[cols[j][i] for j in range(a.shape[1])]
                         for i in range(a.shape[0])])
        return marr([_running(row, 1.0, lambda t, v: t * v) for row in a.data])
    return marr(_running(a._flat(), 1.0, lambda t, v: t * v))


def ediff1d(x):
    # numpy flattens first; diff() kept a 2-D input 2-D
    return diff(asarray(x)._flat())


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
    the real part rather than pass the value through. A scalar input
    gives a float, as numpy does."""
    if isinstance(x, (int, float, complex)):
        return float(x.real) if isinstance(x, complex) else float(x)
    return asarray(x)._map(lambda v: v.real if isinstance(v, complex)
                           else float(v))


def imag(x):
    """Imaginary part.  Returned a hard zero before, which was a silent
    wrong answer for every complex input."""
    if isinstance(x, (int, float, complex)):
        return float(x.imag) if isinstance(x, complex) else 0.0
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
        if v != v:
            return _NAN
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
    return not _bi.any(isinstance(v, complex) for v in asarray(x)._flat())


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
    return _map_unary(x, _ieee(lambda v: 2.0 ** v))


def hypot(a, b):
    return asarray(a)._zip(b, _math.hypot)


def rint(x):
    return _map_unary(x, _ieee(lambda v: float(_bi.round(v))))


def trunc(x):
    return _map_unary(x, _ieee(lambda v: float(_math.trunc(v))))


def negative(x):
    return _map_unary(x, lambda v: -v)


def reciprocal(x):
    a = asarray(x) if not isinstance(x, (int, float)) else None
    if a is not None and getattr(a, "_dt", None) == "int64":
        # numpy integer reciprocal is integer division: 1 for 1, else 0
        return _typed(a._map(lambda v: float(int(1 / v)) if v != 0 else 0.0), int)
    return _map_unary(x, lambda v: _ieee_div(1.0, v))


def cbrt(x):
    return _map_unary(x, lambda v: _math.copysign(_bi.abs(v) ** (1.0 / 3.0), v)
                      if v == v else v)


def ravel(x):
    a = asarray(x)
    return _carry(a, marr(list(a._flat())))


def transpose(x, axes=None):
    del axes
    a = asarray(x)
    if len(a.shape) == 2:
        if 0 in a.shape:
            return _empty2d(a.shape[1], a.shape[0])
        return marr([[a.data[i][j] for i in range(a.shape[0])]
                     for j in range(a.shape[1])])
    return a


def geomspace(a, b, n):
    la, lb = _math.log(a), _math.log(b)
    return marr([_math.exp(la + i * (lb - la) / (n - 1))
                 for i in range_(int(n))])


def split(x, k, axis=0):
    """numpy.split: an integer count of equal pieces, or a list of split
    points; a 2-D input splits its rows (axis 0) or columns (axis 1)."""
    a = asarray(x)
    two_d = len(a.shape) == 2
    n = a.shape[1] if (two_d and axis in (1, -1)) else a.shape[0]
    if isinstance(k, (list, tuple, marr)):
        pts = [int(v) for v in (k._flat() if isinstance(k, marr) else k)]
        bounds = [0] + pts + [n]
    else:
        kk = int(k)
        if n % kk:
            raise ValueError("array split does not result in an equal division")
        step = n // kk
        bounds = [i * step for i in range_(kk + 1)]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        lo, hi = _bi.max(lo, 0), _bi.max(hi, lo)
        if not two_d:
            out.append(marr(a.data[lo:hi]))
        elif axis in (1, -1):
            out.append(marr([row[lo:hi] for row in a.data]))
        else:
            out.append(marr(a.data[lo:hi]) if hi > lo else marr([[]]))
    return out


def empty_like(x, dtype=None):
    return zeros_like(x)


def spacing(x):
    """numpy.spacing: the gap to the next representable float, with the
    sign of x. (The previous body referenced an undefined name and had
    never run.)"""
    def one(v):
        v = float(v)
        if v != v or v in (_INF, -_INF):
            return _NAN
        gap = _math.ulp(_bi.abs(v))
        return gap if v >= 0 else -gap
    return _uf(one)(x)


integer = int
number = float
floating = float


def issubdtype(a, b):
    del a, b
    return True     # all our dtypes are float; callers gate float paths


def array_str(x):
    """numpy's string form: elements padded to a common width, the
    fractional parts left-aligned, so ``[-2.  -0.5  0.   0.5  2. ]``,
    ``[ True False  True]`` and ``[[1. 2.]\n [3. 4.]]``."""
    a = asarray(x)
    is_mask = getattr(a, "_is_mask", False)
    is_int = getattr(a, "_dt", None) == "int64"

    def cells(vals):
        if is_mask:
            return ["True" if v else "False" for v in vals]
        if is_int:
            return [str(int(v)) for v in vals]
        return [_num_str(v) for v in vals]

    def _num_str(v):
        if isinstance(v, complex):
            return repr(v)
        if isinstance(v, bool):
            return "True" if v else "False"
        v = float(v)
        if v != v:
            return "nan"
        if v in (_math.inf, -_math.inf):
            return "inf" if v > 0 else "-inf"
        if v.is_integer() and _bi.abs(v) < 1e16:
            return "%d." % int(v)
        return repr(v)

    flat = [c for row in (a.data if len(a.shape) == 2 else [a.data])
            for c in cells(row)]
    finite = [s for s in flat if "." in s]
    if is_mask or is_int or not finite:
        width = _bi.max([len(s) for s in flat] or [0])

        def fmt(s):
            return s.rjust(width)
    else:
        # nan/inf take the full width of the widest finite cell
        left = _bi.max(len(s.split(".")[0]) for s in finite)
        right = _bi.max(len(s.split(".")[1]) for s in finite)
        total = _bi.max([left + 1 + right] + [len(s) for s in flat])

        def fmt(s):
            if "." not in s:
                return s.rjust(total)
            ip, fp = s.split(".")
            return ip.rjust(left) + "." + fp.ljust(right)

    if len(a.shape) == 2:
        rows = ["[" + " ".join(fmt(s) for s in cells(row)) + "]"
                for row in a.data]
        return "[" + "\n ".join(rows) + "]"
    return "[" + " ".join(fmt(s) for s in flat) + "]"


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
            s = _fsum(low[i][k] * low[j][k] for k in range_(j))
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
    bb = asarray(b)
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

    # A two-dimensional b holds one right-hand side per column, exactly
    # as numpy.linalg.lstsq does; each is solved separately and the
    # solutions are returned as the columns of a k x nrhs matrix.
    two_d = len(bb.shape) == 2
    if two_d:
        if bb.shape[0] != n:
            raise ValueError(
                "lstsq: a has %d rows but b has %d" % (n, bb.shape[0]))
        cols = [[bb.data[r][c] for r in range(n)]
                for c in range(bb.shape[1])]
    else:
        cols = [bb._flat()]

    sols = []
    for bv in cols:
        uy = [_fsum(u.data[r][c] * bv[r] for r in range(n))
              for c in range(len(svl))]
        z = [uy[c] / svl[c] if svl[c] > cut else 0.0
             for c in range(len(svl))]
        sols.append([_fsum(vt.data[c][j] * z[c]
                                for c in range(len(svl)))
                     for j in range(k)])

    if two_d:
        x = marr([[sols[c][j] for c in range(len(sols))]
                  for j in range(k)])
    else:
        x = marr(sols[0])
    resid = marr([])
    rank = _pysum(1 for v in svl if v > cut)
    return x, resid, rank, sv


linalg.lstsq = _lstsq
linalg.svd = _svd


# --------------------------------------------------------------- fft

class carr:
    """Minimal 1-D complex array for FFT results."""

    def __init__(self, data):
        rows = None
        if isinstance(data, carr):
            rows, data = data.rows, data.data
        elif isinstance(data, (list, tuple)) and data and isinstance(
                data[0], (list, tuple, carr, marr)):
            rows = [[complex(v) for v in
                     (r.data if isinstance(r, carr) else
                      r._flat() if isinstance(r, marr) else r)]
                    for r in data]
            data = [v for r in rows for v in r]
        self.rows = rows                      # None => 1-D
        self.data = [complex(v) for v in data]

    def __len__(self):
        return len(self.rows) if self.rows is not None else len(self.data)

    def __iter__(self):
        if self.rows is not None:
            return iter([carr(r) for r in self.rows])
        return iter(self.data)

    def __getitem__(self, i):
        if self.rows is not None:
            return carr(self.rows[i]) if isinstance(i, slice) \
                else carr(self.rows[i])
        if isinstance(i, slice):
            return carr(self.data[i])
        if isinstance(i, (marr, list, tuple)):
            # boolean mask or integer fancy index, as marr supports
            sel = list(i._flat()) if isinstance(i, marr) else list(i)
            if getattr(i, "_is_mask", False) or (
                    sel and _bi.all(isinstance(v, bool) for v in sel)):
                if len(sel) != len(self.data):
                    raise IndexError("boolean index did not match the array")
                return carr([v for v, keep in zip(self.data, sel) if keep])
            return carr([self.data[int(v)] for v in sel])
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

    def copy(self):
        return carr(self)

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
        if self.rows is not None:
            return (len(self.rows), len(self.rows[0]) if self.rows else 0)
        return (len(self.data),)

    @property
    def T(self):
        if self.rows is None:
            return carr(self.data)
        return carr([list(c) for c in zip(*self.rows)])

    def __abs__(self):
        if self.rows is not None:
            return marr([[_bi.abs(v) for v in r] for r in self.rows])
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


def _rows2d(x):
    """The rows of a 2-D input, or None if it is 1-D.

    Without this the transforms below flattened a matrix and returned a
    single spectrum -- a silently wrong answer, not an error.
    """
    if isinstance(x, carr):
        return [r[:] for r in x.rows] if x.rows is not None else None
    if isinstance(x, marr):
        return [[complex(v) for v in r] for r in x.data] \
            if len(x.shape) == 2 else None
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)) and x and \
            isinstance(x[0], (list, tuple)):
        return [[complex(v) for v in r] for r in x]
    return None


def _fft_axis(x, n, axis, one_d):
    """Apply a 1-D transform along `axis`; None if `x` is not 2-D."""
    rows = _rows2d(x)
    if rows is None:
        return None
    if axis in (-1, 1):
        return carr([one_d(r, n) for r in rows])
    if axis == 0:
        cols = [one_d(list(c), n) for c in zip(*rows)]
        return carr([list(r) for r in zip(*cols)])
    raise ValueError("axis %r is out of bounds for a 2-D transform"
                     % (axis,))


def _pad(a, n):
    if n is None:
        return a
    return a[:n] + [0j] * _bi.max(0, n - len(a))


class _FFT:
    @staticmethod
    def _fft1(a, n=None):
        return _fft_any(_pad(a, n), False)

    @staticmethod
    def _ifft1(a, n=None):
        a = _pad(a, n)
        return [v / len(a) for v in _fft_any(a, True)]

    @staticmethod
    def _rfft1(a, n=None):
        a = _pad(a, n)
        return _fft_any(a, False)[:len(a) // 2 + 1]

    @staticmethod
    def fft(x, n=None, axis=-1):
        got = _fft_axis(x, n, axis, _FFT._fft1)
        if got is not None:
            return got
        if axis not in (-1, 0):
            raise ValueError("axis %r is out of bounds for a 1-D "
                             "transform" % (axis,))
        return carr(_FFT._fft1(_tocomplex(x), n))

    @staticmethod
    def ifft(x, n=None, axis=-1):
        got = _fft_axis(x, n, axis, _FFT._ifft1)
        if got is not None:
            return got
        if axis not in (-1, 0):
            raise ValueError("axis %r is out of bounds for a 1-D "
                             "transform" % (axis,))
        return carr(_FFT._ifft1(_tocomplex(x), n))

    @staticmethod
    def rfft(x, n=None, axis=-1):
        got = _fft_axis(x, n, axis, _FFT._rfft1)
        if got is not None:
            return got
        if axis not in (-1, 0):
            raise ValueError("axis %r is out of bounds for a 1-D "
                             "transform" % (axis,))
        return carr(_FFT._rfft1(_tocomplex(x), n))

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
        if a and isinstance(a[0], ndlist):
            # rank-3 input: predicate block by block
            return ndlist(wrapped(marr(b), *a[1:], **k) for b in a[0])
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


def _is_float_dtype(dtype):
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    return str(name).startswith(("float", "f", "complex")) or dtype is float64


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
    if dtype is not None and dtype is not float and not _is_float_dtype(dtype):
        out = logspace(start, stop, num, endpoint, base)
        # numpy casts by truncation toward zero for an integer dtype
        return _typed(marr([float(int(v)) for v in out._flat()]), int)
    if not endpoint:
        step = (stop - start) / num
        return marr([base ** (start + i * step) for i in range(num)])
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
    av, vv = _conv_args(a, v)
    av, vv = list(av), list(vv)
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
    # valid: |n - m| + 1 lags where the shorter input sits fully inside
    # the longer one; numpy gives the same count whichever is shorter.
    lo = _bi.min(n, m) - 1
    return marr(full[lo:lo + abs(n - m) + 1])


def unwrap(p, discont=None, axis=-1, period=2 * _math.pi):
    a = asarray(p)
    if len(a.shape) == 2:
        if axis in (0, -2):
            return transpose(unwrap(transpose(a), discont, -1, period))
        return marr([list(unwrap(row, discont, -1, period)._flat())
                     for row in a.data])
    if period != 2 * _math.pi:
        s = 2 * _math.pi / period
        return unwrap([v * s for v in a._flat()], None if discont is None
                      else discont * s, -1) / s
    v = list(a._flat())
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


def _ieee_pow(x, y):
    """numpy power: a negative base to a fractional exponent is nan, not
    a complex number; overflow is inf."""
    try:
        r = x ** y
    except OverflowError:
        return _INF
    except ZeroDivisionError:
        return _INF
    return _NAN if isinstance(r, complex) else r


def power(a, b):
    return asarray(a)._zip(b, _ieee_pow)


def gradient(f, *varargs, axis=None):
    a = asarray(f)
    if len(a.shape) == 2:
        # a 2-D input: one array per axis (numpy returns the list), or
        # the requested axis alone
        if axis is None:
            return [gradient(a, *varargs, axis=0),
                    gradient(a, *varargs, axis=1)]
        if axis in (1, -1):
            return marr([gradient(row, *varargs).tolist() for row in a.data])
        cols = [gradient([a.data[i][j] for i in range(a.shape[0])],
                         *varargs).tolist() for j in range(a.shape[1])]
        return marr([[cols[j][i] for j in range(a.shape[1])]
                     for i in range(a.shape[0])])
    v = list(a._flat())
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
    # numpy: |x| > 1 is nan, |x| == 1 is +-inf, not a ValueError
    return _map_unary(x, _ieee(lambda v: _math.atanh(v) if _bi.abs(v) != 1.0
                                else _math.copysign(_INF, v)))


def arcsinh(x):
    return _map_unary(x, _ieee(_math.asinh))


def arccosh(x):
    return _map_unary(x, _ieee(_math.acosh))


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
    """numpy.ptp: max - min, NaN if any NaN in the slice."""
    x = asarray(a)
    def _p(vals):
        vals = list(vals)
        return _nan_ext(vals, _bi.max) - _nan_ext(vals, _bi.min)
    if axis is None or len(x.shape) == 1:
        return _p(x._flat())
    if axis in (0, -2):
        return marr([_p(x.data[i][j] for i in range(x.shape[0]))
                     for j in range(x.shape[1])])
    return marr([_p(row) for row in x.data])


def _pad_widths(pad_width, nd):
    """numpy's pad_width forms: int, (before, after), or one pair per axis."""
    if isinstance(pad_width, int):
        return [(pad_width, pad_width)] * nd
    pw = list(pad_width)
    if pw and isinstance(pw[0], (list, tuple)):
        pairs = [tuple(int(v) for v in p) for p in pw]
        return pairs * nd if len(pairs) == 1 else pairs
    lo, hi = pw
    return [(int(lo), int(hi))] * nd


def _pad_axis(seq, lo, hi, mode, c, wrap_scalar):
    """Pad one axis of a nested list (elements are sub-lists or floats)."""
    n = len(seq)
    if mode == "constant":
        return [wrap_scalar(c) for _ in range(lo)] + list(seq) + [wrap_scalar(c) for _ in range(hi)]
    if n == 0:
        raise ValueError("cannot pad an empty axis with mode %r" % mode)
    if mode == "edge":
        return [seq[0]] * lo + list(seq) + [seq[-1]] * hi
    if mode in ("symmetric", "reflect", "wrap"):
        def pick(k):
            if mode == "wrap":
                return seq[k % n]
            if mode == "symmetric":
                period = 2 * n
                k %= period
                return seq[k] if k < n else seq[period - 1 - k]
            if n == 1:
                return seq[0]
            period = 2 * (n - 1)
            k %= period
            return seq[k] if k < n else seq[period - k]
        return [pick(k) for k in range(-lo, 0)] + list(seq) + [pick(k) for k in range(n, n + hi)]
    raise ValueError("unsupported pad mode %r" % (mode,))


def _pad_nested(nested, pairs, mode, c, depth=0):
    if depth == len(pairs) - 1:
        return _pad_axis(nested, pairs[depth][0], pairs[depth][1], mode, c, lambda v: v)
    inner = [_pad_nested(sub, pairs, mode, c, depth + 1) for sub in nested]
    proto = inner[0] if inner else None

    def blank(v):
        def fill(t):
            return [fill(u) for u in t] if isinstance(t, list) else v
        return fill(proto) if proto is not None else v
    return _pad_axis(inner, pairs[depth][0], pairs[depth][1], mode, c, blank)


def pad(a, pad_width, mode="constant", constant_values=0.0, **kw):
    """numpy.pad for 1-D, 2-D and nested rank-3 input; modes constant,
    edge, symmetric, reflect, wrap."""
    del kw
    if isinstance(a, list) and _nested_depth(a) >= 3:
        pairs = _pad_widths(pad_width, 3)
        return ndlist(_pad_nested(a, pairs, mode, float(constant_values)))
    arr = asarray(a)
    if isinstance(arr, ndlist) or (hasattr(arr, "shape") and len(arr.shape) == 3):
        pairs = _pad_widths(pad_width, 3)
        return ndlist(_pad_nested(arr.tolist(), pairs, mode, float(constant_values)))
    if len(arr.shape) == 2:
        pairs = _pad_widths(pad_width, 2)
        out = _pad_nested([row[:] for row in arr.data], pairs, mode, float(constant_values))
        if not out or not out[0]:
            return _empty2d(len(out), len(out[0]) if out else 0)
        return marr(out)
    v = list(arr._flat())
    pairs = _pad_widths(pad_width, 1)
    return marr(_pad_nested(v, pairs, mode, float(constant_values)))


def packbits(a, bitorder="big"):
    bits = [1 if v != 0 else 0 for v in asarray(a)._flat()]
    while len(bits) % 8:
        bits.append(0)
    out = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        byte = 0
        if bitorder == "little":
            for k, b in enumerate(chunk):
                byte |= b << k
        else:
            for b in chunk:
                byte = (byte << 1) | b
        out.append(float(byte))
    res = marr(out)
    res._dt = "uint8"
    return res


def unpackbits(a, bitorder="big"):
    out = []
    order = range(8) if bitorder == "little" else range(7, -1, -1)
    for v in asarray(a)._flat():
        byte = int(v) & 0xFF
        for k in order:
            out.append(float((byte >> k) & 1))
    res = marr(out)
    res._dt = "uint8"
    return res


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


def scalar_out(x, out):
    """Hand a float back when the caller passed a scalar.

    This core has no 0-d array: asarray(0.5) is a one-element marr, so a
    callable that vectorises its input returns marr([v]) where numpy
    returns a scalar. Callables documented as scalar-in, scalar-out pass
    their original argument and their result through here.
    """
    if isinstance(x, (int, float)) and not isinstance(x, bool) \
            and isinstance(out, marr) and out.size == 1:
        return float(out.data[0])
    return out


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
    tol = 1e-9 * _bi.max(1.0, _bi.max(_bi.abs(z) for z in rs))
    if all(_bi.abs(z.imag) <= tol for z in rs):
        # numpy returns real roots as floats, largest first (the
        # companion-matrix eigenvalue order); a complex array made
        # max(np.roots(p)) fail on the comparison
        return marr(sorted((z.real for z in rs), reverse=True))
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

    def __getattr__(self, name):
        # numpy.random.<distribution>(...) for every method the Generator
        # has (negative_binomial, standard_t, ...): the explicit wrappers
        # below cover the common ones, this covers the rest on the global
        # stream. __getattr__ only runs for names not found normally.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_global"), name)

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

    def laplace(self, loc=0.0, scale=1.0, size=None):
        return self._global.laplace(loc, scale, size)

    def choice(self, a, size=None, replace=True, p=None):
        return self._global.choice(a, size, replace, p)

    def shuffle(self, seq):
        return self._global.shuffle(seq)

    def permutation(self, n):
        return self._global.permutation(n)

    @staticmethod
    def default_rng(seed=None):
        if isinstance(seed, _SplitMix64):
            return seed  # numpy: default_rng(generator) is that generator
        if isinstance(seed, Philox):
            seed = seed.key
        return _SplitMix64(seed if seed is not None else 0)


random = _RandomNS()


def frombuffer(buf, dtype="float64", count=-1):
    key = _dtype_name(dtype)
    fmt, size = _DTYPE_FMT.get(key, ("d", 8))
    n = len(buf) // size if count in (-1, None) else int(count)
    vals = _struct.unpack("<%d%s" % (n, fmt), bytes(buf[:n * size]))
    out = marr([float(v) for v in vals])
    out._dt = None if key == "float64" else key
    return out


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


# ------------------------------------------------- campaign gap fills (09-22)

def copy(x):  # numpy.copy
    return asarray(x).copy()


def triu_indices_from(a, k=0):
    sh = asarray(a).shape
    if len(sh) != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(sh[0], k, sh[1])


def tril_indices_from(a, k=0):
    sh = asarray(a).shape
    if len(sh) != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(sh[0], k, sh[1])


def broadcast_shapes(*shapes):
    nd = _bi.max(len(sh) for sh in shapes)
    out = []
    for k in _bi.range(nd):
        dims = {sh[len(sh) - nd + k] for sh in shapes if len(sh) - nd + k >= 0}
        dims.discard(1)
        if len(dims) > 1:
            raise ValueError("shape mismatch: objects cannot be broadcast "
                             "to a single shape %r" % (shapes,))
        out.append(dims.pop() if dims else 1)
    return tuple(out)


def broadcast_arrays(*args):
    arrs = [asarray(a) for a in args]
    shape = broadcast_shapes(*[a.shape for a in arrs])
    return [broadcast_to(a, shape) for a in arrs]


def histogramdd(sample, bins=10, range=None, density=False):  # noqa: A002
    """numpy.histogramdd for a (n, d) sample; returns (counts, edges).

    ``bins`` is an int, a sequence of ints, or a sequence of edge arrays.
    Counts come back as a rank-d nested container (marr for d == 2,
    ndlist above that) so the callers' ``.sum()`` / flattening work.
    """
    if hasattr(sample, "shape"):
        a = atleast_2d(sample)                 # an (N, D) array
    else:
        # numpy: anything without a .shape is a sequence of D coordinate
        # arrays, i.e. (D, N); [x] is N points in one dimension, not one
        # point in N dimensions (which allocated bins ** N cells)
        a = atleast_2d(asarray(sample)).T
    n, d = a.shape
    if isinstance(bins, int):
        bins = [bins] * d
    edges = []
    for j in _bi.range(d):
        col = [a.data[i][j] for i in _bi.range(n)]
        bj = bins[j]
        if isinstance(bj, int):
            lo, hi = (range[j] if range is not None else
                      (_bi.min(col), _bi.max(col)))
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges.append([lo + (hi - lo) * t / bj for t in _bi.range(bj + 1)])
        else:
            edges.append([float(v) for v in asarray(bj)._flat()])
    dims = [len(e) - 1 for e in edges]

    def build(k):
        if k == d:
            return 0.0
        return [build(k + 1) for _ in _bi.range(dims[k])]
    counts = build(0)
    for i in _bi.range(n):
        cell = []
        for j in _bi.range(d):
            e = edges[j]
            v = a.data[i][j]
            if v < e[0] or v > e[-1]:
                cell = None
                break
            b = 0
            while b < dims[j] - 1 and v >= e[b + 1]:
                b += 1
            cell.append(b)
        if cell is None:
            continue
        ref = counts
        for b in cell[:-1]:
            ref = ref[b]
        ref[cell[-1]] += 1.0
    if density:
        total = float(n)

        def scale(node, k, vol):
            if k == d:
                return node / (total * vol)
            return [scale(node[b], k + 1,
                          vol * (edges[k][b + 1] - edges[k][b]))
                    for b in _bi.range(dims[k])]
        counts = scale(counts, 0, 1.0)
    out = marr(counts) if d <= 2 else ndlist(counts)
    return out, [marr(e) for e in edges]


class _StrideTricks:
    @staticmethod
    def sliding_window_view(x, window_shape, axis=None):
        """Windows of a 1-D or 2-D marr; rank 1 -> (n-k+1, k) marr,
        rank 2 -> (n-kh+1, m-kw+1, kh, kw) ndlist. Copies, not views."""
        del axis
        a = asarray(x)
        if len(a.shape) == 1:
            k = int(window_shape[0] if isinstance(window_shape, (tuple, list))
                    else window_shape)
            n = a.shape[0]
            if k > n:
                raise ValueError("window shape cannot be larger than input array shape")
            return marr([a.data[i:i + k] for i in _bi.range(n - k + 1)])
        kh, kw = (int(v) for v in window_shape)
        n, m = a.shape
        if kh > n or kw > m:
            raise ValueError("window shape cannot be larger than input array shape")
        return ndlist([[marr([r[j:j + kw] for r in a.data[i:i + kh]])
                        for j in _bi.range(m - kw + 1)]
                       for i in _bi.range(n - kh + 1)])


class _Lib:
    stride_tricks = _StrideTricks()


lib = _Lib()


def _tag_int(fn):
    """Index-producing functions return integer arrays in numpy; tag the
    marr so the mask heuristics leave them alone."""
    def wrapped(*a, **k):
        out = fn(*a, **k)
        if fn.__name__ == "where" and len(a) + len(k) > 1:
            return out          # where(cond, x, y) returns values
        if isinstance(out, marr):
            _typed(out, int)
        elif isinstance(out, tuple):
            for o in out:
                if isinstance(o, marr):
                    _typed(o, int)
        return out
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


for _iname in ("where", "nonzero", "flatnonzero", "argsort", "argmax",
               "argmin", "searchsorted", "digitize", "argwhere"):
    if _iname in globals() and callable(globals()[_iname]):
        globals()[_iname] = _tag_int(globals()[_iname])
del _iname


def logical_and(a, b):
    return asarray(a) & asarray(b)


def logical_or(a, b):
    return asarray(a) | asarray(b)


def logical_xor(a, b):
    return asarray(a) ^ asarray(b)


def logical_not(a):
    return ~asarray(a)


# ------------------------------------------------ round-four additions (H)

def argwhere(x):
    """Indices of the non-zero elements, one row per element (k, ndim)."""
    a = asarray(x)
    if len(a.shape) == 2:
        out = [[float(i), float(j)] for i, row in enumerate(a.data)
               for j, v in enumerate(row) if v]
        res = marr(out) if out else marr([[]])
        if not out:
            res = zeros((0, 2))
    else:
        out = [[float(i)] for i, v in enumerate(a.data) if v]
        res = marr(out) if out else zeros((0, 1))
    res._is_index = True
    return _typed(res, int)


def argpartition(x, kth, axis=-1, kind=None, order=None):
    """Indices that partition; the sorted permutation satisfies the
    partition contract, as partition() does in this core."""
    del kth, kind, order
    return argsort(x, axis=axis)


def isneginf(x):
    out = _map_unary(x, lambda v: 1.0 if v == -_math.inf else 0.0)
    if isinstance(out, marr):
        out._is_mask = True
    return out


def isposinf(x):
    out = _map_unary(x, lambda v: 1.0 if v == _math.inf else 0.0)
    if isinstance(out, marr):
        out._is_mask = True
    return out


def reshape(x, shape, order="C"):
    del order
    return asarray(x).reshape(shape)


def vdot(a, b):
    """Flattened dot product with the first argument conjugated."""
    fa = list(asarray(a)._flat())
    fb = list(asarray(b)._flat())
    if len(fa) != len(fb):
        raise ValueError("vdot: size mismatch")
    if _bi.any(isinstance(v, complex) for v in fa + fb):
        return _bi.sum((v.conjugate() if isinstance(v, complex) else v) * w
                       for v, w in zip(fa, fb))
    return float(_fsum(v * w for v, w in zip(fa, fb)))


def sinc(x):
    def one(v):
        if v == 0:
            return 1.0
        return _math.sin(_math.pi * v) / (_math.pi * v)
    return _map_unary(x, one)


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply ``func1d`` to every 1-D slice of a 2-D array along ``axis``.
    Scalar results give a vector; vector results are re-stacked along
    the same axis, as numpy does."""
    a = asarray(arr)
    if len(a.shape) == 1:
        return func1d(a, *args, **kwargs)
    if axis in (0, -2):
        slices = [marr([a.data[r][c] for r in range(a.shape[0])])
                  for c in range(a.shape[1])]
    else:
        slices = [marr(row[:]) for row in a.data]
    outs = [func1d(s, *args, **kwargs) for s in slices]
    if _bi.all(not isinstance(o, (list, tuple, marr)) for o in outs):
        return marr([float(o) for o in outs])
    rows = [list(asarray(o)._flat()) for o in outs]
    if axis in (0, -2):
        return transpose(marr(rows))
    return marr(rows)


def moveaxis(a, source, destination):
    """Move an axis to a new position (2-D and nested rank-3 inputs)."""
    x = a if isinstance(a, marr) else asarray(a) if _nested_depth(a) <= 2 \
        else ndlist(a)
    nd = 2 if isinstance(x, marr) and len(x.shape) == 2 else \
        1 if isinstance(x, marr) else _nested_depth(x.tolist() if hasattr(x, "tolist") else x)
    src = source % nd if isinstance(source, int) else list(source)
    dst = destination % nd if isinstance(destination, int) else list(destination)
    if isinstance(src, list):
        order = [i for i in range(nd) if i not in src]
        for s, d in sorted(zip(src, dst), key=lambda t: t[1]):
            order.insert(d, s)
    else:
        order = [i for i in range(nd) if i != src]
        order.insert(dst, src)
    if order == list(range(nd)):
        return x
    if nd == 2:
        return transpose(x)
    nested = x.tolist() if hasattr(x, "tolist") else x
    return transpose(ndlist(nested), axes=order)


class broadcast:
    """numpy.broadcast: the broadcast shape of the arguments, iterable as
    tuples of elements."""

    def __init__(self, *args):
        self._arrs = [asarray(v) for v in args]
        shapes = [tuple(a.shape) for a in self._arrs]
        nd = _bi.max([len(s) for s in shapes] + [0])
        shape = []
        for k in range(nd):
            dims = {s[len(s) - nd + k] for s in shapes if len(s) - nd + k >= 0}
            dims.discard(1)
            if len(dims) > 1:
                raise ValueError("shape mismatch: objects cannot be "
                                 "broadcast to a single shape")
            shape.append(dims.pop() if dims else 1)
        self.shape = tuple(shape)
        self.nd = self.ndim = nd
        self.size = 1
        for d in shape:
            self.size *= d
        self.numiter = len(self._arrs)

    def __iter__(self):
        flats = [list(broadcast_to(a, self.shape)._flat()) if a.shape != self.shape
                 else list(a._flat()) for a in self._arrs]
        return iter(zip(*flats))


class _DType:
    """numpy.dtype descriptor: name, kind, itemsize, equality against
    strings and Python types."""

    _KIND = {"float64": "f", "float32": "f", "float16": "f", "int64": "i",
             "int32": "i", "int16": "i", "int8": "i", "uint8": "u",
             "uint16": "u", "uint32": "u", "uint64": "u", "bool": "b",
             "complex128": "c", "object": "O", "str": "U"}
    _SIZE = {"float64": 8, "float32": 4, "float16": 2, "int64": 8,
             "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "uint16": 2,
             "uint32": 4, "uint64": 8, "bool": 1, "complex128": 16,
             "object": 8, "str": 4}

    def __init__(self, spec):
        if isinstance(spec, _DType):
            name = spec.name
        elif spec is float or spec is float64:
            name = "float64"
        elif spec is int:
            name = "int64"
        elif spec is bool:
            name = "bool"
        elif spec is complex:
            name = "complex128"
        elif spec is object:
            name = "object"
        elif spec is str:
            name = "str"
        elif hasattr(spec, "name") and isinstance(spec.name, str):
            name = spec.name
        else:
            name = str(spec)
            name = {"f8": "float64", "f4": "float32", "f2": "float16",
                    "i8": "int64", "i4": "int32", "i2": "int16", "i1": "int8",
                    "u1": "uint8", "u2": "uint16", "u4": "uint32", "u8": "uint64",
                    "float": "float64", "int": "int64", "bool_": "bool",
                    "?": "bool", "O": "object", "U": "str", "<U": "str",
                    "double": "float64", "single": "float32", "complex": "complex128",
                    "c16": "complex128", "<f8": "float64", "<i8": "int64",
                    "<f4": "float32", "<i4": "int32"}.get(name, name)
            if name.startswith("<U") or name.startswith("U"):
                name = "str"
        self.name = name
        self.kind = self._KIND.get(name, "O")
        self.itemsize = self._SIZE.get(name, 8)
        self.type = {"f": float, "i": int, "u": int, "b": bool,
                     "c": complex, "U": str}.get(self.kind, object)
        self.char = {"float64": "d", "float32": "f", "int64": "l",
                     "int32": "i", "bool": "?"}.get(name, "O")
        self.str = "<" + {"f": "f", "i": "i", "u": "u", "b": "b",
                          "c": "c", "U": "U"}.get(self.kind, "O") + str(self.itemsize)

    def __eq__(self, other):
        try:
            return _DType(other).name == self.name
        except Exception:  # noqa: BLE001
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "dtype('%s')" % self.name

    __str__ = __repr__


dtype = _DType


def savez(file, *args, **kwds):
    """Save arrays into one file (a zip of JSON lists, readable by
    load()); positional arrays are named arr_0, arr_1, ... as in numpy."""
    import json as _json
    import zipfile as _zip
    arrays = {"arr_%d" % i: v for i, v in enumerate(args)}
    arrays.update(kwds)
    path = file if isinstance(file, str) else getattr(file, "name", str(file))
    if isinstance(file, str) and not path.endswith(".npz"):
        path = path + ".npz"
    with _zip.ZipFile(path, "w") as zf:
        for k, v in arrays.items():
            data = v.tolist() if hasattr(v, "tolist") else v
            zf.writestr(k + ".json", _json.dumps(data))
    return path


savez_compressed = savez


class _NpzFile(dict):
    files = property(lambda self: list(self.keys()))

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def load(file, allow_pickle=False, **kw):
    """Read a savez() archive back as a mapping of arrays."""
    del allow_pickle, kw
    import json as _json
    import zipfile as _zip
    path = file if isinstance(file, str) else getattr(file, "name", str(file))
    out = _NpzFile()
    with _zip.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                out[name[:-5]] = array(_json.loads(zf.read(name).decode()))
    return out


def _golub_welsch(n, off, mu0):
    """Nodes and weights of an n-point Gauss rule from the symmetric
    Jacobi matrix with zero diagonal and off-diagonal ``off`` (length
    n-1); mu0 is the total mass of the weight function."""
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")
    if n == 1:
        return marr([0.0]), marr([mu0])
    J = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        J[i][i + 1] = J[i + 1][i] = off[i]
    w, v = _eigh(marr(J))
    nodes = [float(x) for x in w._flat()]
    weights = [mu0 * float(v.data[0][i]) ** 2 for i in range(n)]
    # symmetrise numerically: the rules are exactly symmetric about 0
    for i in range(n // 2):
        j = n - 1 - i
        m = 0.5 * (abs(nodes[i]) + abs(nodes[j]))
        nodes[i], nodes[j] = -m, m
        wm = 0.5 * (weights[i] + weights[j])
        weights[i] = weights[j] = wm
    if n % 2 == 1:
        nodes[n // 2] = 0.0
    return marr(nodes), marr(weights)


class _PolyHermite:
    @staticmethod
    def hermgauss(deg):
        """Gauss-Hermite (physicists', weight exp(-x^2))."""
        n = int(deg)
        return _golub_welsch(n, [_math.sqrt(i / 2.0) for i in range(1, n)],
                             _math.sqrt(_math.pi))

    @staticmethod
    def hermval(x, c):
        cs = list(asarray(c)._flat())

        def one(v):
            h0, h1 = 1.0, 2.0 * v
            tot = cs[0] * h0 if cs else 0.0
            if len(cs) > 1:
                tot += cs[1] * h1
            for k in range(2, len(cs)):
                h0, h1 = h1, 2.0 * v * h1 - 2.0 * (k - 1) * h0
                tot += cs[k] * h1
            return tot
        return _map_unary(x, one)


class _PolyHermiteE:
    @staticmethod
    def hermegauss(deg):
        """Gauss-Hermite (probabilists', weight exp(-x^2/2))."""
        n = int(deg)
        return _golub_welsch(n, [_math.sqrt(float(i)) for i in range(1, n)],
                             _math.sqrt(2.0 * _math.pi))

    @staticmethod
    def hermeval(x, c):
        cs = list(asarray(c)._flat())

        def one(v):
            h0, h1 = 1.0, v
            tot = cs[0] * h0 if cs else 0.0
            if len(cs) > 1:
                tot += cs[1] * h1
            for k in range(2, len(cs)):
                h0, h1 = h1, v * h1 - (k - 1) * h0
                tot += cs[k] * h1
            return tot
        return _map_unary(x, one)


class _PolyLegendre:
    @staticmethod
    def leggauss(deg):
        """Gauss-Legendre on [-1, 1]."""
        n = int(deg)
        return _golub_welsch(
            n, [i / _math.sqrt(4.0 * i * i - 1.0) for i in range(1, n)], 2.0)

    @staticmethod
    def legval(x, c):
        cs = list(asarray(c)._flat())

        def one(v):
            p0, p1 = 1.0, v
            tot = cs[0] * p0 if cs else 0.0
            if len(cs) > 1:
                tot += cs[1] * p1
            for k in range(2, len(cs)):
                p0, p1 = p1, ((2 * k - 1) * v * p1 - (k - 1) * p0) / k
                tot += cs[k] * p1
            return tot
        return _map_unary(x, one)


class _PolyPolynomial:
    """numpy.polynomial.polynomial: coefficients in INCREASING order."""

    @staticmethod
    def polyfit(x, y, deg):
        return polyfit(x, y, deg)[::-1]

    @staticmethod
    def polyval(x, c):
        cs = list(asarray(c)._flat())
        return _map_unary(x, lambda v: _fsum(ck * v ** k for k, ck in enumerate(cs)))

    @staticmethod
    def polyder(c, m=1):
        cs = list(asarray(c)._flat())
        for _ in range(int(m)):
            cs = [k * cs[k] for k in range(1, len(cs))] or [0.0]
        return marr(cs)


class polynomial:  # namespace mirror of numpy.polynomial
    hermite = _PolyHermite
    hermite_e = _PolyHermiteE
    legendre = _PolyLegendre
    polynomial = _PolyPolynomial


# ------------------------------------------------ ufunc.outer (round four)

def _outer_of(fn):
    """numpy's ufunc.outer: fn applied to every pair, shape (n, m)."""
    def outer(a, b):
        fa = list(asarray(a)._flat())
        fb = list(asarray(b)._flat())
        return marr([[fn(x, y) for y in fb] for x in fa])
    return outer


for _name, _op in (("subtract", lambda x, y: x - y), ("multiply", lambda x, y: x * y),
                   ("divide", _ieee_div), ("maximum", lambda x, y: x if x >= y else y),
                   ("minimum", lambda x, y: x if x <= y else y), ("power", lambda x, y: x ** y)):
    _f = globals().get(_name)
    if _f is not None and not hasattr(_f, "outer"):
        try:
            _f.outer = _outer_of(_op)
        except (AttributeError, TypeError):
            pass
if not hasattr(add, "outer"):
    type(add).outer = staticmethod(_outer_of(lambda x, y: x + y))
