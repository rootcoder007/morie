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


class marr:
    """Minimal array: nested lists of floats, 1-D or 2-D."""

    __slots__ = ("data", "shape", "_is_mask", "_is_index")

    def __init__(self, data):
        if isinstance(data, marr):
            self.data = [row[:] for row in data.data] \
                if isinstance(data.data[0], list) else data.data[:]
            self.shape = data.shape
            return
        if isinstance(data, (int, float)):
            data = [float(data)]
        if hasattr(data, "tolist") and not isinstance(data, marr):
            data = data.tolist()                 # foreign arrays (numpy)
            if isinstance(data, (int, float)):
                data = [float(data)]
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
            self.data = [float(v) for v in data]
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
            i, j = idx
            if j is None:                       # x[:, None] -> column
                base = self._flat() if i == slice(None) else None
                if base is None:
                    raise ValueError("unsupported index")
                return marr([[v] for v in base])
            if i is None:                       # x[None, :] -> row
                return marr([self._flat()])
            if len(self.shape) == 2:
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
                return self.data[i][j]
            raise ValueError("unsupported index for 1-D")
        if isinstance(idx, marr):
            vals = idx._flat()
            is_mask = getattr(idx, "_is_mask", False) or (
                not getattr(idx, "_is_index", False)
                and idx.shape == (self.shape[0],)
                and _pyall(v in (0.0, 1.0) for v in vals))
            if is_mask:
                keep = [k for k, m in enumerate(vals) if m != 0]
            else:
                keep = [int(v) for v in vals]   # fancy integer indexing
            return marr([self.data[k] for k in keep])
        if hasattr(idx, "dtype") and hasattr(idx, "tolist"):
            # foreign (numpy) index array
            vals = idx.tolist()
            if getattr(idx.dtype, "kind", "") == "b":
                keep = [k for k, m in enumerate(vals) if m]
            else:
                keep = [int(v) for v in vals]
            return marr([self.data[k] for k in keep])
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

    def copy(self):
        return marr(self)

    def astype(self, dtype=None):
        del dtype
        return marr(self)

    def flatten(self):
        return marr(self._flat())

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        f = self._flat()
        if shape == (-1,) or shape == (len(f),):
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
        if isinstance(idx, tuple) and len(self.shape) == 2:
            i, j = idx
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
            self.data[idx] = float(value)

    def __buffer__(self, flags):
        """PEP 688 buffer protocol (Python >= 3.12): expose the flat
        float64 data so nanobind kernels and memoryview consumers get
        the array without numpy. Snapshot semantics: the exported
        buffer is a copy, matching the immutable-input contract of the
        compiled kernels."""
        del flags
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
    def sum(self, axis=None, keepdims=False):
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

    def mean(self, axis=None, keepdims=False):
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

    def var(self, axis=None, ddof=0):
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

    def std(self, axis=None, ddof=0):
        v = self.var(axis=axis, ddof=ddof)
        if isinstance(v, marr):
            return marr([_math.sqrt(u) for u in v._flat()])
        return _math.sqrt(v)

    def max(self):
        return float(_bi.max(self._flat()))

    def min(self):
        return float(_bi.min(self._flat()))

    def all(self):
        return _bi.all(v != 0 for v in self._flat())

    def any(self):
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
float64 = float


# ------------------------------------------------------------ construction

def asarray(x, dtype=None):
    del dtype
    return x if isinstance(x, marr) else marr(x)


def array(x, dtype=None):
    del dtype
    return marr(x)


def atleast_1d(x):
    return asarray(x)


def atleast_2d(x):
    a = asarray(x)
    return a if len(a.shape) == 2 else marr([a.data])


def arange(start, stop=None, step=1.0, dtype=None):
    del dtype
    if stop is None:
        start, stop = 0.0, start
    out = []
    v = float(start)
    n = _bi.max(0, int(_math.ceil((stop - start) / step - 1e-12)))
    for i in range(n):
        out.append(start + i * step)
    del v
    return marr(out)


def zeros(n, dtype=None):
    if isinstance(n, tuple):
        return marr([[0.0] * n[1] for _ in range(n[0])])
    return marr([0.0] * int(n))


def ones(n, dtype=None):
    if isinstance(n, tuple):
        return marr([[1.0] * n[1] for _ in range(n[0])])
    return marr([1.0] * int(n))


def full(n, v, dtype=None):
    return marr([float(v)] * int(n))


def linspace(a, b, n):
    n = int(n)
    if n == 1:
        return marr([float(a)])
    step = (b - a) / (n - 1)
    return marr([a + i * step for i in range(n)])


def eye(n):
    return marr([[1.0 if i == j else 0.0 for j in range(n)]
                 for i in range(n)])


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


def concatenate(parts):
    out = []
    for p in parts:
        out.extend(asarray(p)._flat())
    return marr(out)


# ------------------------------------------------------------- elementwise

def _uf(fn):
    def wrapped(x):
        a = asarray(x)
        if a.shape == (1,) and not isinstance(x, (list, tuple, marr)):
            return fn(a.data[0])
        return a._map(fn)
    return wrapped


sqrt = _uf(_math.sqrt)
exp = _uf(_math.exp)
log = _uf(_math.log)
log1p = _uf(_math.log1p)
abs = _uf(lambda v: v if v >= 0 else -v)  # noqa: A001


def clip(x, lo, hi):
    return asarray(x)._map(lambda v: lo if v < lo else hi if v > hi else v)


def maximum(x, y):
    return asarray(x)._zip(y, lambda a, b: a if a >= b else b)


def minimum(x, y):
    return asarray(x)._zip(y, lambda a, b: a if a <= b else b)


def where(cond, a=None, b=None):
    c = asarray(cond)
    if a is None:                       # np.where(mask) -> (indices,)
        return (marr([float(i) for i, v in enumerate(c._flat())
                      if v != 0]),)
    aa, bb = asarray(a), asarray(b)
    if aa.shape == (1,):
        aa = full(c.shape[0], aa.data[0])
    if bb.shape == (1,):
        bb = full(c.shape[0], bb.data[0])
    return marr([aa.data[i] if c.data[i] != 0 else bb.data[i]
                 for i in range(c.shape[0])])


def isfinite(x):
    if not isinstance(x, (list, tuple, marr)):
        return _math.isfinite(float(x))
    return asarray(x)._map(lambda v: 1.0 if _math.isfinite(v) else 0.0)


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

def sum(x, axis=None):  # noqa: A001
    return asarray(x).sum(axis=axis)


def mean(x, axis=None):
    return asarray(x).mean(axis=axis)


def std(x, axis=None, ddof=0):
    return asarray(x).std(axis=axis, ddof=ddof)


def var(x, axis=None, ddof=0):
    return asarray(x).var(axis=axis, ddof=ddof)


def max(x):  # noqa: A001
    return asarray(x).max()


def min(x):  # noqa: A001
    return asarray(x).min()


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


def unique(x):
    return marr(sorted(set(asarray(x)._flat())))


def allclose(a, b, atol=1e-8, rtol=1e-5):
    aa, bb = asarray(a)._flat(), asarray(b)._flat()
    if len(aa) != len(bb):
        return False
    return _pyall(_math.isclose(x, y, rel_tol=rtol, abs_tol=atol)
                  for x, y in zip(aa, bb))


_pyall = _bi.all
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
    def lstsq(a, b, rcond=None):
        del rcond
        aa = atleast_2d(a)
        at = aa.T
        beta = _Linalg.solve(matmul(at, aa), matmul(at, asarray(b)))
        return beta, None, None, None


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

class _SplitMix64:
    """Python fallback RNG (SplitMix64 -> floats).

    ponytail: placeholder until the native Philox hook from morie_core is
    wired in; deterministic, well-distributed, NOT the banned gr* LCG.
    """

    def __init__(self, seed):
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
        def one():
            u1 = _pymax(self.uniform(), 1e-300)
            u2 = self.uniform()
            return loc + scale * _math.sqrt(-2 * _math.log(u1)) \
                * _math.cos(2 * _math.pi * u2)
        return self._fill(one, size)

    def _u(self):
        return (self._next() >> 11) / (1 << 53)

    def poisson(self, lam=1.0, size=None):
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

    def choice(self, a, size=None, replace=True, p=None):
        if p is not None:
            raise NotImplementedError("weighted choice not needed yet")
        pool = list(range(int(a))) if isinstance(a, int)             else list(asarray(a)._flat())
        if size is None:
            return pool[self._next() % len(pool)]
        if isinstance(size, (tuple, list)):
            size = 1 if not size else int(
                size[0]) * (int(size[1]) if len(size) > 1 else 1)
        k = int(size)
        if replace:
            return marr([float(pool[self._next() % len(pool)])
                         for _ in range(k)])
        if k > len(pool):
            raise ValueError("cannot sample more than population without "
                             "replacement")
        idx = list(range(len(pool)))
        self.shuffle(idx)
        return marr([float(pool[i]) for i in idx[:k]])

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
    f = asarray(x)._flat()
    return marr(f * int(reps))


def repeat(x, reps):
    out = []
    for v in asarray(x)._flat():
        out.extend([v] * int(reps))
    return marr(out)


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


def _pinv(a, rcond=1e-15):
    """Moore-Penrose pseudoinverse.

    Symmetric input: Jacobi eigendecomposition.  General input: routed
    through the symmetric case via A+ = (A^T A)+ A^T.
    """
    aa = atleast_2d(a)
    n, mcols = aa.shape
    sym = n == mcols and _pyall(
        _bi.abs(aa.data[i][j] - aa.data[j][i]) < 1e-10
        for i in range(n) for j in range(i + 1, n))
    if not sym:
        ata = matmul(aa.T, aa)
        return matmul(_pinv(ata, rcond), aa.T)
    vals, vecs = _jacobi_eigh(aa)
    vmax = _pymax((_bi.abs(x) for x in vals), default=0.0)
    cut = rcond * _pymax(vmax, 1.0) * n
    out = [[0.0] * n for _ in range(n)]
    for k in range(n):
        if _bi.abs(vals[k]) <= cut:
            continue
        inv_l = 1.0 / vals[k]
        for i in range(n):
            for j in range(n):
                out[i][j] += vecs[i][k] * inv_l * vecs[j][k]
    return marr(out)


_LinalgExt.pinv = staticmethod(_pinv)
_LinalgExt.matrix_rank = staticmethod(_matrix_rank)
linalg.matrix_rank = _matrix_rank
linalg.pinv = _pinv
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


float32 = float
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


def argsort(x, kind=None):
    f = asarray(x)._flat()
    return marr([float(i) for i in
                 sorted(range(len(f)), key=lambda k: f[k])])


float16 = float


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


int16 = int
int8 = int
uint8 = int

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


def percentile(x, q):
    """Linear-interpolation percentile (numpy default method)."""
    f = sorted(asarray(x)._flat())
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
    if isinstance(q, (list, tuple)):
        return percentile(x, [100.0 * float(v) for v in q])
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

def corrcoef(x, y=None):
    if y is None:
        a = atleast_2d(x)
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


def stack(parts):
    return marr([asarray(p)._flat() for p in parts])


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


def delete(x, idx):
    f = asarray(x)._flat()
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


def nanmean(x):
    f = _nan_filter(x)
    return float(_math.fsum(f) / len(f))


def nansum(x):
    return float(_math.fsum(_nan_filter(x)))


def nanstd(x, ddof=0):
    f = _nan_filter(x)
    m = _math.fsum(f) / len(f)
    return _math.sqrt(_math.fsum((v - m) ** 2 for v in f) / (len(f) - ddof))


def nanvar(x, ddof=0):
    return nanstd(x, ddof=ddof) ** 2


def nanmax(x):
    return float(_bi.max(_nan_filter(x)))


def nanmin(x):
    return float(_bi.min(_nan_filter(x)))


def nanmedian(x):
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
    return (marr([float(v) for v in asarray(rows)._flat()]),
            marr([float(v) for v in asarray(cols)._flat()]))


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
    return asarray(x)


def imag(x):
    return zeros_like(asarray(x))


def angle(x):
    return asarray(x)._map(lambda v: 0.0 if v >= 0 else _math.pi)


def conjugate(x):
    return asarray(x)


conj = conjugate


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


def partition(x, k):
    return sort(x)   # sorted output satisfies the partition contract


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
    """SVD via eigh of A^T A (values) and A A^T (left vectors)."""
    del full_matrices
    aa = atleast_2d(a)
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
        compiled kernels."""
        del flags
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
    def fft(x, n=None):
        a = _tocomplex(x)
        if n is not None:
            a = a[:n] + [0j] * _bi.max(0, n - len(a))
        return carr(_fft_any(a, False))

    @staticmethod
    def ifft(x, n=None):
        a = _tocomplex(x)
        if n is not None:
            a = a[:n] + [0j] * _bi.max(0, n - len(a))
        out = _fft_any(a, True)
        return carr([v / len(a) for v in out])

    @staticmethod
    def rfft(x, n=None):
        a = _tocomplex(x)
        if n is not None:
            a = a[:n] + [0j] * _bi.max(0, n - len(a))
        full = _fft_any(a, False)
        return carr(full[:len(a) // 2 + 1])

    @staticmethod
    def irfft(x, n=None):
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
               "int64": ("q", 8), "int32": ("i", 4),
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
