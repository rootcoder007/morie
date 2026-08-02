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

    __slots__ = ("data", "shape")

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
            is_mask = (idx.shape == (self.shape[0],)
                       and _pyall(v in (0.0, 1.0) for v in vals))
            if is_mask:
                keep = [k for k, m in enumerate(vals) if m != 0]
            else:
                keep = [int(v) for v in vals]   # fancy integer indexing
            return marr([self.data[k] for k in keep])
        if isinstance(idx, (list, tuple)):
            return marr([self.data[int(k)] for k in idx])
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
        return self._zip(o, lambda a, b: 1.0 if (a != 0 or b != 0) else 0.0)

    def __and__(self, o):
        return self._zip(o, lambda a, b: 1.0 if (a != 0 and b != 0) else 0.0)

    def __bool__(self):
        f = self._flat()
        if len(f) != 1:
            raise ValueError("truth value of multi-element marr is "
                             "ambiguous")
        return f[0] != 0

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple) and len(self.shape) == 2:
            i, j = idx
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
        else:
            self.data[idx] = float(value)

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

    # -- comparisons (return 0/1 masks) -------------------------------
    def __lt__(self, o):
        return self._zip(o, lambda a, b: 1.0 if a < b else 0.0)

    def __le__(self, o):
        return self._zip(o, lambda a, b: 1.0 if a <= b else 0.0)

    def __gt__(self, o):
        return self._zip(o, lambda a, b: 1.0 if a > b else 0.0)

    def __ge__(self, o):
        return self._zip(o, lambda a, b: 1.0 if a >= b else 0.0)

    def __eq__(self, o):
        try:
            return self._zip(o, lambda a, b: 1.0 if a == b else 0.0)
        except (TypeError, ValueError):
            return NotImplemented

    def __ne__(self, o):
        try:
            return self._zip(o, lambda a, b: 1.0 if a != b else 0.0)
        except (TypeError, ValueError):
            return NotImplemented

    __hash__ = None

    def __abs__(self):
        return self._map(lambda v: v if v >= 0 else -v)

    def argmax(self):
        f = self._flat()
        return f.index(_bi.max(f))

    # -- reductions ----------------------------------------------------
    def sum(self, axis=None):
        if axis is None:
            return float(_math.fsum(self._flat()))
        if len(self.shape) != 2:
            raise ValueError("axis reduction needs a 2-D array")
        if axis == 0:
            return marr([_math.fsum(self.data[i][j]
                                    for i in range(self.shape[0]))
                         for j in range(self.shape[1])])
        return marr([_math.fsum(row) for row in self.data])

    def mean(self, axis=None):
        if axis is None:
            f = self._flat()
            return float(_math.fsum(f) / len(f))
        s = self.sum(axis=axis)
        d = self.shape[0] if axis == 0 else self.shape[1]
        return s / float(d)

    def var(self, ddof=0):
        f = self._flat()
        m = _math.fsum(f) / len(f)
        return float(_math.fsum((v - m) ** 2 for v in f) / (len(f) - ddof))

    def std(self, ddof=0):
        return _math.sqrt(self.var(ddof=ddof))

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


def zeros(n):
    if isinstance(n, tuple):
        return marr([[0.0] * n[1] for _ in range(n[0])])
    return marr([0.0] * int(n))


def ones(n):
    if isinstance(n, tuple):
        return marr([[1.0] * n[1] for _ in range(n[0])])
    return marr([1.0] * int(n))


def full(n, v):
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

def sum(x):  # noqa: A001
    return asarray(x).sum()


def mean(x):
    return asarray(x).mean()


def std(x, ddof=0):
    return asarray(x).std(ddof=ddof)


def var(x, ddof=0):
    return asarray(x).var(ddof=ddof)


def max(x):  # noqa: A001
    return asarray(x).max()


def min(x):  # noqa: A001
    return asarray(x).min()


def _axis_reduce(x, axis, red):
    a = atleast_2d(x)
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
                raise ValueError("singular matrix")
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
    def norm(x):
        return _math.sqrt(_math.fsum(v * v for v in asarray(x)._flat()))

    @staticmethod
    def lstsq(a, b, rcond=None):
        del rcond
        aa = atleast_2d(a)
        at = aa.T
        beta = _Linalg.solve(matmul(at, aa), matmul(at, asarray(b)))
        return beta, None, None, None


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

    def uniform(self, low=0.0, high=1.0, size=None):
        def one():
            return low + (high - low) * (self._next() >> 11) / (1 << 53)
        if size is None:
            return one()
        return marr([one() for _ in range(int(size))])

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
        if size is None:
            return one()
        return marr([one() for _ in range(int(size))])

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
        if size is None:
            return one()
        return marr([float(one()) for _ in range(int(size))])


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


def diff(x):
    f = asarray(x)._flat()
    return marr([f[i + 1] - f[i] for i in range(len(f) - 1)])


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


def zeros_like(x):
    a = asarray(x)
    if len(a.shape) == 2:
        return marr([[0.0] * a.shape[1] for _ in range(a.shape[0])])
    return marr([0.0] * a.shape[0])


def ones_like(x):
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


def argmax(x):
    return asarray(x).argmax()


def argmin(x):
    f = asarray(x)._flat()
    return f.index(_bi.min(f))


def argsort(x):
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


def empty(n):
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
