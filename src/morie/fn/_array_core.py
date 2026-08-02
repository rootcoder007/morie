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
        data = list(data)
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
        if o.shape == (1,):
            return self._map(lambda v: fn(v, o.data[0]))
        if self.shape == (1,):
            s = self.data[0]
            return o._map(lambda v: fn(s, v))
        if o.shape != self.shape:
            raise ValueError("shape mismatch %s vs %s"
                             % (self.shape, o.shape))
        if len(self.shape) == 1:
            return marr([fn(a, b) for a, b in zip(self.data, o.data)])
        return marr([[fn(a, b) for a, b in zip(r1, r2)]
                     for r1, r2 in zip(self.data, o.data)])

    # -- python protocol ---------------------------------------------
    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        if len(self.shape) == 1:
            return iter(self.data)
        return iter([marr(row) for row in self.data])

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(self.shape) == 2:
            i, j = idx
            return self.data[i][j]
        out = self.data[idx]
        if isinstance(out, list):
            return marr(out)
        return out

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple) and len(self.shape) == 2:
            i, j = idx
            self.data[i][j] = float(value)
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

    # -- reductions ----------------------------------------------------
    def sum(self):
        return float(_math.fsum(self._flat()))

    def mean(self):
        f = self._flat()
        return float(_math.fsum(f) / len(f))

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


def arange(start, stop=None, step=1.0):
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
    return marr([a.data[i][i] for i in range(min(a.shape))])


def column_stack(cols):
    cs = [asarray(c) for c in cols]
    n = cs[0].shape[0]
    if any(c.shape != (n,) for c in cs):
        raise ValueError("column_stack needs equal-length 1-D inputs")
    return marr([[c.data[i] for c in cs] for i in range(n)])


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


def where(cond, a, b):
    c = asarray(cond)
    aa, bb = asarray(a), asarray(b)
    if aa.shape == (1,):
        aa = full(c.shape[0], aa.data[0])
    if bb.shape == (1,):
        bb = full(c.shape[0], bb.data[0])
    return marr([aa.data[i] if c.data[i] != 0 else bb.data[i]
                 for i in range(c.shape[0])])


def isfinite(x):
    return asarray(x)._map(lambda v: 1.0 if _math.isfinite(v) else 0.0)


def dot(a, b):
    aa, bb = asarray(a), asarray(b)
    if len(aa.shape) == 1 and len(bb.shape) == 1:
        if aa.shape != bb.shape:
            raise ValueError("shape mismatch")
        return float(_math.fsum(x * y for x, y in zip(aa.data, bb.data)))
    return matmul(aa, bb)


def matmul(a, b):
    aa = atleast_2d(a)
    b_arr = asarray(b)
    b_was_1d = len(b_arr.shape) == 1
    bb = marr([[v] for v in b_arr.data]) if b_was_1d else b_arr
    n, k = aa.shape
    k2, m = bb.shape
    if k != k2:
        raise ValueError("shape mismatch")
    out = [[_math.fsum(aa.data[i][t] * bb.data[t][j] for t in range(k))
            for j in range(m)] for i in range(n)]
    if b_was_1d:
        return marr([row[0] for row in out])
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


def all(x):  # noqa: A001
    return asarray(x).all()


def any(x):  # noqa: A001
    return asarray(x).any()


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

    def normal(self, loc=0.0, scale=1.0, size=None):
        def one():
            u1 = _pymax(self.uniform(), 1e-300)
            u2 = self.uniform()
            return loc + scale * _math.sqrt(-2 * _math.log(u1)) \
                * _math.cos(2 * _math.pi * u2)
        if size is None:
            return one()
        return marr([one() for _ in range(int(size))])

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
