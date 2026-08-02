"""morie frame core: pandas subset (DataFrame / Series + module fns).

Native replacement for the pandas surface morie uses (inventory
l14:/tmp/pd_inventory.json + /tmp/pd_methods.json: DataFrame 1726,
Series 374, read_csv 87, concat/to_datetime/crosstab/qcut/get_dummies/
cut ...; methods dominated by sum/mean/astype/dropna/groupby/loc/values).
Pure Python, list-backed columns.  Equivalence-tested against pandas in
tests/fn/test_frame_core.py.

ponytail: positional alignment (RangeIndex semantics) — morie call
sites use default indexes; label-aligned arithmetic when a caller needs
it.
"""

from __future__ import annotations

import csv as _csv
import datetime as _dt
import math as _math

from . import _array_core as _ac

_NAN = float("nan")


def _isnan(v):
    return v is None or (isinstance(v, float) and v != v)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return _NAN


# ===================================================== Series

class Series:
    def __init__(self, data=None, index=None, name=None, dtype=None):
        if isinstance(data, Series):
            self._data = list(data._data)
            index = list(data.index) if index is None else index
            name = data.name if name is None else name
        elif isinstance(data, dict):
            index = list(data.keys()) if index is None else index
            self._data = [data.get(k, _NAN) for k in index]
        elif data is None:
            self._data = []
        elif hasattr(data, "tolist") and not isinstance(data, Series):
            self._data = list(data.tolist())
        elif isinstance(data, (int, float, str, bool)):
            n = len(index) if index is not None else 1
            self._data = [data] * n
        else:
            self._data = list(data)
        self.index = list(index) if index is not None \
            else list(range(len(self._data)))
        self.name = name
        if dtype is not None:
            self._data = _cast_list(self._data, dtype)

    # ---- basics
    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return "Series(%r, name=%r)" % (self._data[:10], self.name)

    @property
    def values(self):
        return _ac.marr([_to_float(v) for v in self._data]) \
            if self._is_numeric() else list(self._data)

    @property
    def empty(self):
        return len(self._data) == 0

    @property
    def size(self):
        return len(self._data)

    @property
    def shape(self):
        return (len(self._data),)

    @property
    def iloc(self):
        return _SeriesILoc(self)

    @property
    def loc(self):
        return _SeriesLoc(self)

    @property
    def str(self):
        return _StrAccessor(self)

    @property
    def dt(self):
        return _DtAccessor(self)

    def _is_numeric(self):
        return all(isinstance(v, (int, float, bool)) or _isnan(v)
                   for v in self._data)

    def to_numpy(self):
        return self.values

    def tolist(self):
        return list(self._data)

    to_list = tolist

    def copy(self):
        return Series(list(self._data), index=list(self.index),
                      name=self.name)

    def head(self, n=5):
        return Series(self._data[:n], index=self.index[:n],
                      name=self.name)

    def tail(self, n=5):
        return Series(self._data[-n:], index=self.index[-n:],
                      name=self.name)

    def items(self):
        return zip(self.index, self._data)

    def keys(self):
        return list(self.index)

    # ---- indexing
    def __getitem__(self, key):
        if isinstance(key, Series):
            key = key.tolist()
        if isinstance(key, list) and key and isinstance(key[0], bool):
            d = [v for v, m in zip(self._data, key) if m]
            ix = [i for i, m in zip(self.index, key) if m]
            return Series(d, index=ix, name=self.name)
        if isinstance(key, slice):
            return Series(self._data[key], index=self.index[key],
                          name=self.name)
        if key in self.index:
            return self._data[self.index.index(key)]
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(key, Series):
            key = key.tolist()
        if isinstance(key, list) and key and isinstance(key[0], bool):
            for i, m in enumerate(key):
                if m:
                    self._data[i] = value
            return
        if key in self.index:
            self._data[self.index.index(key)] = value
        else:
            self.index.append(key)
            self._data.append(value)

    def get(self, key, default=None):
        if key in self.index:
            return self._data[self.index.index(key)]
        return default

    # ---- elementwise ops (positional)
    def _binop(self, other, fn):
        if isinstance(other, Series):
            other = other._data
        if isinstance(other, (list, tuple)):
            d = [fn(a, b) for a, b in zip(self._data, other)]
        else:
            d = [fn(a, other) for a in self._data]
        return Series(d, index=list(self.index), name=self.name)

    def __add__(self, o):
        return self._binop(o, lambda a, b: a + b)

    def __radd__(self, o):
        return self._binop(o, lambda a, b: b + a)

    def __sub__(self, o):
        return self._binop(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._binop(o, lambda a, b: b - a)

    def __mul__(self, o):
        return self._binop(o, lambda a, b: a * b)

    def __rmul__(self, o):
        return self._binop(o, lambda a, b: b * a)

    def __truediv__(self, o):
        return self._binop(o, lambda a, b: a / b if b != 0 else
                           (_NAN if a == 0 else _math.copysign(
                               _math.inf, a) * (1 if b == 0 else 1)))

    def __rtruediv__(self, o):
        return self._binop(o, lambda a, b: b / a)

    def __pow__(self, o):
        return self._binop(o, lambda a, b: a ** b)

    def __neg__(self):
        return Series([-v for v in self._data], index=list(self.index),
                      name=self.name)

    def _cmp(self, other, fn):
        if isinstance(other, Series):
            other = other._data
        if isinstance(other, (list, tuple)):
            d = [bool(fn(a, b)) for a, b in zip(self._data, other)]
        else:
            d = [bool(fn(a, other)) for a in self._data]
        return Series(d, index=list(self.index), name=self.name)

    def __eq__(self, o):  # noqa: D105
        return self._cmp(o, lambda a, b: a == b)

    def __ne__(self, o):
        return self._cmp(o, lambda a, b: a != b)

    def __lt__(self, o):
        return self._cmp(o, lambda a, b: a < b)

    def __le__(self, o):
        return self._cmp(o, lambda a, b: a <= b)

    def __gt__(self, o):
        return self._cmp(o, lambda a, b: a > b)

    def __ge__(self, o):
        return self._cmp(o, lambda a, b: a >= b)

    __hash__ = None

    def __and__(self, o):
        return self._cmp(o, lambda a, b: bool(a) and bool(b))

    def __or__(self, o):
        return self._cmp(o, lambda a, b: bool(a) or bool(b))

    def __invert__(self):
        return Series([not bool(v) for v in self._data],
                      index=list(self.index), name=self.name)

    # ---- reductions (skipna like pandas)
    def _clean(self):
        return [float(v) for v in self._data if not _isnan(v)]

    def sum(self):
        return _math.fsum(self._clean())

    def count(self):
        return len(self._clean())

    def mean(self):
        c = self._clean()
        return _math.fsum(c) / len(c) if c else _NAN

    def var(self, ddof=1):
        c = self._clean()
        n = len(c)
        if n <= ddof:
            return _NAN
        m = _math.fsum(c) / n
        return _math.fsum((v - m) ** 2 for v in c) / (n - ddof)

    def std(self, ddof=1):
        v = self.var(ddof=ddof)
        return _math.sqrt(v) if not _isnan(v) else _NAN

    def sem(self, ddof=1):
        c = self._clean()
        return self.std(ddof=ddof) / _math.sqrt(len(c)) if c else _NAN

    def min(self):
        c = self._clean() if self._is_numeric() \
            else [v for v in self._data if not _isnan(v)]
        return min(c) if c else _NAN

    def max(self):
        c = self._clean() if self._is_numeric() \
            else [v for v in self._data if not _isnan(v)]
        return max(c) if c else _NAN

    def median(self):
        return self.quantile(0.5)

    def quantile(self, q=0.5):
        c = sorted(self._clean())
        if not c:
            return _NAN
        if isinstance(q, (list, tuple)):
            return Series([self.quantile(v) for v in q], index=list(q),
                          name=self.name)
        h = (len(c) - 1) * float(q)
        lo = int(_math.floor(h))
        hi = min(lo + 1, len(c) - 1)
        return c[lo] + (h - lo) * (c[hi] - c[lo])

    def abs(self):
        return Series([abs(v) if not _isnan(v) else v
                       for v in self._data],
                      index=list(self.index), name=self.name)

    def prod(self):
        out = 1.0
        for v in self._clean():
            out *= v
        return out

    def cumsum(self):
        out, acc = [], 0.0
        for v in self._data:
            if _isnan(v):
                out.append(_NAN)
            else:
                acc += v
                out.append(acc)
        return Series(out, index=list(self.index), name=self.name)

    def diff(self, periods=1):
        d = [_NAN] * min(periods, len(self._data))
        for i in range(periods, len(self._data)):
            a, b = self._data[i], self._data[i - periods]
            d.append(_NAN if _isnan(a) or _isnan(b) else a - b)
        return Series(d, index=list(self.index), name=self.name)

    def shift(self, periods=1):
        n = len(self._data)
        if periods >= 0:
            d = [_NAN] * min(periods, n) + self._data[:n - periods]
        else:
            d = self._data[-periods:] + [_NAN] * min(-periods, n)
        return Series(d, index=list(self.index), name=self.name)

    def pct_change(self):
        prev = self.shift(1)
        return Series([_NAN if _isnan(a) or _isnan(b) or b == 0
                       else a / b - 1.0
                       for a, b in zip(self._data, prev._data)],
                      index=list(self.index), name=self.name)

    def clip(self, lower=None, upper=None):
        def one(v):
            if _isnan(v):
                return v
            if lower is not None and v < lower:
                return lower
            if upper is not None and v > upper:
                return upper
            return v
        return Series([one(v) for v in self._data],
                      index=list(self.index), name=self.name)

    def round(self, decimals=0):
        return Series([round(v, decimals) if not _isnan(v) else v
                       for v in self._data],
                      index=list(self.index), name=self.name)

    def any(self):
        return any(bool(v) for v in self._data if not _isnan(v))

    def all(self):
        return all(bool(v) for v in self._data if not _isnan(v))

    def idxmax(self):
        best, bi = None, None
        for i, v in zip(self.index, self._data):
            if not _isnan(v) and (best is None or v > best):
                best, bi = v, i
        return bi

    def idxmin(self):
        best, bi = None, None
        for i, v in zip(self.index, self._data):
            if not _isnan(v) and (best is None or v < best):
                best, bi = v, i
        return bi

    # ---- missing data
    def isna(self):
        return Series([_isnan(v) for v in self._data],
                      index=list(self.index), name=self.name)

    isnull = isna

    def notna(self):
        return Series([not _isnan(v) for v in self._data],
                      index=list(self.index), name=self.name)

    notnull = notna

    def dropna(self):
        pairs = [(i, v) for i, v in zip(self.index, self._data)
                 if not _isnan(v)]
        return Series([v for _, v in pairs], index=[i for i, _ in pairs],
                      name=self.name)

    def fillna(self, value):
        return Series([value if _isnan(v) else v for v in self._data],
                      index=list(self.index), name=self.name)

    # ---- transforms
    def astype(self, dtype):
        return Series(_cast_list(self._data, dtype),
                      index=list(self.index), name=self.name)

    def map(self, fn):
        if isinstance(fn, dict):
            return Series([fn.get(v, _NAN) for v in self._data],
                          index=list(self.index), name=self.name)
        return Series([fn(v) for v in self._data],
                      index=list(self.index), name=self.name)

    def apply(self, fn):
        return self.map(fn)

    def isin(self, values):
        vs = set(values.tolist() if hasattr(values, "tolist")
                 else values)
        return Series([v in vs for v in self._data],
                      index=list(self.index), name=self.name)

    def unique(self):
        seen, out = set(), []
        for v in self._data:
            k = "__nan__" if _isnan(v) else v
            if k not in seen:
                seen.add(k)
                out.append(v)
        return out

    def nunique(self):
        return len({v for v in self._data if not _isnan(v)})

    def value_counts(self, normalize=False, sort=True):
        counts = {}
        for v in self._data:
            if _isnan(v):
                continue
            counts[v] = counts.get(v, 0) + 1
        items = list(counts.items())
        if sort:
            items.sort(key=lambda kv: (-kv[1], str(kv[0])))
        tot = sum(v for _, v in items)
        return Series([v / tot if normalize else v for _, v in items],
                      index=[k for k, _ in items], name=self.name)

    def sort_values(self, ascending=True):
        pairs = sorted(zip(self.index, self._data),
                       key=lambda kv: (kv[1] != kv[1], kv[1]),
                       reverse=not ascending)
        return Series([v for _, v in pairs], index=[i for i, _ in pairs],
                      name=self.name)

    def sort_index(self, ascending=True):
        pairs = sorted(zip(self.index, self._data),
                       key=lambda kv: kv[0], reverse=not ascending)
        return Series([v for _, v in pairs], index=[i for i, _ in pairs],
                      name=self.name)

    def reset_index(self, drop=False):
        if drop:
            return Series(list(self._data), name=self.name)
        return DataFrame({"index": list(self.index),
                          self.name or 0: list(self._data)})

    def rank(self, ascending=True):
        idx = sorted(range(len(self._data)),
                     key=lambda i: self._data[i],
                     reverse=not ascending)
        ranks = [0.0] * len(self._data)
        i = 0
        while i < len(idx):
            j = i
            while (j + 1 < len(idx)
                   and self._data[idx[j + 1]] == self._data[idx[i]]):
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return Series(ranks, index=list(self.index), name=self.name)

    def rename(self, name):
        out = self.copy()
        out.name = name
        return out

    def to_frame(self, name=None):
        return DataFrame({name or self.name or 0: list(self._data)},
                         index=list(self.index))

    def to_dict(self):
        return dict(zip(self.index, self._data))

    def describe(self):
        return Series([self.count(), self.mean(), self.std(),
                       self.min(), self.quantile(0.25), self.median(),
                       self.quantile(0.75), self.max()],
                      index=["count", "mean", "std", "min", "25%",
                             "50%", "75%", "max"], name=self.name)


def _cast_list(data, dtype):
    d = dtype if isinstance(dtype, str) else getattr(
        dtype, "__name__", str(dtype))
    if d in ("float", "float64", "float32"):
        return [_to_float(v) for v in data]
    if d in ("int", "int64", "int32"):
        return [int(v) if not _isnan(v) else v for v in data]
    if d in ("str", "object", "string"):
        return [str(v) for v in data]
    if d in ("bool", "boolean"):
        return [bool(v) for v in data]
    if d == "category":
        return list(data)
    raise ValueError("unsupported dtype %r" % dtype)


class _SeriesILoc:
    def __init__(self, s):
        self._s = s

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Series(self._s._data[i], index=self._s.index[i],
                          name=self._s.name)
        return self._s._data[i]

    def __setitem__(self, i, v):
        self._s._data[i] = v


class _SeriesLoc:
    def __init__(self, s):
        self._s = s

    def __getitem__(self, key):
        return self._s[key]

    def __setitem__(self, key, v):
        self._s[key] = v


class _StrAccessor:
    def __init__(self, s):
        self._s = s

    def _map(self, fn):
        return Series([fn(v) if isinstance(v, str) else _NAN
                       for v in self._s._data],
                      index=list(self._s.index), name=self._s.name)

    def lower(self):
        return self._map(str.lower)

    def upper(self):
        return self._map(str.upper)

    def strip(self):
        return self._map(str.strip)

    def title(self):
        return self._map(str.title)

    def len(self):
        return self._map(len)

    def contains(self, pat, case=True, regex=False):
        if regex:
            import re
            flags = 0 if case else re.IGNORECASE
            rx = re.compile(pat, flags)
            return self._map(lambda v: bool(rx.search(v)))
        if case:
            return self._map(lambda v: pat in v)
        return self._map(lambda v: pat.lower() in v.lower())

    def startswith(self, pat):
        return self._map(lambda v: v.startswith(pat))

    def endswith(self, pat):
        return self._map(lambda v: v.endswith(pat))

    def replace(self, old, new, regex=False):
        if regex:
            import re
            rx = re.compile(old)
            return self._map(lambda v: rx.sub(new, v))
        return self._map(lambda v: v.replace(old, new))

    def split(self, sep=None):
        return self._map(lambda v: v.split(sep))

    def zfill(self, width):
        return self._map(lambda v: v.zfill(width))


class _DtAccessor:
    def __init__(self, s):
        self._s = s

    def _map(self, fn):
        return Series([fn(v) if isinstance(v, (_dt.date, _dt.datetime))
                       else _NAN for v in self._s._data],
                      index=list(self._s.index), name=self._s.name)

    @property
    def year(self):
        return self._map(lambda v: v.year)

    @property
    def month(self):
        return self._map(lambda v: v.month)

    @property
    def day(self):
        return self._map(lambda v: v.day)

    @property
    def dayofweek(self):
        return self._map(lambda v: v.weekday())

    @property
    def hour(self):
        return self._map(lambda v: getattr(v, "hour", 0))


# ===================================================== DataFrame

class DataFrame:
    def __init__(self, data=None, index=None, columns=None):
        self._cols = {}
        if data is None:
            data = {}
        if isinstance(data, DataFrame):
            for c in data.columns:
                self._cols[c] = list(data._cols[c])
            index = list(data.index) if index is None else index
        elif isinstance(data, dict):
            n = None
            for v in data.values():
                if isinstance(v, (list, tuple, Series)) \
                        or hasattr(v, "tolist"):
                    n = len(v.tolist() if hasattr(v, "tolist")
                            and not isinstance(v, Series) else v)
                    break
            if n is None:
                n = len(index) if index is not None else 1
            for k, v in data.items():
                if isinstance(v, Series):
                    self._cols[k] = list(v._data)
                elif hasattr(v, "tolist"):
                    vv = v.tolist()
                    self._cols[k] = list(vv) if isinstance(vv, list) \
                        else [vv] * n
                elif isinstance(v, (list, tuple)):
                    self._cols[k] = list(v)
                else:
                    self._cols[k] = [v] * n
        elif isinstance(data, list) and data \
                and isinstance(data[0], dict):
            keys = []
            for row in data:
                for k in row:
                    if k not in keys:
                        keys.append(k)
            for k in keys:
                self._cols[k] = [row.get(k, _NAN) for row in data]
        elif isinstance(data, list):
            rows = [list(r.tolist() if hasattr(r, "tolist") else r)
                    if isinstance(r, (list, tuple)) or hasattr(
                        r, "tolist") else [r] for r in data]
            ncol = len(rows[0]) if rows else 0
            names = list(columns) if columns is not None \
                else list(range(ncol))
            for j, nm in enumerate(names):
                self._cols[nm] = [r[j] for r in rows]
        elif hasattr(data, "tolist"):
            return DataFrame.__init__(self, data.tolist(), index=index,
                                      columns=columns)
        if columns is not None and isinstance(data, dict):
            self._cols = {c: self._cols.get(
                c, [_NAN] * (len(index) if index else 0))
                for c in columns}
        n = len(next(iter(self._cols.values()))) if self._cols else 0
        self.index = list(index) if index is not None \
            else list(range(n))

    # ---- basics
    @property
    def columns(self):
        return _Columns(self._cols.keys())

    @columns.setter
    def columns(self, names):
        self._cols = dict(zip(list(names), self._cols.values()))

    @property
    def shape(self):
        n = len(next(iter(self._cols.values()))) if self._cols else 0
        return (n, len(self._cols))

    @property
    def empty(self):
        return self.shape[0] == 0

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    @property
    def values(self):
        return _ac.marr([[_to_float(self._cols[c][i])
                          for c in self._cols]
                         for i in range(self.shape[0])])

    @property
    def dtypes(self):
        out = {}
        for c, vals in self._cols.items():
            if all(isinstance(v, bool) for v in vals):
                out[c] = "bool"
            elif all(isinstance(v, (int, float)) or _isnan(v)
                     for v in vals):
                out[c] = "float64"
            else:
                out[c] = "object"
        return Series(list(out.values()), index=list(out.keys()))

    def to_numpy(self):
        return self.values

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return "DataFrame(%d x %d: %s)" % (
            self.shape[0], self.shape[1], list(self._cols)[:8])

    def __contains__(self, key):
        return key in self._cols

    def copy(self):
        return DataFrame({c: list(v) for c, v in self._cols.items()},
                         index=list(self.index))

    def head(self, n=5):
        return self.iloc[slice(0, n)]

    def tail(self, n=5):
        return self.iloc[slice(-n, None)]

    def keys(self):
        return list(self._cols.keys())

    def items(self):
        for c in self._cols:
            yield c, self[c]

    def iterrows(self):
        for i in range(self.shape[0]):
            yield self.index[i], Series(
                [self._cols[c][i] for c in self._cols],
                index=list(self._cols.keys()))

    def itertuples(self, index=True):
        import collections
        fields = (["Index"] if index else []) + [
            str(c) for c in self._cols]
        T = collections.namedtuple("Row", fields, rename=True)
        for i in range(self.shape[0]):
            vals = ([self.index[i]] if index else []) + [
                self._cols[c][i] for c in self._cols]
            yield T(*vals)

    # ---- indexing
    def __getitem__(self, key):
        if isinstance(key, Series):
            key = key.tolist()
        if isinstance(key, list) and key \
                and isinstance(key[0], bool):
            keep = [i for i, m in enumerate(key) if m]
            return self._take(keep)
        if isinstance(key, list):
            return DataFrame({c: list(self._cols[c]) for c in key},
                             index=list(self.index))
        return Series(list(self._cols[key]), index=list(self.index),
                      name=key)

    def __setitem__(self, key, value):
        n = self.shape[0]
        if isinstance(value, Series):
            value = list(value._data)
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            value = [value] * (n if self._cols else 1)
        if self._cols and len(value) != n:
            raise ValueError("length mismatch: %d vs %d"
                             % (len(value), n))
        self._cols[key] = list(value)
        if not self.index and value:
            self.index = list(range(len(value)))

    def _take(self, rows):
        return DataFrame(
            {c: [v[i] for i in rows] for c, v in self._cols.items()},
            index=[self.index[i] for i in rows])

    @property
    def iloc(self):
        return _ILoc(self)

    @property
    def loc(self):
        return _Loc(self)

    def get(self, key, default=None):
        if key in self._cols:
            return self[key]
        return default

    # ---- column/row ops
    def drop(self, labels=None, columns=None, axis=0, errors="raise"):
        if columns is None and axis in (1, "columns"):
            columns = labels
            labels = None
        if columns is not None:
            if not isinstance(columns, (list, tuple)):
                columns = [columns]
            missing = [c for c in columns if c not in self._cols]
            if missing and errors == "raise":
                raise KeyError(missing)
            return DataFrame({c: list(v) for c, v in self._cols.items()
                              if c not in columns},
                             index=list(self.index))
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        keep = [i for i, ix in enumerate(self.index)
                if ix not in labels]
        return self._take(keep)

    def rename(self, columns=None, **kw):
        del kw
        columns = columns or {}
        return DataFrame({columns.get(c, c): list(v)
                          for c, v in self._cols.items()},
                         index=list(self.index))

    def astype(self, dtype):
        if isinstance(dtype, dict):
            out = {c: (_cast_list(v, dtype[c]) if c in dtype
                       else list(v))
                   for c, v in self._cols.items()}
        else:
            out = {c: _cast_list(v, dtype)
                   for c, v in self._cols.items()}
        return DataFrame(out, index=list(self.index))

    def isna(self):
        return DataFrame({c: [_isnan(v) for v in vals]
                          for c, vals in self._cols.items()},
                         index=list(self.index))

    isnull = isna

    def notna(self):
        return DataFrame({c: [not _isnan(v) for v in vals]
                          for c, vals in self._cols.items()},
                         index=list(self.index))

    def dropna(self, subset=None, how="any"):
        cols = subset if subset is not None else list(self._cols)
        keep = []
        for i in range(self.shape[0]):
            miss = [_isnan(self._cols[c][i]) for c in cols]
            bad = any(miss) if how == "any" else all(miss)
            if not bad:
                keep.append(i)
        return self._take(keep)

    def fillna(self, value):
        if isinstance(value, dict):
            return DataFrame(
                {c: [value[c] if c in value and _isnan(v) else v
                     for v in vals]
                 for c, vals in self._cols.items()},
                index=list(self.index))
        return DataFrame({c: [value if _isnan(v) else v for v in vals]
                          for c, vals in self._cols.items()},
                         index=list(self.index))

    def replace(self, to_replace, value=None):
        if isinstance(to_replace, dict) and value is None:
            def one(v):
                return to_replace.get(v, v)
        else:
            def one(v):
                return value if v == to_replace else v
        return DataFrame({c: [one(v) for v in vals]
                          for c, vals in self._cols.items()},
                         index=list(self.index))

    def sort_values(self, by, ascending=True):
        if not isinstance(by, (list, tuple)):
            by = [by]
        if not isinstance(ascending, (list, tuple)):
            ascending = [ascending] * len(by)
        order = list(range(self.shape[0]))
        for col, asc in list(zip(by, ascending))[::-1]:
            vals = self._cols[col]
            order.sort(key=lambda i: (vals[i] != vals[i], vals[i]),
                       reverse=not asc)
        return self._take(order)

    def sort_index(self, ascending=True):
        order = sorted(range(self.shape[0]),
                       key=lambda i: self.index[i],
                       reverse=not ascending)
        return self._take(order)

    def reset_index(self, drop=False):
        out = DataFrame({c: list(v) for c, v in self._cols.items()})
        if not drop:
            out._cols = {"index": list(self.index), **out._cols}
        return out

    def set_index(self, col):
        out = self.drop(columns=[col])
        out.index = list(self._cols[col])
        return out

    def select_dtypes(self, include=None, exclude=None):
        dt = self.dtypes.to_dict()

        def match(kind, spec):
            spec = [spec] if isinstance(spec, str) else list(spec)
            for s in spec:
                if s in ("number", "float", "float64", "int") \
                        and kind == "float64":
                    return True
                if s in ("object", "string") and kind == "object":
                    return True
                if s == "bool" and kind == "bool":
                    return True
            return False
        cols = list(self._cols)
        if include is not None:
            cols = [c for c in cols if match(dt[c], include)]
        if exclude is not None:
            cols = [c for c in cols if not match(dt[c], exclude)]
        return self[cols]

    def duplicated(self, subset=None, keep="first"):
        cols = subset or list(self._cols)
        seen = {}
        flags = [False] * self.shape[0]
        for i in range(self.shape[0]):
            key = tuple(self._cols[c][i] for c in cols)
            if key in seen:
                flags[i] = True
            else:
                seen[key] = i
        if keep == "last":
            flags = [False] * self.shape[0]
            seen = {}
            for i in range(self.shape[0] - 1, -1, -1):
                key = tuple(self._cols[c][i] for c in cols)
                if key in seen:
                    flags[i] = True
                else:
                    seen[key] = i
        return Series(flags, index=list(self.index))

    def drop_duplicates(self, subset=None, keep="first"):
        dup = self.duplicated(subset=subset, keep=keep)
        return self[[not d for d in dup.tolist()]]

    # ---- reductions
    def _reduce(self, fn, numeric_only=True):
        out, ix = [], []
        for c, vals in self._cols.items():
            s = Series(vals)
            if numeric_only and not s._is_numeric():
                continue
            out.append(fn(s))
            ix.append(c)
        return Series(out, index=ix)

    def sum(self, numeric_only=True):
        return self._reduce(lambda s: s.sum(), numeric_only)

    def mean(self, numeric_only=True):
        return self._reduce(lambda s: s.mean(), numeric_only)

    def std(self, ddof=1, numeric_only=True):
        return self._reduce(lambda s: s.std(ddof=ddof), numeric_only)

    def var(self, ddof=1, numeric_only=True):
        return self._reduce(lambda s: s.var(ddof=ddof), numeric_only)

    def median(self, numeric_only=True):
        return self._reduce(lambda s: s.median(), numeric_only)

    def min(self, numeric_only=True):
        return self._reduce(lambda s: s.min(), numeric_only)

    def max(self, numeric_only=True):
        return self._reduce(lambda s: s.max(), numeric_only)

    def count(self):
        return self._reduce(lambda s: s.count(), numeric_only=False)

    def nunique(self):
        return self._reduce(lambda s: s.nunique(), numeric_only=False)

    def quantile(self, q=0.5):
        return self._reduce(lambda s: s.quantile(q))

    def abs(self):
        return DataFrame({c: Series(v).abs().tolist()
                          for c, v in self._cols.items()},
                         index=list(self.index))

    def corr(self):
        cols = [c for c in self._cols if Series(
            self._cols[c])._is_numeric()]
        n = len(cols)
        mat = [[1.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = mat[j][i] = _pearson(
                    self._cols[cols[i]], self._cols[cols[j]])
        return DataFrame(dict(zip(
            cols, [[mat[i][j] for j in range(n)]
                   for i in range(n)])), index=cols) \
            .T if False else DataFrame(
            {cols[j]: [mat[i][j] for i in range(n)]
             for j in range(n)}, index=cols)

    def cov(self, ddof=1):
        cols = [c for c in self._cols if Series(
            self._cols[c])._is_numeric()]
        n = len(cols)
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                mat[i][j] = mat[j][i] = _cov(
                    self._cols[cols[i]], self._cols[cols[j]], ddof)
        return DataFrame({cols[j]: [mat[i][j] for i in range(n)]
                          for j in range(n)}, index=cols)

    def describe(self):
        cols = [c for c in self._cols
                if Series(self._cols[c])._is_numeric()]
        rows = ["count", "mean", "std", "min", "25%", "50%", "75%",
                "max"]
        return DataFrame(
            {c: Series(self._cols[c]).describe().tolist()
             for c in cols}, index=rows)

    def apply(self, fn, axis=0):
        if axis in (1, "columns"):
            out = []
            for _, row in self.iterrows():
                out.append(fn(row))
            return Series(out, index=list(self.index))
        vals, ix = [], []
        for c in self._cols:
            vals.append(fn(self[c]))
            ix.append(c)
        if all(isinstance(v, Series) for v in vals):
            return DataFrame({c: v.tolist()
                              for c, v in zip(ix, vals)},
                             index=list(self.index))
        return Series(vals, index=ix)

    # ---- combine
    def merge(self, right, on=None, how="inner", left_on=None,
              right_on=None, suffixes=("_x", "_y")):
        lk = left_on or on
        rk = right_on or on
        if not isinstance(lk, (list, tuple)):
            lk = [lk]
        if not isinstance(rk, (list, tuple)):
            rk = [rk]
        rmap = {}
        for j in range(right.shape[0]):
            key = tuple(right._cols[c][j] for c in rk)
            rmap.setdefault(key, []).append(j)
        shared_key = (lk == rk)
        rcols = [c for c in right._cols
                 if not (shared_key and c in rk)]
        out = {}
        lnames, rnames = {}, {}
        for c in self._cols:
            lnames[c] = c if c not in rcols else str(c) + suffixes[0]
            out[lnames[c]] = []
        for c in rcols:
            rnames[c] = c if c not in self._cols \
                else str(c) + suffixes[1]
            out[rnames[c]] = []
        for i in range(self.shape[0]):
            key = tuple(self._cols[c][i] for c in lk)
            matches = rmap.get(key, [])
            if not matches:
                if how in ("left", "outer"):
                    for c in self._cols:
                        out[lnames[c]].append(self._cols[c][i])
                    for c in rcols:
                        out[rnames[c]].append(_NAN)
                continue
            for j in matches:
                for c in self._cols:
                    out[lnames[c]].append(self._cols[c][i])
                for c in rcols:
                    out[rnames[c]].append(right._cols[c][j])
        if how in ("right", "outer"):
            lseen = {tuple(self._cols[c][i] for c in lk)
                     for i in range(self.shape[0])}
            for j in range(right.shape[0]):
                key = tuple(right._cols[c][j] for c in rk)
                if key not in lseen:
                    for c in self._cols:
                        out[lnames[c]].append(
                            right._cols[rk[lk.index(c)]][j]
                            if c in lk else _NAN)
                    for c in rcols:
                        out[rnames[c]].append(right._cols[c][j])
        return DataFrame(out)

    def join(self, other, how="left", lsuffix="", rsuffix=""):
        del lsuffix, rsuffix
        rmap = {ix: j for j, ix in enumerate(other.index)}
        out = {c: list(v) for c, v in self._cols.items()}
        for c in other._cols:
            col = []
            for ix in self.index:
                j = rmap.get(ix)
                col.append(other._cols[c][j] if j is not None
                           else _NAN)
            out[c] = col
        del how
        return DataFrame(out, index=list(self.index))

    def groupby(self, by):
        return GroupBy(self, by)

    @property
    def T(self):
        rows = [[self._cols[c][i] for c in self._cols]
                for i in range(self.shape[0])]
        return DataFrame({self.index[i]: rows[i]
                          for i in range(len(rows))},
                         index=list(self._cols.keys()))

    def pivot_table(self, values, index, columns, aggfunc="mean"):
        gb = {}
        for i in range(self.shape[0]):
            key = (self._cols[index][i], self._cols[columns][i])
            gb.setdefault(key, []).append(self._cols[values][i])
        rows = sorted({k[0] for k in gb})
        cols = sorted({k[1] for k in gb})
        agg = {"mean": lambda v: _math.fsum(v) / len(v),
               "sum": _math.fsum, "count": len,
               "min": min, "max": max}[aggfunc]
        return DataFrame(
            {c: [agg(gb[(r, c)]) if (r, c) in gb else _NAN
                 for r in rows] for c in cols}, index=rows)

    # ---- io / export
    def to_dict(self, orient="dict"):
        if orient in ("records",):
            return [{c: self._cols[c][i] for c in self._cols}
                    for i in range(self.shape[0])]
        if orient in ("list",):
            return {c: list(v) for c, v in self._cols.items()}
        return {c: dict(zip(self.index, v))
                for c, v in self._cols.items()}

    def to_csv(self, path=None, index=True, sep=","):
        import io
        buf = io.StringIO()
        w = _csv.writer(buf, delimiter=sep, lineterminator="\n")
        header = ([""] if index else []) + [str(c) for c in self._cols]
        w.writerow(header)
        for i in range(self.shape[0]):
            row = ([self.index[i]] if index else []) + [
                self._cols[c][i] for c in self._cols]
            w.writerow(["" if _isnan(v) else v for v in row])
        s = buf.getvalue()
        if path is None:
            return s
        with open(path, "w") as f:
            f.write(s)

    def to_string(self):
        return self.to_csv(index=True, sep="\t")

    def insert(self, loc, column, value):
        items = list(self._cols.items())
        if isinstance(value, Series):
            value = list(value._data)
        if not isinstance(value, list):
            value = [value] * self.shape[0]
        items.insert(loc, (column, list(value)))
        self._cols = dict(items)

    def pop(self, col):
        s = self[col]
        del self._cols[col]
        return s

    def __delitem__(self, col):
        del self._cols[col]


class _Columns(list):
    def __init__(self, it):
        super().__init__(it)

    def tolist(self):
        return list(self)

    def str_contains(self, pat):
        return [pat in str(c) for c in self]

    def get_loc(self, c):
        return self.index(c)


class _ILoc:
    def __init__(self, df):
        self._df = df

    def __getitem__(self, key):
        df = self._df
        if isinstance(key, tuple):
            rk, ck = key
            cols = list(df._cols)
            if isinstance(ck, int):
                sel = cols[ck]
                sub = df.iloc[rk] if not isinstance(rk, int) else None
                if isinstance(rk, int):
                    return df._cols[sel][rk]
                return sub[sel]
            sel = cols[ck] if isinstance(ck, slice) else \
                [cols[j] for j in ck]
            base = df[sel if isinstance(sel, list) else list(sel)]
            return base.iloc[rk]
        if isinstance(key, int):
            n = self._df.shape[0]
            i = key if key >= 0 else n + key
            return Series([df._cols[c][i] for c in df._cols],
                          index=list(df._cols.keys()),
                          name=df.index[i])
        if isinstance(key, slice):
            rows = list(range(df.shape[0]))[key]
            return df._take(rows)
        return df._take(list(key))


class _Loc:
    def __init__(self, df):
        self._df = df

    def __getitem__(self, key):
        df = self._df
        if isinstance(key, tuple):
            rk, ck = key
            if isinstance(rk, Series):
                rk = rk.tolist()
            if isinstance(rk, list) and rk \
                    and isinstance(rk[0], bool):
                sub = df[rk]
            elif isinstance(rk, slice):
                sub = df.iloc[rk]
            else:
                i = df.index.index(rk)
                if isinstance(ck, str) or (
                        not isinstance(ck, (list, tuple))
                        and ck in df._cols):
                    return df._cols[ck][i]
                return Series([df._cols[c][i] for c in ck], index=ck)
            if isinstance(ck, str):
                return sub[ck]
            return sub[list(ck)]
        if isinstance(key, Series):
            key = key.tolist()
        if isinstance(key, list) and key and isinstance(key[0], bool):
            return df[key]
        i = df.index.index(key)
        return df.iloc[i]

    def __setitem__(self, key, value):
        df = self._df
        if isinstance(key, tuple):
            rk, ck = key
            if isinstance(rk, Series):
                rk = rk.tolist()
            if isinstance(value, Series):
                value = list(value._data)
            if isinstance(rk, list) and rk \
                    and isinstance(rk[0], bool):
                if ck not in df._cols:
                    df._cols[ck] = [_NAN] * df.shape[0]
                vi = 0
                for i, m in enumerate(rk):
                    if m:
                        df._cols[ck][i] = (value[vi]
                                           if isinstance(value, list)
                                           else value)
                        vi += 1
                return
            i = df.index.index(rk)
            df._cols[ck][i] = value
            return
        raise TypeError("unsupported loc assignment")


def _pearson(x, y):
    pairs = [(a, b) for a, b in zip(x, y)
             if not _isnan(a) and not _isnan(b)]
    n = len(pairs)
    if n < 2:
        return _NAN
    mx = _math.fsum(a for a, _ in pairs) / n
    my = _math.fsum(b for _, b in pairs) / n
    sxy = _math.fsum((a - mx) * (b - my) for a, b in pairs)
    sxx = _math.fsum((a - mx) ** 2 for a, _ in pairs)
    syy = _math.fsum((b - my) ** 2 for _, b in pairs)
    if sxx == 0 or syy == 0:
        return _NAN
    return sxy / _math.sqrt(sxx * syy)


def _cov(x, y, ddof=1):
    pairs = [(a, b) for a, b in zip(x, y)
             if not _isnan(a) and not _isnan(b)]
    n = len(pairs)
    if n <= ddof:
        return _NAN
    mx = _math.fsum(a for a, _ in pairs) / n
    my = _math.fsum(b for _, b in pairs) / n
    return _math.fsum((a - mx) * (b - my)
                      for a, b in pairs) / (n - ddof)


# ===================================================== GroupBy

class GroupBy:
    def __init__(self, df, by):
        self._df = df
        self._by = by if isinstance(by, (list, tuple)) else [by]
        self._groups = {}
        for i in range(df.shape[0]):
            key = tuple(df._cols[c][i] for c in self._by)
            self._groups.setdefault(key, []).append(i)

    def __iter__(self):
        for key in sorted(self._groups):
            k = key[0] if len(self._by) == 1 else key
            yield k, self._df._take(self._groups[key])

    def __getitem__(self, col):
        return _GroupBySeries(self, col)

    @property
    def groups(self):
        return {(k[0] if len(self._by) == 1 else k): v
                for k, v in self._groups.items()}

    def get_group(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self._df._take(self._groups[key])

    def size(self):
        keys = sorted(self._groups)
        return Series([len(self._groups[k]) for k in keys],
                      index=[k[0] if len(self._by) == 1 else k
                             for k in keys])

    def ngroups(self):
        return len(self._groups)

    def _agg(self, fn, numeric_only=True):
        keys = sorted(self._groups)
        val_cols = [c for c in self._df._cols if c not in self._by]
        if numeric_only:
            val_cols = [c for c in val_cols
                        if Series(self._df._cols[c])._is_numeric()]
        out = {c: [] for c in val_cols}
        for k in keys:
            rows = self._groups[k]
            for c in val_cols:
                out[c].append(fn(Series(
                    [self._df._cols[c][i] for i in rows])))
        return DataFrame(out, index=[k[0] if len(self._by) == 1
                                     else k for k in keys])

    def mean(self, numeric_only=True):
        return self._agg(lambda s: s.mean(), numeric_only)

    def sum(self, numeric_only=True):
        return self._agg(lambda s: s.sum(), numeric_only)

    def std(self, ddof=1, numeric_only=True):
        return self._agg(lambda s: s.std(ddof=ddof), numeric_only)

    def var(self, ddof=1, numeric_only=True):
        return self._agg(lambda s: s.var(ddof=ddof), numeric_only)

    def median(self, numeric_only=True):
        return self._agg(lambda s: s.median(), numeric_only)

    def min(self, numeric_only=False):
        return self._agg(lambda s: s.min(), numeric_only)

    def max(self, numeric_only=False):
        return self._agg(lambda s: s.max(), numeric_only)

    def count(self):
        return self._agg(lambda s: s.count(), numeric_only=False)

    def agg(self, spec):
        if callable(spec) or isinstance(spec, str):
            fn = _agg_fn(spec)
            return self._agg(fn, numeric_only=False)
        keys = sorted(self._groups)
        out = {}
        for col, how in spec.items():
            fn = _agg_fn(how)
            out[col] = [fn(Series([self._df._cols[col][i]
                                   for i in self._groups[k]]))
                        for k in keys]
        return DataFrame(out, index=[k[0] if len(self._by) == 1
                                     else k for k in keys])

    aggregate = agg

    def apply(self, fn):
        keys = sorted(self._groups)
        out = [fn(self._df._take(self._groups[k])) for k in keys]
        ix = [k[0] if len(self._by) == 1 else k for k in keys]
        if all(isinstance(v, (int, float)) for v in out):
            return Series(out, index=ix)
        return dict(zip(ix, out))

    def transform(self, fn):
        fn = _agg_fn(fn)
        val_cols = [c for c in self._df._cols if c not in self._by]
        out = {c: [_NAN] * self._df.shape[0] for c in val_cols}
        for k, rows in self._groups.items():
            for c in val_cols:
                v = fn(Series([self._df._cols[c][i] for i in rows]))
                for i in rows:
                    out[c][i] = v
        return DataFrame(out, index=list(self._df.index))


def _agg_fn(spec):
    if callable(spec):
        return lambda s: spec(s)
    return {"mean": lambda s: s.mean(), "sum": lambda s: s.sum(),
            "std": lambda s: s.std(), "var": lambda s: s.var(),
            "median": lambda s: s.median(), "min": lambda s: s.min(),
            "max": lambda s: s.max(), "count": lambda s: s.count(),
            "nunique": lambda s: s.nunique(),
            "first": lambda s: s._data[0] if s._data else _NAN,
            "last": lambda s: s._data[-1] if s._data else _NAN,
            }[spec]


class _GroupBySeries:
    def __init__(self, gb, col):
        self._gb = gb
        self._col = col

    def _agg(self, fn):
        gb = self._gb
        keys = sorted(gb._groups)
        vals = [fn(Series([gb._df._cols[self._col][i]
                           for i in gb._groups[k]])) for k in keys]
        return Series(vals, index=[k[0] if len(gb._by) == 1 else k
                                   for k in keys], name=self._col)

    def mean(self):
        return self._agg(lambda s: s.mean())

    def sum(self):
        return self._agg(lambda s: s.sum())

    def std(self, ddof=1):
        return self._agg(lambda s: s.std(ddof=ddof))

    def var(self, ddof=1):
        return self._agg(lambda s: s.var(ddof=ddof))

    def median(self):
        return self._agg(lambda s: s.median())

    def min(self):
        return self._agg(lambda s: s.min())

    def max(self):
        return self._agg(lambda s: s.max())

    def count(self):
        return self._agg(lambda s: s.count())

    def nunique(self):
        return self._agg(lambda s: s.nunique())

    def quantile(self, q=0.5):
        return self._agg(lambda s: s.quantile(q))

    def agg(self, spec):
        return self._agg(_agg_fn(spec))

    def apply(self, fn):
        return self._agg(lambda s: fn(s))

    def unique(self):
        return self._agg(lambda s: s.unique())


# ===================================================== module fns

def isna(v):
    if isinstance(v, Series):
        return v.isna()
    if isinstance(v, DataFrame):
        return v.isna()
    return _isnan(v)


isnull = isna


def notna(v):
    if isinstance(v, (Series, DataFrame)):
        return v.notna()
    return not _isnan(v)


notnull = notna


def unique(values):
    return Series(values).unique()


def factorize(values):
    vals = list(values.tolist() if hasattr(values, "tolist")
                else values)
    seen = {}
    codes = []
    uniq = []
    for v in vals:
        if _isnan(v):
            codes.append(-1)
            continue
        if v not in seen:
            seen[v] = len(uniq)
            uniq.append(v)
        codes.append(seen[v])
    return _ac.marr([float(c) for c in codes]), uniq


def concat(objs, axis=0, ignore_index=False):
    objs = [o for o in objs if o is not None]
    if all(isinstance(o, Series) for o in objs):
        if axis == 1:
            return DataFrame({(o.name if o.name is not None else i):
                              o.tolist() for i, o in enumerate(objs)})
        data, ix = [], []
        for o in objs:
            data += o.tolist()
            ix += list(o.index)
        return Series(data, index=list(range(len(data)))
                      if ignore_index else ix)
    if axis == 1:
        out = {}
        index = objs[0].index
        used = 0
        for o in objs:
            if isinstance(o, Series):
                key = o.name if o.name is not None else used
                out[key] = list(o._data)
            else:
                for c in o._cols:
                    out[c] = list(o._cols[c])
            used += 1
        return DataFrame(out, index=list(index))
    cols = []
    for o in objs:
        for c in o._cols:
            if c not in cols:
                cols.append(c)
    out = {c: [] for c in cols}
    ix = []
    for o in objs:
        n = o.shape[0]
        for c in cols:
            out[c] += list(o._cols.get(c, [_NAN] * n))
        ix += list(o.index)
    return DataFrame(out, index=list(range(len(ix)))
                     if ignore_index else ix)


def merge(left, right, **kw):
    return left.merge(right, **kw)


def to_numeric(arg, errors="raise"):
    vals = arg.tolist() if hasattr(arg, "tolist") else list(arg)
    out = []
    for v in vals:
        try:
            f = float(v)
            out.append(int(f) if f == int(f) and not isinstance(
                v, float) and "." not in str(v) and "e" not in
                str(v).lower() else f)
        except (TypeError, ValueError):
            if errors == "coerce":
                out.append(_NAN)
            elif errors == "ignore":
                out.append(v)
            else:
                raise ValueError("Unable to parse %r" % (v,))
    if isinstance(arg, Series):
        return Series(out, index=list(arg.index), name=arg.name)
    return Series(out)


_DT_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
               "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d",
               "%d-%b-%Y", "%B %d, %Y", "%b %d, %Y"]


def _parse_dt(v, fmt=None):
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v if isinstance(v, _dt.datetime) else _dt.datetime(
            v.year, v.month, v.day)
    if _isnan(v):
        return _NAN
    s = str(v).strip()
    if fmt is not None:
        return _dt.datetime.strptime(s, fmt)
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        pass
    for f in _DT_FORMATS:
        try:
            return _dt.datetime.strptime(s, f)
        except ValueError:
            continue
    raise ValueError("cannot parse datetime %r" % (v,))


def to_datetime(arg, errors="raise", format=None):
    scalar = not (isinstance(arg, (list, tuple, Series))
                  or hasattr(arg, "tolist"))
    vals = [arg] if scalar else (
        arg.tolist() if hasattr(arg, "tolist") else list(arg))
    out = []
    for v in vals:
        try:
            out.append(_parse_dt(v, format))
        except ValueError:
            if errors == "coerce":
                out.append(_NAN)
            else:
                raise
    if scalar:
        return out[0]
    if isinstance(arg, Series):
        return Series(out, index=list(arg.index), name=arg.name)
    return Series(out)


Timestamp = _dt.datetime
NaT = _NAN


def date_range(start, periods, freq="D"):
    s = _parse_dt(start)
    step = {"D": _dt.timedelta(days=1), "H": _dt.timedelta(hours=1),
            "W": _dt.timedelta(weeks=1)}[freq]
    return Series([s + i * step for i in range(int(periods))])


def crosstab(index, columns, normalize=False):
    iv = index.tolist() if hasattr(index, "tolist") else list(index)
    cv = columns.tolist() if hasattr(columns, "tolist") \
        else list(columns)
    rows = sorted({v for v in iv if not _isnan(v)}, key=str)
    cols = sorted({v for v in cv if not _isnan(v)}, key=str)
    counts = {(r, c): 0 for r in rows for c in cols}
    tot = 0
    for a, b in zip(iv, cv):
        if _isnan(a) or _isnan(b):
            continue
        counts[(a, b)] += 1
        tot += 1
    out = {c: [counts[(r, c)] for r in rows] for c in cols}
    if normalize:
        out = {c: [v / tot for v in vals] for c, vals in out.items()}
    return DataFrame(out, index=rows)


def cut(x, bins, labels=None, right=True, include_lowest=False):
    vals = x.tolist() if hasattr(x, "tolist") else list(x)
    if isinstance(bins, int):
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.001 or 0.001
        edges = [lo - pad] + [lo + (hi - lo) * (i + 1) / bins
                              for i in range(bins)]
    else:
        edges = [float(b) for b in bins]
    out = []
    for v in vals:
        if _isnan(v):
            out.append(_NAN)
            continue
        placed = _NAN
        for i in range(len(edges) - 1):
            lo_e, hi_e = edges[i], edges[i + 1]
            if right:
                ok = (lo_e < v <= hi_e) or (
                    include_lowest and i == 0 and v == lo_e)
            else:
                ok = lo_e <= v < hi_e or (
                    i == len(edges) - 2 and v == hi_e)
            if ok:
                placed = (labels[i] if labels is not None
                          else "(%g, %g]" % (lo_e, hi_e) if right
                          else "[%g, %g)" % (lo_e, hi_e))
                break
        out.append(placed)
    if isinstance(x, Series):
        return Series(out, index=list(x.index), name=x.name)
    return Series(out)


def qcut(x, q, labels=None):
    vals = x.tolist() if hasattr(x, "tolist") else list(x)
    s = Series(vals)
    if isinstance(q, int):
        qs = [i / q for i in range(q + 1)]
    else:
        qs = list(q)
    edges = [s.quantile(v) for v in qs]
    # dedupe equal edges like pandas duplicates='drop' would complain
    return cut(x, edges, labels=labels, right=True,
               include_lowest=True)


def get_dummies(data, prefix=None, drop_first=False, columns=None,
                dtype=None):
    del dtype
    if isinstance(data, DataFrame):
        cols = columns if columns is not None else [
            c for c in data._cols
            if not Series(data._cols[c])._is_numeric()]
        out = DataFrame({c: list(v) for c, v in data._cols.items()
                         if c not in cols}, index=list(data.index))
        for c in cols:
            dm = get_dummies(data[c], prefix=str(c),
                             drop_first=drop_first)
            for dc in dm._cols:
                out[dc] = dm._cols[dc]
        return out
    vals = data.tolist() if hasattr(data, "tolist") else list(data)
    cats = sorted({v for v in vals if not _isnan(v)}, key=str)
    if drop_first:
        cats = cats[1:]
    name = prefix          # pandas ignores Series name unless prefix=
    out = {}
    for c in cats:
        key = "%s_%s" % (name, c) if name is not None else c
        out[key] = [1 if v == c else 0 for v in vals]
    return DataFrame(out)


class Categorical(list):
    def __init__(self, values, categories=None, ordered=False):
        vals = values.tolist() if hasattr(values, "tolist") \
            else list(values)
        super().__init__(vals)
        self.categories = categories if categories is not None \
            else sorted({v for v in vals if not _isnan(v)}, key=str)
        self.ordered = ordered

    @property
    def codes(self):
        pos = {c: i for i, c in enumerate(self.categories)}
        return _ac.marr([float(pos.get(v, -1)) for v in self])


class CategoricalDtype:
    def __init__(self, categories=None, ordered=False):
        self.categories = categories
        self.ordered = ordered


def read_csv(path, sep=",", header=0, names=None, dtype=None,
             na_values=None, skiprows=0, nrows=None, usecols=None,
             encoding=None, **kw):
    del kw
    na_extra = set(na_values or [])
    na_default = {"", "NA", "N/A", "NaN", "nan", "NULL", "null",
                  "None", "#N/A"}
    f = open(path, newline="", encoding=encoding or "utf-8")
    try:
        rows = list(_csv.reader(f, delimiter=sep))
    finally:
        f.close()
    rows = rows[skiprows:]
    if names is not None:
        cols = list(names)
        if header == 0:
            rows = rows[1:]
    elif header is None:
        cols = list(range(len(rows[0]) if rows else 0))
    else:
        cols = rows[header]
        rows = rows[header + 1:]
    if nrows is not None:
        rows = rows[:nrows]

    def conv(v):
        if v in na_default or v in na_extra:
            return _NAN
        try:
            iv = int(v)
            return iv
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v
    data = {}
    for j, c in enumerate(cols):
        if usecols is not None and c not in usecols \
                and j not in (usecols or []):
            continue
        data[c] = [conv(r[j]) if j < len(r) else _NAN for r in rows]
    df = DataFrame(data)
    if dtype is not None:
        df = df.astype(dtype)
    return df


class _PdApiTypes:
    @staticmethod
    def is_numeric_dtype(obj):
        if isinstance(obj, Series):
            return obj._is_numeric()
        return isinstance(obj, (int, float))

    @staticmethod
    def is_string_dtype(obj):
        if isinstance(obj, Series):
            return all(isinstance(v, str) or _isnan(v)
                       for v in obj._data)
        return isinstance(obj, str)

    @staticmethod
    def is_categorical_dtype(obj):
        return isinstance(obj, Categorical)

    @staticmethod
    def is_datetime64_any_dtype(obj):
        if isinstance(obj, Series):
            return any(isinstance(v, (_dt.date, _dt.datetime))
                       for v in obj._data)
        return isinstance(obj, (_dt.date, _dt.datetime))

    @staticmethod
    def is_object_dtype(obj):
        if isinstance(obj, Series):
            return not obj._is_numeric()
        return isinstance(obj, str)

    @staticmethod
    def is_float_dtype(obj):
        if isinstance(obj, Series):
            return obj._is_numeric() and any(
                isinstance(v, float) for v in obj._data)
        return isinstance(obj, float)

    @staticmethod
    def is_bool_dtype(obj):
        if isinstance(obj, Series):
            return all(isinstance(v, bool) for v in obj._data)
        return isinstance(obj, bool)

    @staticmethod
    def is_integer_dtype(obj):
        if isinstance(obj, Series):
            return all(isinstance(v, int) and not isinstance(v, bool)
                       for v in obj._data)
        return isinstance(obj, int)


class _PdApi:
    types = _PdApiTypes()


api = _PdApi()
__version__ = "0.0-morie-native"


# ===================================================== io tail

def read_json(path_or_buf, orient=None, lines=False):
    import json as _json
    if hasattr(path_or_buf, "read"):
        raw = path_or_buf.read()
    elif isinstance(path_or_buf, str) and path_or_buf.lstrip()[:1] \
            in ("[", "{"):
        raw = path_or_buf
    else:
        with open(path_or_buf) as fh:
            raw = fh.read()
    if lines:
        rows = [_json.loads(ln) for ln in raw.splitlines()
                if ln.strip()]
        return DataFrame(rows)
    obj = _json.loads(raw)
    if isinstance(obj, list):
        return DataFrame(obj)
    if orient == "index":
        rows = [{"index": k, **v} for k, v in obj.items()]
        df = DataFrame(rows)
        return df.set_index("index")
    return DataFrame(obj)


def read_sql(sql, con, params=None):
    """Works with any DB-API connection (sqlite3 etc.)."""
    cur = con.cursor() if hasattr(con, "cursor") else con
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return DataFrame({c: [r[i] for r in rows]
                      for i, c in enumerate(cols)})


read_sql_query = read_sql


def _xlsx_shared_strings(zf):
    import xml.etree.ElementTree as ET
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    ns = {"m": "http://schemas.openxmlformats.org/"
               "spreadsheetml/2006/main"}
    root = ET.fromstring(raw)
    out = []
    for si in root.findall("m:si", ns):
        text = "".join(t.text or "" for t in si.iter(
            "{http://schemas.openxmlformats.org/spreadsheetml/2006/"
            "main}t"))
        out.append(text)
    return out


def _xlsx_col_index(ref):
    col = 0
    for ch in ref:
        if ch.isalpha():
            col = col * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return col - 1


def read_excel(path, sheet_name=0, header=0, **kw):
    """Native .xlsx reader (zip + XML, stdlib only)."""
    del kw
    import xml.etree.ElementTree as ET
    import zipfile
    zf = zipfile.ZipFile(path)
    shared = _xlsx_shared_strings(zf)
    sheets = sorted(n for n in zf.namelist()
                    if n.startswith("xl/worksheets/sheet")
                    and n.endswith(".xml"))
    if isinstance(sheet_name, int):
        target = sheets[sheet_name]
    else:
        # map workbook sheet names to files via workbook.xml order
        ns = {"m": "http://schemas.openxmlformats.org/"
                   "spreadsheetml/2006/main"}
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        names = [s.get("name") for s in wb.find("m:sheets", ns)]
        target = sheets[names.index(sheet_name)]
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    root = ET.fromstring(zf.read(target))
    grid = {}
    maxc = 0
    for row in root.iter(ns + "row"):
        r = int(row.get("r")) - 1
        for cell in row.iter(ns + "c"):
            ref = cell.get("r")
            c = _xlsx_col_index(ref)
            t = cell.get("t")
            if t == "inlineStr":
                tnode = cell.find(ns + "is/" + ns + "t")
                grid[(r, c)] = tnode.text if tnode is not None else ""
                maxc = max(maxc, c + 1)
                continue
            vnode = cell.find(ns + "v")
            if vnode is None:
                continue
            raw = vnode.text
            if t == "s":
                val = shared[int(raw)]
            elif t == "b":
                val = bool(int(raw))
            else:
                try:
                    fv = float(raw)
                    val = int(fv) if fv == int(fv) else fv
                except (TypeError, ValueError):
                    val = raw
            grid[(r, c)] = val
            maxc = max(maxc, c + 1)
    if not grid:
        return DataFrame({})
    maxr = max(r for r, _ in grid) + 1
    rows = [[grid.get((r, c), _NAN) for c in range(maxc)]
            for r in range(maxr)]
    if header is None:
        return DataFrame(rows)
    cols = [str(v) for v in rows[header]]
    body = rows[header + 1:]
    return DataFrame({cols[j]: [row[j] for row in body]
                      for j in range(maxc)})


class ExcelFile:
    def __init__(self, path):
        import xml.etree.ElementTree as ET
        import zipfile
        self._path = path
        zf = zipfile.ZipFile(path)
        ns = {"m": "http://schemas.openxmlformats.org/"
                   "spreadsheetml/2006/main"}
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        self.sheet_names = [s.get("name")
                            for s in wb.find("m:sheets", ns)]

    def parse(self, sheet_name=0, **kw):
        return read_excel(self._path, sheet_name=sheet_name, **kw)


def read_parquet(path, **kw):
    raise ImportError(
        "read_parquet requires a parquet engine; morie stores its "
        "native data as CSV/JSON — convert the file or install "
        "pyarrow for this one path (path=%r)" % (path,))


def pivot_table(data, values=None, index=None, columns=None,
                aggfunc="mean"):
    return data.pivot_table(values=values, index=index,
                            columns=columns, aggfunc=aggfunc)


class MultiIndex:
    """Minimal from_tuples/from_product holder."""

    def __init__(self, tuples):
        self.tuples = list(tuples)

    @classmethod
    def from_tuples(cls, tuples, names=None):
        obj = cls(tuples)
        obj.names = names
        return obj

    @classmethod
    def from_product(cls, iterables, names=None):
        out = [()]
        for it in iterables:
            out = [t + (v,) for t in out for v in it]
        obj = cls(out)
        obj.names = names
        return obj

    def __iter__(self):
        return iter(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def tolist(self):
        return list(self.tuples)
