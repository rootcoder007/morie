"""jsonlt -- jsonlite's JSON mapping, implemented natively.

The R arms used to call jsonlite for JSON. That was the last runtime
dependency doing real work, and it is gone: aaa_helpers_s03.R carries a
codec covering the four functions that were in use. This module is the
rest of it -- jsonlite's actual surface, which is much wider than four
functions, in both languages so the mapping can be checked by execution
rather than asserted.

What makes this more than "call json.dumps" is that jsonlite is not a JSON
library, it is a MAPPING between R's data model and JSON's. R has no
scalars, so every length-one value is ambiguous; a data.frame is a list of
columns that most people want written as an array of rows; NA is neither
null nor a number; a factor is an integer vector wearing a string coat.
jsonlite resolves each of those with a documented, switchable rule, and it
is those rules -- not the bracket-and-comma part -- that a second
implementation has to reproduce.

The R data model is carried here by small tagged wrappers (RVector,
DataFrame, Matrix, Factor, RawVec, Boxed, Unboxed, NA). They exist so the
Python arm can express an R value exactly; nothing else in the package
needs them.

Reference
  Ooms, J. (2014) "The jsonlite Package: A Practical and Consistent Mapping
    Between JSON Data and R Objects." arXiv:1403.2805.
  Bray, T. (ed.) (2017) "The JavaScript Object Notation (JSON) Data
    Interchange Format." RFC 8259.
"""

import datetime
import math
import re

__all__ = [
    "jsonlt", "to_json", "from_json", "prettify", "minify", "flatten",
    "serialize_json", "unserialize_json", "box", "unbox",
    "RVector", "DataFrame", "Matrix", "Factor", "RawVec", "Boxed",
    "Unboxed", "NA", "Sig",
]


# ---------------------------------------------------------------- R model

class _NAType(object):
    """R's NA. Not None (that is NULL) and not NaN (that is a number)."""

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super(_NAType, cls).__new__(cls)
        return cls._inst

    def __repr__(self):
        return "NA"

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return hash("morie.NA")


NA = _NAType()


class RVector(object):
    """An R atomic vector: values plus one type for all of them."""

    def __init__(self, values, rtype=None):
        self.values = list(values)
        self.rtype = rtype or _infer_type(self.values)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return "RVector(%r, %r)" % (self.values, self.rtype)


class DataFrame(object):
    """Columns, in order. Every column the same length."""

    def __init__(self, columns):
        if isinstance(columns, dict):
            columns = list(columns.items())
        self.columns = [(str(k), v) for k, v in columns]
        lens = set(len(_as_values(v)) for _, v in self.columns)
        if len(lens) > 1:
            raise ValueError("jsonlt: data.frame columns differ in length: %s"
                             % sorted(lens))
        self.nrow = lens.pop() if lens else 0

    def names(self):
        return [k for k, _ in self.columns]

    def col(self, name):
        for k, v in self.columns:
            if k == name:
                return v
        raise KeyError(name)


class Matrix(object):
    """A rectangular numeric/character array, stored row by row."""

    def __init__(self, rows, rtype=None):
        self.rows = [list(r) for r in rows]
        w = set(len(r) for r in self.rows)
        if len(w) > 1:
            raise ValueError("jsonlt: matrix rows differ in length")
        self.ncol = w.pop() if w else 0
        flat = [v for r in self.rows for v in r]
        self.rtype = rtype or _infer_type(flat)

    @property
    def nrow(self):
        return len(self.rows)


class Factor(object):
    """Integer codes (1-based, R's own) over a level table."""

    def __init__(self, codes, levels):
        self.codes = list(codes)
        self.levels = [str(v) for v in levels]

    def as_strings(self):
        out = []
        for c in self.codes:
            if c is NA or c is None:
                out.append(NA)
            else:
                out.append(self.levels[int(c) - 1])
        return out


class RawVec(object):
    """R's raw vector: bytes."""

    def __init__(self, data):
        if isinstance(data, (bytes, bytearray)):
            self.data = bytes(data)
        else:
            self.data = bytes(bytearray(int(b) & 0xFF for b in data))


class Boxed(object):
    """I(x): never unboxed, however long it is. jsonlite's AsIs."""

    def __init__(self, value):
        self.value = value


class Unboxed(object):
    """unbox(x): always a bare scalar. Errors on length != 1."""

    def __init__(self, value):
        n = len(_as_values(value))
        if n != 1:
            raise ValueError("jsonlt: unbox() needs length 1, got %d" % n)
        self.value = value


class Sig(object):
    """digits = Sig(n): n SIGNIFICANT digits, jsonlite's digits = I(n)."""

    def __init__(self, n):
        self.n = int(n)


def box(x):
    return Boxed(x)


def unbox(x):
    return Unboxed(x)


_TYPE_ORDER = ["logical", "integer", "double", "character"]


def _infer_type(values):
    seen = "logical"
    any_val = False
    for v in values:
        if v is NA or v is None:
            continue
        any_val = True
        if isinstance(v, bool):
            t = "logical"
        elif isinstance(v, int):
            t = "integer"
        elif isinstance(v, float):
            t = "double"
        elif isinstance(v, complex):
            t = "complex"
        elif isinstance(v, (datetime.datetime,)):
            t = "POSIXt"
        elif isinstance(v, (datetime.date,)):
            t = "Date"
        else:
            t = "character"
        if t in ("complex", "POSIXt", "Date"):
            return t
        if _TYPE_ORDER.index(t) > _TYPE_ORDER.index(seen):
            seen = t
    return seen if any_val else "logical"


def _as_values(x):
    """The element list behind any of the wrappers, for length questions."""
    if isinstance(x, RVector):
        return x.values
    if isinstance(x, Factor):
        return x.codes
    if isinstance(x, RawVec):
        return list(x.data)
    if isinstance(x, Matrix):
        return [v for r in x.rows for v in r]
    if isinstance(x, DataFrame):
        return list(range(x.nrow))
    if isinstance(x, (Boxed, Unboxed)):
        return _as_values(x.value)
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None:
        return []
    return [x]


# ---------------------------------------------------------------- numbers

_EPOCH_D = datetime.date(1970, 1, 1)


def _fmt_num(x, digits):
    """The one place a double becomes text.

    digits = None keeps every digit that round-trips; an int rounds to that
    many DECIMAL places (jsonlite's rule, and its default is 4); Sig(n)
    keeps n SIGNIFICANT digits, which is jsonlite's digits = I(n).

    Both arms go through C's printf with the same format, so the bytes are
    the same bytes. Doing this with each language's own float-to-string
    would put the two arms one ulp apart and call it a parity failure.
    """
    if x != x:
        return '"NaN"'
    if x == float("inf"):
        return '"Inf"'
    if x == float("-inf"):
        return '"-Inf"'
    if digits is None:
        s = "%.17g" % x
    elif isinstance(digits, Sig):
        s = "%.*g" % (max(1, digits.n), x)
    else:
        d = int(digits)
        if d < 0:
            d = 0
        s = "%.*f" % (d, x)
    return _tidy(s)


def _tidy(s):
    """Trim a printf result to jsonlite's shape: no trailing zeros, no
    leading plus in the exponent, no bare trailing dot."""
    if "e" in s or "E" in s:
        mant, _, expo = s.partition("e") if "e" in s else s.partition("E")
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        sign = ""
        if expo and expo[0] in "+-":
            sign = "-" if expo[0] == "-" else ""
            expo = expo[1:]
        expo = expo.lstrip("0") or "0"
        return "%se%s%s" % (mant, sign, expo)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


# ---------------------------------------------------------------- escaping

_ESC = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f",
        "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _esc(s):
    out = []
    for c in s:
        e = _ESC.get(c)
        if e is not None:
            out.append(e)
        elif ord(c) < 0x20:
            out.append("\\u%04x" % ord(c))
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


_B64 = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")


def _b64(data):
    """Base64 without a library, so both arms produce the same string."""
    out = []
    i = 0
    n = len(data)
    while i + 2 < n:
        v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
        out.append(_B64[(v >> 18) & 63] + _B64[(v >> 12) & 63] +
                   _B64[(v >> 6) & 63] + _B64[v & 63])
        i += 3
    rem = n - i
    if rem == 1:
        v = data[i] << 16
        out.append(_B64[(v >> 18) & 63] + _B64[(v >> 12) & 63] + "==")
    elif rem == 2:
        v = (data[i] << 16) | (data[i + 1] << 8)
        out.append(_B64[(v >> 18) & 63] + _B64[(v >> 12) & 63] +
                   _B64[(v >> 6) & 63] + "=")
    return "".join(out)


def _unb64(s):
    s = "".join(c for c in s if not c.isspace())
    pad = s.count("=")
    s = s.rstrip("=")
    bits = 0
    acc = 0
    out = bytearray()
    for c in s:
        acc = (acc << 6) | _B64.index(c)
        bits += 6
        if bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
    del pad
    return bytes(out)


# ---------------------------------------------------------------- encoder

class _Opt(object):
    pass


def _opts(**kw):
    o = _Opt()
    o.dataframe = kw.get("dataframe", "rows")
    o.matrix = kw.get("matrix", "rowmajor")
    o.Date = kw.get("Date", "ISO8601")
    o.POSIXt = kw.get("POSIXt", "string")
    o.factor = kw.get("factor", "string")
    o.complex = kw.get("complex", "string")
    o.raw = kw.get("raw", "base64")
    o.null = kw.get("null", "list")
    o.na = kw.get("na", None)
    o.auto_unbox = kw.get("auto_unbox", False)
    o.digits = kw.get("digits", 4)
    o.force = kw.get("force", False)
    for name, allowed in (("dataframe", ("rows", "columns", "values")),
                          ("matrix", ("rowmajor", "columnmajor")),
                          ("Date", ("ISO8601", "epoch")),
                          ("POSIXt", ("string", "ISO8601", "epoch", "mongo")),
                          ("factor", ("string", "integer")),
                          ("complex", ("string", "list")),
                          ("raw", ("base64", "hex", "int", "mongo")),
                          ("null", ("list", "null")),
                          ("na", (None, "null", "string"))):
        v = getattr(o, name)
        if v not in allowed:
            raise ValueError("jsonlt: %s = %r; expected one of %s"
                             % (name, v, ", ".join(map(repr, allowed))))
    return o


def _na_token(o, rtype):
    """What an NA becomes. None means "leave this field out", which is what
    jsonlite does inside data.frame rows when na is unset."""
    if o.na == "string":
        return '"NA"'
    if o.na == "null":
        return "null"
    if o.na is None and rtype == "character":
        return "null"
    return "null"


def _scalar(v, rtype, o):
    if v is NA or v is None:
        return _na_token(o, rtype)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, complex):
        if o.complex == "string":
            re_s = _fmt_num(v.real, o.digits)
            im_s = _fmt_num(abs(v.imag), o.digits)
            sign = "-" if v.imag < 0 else "+"
            return _esc("%s%s%si" % (re_s, sign, im_s))
        return None                      # handled vector-wise
    if isinstance(v, datetime.datetime):
        return _posix(v, o)
    if isinstance(v, datetime.date):
        if o.Date == "epoch":
            return _fmt_num(float((v - _EPOCH_D).days), o.digits)
        return _esc(v.isoformat())
    if isinstance(v, float):
        return _fmt_num(v, o.digits)
    if isinstance(v, int):
        return "%d" % v
    return _esc(str(v))


def _posix(v, o):
    if v.tzinfo is None:
        v = v.replace(tzinfo=datetime.timezone.utc)
    secs = (v - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc))
    ms = secs.total_seconds() * 1000.0
    if o.POSIXt == "epoch":
        return _fmt_num(ms, o.digits)
    if o.POSIXt == "mongo":
        return '{"$date":%s}' % _fmt_num(ms, o.digits)
    u = v.astimezone(datetime.timezone.utc)
    if o.POSIXt == "ISO8601":
        return _esc(u.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return _esc(u.strftime("%Y-%m-%d %H:%M:%S"))


def _atomic(values, rtype, o, unbox_ok):
    if rtype == "complex" and o.complex == "list":
        reals = [v.real if isinstance(v, complex) else v for v in values]
        imags = [v.imag if isinstance(v, complex) else v for v in values]
        return '{"r":%s,"i":%s}' % (
            _atomic(reals, "double", o, False),
            _atomic(imags, "double", o, False))
    parts = [_scalar(v, rtype, o) for v in values]
    if len(parts) == 1 and unbox_ok and o.auto_unbox:
        return parts[0]
    return "[" + ",".join(parts) + "]"


def _raw_enc(rv, o):
    if o.raw == "int":
        return "[" + ",".join("%d" % b for b in rv.data) + "]"
    if o.raw == "hex":
        return _esc("".join("%02x" % b for b in rv.data))
    if o.raw == "mongo":
        return '{"$binary":%s,"$type":"00"}' % _esc(_b64(rv.data))
    return _esc(_b64(rv.data))


def _df_rows(df, o):
    rows = []
    for i in range(df.nrow):
        parts = []
        for name, col in df.columns:
            cell = _cell(col, i, o)
            if cell is None:                 # NA, na unset: field omitted
                continue
            parts.append("%s:%s" % (_esc(name), cell))
        rows.append("{" + ",".join(parts) + "}")
    return "[" + ",".join(rows) + "]"


def _cell(col, i, o):
    """One data.frame cell. None means "omit this field"."""
    if isinstance(col, DataFrame):
        sub = DataFrame([(k, _slice(v, i)) for k, v in col.columns])
        return _df_rows(sub, o)
    if isinstance(col, Matrix):
        return _atomic(col.rows[i], col.rtype, o, False)
    if isinstance(col, Factor):
        vals = col.as_strings() if o.factor == "string" else col.codes
        rt = "character" if o.factor == "string" else "integer"
        v = vals[i]
        if (v is NA or v is None) and o.na is None:
            return None
        return _scalar(v, rt, o)
    if isinstance(col, RawVec):
        return _raw_enc(RawVec([col.data[i]]), o)
    if isinstance(col, (list, tuple)) and col and isinstance(col[0], (list, DataFrame, dict)):
        return _encode(col[i], o, True)
    vals = _as_values(col)
    rt = col.rtype if isinstance(col, RVector) else _infer_type(vals)
    v = vals[i]
    if (v is NA or v is None) and o.na is None:
        return None
    return _scalar(v, rt, o)


def _slice(col, i):
    if isinstance(col, RVector):
        return RVector([col.values[i]], col.rtype)
    if isinstance(col, DataFrame):
        return DataFrame([(k, _slice(v, i)) for k, v in col.columns])
    return [_as_values(col)[i]]


def _df_columns(df, o):
    parts = []
    for name, col in df.columns:
        parts.append("%s:%s" % (_esc(name), _encode(col, o, False)))
    return "{" + ",".join(parts) + "}"


def _df_values(df, o):
    rows = []
    for i in range(df.nrow):
        parts = []
        for _, col in df.columns:
            cell = _cell(col, i, o)
            parts.append("null" if cell is None else cell)
        rows.append("[" + ",".join(parts) + "]")
    return "[" + ",".join(rows) + "]"


def _encode(x, o, unbox_ok):
    if x is None:
        return "{}" if o.null == "list" else "null"
    if isinstance(x, Unboxed):
        vals = _as_values(x.value)
        rt = (x.value.rtype if isinstance(x.value, RVector)
              else _infer_type(vals))
        return _scalar(vals[0], rt, o)
    if isinstance(x, Boxed):
        inner = x.value
        if isinstance(inner, (RVector, list, tuple)):
            vals = _as_values(inner)
            rt = (inner.rtype if isinstance(inner, RVector)
                  else _infer_type(vals))
            return _atomic(vals, rt, o, False)
        return _encode(inner, o, False)
    if isinstance(x, DataFrame):
        if o.dataframe == "columns":
            return _df_columns(x, o)
        if o.dataframe == "values":
            return _df_values(x, o)
        return _df_rows(x, o)
    if isinstance(x, Matrix):
        if o.matrix == "columnmajor":
            cols = [[x.rows[r][c] for r in range(x.nrow)]
                    for c in range(x.ncol)]
            return "[" + ",".join(_atomic(c, x.rtype, o, False)
                                  for c in cols) + "]"
        return "[" + ",".join(_atomic(r, x.rtype, o, False)
                              for r in x.rows) + "]"
    if isinstance(x, Factor):
        if o.factor == "integer":
            return _atomic(x.codes, "integer", o, unbox_ok)
        return _atomic(x.as_strings(), "character", o, unbox_ok)
    if isinstance(x, RawVec):
        return _raw_enc(x, o)
    if isinstance(x, RVector):
        return _atomic(x.values, x.rtype, o, unbox_ok)
    if isinstance(x, dict):
        parts = ["%s:%s" % (_esc(str(k)), _encode(v, o, True))
                 for k, v in x.items()]
        return "{" + ",".join(parts) + "}"
    if isinstance(x, (list, tuple)):
        # A plain list is an R LIST, not an atomic vector, and jsonlite
        # writes list(1, 2, 3) as [[1],[2],[3]] because each element is
        # itself a length-one vector. Collapsing it here would erase the
        # only distinction R draws between the two -- RVector is how an
        # atomic vector is spelled.
        return "[" + ",".join(_encode(v, o, True) for v in x) + "]"
    if _is_scalar(x):
        return _atomic([x], _infer_type([x]), o, unbox_ok)
    if o.force:
        return _esc(str(x))
    raise TypeError("jsonlt: no JSON mapping for %s; pass force = True to "
                    "write it as a string" % type(x).__name__)


def _is_scalar(v):
    return (v is NA or v is None or
            isinstance(v, (bool, int, float, complex, str,
                           datetime.date, datetime.datetime)))


def to_json(x, **kw):
    """jsonlite::toJSON. Every option it documents, same defaults."""
    pretty = kw.pop("pretty", False)
    o = _opts(**kw)
    s = _encode(x, o, True)
    if pretty is True:
        return prettify(s, 4)
    if pretty:
        return prettify(s, int(pretty))
    return s


# ---------------------------------------------------------------- parser

class _P(object):
    def __init__(self, s):
        self.s = s
        self.i = 0
        self.n = len(s)

    def ws(self):
        while self.i < self.n and self.s[self.i] in " \t\n\r":
            self.i += 1

    def err(self, msg):
        raise ValueError("jsonlt: %s at character %d" % (msg, self.i + 1))

    def value(self):
        self.ws()
        if self.i >= self.n:
            self.err("unexpected end of input")
        c = self.s[self.i]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c == '"':
            return self.string()
        if self.s.startswith("true", self.i):
            self.i += 4
            return True
        if self.s.startswith("false", self.i):
            self.i += 5
            return False
        if self.s.startswith("null", self.i):
            self.i += 4
            return None
        return self.number()

    def obj(self):
        self.i += 1
        out = {}
        self.ws()
        if self.i < self.n and self.s[self.i] == "}":
            self.i += 1
            return out
        while True:
            self.ws()
            k = self.string()
            self.ws()
            if self.i >= self.n or self.s[self.i] != ":":
                self.err("expected ':'")
            self.i += 1
            out[k] = self.value()
            self.ws()
            if self.i < self.n and self.s[self.i] == ",":
                self.i += 1
                continue
            if self.i < self.n and self.s[self.i] == "}":
                self.i += 1
                return out
            self.err("expected ',' or '}'")

    def arr(self):
        self.i += 1
        out = []
        self.ws()
        if self.i < self.n and self.s[self.i] == "]":
            self.i += 1
            return out
        while True:
            out.append(self.value())
            self.ws()
            if self.i < self.n and self.s[self.i] == ",":
                self.i += 1
                continue
            if self.i < self.n and self.s[self.i] == "]":
                self.i += 1
                return out
            self.err("expected ',' or ']'")

    def string(self):
        if self.i >= self.n or self.s[self.i] != '"':
            self.err("expected a string")
        self.i += 1
        out = []
        while self.i < self.n and self.s[self.i] != '"':
            c = self.s[self.i]
            if c == "\\":
                self.i += 1
                e = self.s[self.i]
                if e == "u":
                    cp = int(self.s[self.i + 1:self.i + 5], 16)
                    self.i += 4
                    if 0xD800 <= cp <= 0xDBFF and \
                       self.s[self.i + 1:self.i + 3] == "\\u":
                        lo = int(self.s[self.i + 3:self.i + 7], 16)
                        self.i += 6
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    out.append(chr(cp))
                else:
                    out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b",
                                "f": "\f"}.get(e, e))
            else:
                out.append(c)
            self.i += 1
        self.i += 1
        return "".join(out)

    def number(self):
        m = re.compile(r"-?(?:0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?").match(
            self.s, self.i)
        if not m:
            self.err("not a number")
        self.i = m.end()
        txt = m.group(0)
        if m.group(1) is None and m.group(2) is None:
            v = int(txt)
            if -2147483647 <= v <= 2147483647:
                return v
            return float(v)
        return float(txt)


def _all_scalar(xs):
    return all(x is None or isinstance(x, (bool, int, float, str))
               for x in xs)


def _simplify(x, sv, sdf, sm, flat):
    if isinstance(x, dict):
        return dict((k, _simplify(v, sv, sdf, sm, flat))
                    for k, v in x.items())
    if not isinstance(x, list):
        return x
    kids = [_simplify(v, sv, sdf, sm, flat) for v in x]
    if not kids:
        return kids
    if sdf and all(isinstance(k, dict) for k in kids):
        names = []
        for k in kids:
            for nm in k:
                if nm not in names:
                    names.append(nm)
        cols = []
        for nm in names:
            col = [k.get(nm, NA) for k in kids]
            cols.append((nm, RVector(col) if _all_scalar(
                [c for c in col if c is not NA]) else col))
        df = DataFrame(cols)
        return flatten(df) if flat else df
    if sm and all(isinstance(k, RVector) for k in kids) and \
            len(set(len(k) for k in kids)) == 1 and len(kids[0]) > 0:
        return Matrix([k.values for k in kids])
    if sv and _all_scalar(kids):
        return RVector(kids)
    return kids


def from_json(txt, simplify_vector=True, simplify_data_frame=True,
              simplify_matrix=True, flatten=False, simplify=None):
    """jsonlite::fromJSON. simplify = False turns all three switches off,
    which is what simplifyVector = FALSE does there."""
    if simplify is False:
        simplify_vector = simplify_data_frame = simplify_matrix = False
    if isinstance(txt, (list, tuple)):
        txt = "\n".join(txt)
    p = _P(txt)
    v = p.value()
    p.ws()
    if p.i != p.n:
        p.err("trailing content")
    return _simplify(v, simplify_vector, simplify_data_frame,
                     simplify_matrix, flatten)


# ---------------------------------------------------------------- text ops

def _walk_text(txt, on_open, on_close, on_comma, on_colon, on_other):
    out = []
    depth = 0
    instr = False
    esc = False
    for c in txt:
        if instr:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
            out.append(c)
        elif c in "{[":
            depth += 1
            out.append(on_open(c, depth))
        elif c in "}]":
            depth -= 1
            out.append(on_close(c, depth))
        elif c == ",":
            out.append(on_comma(depth))
        elif c == ":":
            out.append(on_colon())
        else:
            out.append(on_other(c))
    return "".join(out)


def prettify(txt, indent=4):
    """jsonlite::prettify. Structure only -- a comma inside a string stays
    a comma, which is the whole reason this is not a regex."""
    ind = int(indent)
    empty = re.compile(r"([\[{])(\s*)([\]}])")

    def pad(d):
        return "\n" + " " * (d * ind)

    s = _walk_text(
        txt,
        lambda c, d: c + pad(d),
        lambda c, d: pad(d) + c,
        lambda d: "," + pad(d),
        lambda: ": ",
        lambda c: "" if c in " \t\n\r" else c)
    prev = None
    while prev != s:
        prev = s
        s = empty.sub(r"\1\3", s)
    return s


def minify(txt):
    """jsonlite::minify. Every byte of whitespace outside a string goes."""
    return _walk_text(
        txt,
        lambda c, d: c,
        lambda c, d: c,
        lambda d: ",",
        lambda: ":",
        lambda c: "" if c in " \t\n\r" else c)


def flatten(df, recursive=True):
    """jsonlite::flatten. A data.frame column that is itself a data.frame
    becomes columns named outer.inner."""
    if not isinstance(df, DataFrame):
        return df
    cols = []
    for name, col in df.columns:
        if isinstance(col, DataFrame):
            inner = flatten(col) if recursive else col
            for nm2, c2 in inner.columns:
                cols.append(("%s.%s" % (name, nm2), c2))
        else:
            cols.append((name, col))
    return DataFrame(cols)


# ---------------------------------------------------- serialize/unserialize

_RTYPE_JSON = {"logical": "logical", "integer": "integer",
               "double": "double", "character": "character",
               "complex": "complex"}


def serialize_json(x, pretty=False):
    """jsonlite::serializeJSON. Lossless: type and attributes travel with
    the value, so unserialize_json returns the same object rather than
    something that merely prints the same."""
    s = _ser(x)
    return prettify(s, 4) if pretty else s


def _ser_attr(pairs):
    if not pairs:
        return "{}"
    return "{" + ",".join("%s:%s" % (_esc(k), _ser(v))
                          for k, v in pairs) + "}"


def _ser(x):
    o = _opts(digits=None, na="null", auto_unbox=False)
    if x is None:
        return '{"type":"NULL","attributes":{},"value":{}}'
    if isinstance(x, DataFrame):
        attrs = [("names", RVector(df_names(x), "character")),
                 ("class", RVector(["data.frame"], "character")),
                 ("row.names", RVector(list(range(1, x.nrow + 1)),
                                       "integer"))]
        vals = ",".join(_ser(c) for _, c in x.columns)
        return ('{"type":"list","attributes":%s,"value":[%s]}'
                % (_ser_attr(attrs), vals))
    if isinstance(x, Matrix):
        flat = [x.rows[r][c] for c in range(x.ncol) for r in range(x.nrow)]
        attrs = [("dim", RVector([x.nrow, x.ncol], "integer"))]
        return ('{"type":"%s","attributes":%s,"value":%s}'
                % (_RTYPE_JSON.get(x.rtype, "character"), _ser_attr(attrs),
                   _atomic(flat, x.rtype, o, False)))
    if isinstance(x, Factor):
        attrs = [("levels", RVector(x.levels, "character")),
                 ("class", RVector(["factor"], "character"))]
        return ('{"type":"integer","attributes":%s,"value":%s}'
                % (_ser_attr(attrs), _atomic(x.codes, "integer", o, False)))
    if isinstance(x, RawVec):
        return ('{"type":"raw","attributes":{},"value":%s}'
                % _esc(_b64(x.data)))
    if isinstance(x, (Boxed, Unboxed)):
        return _ser(x.value)
    if isinstance(x, dict):
        attrs = [("names", RVector([str(k) for k in x], "character"))]
        vals = ",".join(_ser(v) for v in x.values())
        return ('{"type":"list","attributes":%s,"value":[%s]}'
                % (_ser_attr(attrs), vals))
    if isinstance(x, RVector):
        return ('{"type":"%s","attributes":{},"value":%s}'
                % (_RTYPE_JSON.get(x.rtype, "character"),
                   _atomic(x.values, x.rtype, o, False)))
    if isinstance(x, (list, tuple)):
        return ('{"type":"list","attributes":{},"value":[%s]}'
                % ",".join(_ser(v) for v in x))
    return _ser(RVector([x]))


def df_names(df):
    return df.names()


def unserialize_json(txt):
    return _unser(from_json(txt, simplify_vector=False,
                            simplify_data_frame=False,
                            simplify_matrix=False))


def _coerce(v, t):
    if v is None:
        return NA
    if t == "logical":
        return bool(v)
    if t == "integer":
        return int(v)
    if t == "double":
        if isinstance(v, str):
            return {"Inf": float("inf"), "-Inf": float("-inf"),
                    "NaN": float("nan")}.get(v, float(v))
        return float(v)
    return str(v)


def _unser(node):
    if not isinstance(node, dict) or "type" not in node:
        raise ValueError("jsonlt: not a serializeJSON document")
    t = node["type"]
    raw_attrs = node.get("attributes") or {}
    # An attribute is itself a serialized R value, so it needs the same
    # walk. Reading its "value" field directly is how a names vector comes
    # back as the literal string "type".
    attrs = dict((k, _as_values(_unser(v)))
                 for k, v in raw_attrs.items() if isinstance(v, dict))
    val = node.get("value")
    if t == "NULL":
        return None
    if t == "raw":
        return RawVec(_unb64(val))
    if t == "list":
        kids = [_unser(v) for v in (val or [])]
        cls = attrs.get("class")
        names = attrs.get("names")
        names = [str(n) for n in (names or [])]
        if cls and "data.frame" in [str(c) for c in cls]:
            return DataFrame(list(zip(names, kids)))
        if names and len(names) == len(kids):
            return dict(zip(names, kids))
        return kids
    vals = val if isinstance(val, list) else [val]
    vals = [_coerce(v, t) for v in vals]
    if "levels" in attrs:
        return Factor(vals, [str(l) for l in attrs["levels"]])
    if "dim" in attrs:
        d = [int(v) for v in attrs["dim"]]
        nr, nc = d[0], d[1]
        rows = [[vals[c * nr + r] for c in range(nc)] for r in range(nr)]
        return Matrix(rows, t)
    return RVector(vals, t)


# ---------------------------------------------------------------- route

_ROUTES = ("to_json", "from_json", "prettify", "minify", "flatten",
           "serialize", "unserialize")


def jsonlt(x=None, route="to_json", **kw):
    """One entry point over the routes, so the module can be driven the
    same way from either arm.

    route: to_json | from_json | prettify | minify | flatten | serialize |
           unserialize
    """
    if route not in _ROUTES:
        raise ValueError("jsonlt: route = %r; expected one of %s"
                         % (route, ", ".join(_ROUTES)))
    if route == "to_json":
        out = to_json(x, **kw)
    elif route == "from_json":
        out = from_json(x, **kw)
    elif route == "prettify":
        out = prettify(x, kw.get("indent", 4))
    elif route == "minify":
        out = minify(x)
    elif route == "flatten":
        out = flatten(x, kw.get("recursive", True))
    elif route == "serialize":
        out = serialize_json(x, kw.get("pretty", False))
    else:
        out = unserialize_json(x)
    return {"route": route, "result": out,
            "method": "jsonlite mapping (Ooms 2014), RFC 8259"}
