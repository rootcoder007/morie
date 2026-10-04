"""Read R's ``.rds`` files natively: the serialization format of R Internals, section 1.8.

``read_rds(path)`` returns a data frame (a ``morie.fn._frame_core.DataFrame``) for an R
``data.frame`` / ``data.table`` / tibble, with factors as their labels, and plain Python values
otherwise (a vector as a list, a named list as a dict). XDR (big-endian) format, versions 2 and 3,
compressed with gzip, bzip2 or xz or not at all -- what ``saveRDS()`` writes by default.

Examples:
    >>> import gzip, io, struct, tempfile, os
    >>> def i(*v): return struct.pack(f">{len(v)}i", *v)
    >>> def chars(s): b = s.encode(); return i(0x00040009, len(b)) + b   # CHARSXP, UTF-8
    >>> body = b"X\\n" + i(2, 0x040200, 0x020300)
    >>> body += i(0x0000020D, 3, 1, 2, -2147483648)                    # INTSXP with attributes
    >>> body += i(0x00000402) + i(0x00000001) + chars("names")         # pairlist: tag "names"
    >>> body += i(0x00000010, 3) + chars("a") + chars("b") + chars("c")
    >>> body += i(0x000000FE)                                          # end of attributes
    >>> p = os.path.join(tempfile.mkdtemp(), "x.rds")
    >>> with gzip.open(p, "wb") as fh: _ = fh.write(body)
    >>> read_rds(p)
    {'a': 1, 'b': 2, 'c': None}
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import math
import struct
from pathlib import Path
from typing import Any

_NA_INT = -2147483648


def _open_bytes(path) -> bytes:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(raw)
    return raw


class _RObj:
    """An R value with its attributes (names, class, levels, row.names, ...)."""

    __slots__ = ("value", "attrs", "rtype")

    def __init__(self, value, attrs=None, rtype=None):
        self.value, self.attrs, self.rtype = value, attrs or {}, rtype


class _Reader:
    def __init__(self, buf: bytes):
        self.b = buf
        self.p = 0
        self.refs: list[Any] = []

    def int(self) -> int:
        v = struct.unpack_from(">i", self.b, self.p)[0]
        self.p += 4
        return v

    def ints(self, n: int) -> tuple:
        v = struct.unpack_from(f">{n}i", self.b, self.p)
        self.p += 4 * n
        return v

    def doubles(self, n: int) -> tuple:
        v = struct.unpack_from(f">{n}d", self.b, self.p)
        self.p += 8 * n
        return v

    def length(self) -> int:
        n = self.int()
        if n == -1:  # a long vector: two more ints
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def charsxp(self, flags: int):
        n = self.int()
        if n == -1:
            return None  # NA_character_
        s = self.b[self.p : self.p + n]
        self.p += n
        levels = flags >> 12
        enc = "latin-1" if levels & 4 else "utf-8"  # LATIN1_MASK; UTF8, ASCII and native read as UTF-8
        return s.decode(enc, errors="replace")

    def attributes(self) -> dict:
        out = self.item()
        return out if isinstance(out, dict) else {}

    def item(self):
        flags = self.int()
        t = flags & 0xFF
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if t == 254:  # NILVALUE_SXP
            return None
        if t in (253, 242, 241, 251, 252):  # global / empty / base env, missing arg, unbound
            return None
        if t == 255:  # REFSXP
            idx = flags >> 8 or self.int()
            return self.refs[idx - 1]
        if t == 1:  # SYMSXP
            name = self.item()
            self.refs.append(name)
            return name
        if t == 9:
            return self.charsxp(flags)
        if t in (2, 4, 6, 17):  # pairlist (LISTSXP, CLOSXP, LANGSXP, DOTSXP) as a dict / list
            out: dict = {}
            pos = 0
            while True:
                if has_attr:
                    self.attributes()
                tag = self.item() if has_tag else None
                val = self.item()
                out[tag if tag is not None else pos] = val
                pos += 1
                flags = self.int()
                t2 = flags & 0xFF
                if t2 == 254:
                    break
                has_attr = bool(flags & (1 << 9))
                has_tag = bool(flags & (1 << 10))
            return out
        if t == 22:  # EXTPTRSXP (data.table's .internal.selfref)
            self.refs.append(None)
            self.item()  # prot
            self.item()  # tag
            if has_attr:
                self.attributes()
            return None
        if t == 4:  # ENVSXP (not reached: 4 is LISTSXP-like above for closures)
            pass
        if t == 238:  # ALTREP_SXP
            info = self.item()
            state = self.item()
            attrs = self.attributes()
            cls = info.get(0) if isinstance(info, dict) else None
            val = _altrep(cls, state)
            return _RObj(val, attrs) if attrs else val
        if t in (10, 13):  # LGLSXP, INTSXP
            n = self.length()
            v = [None if x == _NA_INT else x for x in self.ints(n)]
            if t == 10:
                v = [None if x is None else bool(x) for x in v]
        elif t == 14:  # REALSXP
            v = list(self.doubles(self.length()))
        elif t == 15:  # CPLXSXP
            d = self.doubles(2 * self.length())
            v = [complex(d[k], d[k + 1]) for k in range(0, len(d), 2)]
        elif t == 16:  # STRSXP
            n = self.length()
            v = []
            for _ in range(n):
                f = self.int()
                v.append(self.charsxp(f))
        elif t in (19, 20):  # VECSXP, EXPRSXP
            v = [self.item() for _ in range(self.length())]
        elif t == 24:  # RAWSXP
            n = self.length()
            v = self.b[self.p : self.p + n]
            self.p += n
        else:
            raise ValueError(f"R object type {t} is not supported by read_rds (a function, environment or S4 object?)")
        attrs = self.attributes() if has_attr else {}
        return _RObj(v, attrs, t) if attrs else v


def _altrep(cls, state):
    if cls in ("compact_intseq", "compact_realseq"):
        n, start, step = state[0], state[1], state[2]
        seq = [start + k * step for k in range(int(n))]
        return [int(x) for x in seq] if cls == "compact_intseq" else seq
    if isinstance(cls, str) and cls.startswith("wrap_"):
        x = state[0] if isinstance(state, list) else state
        return x.value if isinstance(x, _RObj) else x
    if cls == "deferred_string":
        x = state[0] if isinstance(state, list) else state
        x = x.value if isinstance(x, _RObj) else x
        return [None if v is None else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)) for v in x]
    raise ValueError(f"ALTREP class {cls!r} is not supported by read_rds")


def _plain(x):
    """An R value as plain Python: factors as labels, NA as None, named vectors/lists as dicts."""
    if not isinstance(x, _RObj):
        return x
    a = x.attrs
    cls = a.get("class")
    cls = [cls] if isinstance(cls, str) else list(_plain(cls) or [])
    if "factor" in cls:
        lev = _plain(a.get("levels")) or []
        return [None if c is None else lev[c - 1] for c in x.value]
    v = x.value
    if isinstance(v, list) and x.rtype == 19:
        v = [_plain(e) for e in v]
    names = _plain(a.get("names"))
    if names is not None and isinstance(v, list) and len(names) == len(v) and "data.frame" not in cls:
        return dict(zip(names, v))
    return v


def _parse(buf: bytes, name: str):
    if buf[:2] != b"X\n":
        raise ValueError(f"{name}: not XDR-serialized R data (ASCII and native-binary saves are not read)")
    r = _Reader(buf)
    r.p = 2
    version = r.int()
    r.int()  # writer's R version
    r.int()  # minimal reader version
    if version == 3:
        n = r.int()
        r.p += n  # native encoding name
    elif version != 2:
        raise ValueError(f"unsupported R serialization version {version}")
    return r.item()


def _frame_or_plain(obj):
    if isinstance(obj, _RObj):
        cls = obj.attrs.get("class")
        cls = [cls] if isinstance(cls, str) else list(_plain(cls) or [])
        if "data.frame" in cls:
            from morie.fn import _frame_core as pd

            names = _plain(obj.attrs.get("names")) or [f"V{k + 1}" for k in range(len(obj.value))]
            cols = {}
            for nm, col in zip(names, obj.value):
                vals = _plain(col)
                if vals and all(v is None or isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                    if any(v is None for v in vals):
                        vals = [math.nan if v is None else float(v) for v in vals]
                cols[nm] = vals
            return pd.DataFrame(cols)
    return _plain(obj)


def read_rdata(path) -> dict:
    """The objects of an ``.RData`` / ``.rda`` workspace (``save()``), by name: data frames as DataFrames."""
    buf = _open_bytes(path)
    if buf[:5] not in (b"RDX2\n", b"RDX3\n"):
        raise ValueError(f"{Path(path).name}: not an R workspace (RDX2/RDX3)")
    objs = _parse(buf[5:], Path(path).name)
    if not isinstance(objs, dict):
        return {}
    return {str(k): _frame_or_plain(v) for k, v in objs.items()}


def read_rds(path):
    """Read an ``.rds`` file written by ``saveRDS()``: a data frame comes back as a DataFrame.

    Factors become their labels; integer and logical NA become None (NaN in a numeric column of
    the frame); dates stay numbers (days since 1970-01-01, R's own value).
    """
    return _frame_or_plain(_parse(_open_bytes(path), Path(path).name))
