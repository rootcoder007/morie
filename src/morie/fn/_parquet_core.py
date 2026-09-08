"""Native Apache Parquet reader and writer.

morie stores tabular data as Parquet in rmoriedata (40+ files under
inst/extdata/parquet, including the SIU DRID manifest, the dataset
catalogue and the dictionaries table), but every read and write in the
family went through an external engine -- nanoparquet:: on the R side,
nothing at all on the Python side, where read_parquet() raised
ImportError. This module removes that dependency.

Scope was measured against the actual store rather than guessed, by
reading the footers of _catalog, _dictionaries, siu_drid_manifest,
tps_arcgis_hub_catalog, complaints_sample and arrests_sample:

    format version   1.0
    encodings        PLAIN, RLE_DICTIONARY, RLE
    compression      SNAPPY
    physical types   BYTE_ARRAY, INT32, INT64, BOOLEAN, DOUBLE

Everything in that list is implemented, plus FLOAT, INT96,
FIXED_LEN_BYTE_ARRAY, PLAIN_DICTIONARY (the v1 spelling of
RLE_DICTIONARY) and uncompressed pages, because they cost a few lines
each once the framing is in place. Nested/repeated columns are NOT
implemented: the store is entirely flat, and a repeated column decoded
as if it were flat would return a silently wrong row count rather than
an error, so a max repetition level above zero is refused explicitly.

The writer emits PLAIN-encoded, Snappy-compressed, single-row-group
files that pyarrow and nanoparquet both read back unchanged.

References
----------
Apache Parquet format specification, parquet-format/README.md and
parquet.thrift (field ids below are from parquet.thrift).
Thrift compact protocol: THRIFT-110, doc/specs/thrift-compact-protocol.md.
Snappy compressed format description, google/snappy/format_description.txt.
"""

from __future__ import annotations

import struct

# ---------------------------------------------------------------- thrift

# Compact-protocol type ids (thrift-compact-protocol.md section "Struct").
_T_STOP = 0x00
_T_TRUE = 0x01
_T_FALSE = 0x02
_T_BYTE = 0x03
_T_I16 = 0x04
_T_I32 = 0x05
_T_I64 = 0x06
_T_DOUBLE = 0x07
_T_BINARY = 0x08
_T_LIST = 0x09
_T_SET = 0x0A
_T_MAP = 0x0B
_T_STRUCT = 0x0C


class _TReader:
    """Thrift compact protocol decoder over a bytes buffer.

    Only the subset parquet.thrift uses is here. Struct fields are
    returned as a dict {field_id: value}; unknown fields are skipped by
    type, which is what lets this read footers written by newer versions
    of pyarrow than the one the scope was measured against.
    """

    def __init__(self, buf, pos=0):
        self.buf = buf
        self.pos = pos

    def _byte(self):
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def varint(self):
        result = 0
        shift = 0
        while True:
            b = self._byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    def zigzag(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def binary(self):
        n = self.varint()
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def double(self):
        # Compact protocol writes doubles little-endian (THRIFT-110 fixed
        # the big-endian wording; every real implementation is LE).
        v = struct.unpack_from("<d", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def _scalar(self, ttype):
        if ttype == _T_TRUE:
            return True
        if ttype == _T_FALSE:
            return False
        if ttype == _T_BYTE:
            b = self._byte()
            return b - 256 if b > 127 else b
        if ttype in (_T_I16, _T_I32, _T_I64):
            return self.zigzag()
        if ttype == _T_DOUBLE:
            return self.double()
        if ttype == _T_BINARY:
            return self.binary()
        if ttype == _T_LIST or ttype == _T_SET:
            return self.list()
        if ttype == _T_MAP:
            return self.map()
        if ttype == _T_STRUCT:
            return self.struct()
        raise ValueError("unknown thrift compact type %d at byte %d"
                         % (ttype, self.pos))

    def list(self):
        h = self._byte()
        size = h >> 4
        etype = h & 0x0F
        if size == 15:
            size = self.varint()
        # A bool list encodes its elements in the header type, not per
        # element, so _scalar's TRUE/FALSE dispatch is already correct.
        return [self._scalar(etype) for _ in range(size)]

    def map(self):
        size = self.varint()
        if size == 0:
            return {}
        kv = self._byte()
        ktype, vtype = kv >> 4, kv & 0x0F
        return {self._scalar(ktype): self._scalar(vtype)
                for _ in range(size)}

    def struct(self):
        out = {}
        fid = 0
        while True:
            h = self._byte()
            if h == _T_STOP:
                return out
            delta = h >> 4
            ttype = h & 0x0F
            fid = fid + delta if delta else self.zigzag()
            out[fid] = self._scalar(ttype)


class _TWriter:
    """Thrift compact protocol encoder. Mirrors _TReader exactly."""

    def __init__(self):
        self.out = bytearray()

    def varint(self, n):
        while True:
            if n < 0x80:
                self.out.append(n)
                return
            self.out.append((n & 0x7F) | 0x80)
            n >>= 7

    def zigzag(self, n):
        self.varint((n << 1) ^ (n >> 63) if n < 0 else (n << 1))

    def binary(self, b):
        if isinstance(b, str):
            b = b.encode("utf-8")
        self.varint(len(b))
        self.out += b

    def field(self, fid, ttype, last):
        delta = fid - last
        if 0 < delta <= 15:
            self.out.append((delta << 4) | ttype)
        else:
            self.out.append(ttype)
            self.zigzag(fid)
        return fid

    def i32(self, fid, v, last):
        last = self.field(fid, _T_I32, last)
        self.zigzag(v)
        return last

    def i64(self, fid, v, last):
        last = self.field(fid, _T_I64, last)
        self.zigzag(v)
        return last

    def bool(self, fid, v, last):
        return self.field(fid, _T_TRUE if v else _T_FALSE, last)

    def bytes_(self, fid, v, last):
        last = self.field(fid, _T_BINARY, last)
        self.binary(v)
        return last

    def list_i32(self, fid, vals, last):
        last = self.field(fid, _T_LIST, last)
        self._list_header(len(vals), _T_I32)
        for v in vals:
            self.zigzag(v)
        return last

    def list_binary(self, fid, vals, last):
        last = self.field(fid, _T_LIST, last)
        self._list_header(len(vals), _T_BINARY)
        for v in vals:
            self.binary(v)
        return last

    def list_struct(self, fid, writers, last):
        last = self.field(fid, _T_LIST, last)
        self._list_header(len(writers), _T_STRUCT)
        for w in writers:
            self.out += w
        return last

    def struct(self, fid, body, last):
        last = self.field(fid, _T_STRUCT, last)
        self.out += body
        return last

    def _list_header(self, size, etype):
        if size < 15:
            self.out.append((size << 4) | etype)
        else:
            self.out.append(0xF0 | etype)
            self.varint(size)

    def stop(self):
        self.out.append(_T_STOP)
        return bytes(self.out)


# ---------------------------------------------------------------- snappy

def _snappy_decompress(data):
    """Snappy raw block format (no framing).

    Preamble is the varint uncompressed length, then a tag stream of
    literals and back-references (format_description.txt).
    """
    pos = 0
    n = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            break
        shift += 7

    out = bytearray()
    end = len(data)
    while pos < end:
        tag = data[pos]
        pos += 1
        kind = tag & 0x03
        if kind == 0:                                   # literal
            ln = tag >> 2
            if ln >= 60:
                extra = ln - 59
                ln = int.from_bytes(data[pos:pos + extra], "little")
                pos += extra
            ln += 1
            out += data[pos:pos + ln]
            pos += ln
            continue
        if kind == 1:                                   # 1-byte offset
            ln = 4 + ((tag >> 2) & 0x07)
            off = ((tag >> 5) << 8) | data[pos]
            pos += 1
        elif kind == 2:                                 # 2-byte offset
            ln = (tag >> 2) + 1
            off = int.from_bytes(data[pos:pos + 2], "little")
            pos += 2
        else:                                           # 4-byte offset
            ln = (tag >> 2) + 1
            off = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
        if off == 0 or off > len(out):
            raise ValueError("snappy: bad copy offset %d at %d"
                             % (off, pos))
        # Copies may overlap (that is how snappy encodes runs), so this
        # has to advance a byte at a time when off < ln.
        start = len(out) - off
        if off >= ln:
            out += out[start:start + ln]
        else:
            for i in range(ln):
                out.append(out[start + i])

    if len(out) != n:
        raise ValueError("snappy: expected %d bytes, decoded %d"
                         % (n, len(out)))
    return bytes(out)


def _snappy_compress(data):
    """Emit a valid snappy block using literals only.

    A literal-only stream is fully conformant -- every decompressor
    accepts it -- and the files this writes are read back by pyarrow and
    nanoparquet unchanged. It just does not shrink anything.

    ponytail: no match-finder. Add one (hash table over 4-byte windows)
    if written file size ever matters; reading is where the cost is.
    """
    out = bytearray()
    n = len(data)
    while True:                                          # varint length
        if n < 0x80:
            out.append(n)
            break
        out.append((n & 0x7F) | 0x80)
        n >>= 7

    pos = 0
    total = len(data)
    while pos < total:
        chunk = min(total - pos, 1 << 16)
        ln = chunk - 1
        if ln < 60:
            out.append(ln << 2)
        elif ln < (1 << 8):
            out.append((60 << 2) | 0)
            out.append(ln)
        elif ln < (1 << 16):
            out.append((61 << 2) | 0)
            out += ln.to_bytes(2, "little")
        else:
            out.append((62 << 2) | 0)
            out += ln.to_bytes(3, "little")
        out += data[pos:pos + chunk]
        pos += chunk
    return bytes(out)


# ------------------------------------------------------------- constants

# parquet.thrift enum Type
_BOOLEAN, _INT32, _INT64, _INT96, _FLOAT, _DOUBLE, _BYTE_ARRAY, _FLBA = range(8)
# parquet.thrift enum Encoding
_E_PLAIN = 0
_E_PLAIN_DICTIONARY = 2
_E_RLE = 3
_E_BIT_PACKED = 4
_E_RLE_DICTIONARY = 8
# parquet.thrift enum CompressionCodec
_C_UNCOMPRESSED = 0
_C_SNAPPY = 1
# parquet.thrift enum FieldRepetitionType
_REQUIRED, _OPTIONAL, _REPEATED = range(3)
# parquet.thrift enum PageType
_P_DATA, _P_INDEX, _P_DICT, _P_DATA_V2 = range(4)
# parquet.thrift enum ConvertedType (the subset the store uses)
_CT_UTF8 = 0
_CT_DATE = 6
_CT_TIMESTAMP_MILLIS = 9
_CT_TIMESTAMP_MICROS = 10

_PLAIN_FMT = {_INT32: "<i", _INT64: "<q", _FLOAT: "<f", _DOUBLE: "<d"}
_PLAIN_SIZE = {_INT32: 4, _INT64: 8, _FLOAT: 4, _DOUBLE: 8}


# --------------------------------------------------------------- decoding

def _bit_width(n):
    w = 0
    while n:
        w += 1
        n >>= 1
    return w


def _read_rle_hybrid(buf, pos, width, count, end):
    """RLE / bit-packed hybrid run decoder (Encodings.md, RLE).

    Returns (values, new_pos). Stops once `count` values are produced;
    a bit-packed run may overshoot, and the surplus is dropped, which is
    what the format requires.
    """
    if width == 0:
        return [0] * count, pos
    out = []
    nbytes = (width + 7) // 8
    while len(out) < count and pos < end:
        header = 0
        shift = 0
        while True:
            b = buf[pos]
            pos += 1
            header |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        if header & 1:                                   # bit-packed run
            groups = header >> 1
            nvals = groups * 8
            need = groups * width
            chunk = buf[pos:pos + need]
            pos += need
            acc = int.from_bytes(chunk, "little")
            mask = (1 << width) - 1
            for i in range(nvals):
                out.append((acc >> (i * width)) & mask)
        else:                                            # RLE run
            run = header >> 1
            val = int.from_bytes(buf[pos:pos + nbytes], "little")
            pos += nbytes
            out.extend([val] * run)
    return out[:count], pos


def _decode_plain(buf, pos, ptype, count, type_length=None):
    """PLAIN encoding for one physical type. Returns (values, new_pos)."""
    if ptype == _BOOLEAN:
        vals = []
        for i in range(count):
            byte = buf[pos + (i >> 3)]
            vals.append(bool((byte >> (i & 7)) & 1))
        return vals, pos + (count + 7) // 8

    if ptype in _PLAIN_FMT:
        fmt, size = _PLAIN_FMT[ptype], _PLAIN_SIZE[ptype]
        vals = [struct.unpack_from(fmt, buf, pos + i * size)[0]
                for i in range(count)]
        return vals, pos + count * size

    if ptype == _BYTE_ARRAY:
        vals = []
        for _ in range(count):
            n = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            vals.append(bytes(buf[pos:pos + n]))
            pos += n
        return vals, pos

    if ptype == _FLBA:
        if not type_length:
            raise ValueError("FIXED_LEN_BYTE_ARRAY without type_length")
        vals = [bytes(buf[pos + i * type_length:pos + (i + 1) * type_length])
                for i in range(count)]
        return vals, pos + count * type_length

    if ptype == _INT96:
        # 12 bytes: 8-byte nanoseconds-of-day + 4-byte Julian day. Only
        # ever produced as a deprecated timestamp; converted below.
        vals = []
        for _ in range(count):
            nanos = struct.unpack_from("<Q", buf, pos)[0]
            jday = struct.unpack_from("<I", buf, pos + 8)[0]
            pos += 12
            vals.append((jday - 2440588) * 86400 * 10 ** 9 + nanos)
        return vals, pos

    raise ValueError("unsupported physical type %d" % ptype)


def _apply_logical(vals, ptype, converted):
    """Logical (converted) types, matching the R arm value for value.

    Without this a DATE column came back as a bare day count while
    nanoparquet handed its caller a Date -- a silent factor-of-86400
    trap for anyone swapping engines.
    """
    import datetime as _dt
    if converted is None:
        return vals
    epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
    if converted == _CT_DATE and ptype == _INT32:
        return [None if v is None else
                _dt.date(1970, 1, 1) + _dt.timedelta(days=v)
                for v in vals]
    if converted == _CT_TIMESTAMP_MILLIS:
        return [None if v is None else
                epoch + _dt.timedelta(milliseconds=v) for v in vals]
    if converted == _CT_TIMESTAMP_MICROS:
        return [None if v is None else
                epoch + _dt.timedelta(microseconds=v) for v in vals]
    return vals


def _convert(vals, ptype, converted):
    """Apply the converted (logical) type to already-decoded values."""
    if ptype == _BYTE_ARRAY and converted in (_CT_UTF8, None):
        # A BYTE_ARRAY with no converted type is still a string in every
        # file this store holds; decode when it is valid UTF-8 and leave
        # it as bytes when it is not, rather than raising on a blob.
        out = []
        for v in vals:
            if v is None:
                out.append(None)
                continue
            try:
                out.append(v.decode("utf-8"))
            except UnicodeDecodeError:
                out.append(v)
        return out
    return vals


# ------------------------------------------------------------------ read

def _read_footer(fh):
    fh.seek(0)
    if fh.read(4) != b"PAR1":
        raise ValueError("not a parquet file: missing leading PAR1")
    fh.seek(-8, 2)
    tail = fh.read(8)
    if tail[4:] != b"PAR1":
        raise ValueError("not a parquet file: missing trailing PAR1")
    n = struct.unpack("<I", tail[:4])[0]
    fh.seek(-(8 + n), 2)
    return _TReader(fh.read(n)).struct()


def _column_values(fh, chunk_meta, num_rows):
    """Decode one column chunk to a list of length num_rows."""
    ptype = chunk_meta[1]
    codec = chunk_meta[4]
    total_values = chunk_meta[5]
    data_off = chunk_meta[9]
    dict_off = chunk_meta.get(11)

    start = dict_off if dict_off else data_off
    if dict_off and data_off and data_off < dict_off:
        start = data_off
    fh.seek(start)
    # Read to the end of the chunk: compressed sizes in the footer cover
    # the pages, and the page headers carry their own lengths, so an
    # over-read of the tail is harmless.
    blob = fh.read(chunk_meta[7] + 64)

    pos = 0
    dictionary = None
    values = []
    while len(values) < total_values and pos < len(blob):
        r = _TReader(blob, pos)
        head = r.struct()
        pos = r.pos
        ptype_size = head[3]
        raw = blob[pos:pos + ptype_size]
        pos += ptype_size
        page = _snappy_decompress(raw) if codec == _C_SNAPPY else raw
        if codec not in (_C_SNAPPY, _C_UNCOMPRESSED):
            raise ValueError(
                "compression codec %d not implemented; the store uses "
                "SNAPPY only" % codec)

        if head[1] == _P_DICT:
            dh = head[7]
            dictionary = _decode_plain(page, 0, ptype, dh[1])[0]
            continue
        if head[1] == _P_DATA_V2:
            raise ValueError("data page v2 not implemented (store is v1)")
        if head[1] != _P_DATA:
            continue

        dph = head[5]
        n = dph[1]
        encoding = dph[2]
        p = 0

        # Definition levels. A flat OPTIONAL column has max level 1;
        # REQUIRED has 0 and writes nothing.
        if chunk_meta.get("_maxdef", 1):
            width = _bit_width(chunk_meta.get("_maxdef", 1))
            if dph[3] == _E_RLE:
                ln = struct.unpack_from("<I", page, p)[0]
                p += 4
                defs, _ = _read_rle_hybrid(page, p, width, n, p + ln)
                p += ln
            elif dph[3] == _E_BIT_PACKED:
                raise ValueError("BIT_PACKED levels not implemented")
            else:
                defs = [1] * n
        else:
            defs = [1] * n

        present = sum(1 for d in defs if d)
        if encoding in (_E_PLAIN_DICTIONARY, _E_RLE_DICTIONARY):
            if dictionary is None:
                raise ValueError("dictionary-encoded page with no "
                                 "dictionary page")
            width = page[p]
            p += 1
            idx, _ = _read_rle_hybrid(page, p, width, present, len(page))
            vals = [dictionary[i] for i in idx]
        elif encoding == _E_PLAIN:
            vals, p = _decode_plain(page, p, ptype, present,
                                    chunk_meta.get("_typelen"))
        else:
            raise ValueError(
                "encoding %d not implemented; the store uses PLAIN and "
                "RLE_DICTIONARY" % encoding)

        it = iter(vals)
        values.extend(next(it) if d else None for d in defs)

    if len(values) < num_rows:
        values.extend([None] * (num_rows - len(values)))
    return values[:num_rows]


def read_parquet(path, columns=None):
    """Read a Parquet file into a DataFrame.

    Parameters
    ----------
    path : str
        File to read.
    columns : list of str, optional
        Subset of column names; the rest are not decoded at all.
    """
    from ._frame_core import DataFrame

    with open(path, "rb") as fh:
        meta = _read_footer(fh)
        schema = meta[2]
        num_rows = meta[3]
        row_groups = meta.get(4) or []

        # schema[0] is the root; one element per leaf follows. Flat only.
        leaves = []
        for el in schema[1:]:
            if el.get(5):                                # num_children
                raise ValueError(
                    "nested schema not implemented: group field %r has "
                    "%d children" % (el[4].decode(), el[5]))
            rep = el.get(3, _REQUIRED)
            if rep == _REPEATED:
                raise ValueError(
                    "repeated column %r not implemented; decoding it as "
                    "flat would silently change the row count"
                    % el[4].decode())
            leaves.append({
                "name": el[4].decode("utf-8"),
                "type": el.get(1),
                "typelen": el.get(2),
                "converted": el.get(6),
                "maxdef": 1 if rep == _OPTIONAL else 0,
            })

        wanted = list(range(len(leaves)))
        if columns is not None:
            byname = {c["name"]: i for i, c in enumerate(leaves)}
            missing = [c for c in columns if c not in byname]
            if missing:
                raise KeyError("no such column(s) in %s: %s"
                               % (path, ", ".join(missing)))
            wanted = [byname[c] for c in columns]

        data = {}
        for i in wanted:
            leaf = leaves[i]
            col = []
            for rg in row_groups:
                chunk = rg[1][i]
                cm = dict(chunk[3])
                cm["_maxdef"] = leaf["maxdef"]
                cm["_typelen"] = leaf["typelen"]
                col.extend(_column_values(fh, cm, rg[3]))
            col = _convert(col, leaf["type"], leaf["converted"])
            data[leaf["name"]] = _apply_logical(col, leaf["type"],
                                                leaf["converted"])

    df = DataFrame(data)
    if len(df) != num_rows:
        raise ValueError("footer says %d rows, decoded %d"
                         % (num_rows, len(df)))
    return df


# ----------------------------------------------------------------- write

def _infer(values):
    """Pick a physical + converted type for a column of Python values.

    date and datetime must keep their logical type on the way out;
    without that a column read as TIMESTAMP_MICROS came back as a bare
    number and stopped being a timestamp to any other engine.
    """
    import datetime as _dt
    live = [v for v in values if v is not None]
    if live and all(isinstance(v, _dt.datetime) for v in live):
        return _INT64, _CT_TIMESTAMP_MILLIS
    if live and all(isinstance(v, _dt.date) for v in live):
        return _INT32, _CT_DATE
    seen = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            seen.add("bool")
        elif isinstance(v, int):
            seen.add("int")
        elif isinstance(v, float):
            seen.add("float")
        elif isinstance(v, (str, bytes)):
            seen.add("str")
        else:
            seen.add("str")
    if not seen or seen == {"bool"}:
        return _BOOLEAN, None
    if seen == {"int"}:
        lo = min((v for v in values if v is not None), default=0)
        hi = max((v for v in values if v is not None), default=0)
        if -(2 ** 31) <= lo and hi < 2 ** 31:
            return _INT32, None
        return _INT64, None
    if seen <= {"int", "float"}:
        return _DOUBLE, None
    return _BYTE_ARRAY, _CT_UTF8


def _prep_write(values, converted):
    """Turn date/datetime back into the integers PLAIN encodes."""
    import datetime as _dt
    if converted == _CT_DATE:
        base = _dt.date(1970, 1, 1)
        return [(v - base).days for v in values]
    if converted == _CT_TIMESTAMP_MILLIS:
        epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
        out = []
        for v in values:
            if v.tzinfo is None:
                v = v.replace(tzinfo=_dt.timezone.utc)
            out.append(round((v - epoch).total_seconds() * 1000))
        return out
    return values


def _encode_plain(values, ptype):
    out = bytearray()
    if ptype == _BOOLEAN:
        cur = 0
        for i, v in enumerate(values):
            if v:
                cur |= 1 << (i & 7)
            if (i & 7) == 7:
                out.append(cur)
                cur = 0
        if len(values) & 7:
            out.append(cur)
        return bytes(out)
    if ptype in _PLAIN_FMT:
        fmt = _PLAIN_FMT[ptype]
        caster = float if ptype in (_FLOAT, _DOUBLE) else int
        for v in values:
            out += struct.pack(fmt, caster(v))
        return bytes(out)
    if ptype == _BYTE_ARRAY:
        for v in values:
            b = v.encode("utf-8") if isinstance(v, str) else bytes(v)
            out += struct.pack("<I", len(b)) + b
        return bytes(out)
    raise ValueError("cannot PLAIN-encode physical type %d" % ptype)


def _encode_rle_levels(levels, width):
    """Levels as a single bit-packed hybrid run, length-prefixed."""
    if width == 0:
        return b""
    body = bytearray()
    groups = (len(levels) + 7) // 8
    padded = list(levels) + [0] * (groups * 8 - len(levels))
    body_bits = 0
    for i, lv in enumerate(padded):
        body_bits |= (lv & ((1 << width) - 1)) << (i * width)
    nbytes = groups * width
    header = bytearray()
    h = (groups << 1) | 1
    while True:
        if h < 0x80:
            header.append(h)
            break
        header.append((h & 0x7F) | 0x80)
        h >>= 7
    body += body_bits.to_bytes(nbytes, "little")
    payload = bytes(header) + bytes(body)
    return struct.pack("<I", len(payload)) + payload


def to_parquet(df, path, compression="snappy"):
    """Write a DataFrame to a single-row-group Parquet file.

    Columns are PLAIN-encoded and OPTIONAL (nullable) throughout, which
    is the shape every reader accepts without negotiation.
    """
    if compression not in ("snappy", None, "none", "uncompressed"):
        raise ValueError("compression must be 'snappy' or None; got %r"
                         % (compression,))
    codec = _C_SNAPPY if compression == "snappy" else _C_UNCOMPRESSED
    compress = _snappy_compress if codec == _C_SNAPPY else (lambda b: b)

    names = list(df.columns)
    cols = {n: list(df[n]) for n in names}
    nrows = len(df)

    fh = open(path, "wb")
    try:
        fh.write(b"PAR1")
        chunks = []
        for name in names:
            values = cols[name]
            ptype, converted = _infer(values)
            defs = [0 if v is None else 1 for v in values]
            present = _prep_write([v for v in values if v is not None],
                                  converted)

            body = _encode_rle_levels(defs, 1) + _encode_plain(present,
                                                               ptype)
            payload = compress(body)

            # PageHeader{1:type, 2:uncompressed, 3:compressed,
            #            5:DataPageHeader{1:num_values, 2:encoding,
            #                             3:def_enc, 4:rep_enc}}
            dph = _TWriter()
            last = dph.i32(1, nrows, 0)
            last = dph.i32(2, _E_PLAIN, last)
            last = dph.i32(3, _E_RLE, last)
            last = dph.i32(4, _E_RLE, last)
            dph_body = dph.stop()

            ph = _TWriter()
            last = ph.i32(1, _P_DATA, 0)
            last = ph.i32(2, len(body), last)
            last = ph.i32(3, len(payload), last)
            last = ph.struct(5, dph_body, last)
            ph_body = ph.stop()

            offset = fh.tell()
            fh.write(ph_body)
            fh.write(payload)
            total = fh.tell() - offset

            # ColumnMetaData{1:type, 2:encodings, 3:path_in_schema,
            #                4:codec, 5:num_values, 6:uncompressed,
            #                7:compressed, 9:data_page_offset}
            cmd = _TWriter()
            last = cmd.i32(1, ptype, 0)
            last = cmd.list_i32(2, [_E_RLE, _E_PLAIN], last)
            last = cmd.list_binary(3, [name], last)
            last = cmd.i32(4, codec, last)
            last = cmd.i64(5, nrows, last)
            last = cmd.i64(6, len(ph_body) + len(body), last)
            last = cmd.i64(7, total, last)
            last = cmd.i64(9, offset, last)
            cmd_body = cmd.stop()

            cc = _TWriter()
            last = cc.i64(2, offset, 0)
            last = cc.struct(3, cmd_body, last)
            chunks.append((cc.stop(), ptype, converted, name, total))

        # FileMetaData{1:version, 2:schema, 3:num_rows, 4:row_groups,
        #              6:created_by}
        root = _TWriter()
        last = root.bytes_(4, "schema", 0)
        last = root.i32(5, len(names), last)
        schema_structs = [root.stop()]
        for _, ptype, converted, name, _t in chunks:
            se = _TWriter()
            last = se.i32(1, ptype, 0)
            last = se.i32(3, _OPTIONAL, last)
            last = se.bytes_(4, name, last)
            if converted is not None:
                last = se.i32(6, converted, last)
            schema_structs.append(se.stop())

        rg = _TWriter()
        last = rg.list_struct(1, [c[0] for c in chunks], 0)
        last = rg.i64(2, sum(c[4] for c in chunks), last)
        last = rg.i64(3, nrows, last)
        rg_body = rg.stop()

        fm = _TWriter()
        last = fm.i32(1, 1, 0)
        last = fm.list_struct(2, schema_structs, last)
        last = fm.i64(3, nrows, last)
        last = fm.list_struct(4, [rg_body], last)
        last = fm.bytes_(6, "morie native parquet writer", last)
        footer = fm.stop()

        fh.write(footer)
        fh.write(struct.pack("<I", len(footer)))
        fh.write(b"PAR1")
    finally:
        fh.close()
    return path


def _demo():
    """Round-trip check: write, read back, compare cell by cell."""
    import os
    import tempfile
    from ._frame_core import DataFrame

    df = DataFrame({
        "case": ["23-OCI-001", "23-OCI-002", None, "23-OFP-9"],
        "year": [2023, 2023, 2024, 2024],
        "rate": [0.5, -1.25, None, 3.75],
        "closed": [True, False, True, None],
    })
    path = os.path.join(tempfile.mkdtemp(), "t.parquet")
    to_parquet(df, path)
    back = read_parquet(path)
    assert list(back.columns) == list(df.columns), back.columns
    for c in df.columns:
        a, b = list(df[c]), list(back[c])
        assert a == b, (c, a, b)
    sub = read_parquet(path, columns=["year"])
    assert list(sub.columns) == ["year"] and list(sub["year"]) == \
        [2023, 2023, 2024, 2024]
    with open(path, "rb") as fh:
        assert fh.read(4) == b"PAR1"
    print("round-trip ok:", len(df), "rows,", len(df.columns), "columns")


if __name__ == "__main__":
    _demo()
