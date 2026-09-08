# morie -- native ESRI shapefile reader (rootcoder007/morie)
"""Native reader for ESRI shapefiles (.shp/.dbf), no pyshp.

Format per the ESRI Shapefile Technical Description (ESRI White Paper,
July 1998): the .shp main file is a 100-byte header (big-endian file
code 9994, file length in 16-bit words; little-endian shape type and
bounding box) followed by records, each an 8-byte big-endian record
header (number, content length in words) and little-endian shape
content. The companion .dbf is a dBASE III table: 32-byte header
(record count, header size, record size), 32-byte field descriptors
terminated by 0x0D, then fixed-width text records each prefixed by a
deletion flag byte.

Covered shape types (the ones morie's TPS ingest reads): 0 Null,
1 Point, 3 PolyLine, 5 Polygon, 8 MultiPoint, and their *M/*Z
variants (measures/z ignored, coordinates kept).
"""

from __future__ import annotations

import struct
from pathlib import Path

_SHAPE_NAMES = {
    0: "NULL", 1: "POINT", 3: "POLYLINE", 5: "POLYGON",
    8: "MULTIPOINT", 11: "POINTZ", 13: "POLYLINEZ", 15: "POLYGONZ",
    18: "MULTIPOINTZ", 21: "POINTM", 23: "POLYLINEM", 25: "POLYGONM",
    28: "MULTIPOINTM",
}


class ShapeRecord:
    """One geometry + its attribute record."""

    def __init__(self, shape_type, points, parts, attributes):
        self.shape_type = shape_type
        self.shape_type_name = _SHAPE_NAMES.get(shape_type,
                                                str(shape_type))
        self.points = points          # [(x, y), ...]
        self.parts = parts            # part start indices into points
        self.record = attributes      # dict of dbf fields

    @property
    def bbox(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))


def _read_shp(path):
    data = Path(path).read_bytes()
    (code,) = struct.unpack(">i", data[0:4])
    if code != 9994:
        raise ValueError("%s is not a shapefile (file code %d != 9994)"
                         % (path, code))
    shapes = []
    off = 100
    n = len(data)
    while off + 8 <= n:
        (_recno, content_words) = struct.unpack(">ii", data[off:off + 8])
        off += 8
        end = off + content_words * 2
        (stype,) = struct.unpack("<i", data[off:off + 4])
        base = stype % 10 if stype else 0
        pts, parts = [], [0]
        if base == 1 and stype != 0:                       # Point*
            x, y = struct.unpack("<dd", data[off + 4:off + 20])
            pts = [(x, y)]
            parts = [0]
        elif base in (3, 5):                               # Poly*
            nparts, npts = struct.unpack("<ii",
                                         data[off + 36:off + 44])
            p0 = off + 44
            parts = list(struct.unpack("<%di" % nparts,
                                       data[p0:p0 + 4 * nparts]))
            q0 = p0 + 4 * nparts
            flat = struct.unpack("<%dd" % (2 * npts),
                                 data[q0:q0 + 16 * npts])
            pts = [(flat[2 * i], flat[2 * i + 1])
                   for i in range(npts)]
        elif base == 8:                                    # MultiPoint*
            (npts,) = struct.unpack("<i", data[off + 36:off + 40])
            q0 = off + 40
            flat = struct.unpack("<%dd" % (2 * npts),
                                 data[q0:q0 + 16 * npts])
            pts = [(flat[2 * i], flat[2 * i + 1])
                   for i in range(npts)]
            parts = [0]
        shapes.append((stype, pts, parts))
        off = end
    return shapes


def _read_dbf(path):
    p = Path(path)
    if not p.exists():
        return None, []
    data = p.read_bytes()
    nrec, hdr_size, rec_size = struct.unpack("<IHH", data[4:12])
    fields = []
    off = 32
    while off < hdr_size - 1 and data[off] != 0x0D:
        raw = data[off:off + 32]
        name = raw[0:11].split(b"\x00")[0].decode("ascii", "replace")
        ftype = chr(raw[11])
        flen = raw[16]
        fdec = raw[17]
        fields.append((name, ftype, flen, fdec))
        off += 32
    records = []
    off = hdr_size
    for _ in range(nrec):
        if off + rec_size > len(data):
            break
        row = data[off:off + rec_size]
        off += rec_size
        if row[0:1] == b"*":            # deleted record
            continue
        rec = {}
        pos = 1
        for name, ftype, flen, fdec in fields:
            cell = row[pos:pos + flen]
            pos += flen
            txt = cell.decode("latin-1").strip()
            if ftype in ("N", "F") and txt:
                try:
                    rec[name] = float(txt) if ("." in txt or fdec) \
                        else int(txt)
                except ValueError:
                    rec[name] = txt
            elif ftype == "L":
                rec[name] = txt.upper() in ("T", "Y")
            else:
                rec[name] = txt
        records.append(rec)
    return fields, records


class Reader:
    """pyshp-compatible surface for the read paths morie uses.

    ``Reader(path).shapeRecords()`` -> list of :class:`ShapeRecord`
    with ``.points``, ``.parts``, ``.record`` (attribute dict);
    ``.fields``; ``.shapes()``; iteration.
    """

    def __init__(self, path):
        p = str(path)
        if p.lower().endswith((".shp", ".dbf", ".shx")):
            p = p[:-4]
        self._base = p
        self._shapes = _read_shp(p + ".shp")
        self.fields, self._records = _read_dbf(p + ".dbf")

    def __len__(self):
        return len(self._shapes)

    def shapeRecords(self):
        out = []
        for i, (stype, pts, parts) in enumerate(self._shapes):
            attrs = self._records[i] if i < len(self._records) else {}
            out.append(ShapeRecord(stype, pts, parts, attrs))
        return out

    def shapes(self):
        return [ShapeRecord(st, pts, parts, {})
                for st, pts, parts in self._shapes]

    def __iter__(self):
        return iter(self.shapeRecords())
