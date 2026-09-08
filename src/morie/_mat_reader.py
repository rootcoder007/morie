# morie -- native MATLAB .mat (v5) reader (rootcoder007/morie)
"""Read MATLAB Level 5 MAT-files without pymatreader/scipy.

Format per MathWorks, "MAT-File Format" (R2023b, chapter on the
Level 5 MAT-file): a 128-byte text+version header, then a sequence of
data elements, each an 8-byte tag (miTYPE uint32, byte count uint32,
with the "small data element" packing when the count fits in 4 bytes)
followed by payload padded to 8 bytes. ``miCOMPRESSED`` elements wrap
a zlib stream of further elements. ``miMATRIX`` payloads hold array
flags, dimensions, name, and value subelements.

Covered: numeric arrays of every mi/mx numeric class (returned as
nested Python float lists, column-major decoded to row-major),
logical arrays, char arrays (returned as strings), cell arrays and
structs (returned as dicts / lists). HDF5-based v7.3 files raise a
clear error naming the format.
"""

from __future__ import annotations

import struct
import zlib

_MI = {1: ("b", 1), 2: ("B", 1), 3: ("h", 2), 4: ("H", 2),
       5: ("i", 4), 6: ("I", 4), 7: ("f", 4), 9: ("d", 8),
       12: ("q", 8), 13: ("Q", 8)}


def _elements(buf):
    """Yield (mi_type, payload) data elements from a buffer."""
    i = 0
    n = len(buf)
    while i + 8 <= n:
        mtype, count = struct.unpack_from("<II", buf, i)
        if mtype >> 16:                     # small data element
            count = mtype >> 16
            mtype &= 0xFFFF
            payload = buf[i + 4:i + 4 + count]
            i += 8
        else:
            payload = buf[i + 8:i + 8 + count]
            i += 8 + count
            if mtype != 15:
                # 8-byte alignment -- except miCOMPRESSED, which the
                # MAT-File Format doc exempts from padding: the next
                # element begins immediately after the zlib stream.
                i += (-i) % 8
        yield mtype, payload


def _numeric(mtype, payload):
    fmt, w = _MI[mtype]
    k = len(payload) // w
    return list(struct.unpack("<%d%s" % (k, fmt), payload[:k * w]))


def _reshape_colmajor(flat, dims):
    if len(dims) <= 1:
        return list(flat)
    if len(dims) == 2:
        r, c = dims
        return [[flat[j * r + i] for j in range(c)] for i in range(r)]
    # higher ranks: nested by last dimension
    step = 1
    for d in dims[:-1]:
        step *= d
    return [_reshape_colmajor(flat[k * step:(k + 1) * step],
                              dims[:-1]) for k in range(dims[-1])]


def _matrix(payload):
    subs = list(_elements(payload))
    flags = _numeric(subs[0][0], subs[0][1])
    mclass = int(flags[0]) & 0xFF
    dims = [int(v) for v in _numeric(subs[1][0], subs[1][1])]
    name = subs[2][1].decode("latin-1").rstrip("\x00")
    if mclass == 4:                                       # mxCHAR
        raw = subs[3][1] if len(subs) > 3 else b""
        if subs[3][0] == 4:                               # miUINT16
            txt = raw.decode("utf-16-le", "replace")
        else:
            txt = raw.decode("latin-1", "replace")
        return name, txt.rstrip("\x00")
    if mclass == 1:                                       # mxCELL
        cells = []
        for mt, pl in subs[3:]:
            if mt == 14:
                cells.append(_matrix(pl)[1])
        return name, cells
    if mclass == 2:                                       # mxSTRUCT
        (fl,) = struct.unpack("<i", subs[3][1][:4])
        raw_names = subs[4][1]
        fields = [raw_names[i:i + fl].decode("latin-1").rstrip("\x00")
                  for i in range(0, len(raw_names), fl)]
        vals = [pl for mt, pl in subs[5:] if mt == 14]
        out = {}
        for fname, pl in zip(fields, vals):
            out[fname] = _matrix(pl)[1]
        return name, out
    # numeric / logical classes: real part is the 4th subelement
    if len(subs) > 3:
        real = _numeric(subs[3][0], subs[3][1])
    else:
        real = []
    return name, _reshape_colmajor([float(v) for v in real], dims)


def read_mat(path):
    """Load a v5 .mat file as {variable name: value}."""
    data = open(path, "rb").read()
    if data[:8] == b"\x89HDF\r\n\x1a\n" or data[:4] == b"\x89HDF":
        raise ValueError(
            "%s is a MATLAB v7.3 (HDF5) file; save it with "
            "-v7 in MATLAB, or convert it, to load natively" % path)
    if len(data) < 128:
        raise ValueError("%s is too short to be a MAT-file" % path)
    (version, endian) = struct.unpack_from("<HH", data, 124)
    del version
    if endian not in (0x4D49, 0x494D):
        raise ValueError("%s has no MAT v5 endian marker" % path)
    out = {}
    for mtype, payload in _elements(data[128:]):
        if mtype == 15:                                  # miCOMPRESSED
            inner = zlib.decompress(payload)
            for mt2, pl2 in _elements(inner):
                if mt2 == 14:
                    k, v = _matrix(pl2)
                    out[k] = v
        elif mtype == 14:                                # miMATRIX
            k, v = _matrix(payload)
            out[k] = v
    return out
