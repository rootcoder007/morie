# morie -- native PDF text extractor (rootcoder007/morie)
"""Native PDF text extraction, no pypdf.

Reads the subset of ISO 32000-1 (PDF 1.7) that digital-text documents
use:

* objects located by scanning ``N G obj`` (robust to damaged xref
  tables) plus expansion of object streams (``/Type /ObjStm``,
  sec. 7.5.7), so both classic and compressed-xref files load;
* ``/FlateDecode`` streams inflated with stdlib zlib (sec. 7.4.4);
* the page tree walked from the catalog (sec. 7.7.3), falling back to
  every ``/Type /Page`` object when the tree is damaged;
* text shown by the ``Tj``, ``'``, ``"`` and ``TJ`` operators inside
  ``BT``/``ET`` blocks (sec. 9.4.3), with ``Td``/``TD``/``T*``
  treated as line breaks and large negative ``TJ`` kerns as spaces;
* literal strings with the sec. 7.3.4.2 escapes, hex strings, and
  UTF-16BE (BOM fe ff) / PDFDocEncoding decoding.

Scanned (image-only) pages yield no text -- the same behaviour pypdf
has without OCR. The public surface mirrors what morie used:
``PdfReader(path).pages`` and ``page.extract_text()``.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

_WS = b"\x00\t\n\x0c\r "
_DELIM = b"()<>[]{}/%"


class _Ref:
    __slots__ = ("num",)

    def __init__(self, num):
        self.num = num


class _Lexer:
    def __init__(self, data, pos=0):
        self.d = data
        self.i = pos

    def _skip(self):
        d, i = self.d, self.i
        n = len(d)
        while i < n:
            c = d[i:i + 1]
            if c in _WS:
                i += 1
            elif c == b"%":
                j = d.find(b"\n", i)
                i = n if j == -1 else j + 1
            else:
                break
        self.i = i

    def value(self):
        """Parse one PDF object value at the cursor."""
        self._skip()
        d, i = self.d, self.i
        c = d[i:i + 1]
        if c == b"<":
            if d[i + 1:i + 2] == b"<":
                return self._dict()
            return self._hexstring()
        if c == b"(":
            return self._litstring()
        if c == b"/":
            return self._name()
        if c == b"[":
            self.i += 1
            arr = []
            while True:
                self._skip()
                if self.d[self.i:self.i + 1] == b"]":
                    self.i += 1
                    return arr
                arr.append(self.value())
        if c in b"+-.0123456789":
            return self._number()
        m = re.match(rb"(true|false|null)", d[i:i + 5])
        if m:
            self.i += len(m.group(1))
            return {b"true": True, b"false": False,
                    b"null": None}[m.group(1)]
        # bare keyword (operator etc.) -- consume a token
        j = i
        n = len(d)
        while j < n and d[j:j + 1] not in _WS \
                and d[j:j + 1] not in _DELIM:
            j += 1
        tok = d[i:j]
        self.i = j if j > i else i + 1
        return tok

    def _number(self):
        m = re.match(rb"[+-]?\d*\.?\d+", self.d[self.i:self.i + 32])
        txt = m.group(0)
        # reference?  "N G R"
        save = self.i
        self.i += len(txt)
        if b"." not in txt:
            self._skip()
            m2 = re.match(rb"(\d+)\s+R(?![A-Za-z])",
                          self.d[self.i:self.i + 24])
            if m2:
                self.i += m2.end()
                return _Ref(int(txt))
            self.i = save + len(txt)
        return float(txt) if b"." in txt else int(txt)

    def _name(self):
        j = self.i + 1
        d = self.d
        n = len(d)
        while j < n and d[j:j + 1] not in _WS \
                and d[j:j + 1] not in _DELIM:
            j += 1
        raw = d[self.i + 1:j]
        self.i = j
        # #xx escapes in names (sec. 7.3.5)
        raw = re.sub(rb"#([0-9A-Fa-f]{2})",
                     lambda m: bytes([int(m.group(1), 16)]), raw)
        return "/" + raw.decode("latin-1")

    def _dict(self):
        self.i += 2
        out = {}
        while True:
            self._skip()
            if self.d[self.i:self.i + 2] == b">>":
                self.i += 2
                return out
            key = self.value()
            out[key] = self.value()

    def _litstring(self):
        d = self.d
        i = self.i + 1
        depth = 1
        out = bytearray()
        n = len(d)
        while i < n:
            c = d[i:i + 1]
            if c == b"\\":
                nxt = d[i + 1:i + 2]
                esc = {b"n": b"\n", b"r": b"\r", b"t": b"\t",
                       b"b": b"\b", b"f": b"\x0c", b"(": b"(",
                       b")": b")", b"\\": b"\\"}
                if nxt in esc:
                    out += esc[nxt]
                    i += 2
                elif nxt.isdigit():
                    m = re.match(rb"[0-7]{1,3}", d[i + 1:i + 4])
                    out.append(int(m.group(0), 8) & 0xFF)
                    i += 1 + len(m.group(0))
                elif nxt in (b"\n", b"\r"):
                    i += 2                     # line continuation
                else:
                    i += 1
            elif c == b"(":
                depth += 1
                out += c
                i += 1
            elif c == b")":
                depth -= 1
                if depth == 0:
                    self.i = i + 1
                    return bytes(out)
                out += c
                i += 1
            else:
                out += c
                i += 1
        self.i = i
        return bytes(out)

    def _hexstring(self):
        j = self.d.find(b">", self.i)
        hx = re.sub(rb"[^0-9A-Fa-f]", b"", self.d[self.i + 1:j])
        if len(hx) % 2:
            hx += b"0"
        self.i = j + 1
        return bytes.fromhex(hx.decode("ascii"))


def _decode_pdf_text(raw):
    if raw[:2] == b"\xfe\xff":
        return raw[2:].decode("utf-16-be", "replace")
    return raw.decode("latin-1", "replace")


class _Page:
    def __init__(self, doc, obj):
        self._doc = doc
        self._obj = obj

    def extract_text(self):
        contents = self._doc._resolve(self._obj.get("/Contents"))
        chunks = []
        if isinstance(contents, list):
            for c in contents:
                c = self._doc._resolve(c)
                if isinstance(c, tuple):
                    chunks.append(self._doc._stream_data(c))
        elif isinstance(contents, tuple):
            chunks.append(self._doc._stream_data(contents))
        return _extract_from_content(b"\n".join(chunks))


def _extract_from_content(data):
    lex = _Lexer(data)
    out = []
    stack = []
    n = len(data)
    while lex.i < n:
        try:
            tok = lex.value()
        except Exception:
            break
        if isinstance(tok, bytes) and tok in (
                b"Tj", b"'", b'"', b"TJ", b"Td", b"TD", b"T*",
                b"Tm", b"BT", b"ET"):
            if tok == b"Tj" or tok == b"'":
                if stack and isinstance(stack[-1], bytes):
                    out.append(_decode_pdf_text(stack[-1]))
                if tok == b"'":
                    out.append("\n")
            elif tok == b'"':
                if stack and isinstance(stack[-1], bytes):
                    out.append(_decode_pdf_text(stack[-1]))
                out.append("\n")
            elif tok == b"TJ":
                if stack and isinstance(stack[-1], list):
                    for el in stack[-1]:
                        if isinstance(el, bytes):
                            out.append(_decode_pdf_text(el))
                        elif isinstance(el, (int, float)) \
                                and el < -180:
                            out.append(" ")     # big kern = space
            elif tok in (b"Td", b"TD", b"T*"):
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
            elif tok == b"Tm":
                if out and not out[-1].endswith("\n"):
                    out.append("\n")
            stack = []
        else:
            stack.append(tok)
            if len(stack) > 16:
                stack = stack[-16:]
    return "".join(out)


class PdfReader:
    """Native reader with the pypdf read surface morie uses."""

    def __init__(self, path):
        data = Path(path).read_bytes() if not hasattr(path, "read") \
            else path.read()
        self._objs = {}
        for m in re.finditer(rb"(\d+)\s+\d+\s+obj\b", data):
            num = int(m.group(1))
            self._objs[num] = m.end()
        self._data = data
        self._parsed = {}
        self._expand_object_streams()
        self.pages = self._collect_pages()

    # ------------------------------------------------------ objects
    def _object(self, num):
        if num in self._parsed:
            return self._parsed[num]
        pos = self._objs.get(num)
        if pos is None:
            return None
        lex = _Lexer(self._data, pos)
        val = lex.value()
        if isinstance(val, dict):
            lex._skip()
            if self._data[lex.i:lex.i + 6] == b"stream":
                j = lex.i + 6
                if self._data[j:j + 2] == b"\r\n":
                    j += 2
                elif self._data[j:j + 1] == b"\n":
                    j += 1
                ln = self._resolve(val.get("/Length"))
                if isinstance(ln, (int, float)):
                    end = j + int(ln)
                else:
                    end = self._data.find(b"endstream", j)
                val = (val, self._data[j:end])
        self._parsed[num] = val
        return val

    def _resolve(self, v):
        seen = 0
        while isinstance(v, _Ref) and seen < 32:
            v = self._object(v.num)
            seen += 1
        return v

    def _stream_data(self, tup):
        d, raw = tup
        filt = self._resolve(d.get("/Filter"))
        filters = filt if isinstance(filt, list) else \
            ([filt] if filt else [])
        for f in filters:
            if f == "/FlateDecode":
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    raw = zlib.decompressobj().decompress(raw)
            # other filters (DCT, CCITT) carry images, not text
        return raw

    def _expand_object_streams(self):
        for num in list(self._objs):
            obj = self._object(num)
            if isinstance(obj, tuple) and \
                    obj[0].get("/Type") == "/ObjStm":
                try:
                    body = self._stream_data(obj)
                    n = int(self._resolve(obj[0].get("/N")))
                    first = int(self._resolve(obj[0].get("/First")))
                    head = _Lexer(body[:first])
                    pairs = []
                    for _ in range(n):
                        onum = head.value()
                        ooff = head.value()
                        pairs.append((int(onum), int(ooff)))
                    for onum, ooff in pairs:
                        lx = _Lexer(body, first + ooff)
                        self._parsed[onum] = lx.value()
                        self._objs.setdefault(onum, -1)
                except Exception:
                    continue

    # -------------------------------------------------------- pages
    def _collect_pages(self):
        pages = []
        root = None
        for num in list(self._objs):
            obj = self._object(num)
            d = obj[0] if isinstance(obj, tuple) else obj
            if isinstance(d, dict) and d.get("/Type") == "/Catalog":
                root = d
                break
        if root is not None:
            tree = self._resolve(root.get("/Pages"))
            if isinstance(tree, dict):
                self._walk(tree, pages, 0)
        if not pages:            # damaged tree: brute collect
            for num in sorted(self._objs):
                obj = self._object(num)
                d = obj[0] if isinstance(obj, tuple) else obj
                if isinstance(d, dict) and d.get("/Type") == "/Page":
                    pages.append(_Page(self, d))
        return pages

    def _walk(self, node, pages, depth):
        if depth > 64:
            return
        if node.get("/Type") == "/Page":
            pages.append(_Page(self, node))
            return
        for kid in (self._resolve(node.get("/Kids")) or []):
            kid = self._resolve(kid)
            if isinstance(kid, tuple):
                kid = kid[0]
            if isinstance(kid, dict):
                self._walk(kid, pages, depth + 1)
