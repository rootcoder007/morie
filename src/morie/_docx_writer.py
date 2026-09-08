# morie -- native .docx writer (rootcoder007/morie)
"""Native Word-document writer, no python-docx.

A .docx file is an OOXML package (ECMA-376 Part 2: Open Packaging
Conventions): a zip holding ``[Content_Types].xml``, ``_rels/.rels``
and ``word/document.xml`` (WordprocessingML, ECMA-376 Part 1 sec. 17).
This module writes that package directly with ``zipfile`` and covers
the python-docx surface morie's export path uses: ``Document()``,
``add_heading``, ``add_paragraph`` (with bold/italic runs),
``add_table`` (with cell text), ``add_page_break``, ``save``.
``Inches``/``Pt`` are unit helpers kept for signature compatibility
(EMU: 914400 per inch; half-points for font sizes).
"""

from __future__ import annotations

import zipfile
from xml.sax.saxutils import escape

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def Inches(v):
    """Inches -> EMU (ECMA-376: 914400 EMU per inch)."""
    return int(round(float(v) * 914400))


def Pt(v):
    """Points -> half-points (WordprocessingML w:sz unit)."""
    return float(v)


class _Run:
    def __init__(self, text, bold=False, italic=False, size=None):
        self.text = text
        self.bold = bold
        self.italic = italic
        self.size = size

    def xml(self):
        props = ""
        if self.bold:
            props += "<w:b/>"
        if self.italic:
            props += "<w:i/>"
        if self.size is not None:
            hp = int(round(2 * float(self.size)))
            props += '<w:sz w:val="%d"/>' % hp
        rpr = "<w:rPr>%s</w:rPr>" % props if props else ""
        return ('<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>'
                % (rpr, escape(self.text)))


class _Paragraph:
    def __init__(self, style=None):
        self.runs = []
        self.style = style

    def add_run(self, text, bold=False, italic=False):
        r = _Run(str(text), bold=bold, italic=italic)
        self.runs.append(r)
        return r

    @property
    def text(self):
        return "".join(r.text for r in self.runs)

    def xml(self):
        ppr = ('<w:pPr><w:pStyle w:val="%s"/></w:pPr>' % self.style
               if self.style else "")
        return "<w:p>%s%s</w:p>" % (ppr,
                                    "".join(r.xml() for r in self.runs))


class _Cell:
    def __init__(self):
        self.paragraphs = [_Paragraph()]

    @property
    def text(self):
        return self.paragraphs[0].text

    @text.setter
    def text(self, v):
        self.paragraphs = [_Paragraph()]
        self.paragraphs[0].add_run(str(v))

    def xml(self):
        body = "".join(p.xml() for p in self.paragraphs) or "<w:p/>"
        return ("<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/>"
                "</w:tcPr>%s</w:tc>" % body)


class _Row:
    def __init__(self, ncols):
        self.cells = [_Cell() for _ in range(ncols)]

    def xml(self):
        return "<w:tr>%s</w:tr>" % "".join(c.xml() for c in self.cells)


class _Table:
    def __init__(self, rows, cols, style=None):
        self.style = style
        self.rows = [_Row(cols) for _ in range(rows)]
        self._ncols = cols

    def cell(self, r, c):
        return self.rows[r].cells[c]

    def add_row(self):
        row = _Row(self._ncols)
        self.rows.append(row)
        return row

    def xml(self):
        borders = ("<w:tblBorders>" + "".join(
            '<w:%s w:val="single" w:sz="4" w:color="auto"/>' % side
            for side in ("top", "left", "bottom", "right",
                         "insideH", "insideV")) + "</w:tblBorders>")
        return ('<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>%s'
                "</w:tblPr>%s</w:tbl>"
                % (borders, "".join(r.xml() for r in self.rows)))


class _PageBreak:
    @staticmethod
    def xml():
        return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="%s">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
    <w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>
    <w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>
</w:styles>""" % _W


class Document:
    """Minimal python-docx-compatible document writer."""

    def __init__(self, path=None):
        del path  # template loading not supported; always fresh
        self._body = []
        self.paragraphs = []
        self.tables = []

    def add_heading(self, text="", level=1):
        style = "Title" if level == 0 else "Heading%d" % min(level, 3)
        p = _Paragraph(style=style)
        p.add_run(str(text), bold=False)
        self._body.append(p)
        self.paragraphs.append(p)
        return p

    def add_paragraph(self, text="", style=None):
        p = _Paragraph(style=style)
        if text:
            p.add_run(str(text))
        self._body.append(p)
        self.paragraphs.append(p)
        return p

    def add_table(self, rows, cols, style=None):
        t = _Table(int(rows), int(cols), style=style)
        self._body.append(t)
        self.tables.append(t)
        return t

    def add_page_break(self):
        self._body.append(_PageBreak())

    def save(self, path):
        doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<w:document xmlns:w="%s"><w:body>%s'
               "<w:sectPr/></w:body></w:document>"
               % (_W, "".join(el.xml() for el in self._body)))
        ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<Types xmlns="http://schemas.openxmlformats.org/'
              'package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/'
              'vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="xml" ContentType="application/xml"/>'
              '<Override PartName="/word/document.xml" ContentType='
              '"application/vnd.openxmlformats-officedocument.'
              'wordprocessingml.document.main+xml"/>'
              '<Override PartName="/word/styles.xml" ContentType='
              '"application/vnd.openxmlformats-officedocument.'
              'wordprocessingml.styles+xml"/></Types>')
        rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.'
                'org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.'
                'openxmlformats.org/officeDocument/2006/relationships/'
                'officeDocument" Target="word/document.xml"/>'
                "</Relationships>")
        drels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.'
                 'org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.'
                 'openxmlformats.org/officeDocument/2006/relationships/'
                 'styles" Target="styles.xml"/></Relationships>')
        with zipfile.ZipFile(str(path), "w",
                             zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", ct)
            z.writestr("_rels/.rels", rels)
            z.writestr("word/_rels/document.xml.rels", drels)
            z.writestr("word/styles.xml", _STYLES)
            z.writestr("word/document.xml", doc)
