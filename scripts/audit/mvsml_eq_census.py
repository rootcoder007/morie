"""Census numbered display equations per chapter.

The PDF's math font emits one glyph per line and maps the decimal
point to a colon, so an equation tag appears in the extracted text as
"(\\n7\\n:\\n5\\n)".  Collapse whitespace first, then accept either
separator."""
import sys, re, glob, json, collections
sys.path.insert(0, "src")
from morie._pdf_reader import PdfReader

CH = {"1": 1, "35": 2, "71": 3, "109": 4, "141": 5, "171": 6,
      "209": 7, "251": 8}

found = collections.defaultdict(dict)
for p in sorted(glob.glob("/home/rootcoder/work/ledger/mvsml_pdf/*.pdf")):
    m = re.search(r"Pages (\d+)-(\d+)", p)
    ch = CH.get(m.group(1))
    if ch is None:
        continue
    r = PdfReader(p)
    for i, page in enumerate(r.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            continue
        flat = re.sub(r"\s+", "", t)
        book_page = int(m.group(1)) + i - 1
        for mm in re.finditer(r"\(%d[.:](\d{1,2}[a-z]?)\)" % ch, flat):
            found[ch].setdefault(mm.group(1), book_page)
json.dump({str(k): v for k, v in found.items()},
          open("/tmp/eq_census.json", "w"))
def key(s):
    return (int(re.match(r"\d+", s).group()), s)
for ch in sorted(found):
    ids = sorted(found[ch], key=key)
    print("ch%-2d %2d displays: %s" % (ch, len(ids), " ".join(ids)))
