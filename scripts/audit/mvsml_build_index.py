"""Build the MVSML shelf index: one row per distinct METHOD (chapter +
equation), listing the canonical implementation and every file that
re-exports it, plus the artefact stubs that carry no equation."""
import json, re, glob, collections
from pathlib import Path

inv = json.load(open("/tmp/msm_inv.json"))       # pre-shelf snapshot
census = json.load(open("/tmp/eq_census.json"))

def content_class(f):
    t = f.lower()
    if re.search(r"\brm\(list=|library\(|bglr\(|mmer\(|cv\.glmnet|"
                 r"model\.matrix|read\.csv|load\(|data\.frame|<-", f):
        return "R-code artefact"
    if "<latex>" in t:
        return "LaTeX-fail artefact"
    if not f.strip():
        return "empty artefact"
    if re.search(r"[=<>\u2264\u2265]|\\frac|\\sum|prod|exp\(|log\(", f):
        return "equation"
    return "prose"

canon = {}
shims = collections.defaultdict(list)
for p in sorted(glob.glob("src/morie/fn/msm*.py")):
    mod = Path(p).stem
    s = Path(p).read_text()
    if "result = float(np.mean(" in s:
        continue
    m = re.search(r"re-exported from :mod:`morie\.fn\.(\w+)`", s)
    if m:
        shims[m.group(1)].append(mod)
    else:
        fn = re.search(r'__all__ = \["([a-z_0-9]+)"', s)
        cite = re.search(r"Implements (.+?) of Montesinos", s, re.S)
        canon[mod] = (fn.group(1) if fn else "",
                      " ".join(cite.group(1).split()) if cite else "")

byfile = {r["file"]: r for r in inv}
rows = []
for mod, (fn, cite) in sorted(canon.items()):
    eq = re.search(r"eq_(\d+)_(\d+[a-z]?)", fn)
    rows.append({
        "method": cite or fn,
        "chapter": eq.group(1) if eq else "?",
        "equation": eq.group(2) if eq else "?",
        "canonical_module": mod,
        "entry_point": fn,
        "reexported_by": sorted(shims.get(mod, [])),
        "stub_content": content_class(byfile.get(mod, {})
                                      .get("formula", "")),
    })

out = ["# MVSML shelf index (one row per distinct method)", "",
       "Book: Montesinos Lopez, Montesinos Lopez & Crossa (2022),",
       "*Multivariate Statistical Machine Learning Methods for Genomic",
       "Prediction*, Springer, DOI 10.1007/978-3-030-89010-0.", "",
       "The auto-generated stubs are NOT one-per-method: the extractor",
       "stamped many different page fragments with the same function",
       "name. This index is keyed by the book equation actually",
       "implemented. `reexported by` lists the stub files that resolve",
       "to the same implementation, so nothing is written twice.", "",
       "| ch | eq | method | canonical module | entry point | re-exported by | stub content |",
       "|----|----|--------|------------------|-------------|----------------|--------------|"]
for r in sorted(rows, key=lambda r: (r["chapter"].zfill(2),
                                     r["equation"].zfill(3))):
    out.append("| %s | %s | %s | %s | `%s` | %s | %s |" % (
        r["chapter"], r["equation"], r["method"],
        r["canonical_module"], r["entry_point"],
        (", ".join(r["reexported_by"]) or "-"), r["stub_content"]))

out += ["", "## Equation census taken from the PDFs",
        "", "Extracted with morie's own `_pdf_reader`; the math font",
        "emits one glyph per line and renders the decimal point as a",
        "colon, so tags appear as `(\\n7\\n:\\n5\\n)`. Counts are a",
        "LOWER bound (some tags do not survive extraction).", "",
        "| chapter | numbered displays found | implemented |",
        "|---------|-------------------------|-------------|"]
impl_by_ch = collections.defaultdict(set)
for r in rows:
    impl_by_ch[r["chapter"]].add(r["equation"])
def k(s):
    return (int(re.match(r"\d+", s).group()), s)
for ch in sorted(census, key=int):
    out.append("| %s | %s | %s |" % (
        ch, " ".join(sorted(census[ch], key=k)),
        " ".join(sorted(impl_by_ch.get(ch, []), key=k)) or "-"))

Path("scripts/audit/MVSML_INDEX.md").write_text("\n".join(out) + "\n")
json.dump(rows, open("scripts/audit/mvsml_index.json", "w"), indent=1)
print("distinct methods indexed:", len(rows))
print("stub files re-exporting them:",
      sum(len(r["reexported_by"]) for r in rows))
cc = collections.Counter(r["stub_content"] for r in rows)
print("canonical modules by original stub content:", dict(cc))
