import ast, pathlib, re, collections, json
FN = pathlib.Path("src/morie/fn")

CODE = re.compile(r"cat\(|print\(|<-|%>%|library\(|require\(|function\s*\(|\bdata\.frame\(|"
                  r"ggplot\(|str\(|head\(|paste0?\(|c\(\s*[\"']|\$<-|::")
LATEX_FAIL = re.compile(r"<LATEX>|\[EQ\]\s*$|\[EQ\]\s*\[EQ\]")
MATHY = re.compile(r"[=<>]\s*[^=]|\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|\\mu|\\sigma|\\theta|"
                   r"[∑∫√±≤≥≠αβγμσθλπ]|\^|_\{|\bP\(|\bE\[|\bVar\(|\bCov\(")

rows = []
for f in sorted(FN.glob("*.py")):
    if f.name.startswith("_"):
        continue
    try:
        src = f.read_text()
    except Exception:
        continue
    doc = ast.get_docstring(ast.parse(src)) or ""
    if "auto-extracted" not in doc:
        continue
    m = re.search(r"^\s*Formula:\s*(.+)$", src, re.M)
    formula = (m.group(1).strip() if m else "")
    ref = ""
    rm = re.search(r"References\n\s*-+\n\s*(.+)", src)
    if rm:
        ref = rm.group(1).strip()
    book = re.sub(r"_th?[0-9].*$", "", f.stem)

    if not formula or LATEX_FAIL.search(formula):
        cls = "artefact-latex-fail"
    elif CODE.search(formula):
        cls = "artefact-code-fragment"
    elif MATHY.search(formula):
        cls = "equation-candidate"
    else:
        cls = "unclear"
    rows.append({"mod": f.stem, "book": book, "cls": cls, "formula": formula[:110], "ref": ref[:90]})

print(f"auto-extracted modules examined: {len(rows)}\n")
c = collections.Counter(r["cls"] for r in rows)
for k, v in c.most_common():
    print(f"  {k:26s} {v:6d}  {100*v/len(rows):5.1f}%")
json.dump(rows, open("/tmp/census.json", "w"))
