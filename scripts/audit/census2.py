import json, re, collections
rows = json.load(open("/tmp/census.json"))

CODE = re.compile(r"cat\(|print\(|<-|%>%|library\(|require\(|\bdata\.frame\(|ggplot\(|"
                  r"paste0?\(|lm\s*\(|list\(|control\.|\.results|~\s*\w+\s*[,+]|::|TRUE|FALSE")
LATEXFAIL = re.compile(r"<LATEX>")
# Prose: reads as a sentence rather than a statement of an equation.
PROSE = re.compile(r"\b(so,? for example|for example|where(,| the| ˜)|respectively|"
                   r"^(1|2|3)\.\s|let\s+\w|is the|are as follows|not (declared|equivalent)|"
                   r"we (have|obtain|get)|note that|denote|Test distribution|deg\b|"
                   r"generation of|the value)\b", re.I)
# Truncated: ends mid-expression.
TRUNC = re.compile(r"(=|\+|-|·|\*|/|,|⇒|<|>|\||′|\(|\[)\s*$")
MATH = re.compile(r"[=<>≤≥≠⇒]|\\frac|\\sum|[∑∫√±αβγδλμνπρσθφω]|\^")

def unbalanced(s):
    return (s.count("(") != s.count(")")) or (s.count("[") != s.count("]"))

for r in rows:
    f = r["formula"].replace("[EQ]", "").strip()
    if not f:
        r["cls2"] = "empty"
    elif LATEXFAIL.search(f):
        r["cls2"] = "artefact-latex-fail"
    elif CODE.search(f):
        r["cls2"] = "artefact-code"
    elif PROSE.search(f):
        r["cls2"] = "artefact-prose"
    elif TRUNC.search(f) or unbalanced(f):
        r["cls2"] = "truncated"
    elif MATH.search(f):
        r["cls2"] = "equation-usable"
    else:
        r["cls2"] = "no-math"

n = len(rows)
c = collections.Counter(r["cls2"] for r in rows)
print(f"auto-extracted modules: {n}\n")
usable = c["equation-usable"]
for k, v in c.most_common():
    print(f"  {k:22s} {v:6d}  {100*v/n:5.1f}%")
print(f"\n  ==> usable equations       {usable:6d}  ({100*usable/n:.1f}%)")
print(f"  ==> not implementable      {n-usable:6d}  ({100*(n-usable)/n:.1f}%)")
json.dump(rows, open("/tmp/census2.json", "w"))
