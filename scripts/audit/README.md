# Auto-extracted module audit

Tooling for the 2026-07 audit of `src/morie/fn`. Run from the repo root
with `PYTHONPATH=src`.

| script | what it does |
|---|---|
| `triage.py` | clusters placeholder modules by statistical domain |
| `census.py` | separates auto-extracted book-equation modules from named methods and dumps `/tmp/census.json` |
| `census2.py` | classifies the auto-extracted set into usable equations vs extraction artefacts |

## What the audit found

`fn` holds two distinct populations. **Named methods** carry a real
method name and a citable source. **Auto-extracted modules** were minted
one per equation scraped from the reference library; there are 22,374 of
them, named after their source book.

The auto-extracted set is damaged upstream of any implementation. A
hand-classified random sample of 40 (seed 2027) came out as:

| what the `Formula:` field actually contains | n/40 | share |
|---|---|---|
| truncated or garbled mathematics | 16 | 40% |
| prose, worked examples, exercise text | 13 | 33% |
| R code copied from the book | 5 | 13% |
| a complete, implementable equation | 6 | 15% |

At 15% (Wilson 95% interval roughly 7-29%) that is on the order of
1,600-6,500 modules carrying a usable equation out of 22,374.

Three failure modes, with examples from the sample:

- **Truncation.** `= V^-1 (Z(s) - mu(beta)),` has no left-hand side;
  `P(X=i; Y=j) =` has no right-hand side. Nothing can be implemented
  from either.
- **Wrong material.** `qda(Direction ~ Lag1 + Lag2, data = Smarket)` and
  `if(res[3]$p.value<=0.05) val=val+1` are R code, not equations.
  `Advanced R` is a book about programming; its chapter 10 is *Function
  factories* and contains no statistics, yet it contributed hundreds of
  modules whose placeholder body is a Kolmogorov-Smirnov test.
- **Prose.** `So, for example, if mu = 4 and sigma = 3 ...` is a
  sentence, not a specification.

The scraper also had no equation numbers to work with: references read
`ch.10 (unnumbered)`, so the fragment cannot be located in the source
even with the book open.

## Consequence

The `Formula:` strings are not a usable specification. Where a real
equation is wanted, the source PDFs are on disk and re-extraction from
them is the only sound route; salvaging the mangled strings is not.
