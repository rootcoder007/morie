# Auto-extracted placeholder triage — 11,219 modules

Ran `scripts/audit/triage_autoextracted.py` over every
`<book><chapter>u<equation>.py` module in `src/morie/fn` that carries
the placeholder template (`result = float(np.mean(...))` +
`se = float(np.std(...))`). Full per-module results in
`scripts/audit/autoextracted_triage.csv` (11,219 rows, written with a
per-row fsync so an interrupted run keeps everything it reached).

## Verdict

**None of the 11,219 are implementable. They should all be deleted.**

| Verdict | Count | Share |
|---|---:|---:|
| formula-only | 11,131 | 99.2% |
| prose-artifact | 75 | 0.7% |
| no-formula | 7 | 0.1% |
| implementable | 6 | 0.1% |

The 6 "implementable" are **false positives of my own classifier** —
prose fragments that happened to contain an operator character and
more than one parameter, e.g. `ca8u273.py`: *"effect size. To
determine β and by extension statistical power, or 1 −"*. There is no
equation in any of them.

## Why they cannot be implemented

**Every module has the same degenerate signature.** 11,213 of 11,219
take exactly one parameter named `x`; the remaining 6 take `x1, x2`.
There is no call contract to implement against — the harvester did not
capture which quantities an equation consumes.

**The "Formula:" lines are OCR fragments of running prose, not
equations.** A random sample:

```
over 38Â° (LR+ = 3) or leukocyte count> 15[·109/L] (LR+ = 7), can gradually contribute t
with 0< S0< N, for which each step is rightwards with probabilityp where 0< p = 1− q<
[EQ] 2 z8 = 2 2 z8 = 2 2 z6 = 2
[EQ] • No defiers: 𝑃(𝐷+ ≥ 𝐷−|𝑋 = 𝑐, 𝑀 = 0) = 1.
where ˆθ = w ˆy + (1− w)θ0, σ 2
Assuming that j − 1 ∑j − 1
satisfy a +b =b +a.]
T0 = 0, Tn = X1 + X2 + · · · + Xn for n ≥ 1,
```

They are cut mid-sentence, carry mojibake (`38Â°`), duplicated OCR
runs (`2 z8 = 2 2 z8 = 2`), and mathematical alphanumeric symbols the
extractor did not normalise. 17.8% contain no `=` or LaTeX at all.

**Deduplication would not help.** 11,150 distinct formula hashes
across 11,212 rows — the fragments are nearly all unique, because they
are arbitrary text windows rather than repeated equations.

## What these modules currently do

Each exports a function whose docstring states a book "formula" and
whose body returns the **mean and standard error of its single
argument**, with a RichResult title reading
`"<Topic> expression (auto-extracted; see ref)"`. That is a
documented claim the code does not meet — 11,219 times over.

## Recommendation

Delete all 11,219 modules and their generated tests. This removes
roughly 31% of the `morie.fn` module count and eliminates the largest
block of documentation that misdescribes its own behaviour. Nothing is
lost: no equation was successfully captured, so there is nothing to
re-implement later from these files. The source PDFs remain in the
library and are already the spec for the hand-named shelves.

The hand-named placeholders (5,002, the `gb*`/`ksr*`/`vol*`/`cop*`
families) are a **separate population** with real docstrings, stated
formulas and meaningful signatures — those are the ones the book-as-
spec batches work through, and they stay.

## Note on the classifier

`triage_autoextracted.py`'s `implementable` rule (formula text with an
operator + more than one data parameter) produced a 100% false-positive
rate on this corpus. It is left in the script as-written because the
CSV records the reason for every row, so the misfires are auditable
rather than hidden — but the rule should not be reused on another
corpus without tightening it to require a parseable left-hand side.
