# `tests/fn` audit — the 339 reds

Started 2026-07-26. This directory is the durable record of the audit; it
is the worklist for the re-implementation series, not generated output that
can be recreated cheaply.

## Files

| file | what |
|---|---|
| `red_functions.csv` | the worklist — 339 rows, one per failing testcase, parsed from the junit XML |
| `fn-full-2026-07-26.log.gz` | full pytest log of the complete run |
| `trivial_tests.csv` | the 3,621 tests whose assertions cannot fail, with each one's target classified stub-vs-real |
| `harvest_trivial.py` | regenerates that CSV, and `--check` fails if the count grows |

The 11 MB junit XML is too large for the repo and lives at
`/Volumes/VSR/rootcoderfiles/morie-audit-artifacts/fn-full-2026-07-26.xml`.

## How the run was produced

L14 (Fedora), CPython **3.12.13** — matched to CI exactly, because L14's
system Python is 3.14 and building there surfaces dependency failures that
are not our bug. Virtualenv `.venv-fn`, built with `uv`.

```sh
pytest tests/fn -q -o addopts="" -p no:cacheprovider --timeout=120 \
  -n 6 --dist=loadfile -rf --junitxml=fn-full.xml
```

`-o addopts=""` is required: `pyproject.toml` excludes `tests/fn` by
default. `-n 6` not 8 — 8 workers of numpy/scipy against L14's 15 GB is how
the OOM killer took out the CI runner on 2026-07-23.

Result, 11m27s:

```
339 failed, 78811 passed, 14 skipped, 1 xfailed, 2 xpassed, 3858 warnings
```

## Why this audit exists

The nightly `fn-sampled` workflow samples **500 of 36,286** files with a
date-derived seed, so a green night is evidence about 1.4% of the suite,
not about the suite. Failures had been arriving one per night for months
and reading as flaky CI. They are not flaky — the sampler is a working
detector draining a real backlog.

The full run showed the reds are the small part of the problem. Clustering
every function body by AST skeleton (identifiers and literals erased)
gives only **3,902 distinct skeletons for 36,407 functions**; the top two
cover 64%. The dominant body returns `mean(x)` and a standard error
whatever the function is named or documented to do.

Cross-tabulating test outcome against "is the target a shared skeleton
(used by >= 20 functions)":

```
              stub impl    real impl
FAIL                 84          255
PASS             65,517       13,296
```

The 65,517 passing-against-a-stub cell is the false-negative quadrant —
83% of the suite. It is **out of scope** for this series and must not be
touched; converting it to honest failures in one pass would crater the
suite and tell a reviewer nothing they can act on.

Independently, the suite's own `FALSE-POSITIVE RISK` warning (trivially-true
assertions, zero domain-specific checks) fired on **3,621 distinct tests**
in the same run. Nothing reads it, because warnings do not fail a build.

## The trivial-assertion harvest

`trivial_tests.csv` is that warning made durable. The detector in
`tests/conftest.py` is a pure source-inspection fixture — it reads
`inspect.getsource(request.function)` and looks at the `assert` lines — so it
can be replayed statically over the whole tree without running pytest.
`harvest_trivial.py` does exactly that, and lands on **3,621**, the same
number the 11-minute run reported. That equality is the evidence the replay is
faithful; `tests/test_audit_trivial_harvest.py` pins the two pattern sets
together so a later edit to the fixture cannot silently change what the CSV
counts.

The CSV's value is not the count, which was already known. It is the
`target_kind` column, which splits the 3,621 into work and non-work by
skeletonising every function in `morie.fn` (identifiers and literals erased)
and asking whether the target's body is shared with >= 20 siblings:

| `target_kind` | rows | meaning |
|---|---|---|
| `real` | 126 | trivial test of a REAL implementation — **actionable** |
| `stub` | 3,491 | trivial test of a generated stub body — blocked on the stub |
| `unknown` | 4 | target not resolvable from the module's imports |

**96% are blocked, not neglected.** Tightening a test against `abblc` — whose
docstring says "Black carbon spatial" and whose body is `float(np.mean(data))`
— converts a passing test into a failing one and tells a reviewer nothing the
skeleton census above does not already say. Those rows stay in the
false-negative quadrant, out of scope for this series.

The **126 `real` rows are the actionable set**, and they are tractable: 117
modules, mostly one test each, clustered in `gwr*` (16), plus `igrav*`,
`gns*`, `dif*` and the IRT/MDS group. These are real implementations whose
only assertion is `assert r.value is not None` — the function could return the
wrong number, the wrong units, or a constant, and the test would still pass.
They grade the same way as a red: find the book, derive the value, pin it.

The four `unknown` rows (`ghsrv`, `sgedg`, `sgint`) do not import their target
from `morie.fn.*` in the form the resolver expects, and need reading by hand.

Regenerate and gate:

```sh
python tests/fn/_audit/harvest_trivial.py          # rewrite the CSV
python tests/fn/_audit/harvest_trivial.py --check  # exit 1 if the count grew
```

The `--check` form runs nightly in `.github/workflows/fn-sampled.yml`, so the
baseline can fall but cannot silently rise. Lowering it is a normal commit.

## The source hierarchy

The spec for a function is the first of these that actually covers the method
it implements. Record which tier was used in `book_reference`, and cite it in
the docstring `References:` block.

1. **The PDF in the library.** Authoritative. Chapter, section, equation
   number and printed page.
2. **The primary methodology paper**, when the library holds only a secondary
   source that cites it. Where the two disagree the primary wins, and the
   secondary's value is recorded as the alternative (`gmatv`, where the
   secondary contradicted its own worked example twice).
3. **Official reference documentation** — CRAN package manuals / vignettes,
   the Python / NumPy / SciPy / PyWavelets reference docs — when the library
   has no book covering the method, or when the function's behaviour is
   *defined by* the library routine it calls rather than by a textbook.
4. **STOP and ask.** Nothing above applies.

**The txt extractions are never a tier.** They are a search index for
locating material fast across 85 MB. Every equation, every fixture value and
every page citation is read from the PDF. The extraction destroys
mathematics — `G ¼ 1 p XX T` for `G = (1/p)XXᵀ`, `zij ¼ xij  2p j = 2p j 1  p j`
for `z = (x−2p)/√(2p(1−p))` — and it silently truncates tables. Both bugs
found in Rangayyan batch 1 lived exactly where the extraction was unusable.

Tier 3 is not a fallback for laziness; it is the correct tier when the
function's contract *is* the library's contract. `rgfir` is the worked case:
Rangayyan Ch. 3 covers Butterworth IIR filtering and has no windowed-sinc FIR
design section, so the SciPy `firwin` documentation is the specification —
and it supplied both the missing error behaviour ("raises ValueError if any
value in cutoff is ... greater than or equal to fs/2", which the function was
masking with a clip) and the exact identity to pin it with (`scale=True`
normalises unity gain at DC, therefore `sum(taps) == 1`).

Quote tier-3 documentation verbatim in the docstring the same way a book is
quoted, with the URL, so a reviewer can check the claim without guessing
which version of the docs was read.

**Bank every tier-2/tier-3 source on first use.** A spec fetched from outside
the library gets written into the library immediately, so no later session
re-fetches it:

- real PDF -> `data/datasets/userguides/other/pdf/`
- online docs -> verbatim snapshot in
  `data/datasets/userguides/other/docs-snapshots/<topic>.md`, with URL and
  retrieval date (web docs are mutable; the snapshot is what was actually read)
- paywalled / HTTP 403 -> a row in `docs-snapshots/ACQUISITION.md` with the
  DOI, the function that needs it, and how the definition was verified meanwhile

Then add the row to `books.csv`. Publishers block scripted download hard --
PNAS and the American Physiological Society both return 403 with an HTML body
that lands under a `.pdf` name, so `file` the download before trusting it.

## The irtgr precedent

The first fix in this series is the worked example of why "the test passes
and it's faster" is not evidence of correctness.

`irtgr` (Samejima GRM) was blowing the 120s CI cap; the real cost was
**314.0s**. The first vectorisation collapsed the M-step reduction into a
single `np.sum` over a 2-D array and ran in **19.1s** — and it was wrong.
numpy's `np.sum` uses pairwise summation where the original accumulated
sequentially over quadrature points. That perturbed the last bits, which
moved L-BFGS-B's finite-difference gradients, which shifted EM
convergence: `n_iter` 59 -> 58, loglik delta 5.1e-05, thresholds off by
2.4e-03. Every test still passed.

It was backed out. The reduction stays a Python loop over quadrature
points; the speed lives in the vectorised probability call, not in the
reduction. Final: **42.2s (7.4x), bit-identical** — loglik, theta, info,
discriminations and thresholds all `0.000e+00` delta against a baseline
saved *before* the code was touched.

Three rules follow, and they bind for every function in this series:

1. Save every output to disk **before** editing. "Should be speed-only" is
   a hypothesis; a diff against a saved baseline is a result.
2. A green test does not distinguish a correct change from a change whose
   error is smaller than the assertions. Only the baseline diff does.
3. Reduction order is part of the contract. Anything feeding an iterative
   optimiser can turn a last-bit difference into a different answer.

Commit `8cebb72491`.

## Column meanings

`priority` — 1 = non-skeleton (255), 2 = shared-skeleton (84). P1 first.

`skeleton` — yes if the target shares a body skeleton with >= 20 siblings.

`book_reference` / `book_chapter` / `book_equation` — the spec. The book is
the spec, not the docstring and not the test. If the book cannot be found,
the row is `MISSING - NEEDS USER` and the function is **not** implemented.

`docstring_consistent_with_book` / `test_consistent_with_book` — where they
disagree with the book, the book wins and the other is the bug.

Planned additions before batch 1: `language_layer`, `mirrored_in_rmorie`,
`book_value_at_test_input`, `mirt_value_at_test_input`,
`convention_drift_yes_no`, `test_quality_class`.

## P1 by the book each docstring already cites

| spec | modules | P1 tests |
|---|---|---|
| Rangayyan, *Biomedical Signal Analysis* | 20 (+`wavts` = 21) | 36-38 |
| Armstrong, *Analyzing Spatial Models of Choice and Judgment* | 13 | 21 |
| Chakraborti & Gibbons, *Nonparametric Statistical Inference* 5e | 12 | 17 |
| DL/LLM papers | 28 | 53 (out of scope) |
| Fauzi / Horowitz | 6 | 8 |
| other / paper-cited | 56 | 113 |

No P1 module is blocked on a missing book. Three that looked missing were
filed under non-obvious names: Kosorok is `978-0-387-74978-5.txt` (ISBN),
VanRaden's GRM is section 2.4 pp. 49-52 of the *Multivariate Statistical
Machine Learning* `Pages 35-70` volume, and the wavelet spec is Rangayyan
Ch. 10. Search the corpus by content, never by filename.
