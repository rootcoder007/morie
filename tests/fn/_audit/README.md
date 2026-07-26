# `tests/fn` audit — the 339 reds

Started 2026-07-26. This directory is the durable record of the audit; it
is the worklist for the re-implementation series, not generated output that
can be recreated cheaply.

## Files

| file | what |
|---|---|
| `red_functions.csv` | the worklist — 339 rows, one per failing testcase, parsed from the junit XML |
| `fn-full-2026-07-26.log.gz` | full pytest log of the complete run |

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
