# Phase 0 baseline audit

Local execution on Mac (R 4.6.0, all Suggests installed) — 2026-05-23.
Supplements the remote session's `audit/` artifacts with the
R-equipped pieces (`covr`, `lintr`, `R CMD check`) that the container
couldn't run.

## Coverage

- Tool: `covr::package_coverage()`
- Overall: **80.09%** (line-weighted)
- Per-file CSV: `audit/coverage-baseline.csv` (425 files)
  - 237 files at 100%
  - 3 files at literal 0% (`arsau_analyze.R`, `tpsuof.R`, `vertex.R`)
- Top remaining gaps to close for 99%+:
  - tps_hawkes_advanced.R 8.68%
  - crypto_keystore.R 10.20%
  - fairness_metrics.R 13.98%
  - otis.R 17.95%
  - tps_io.R, tps_fetch.R 19-22%
  - tps_statphysics.R, spatial_voting.R 26-28%
  - otis_analyze.R, dsp_filters.R, dsp_spectral.R 33-39%

The 80.09% is the recovered number after today's test work landed
on `release/v0.9.5.5` (HEAD `6c26efcc5`). The pre-port-marathon
baseline is >99% (see `feedback_coverage_baseline_99pct.md`); the
gap-to-baseline is the work Phase 1 closes by dropping
`skip_if_not_installed()` and Phase 2 closes by giving the dataset
tests synthetic fixtures.

## Lint

- Tool: `lintr::lint_package("r-package/morie")`
- Per-rule CSV: `audit/lint-baseline.csv`
- Saved object: `/tmp/morie-lintr.rds`
- Headline: see CSV; non-blocking style issues at this scale.

## R CMD check --as-cran

- Tool: `R CMD build` + `R CMD check --as-cran --no-manual
  --no-build-vignettes`
- Full log: `/tmp/morie-rcheck/check.log`
- Summary: `/tmp/morie-rcheck/result.log`
- Pass criterion (Phase 6): 0 ERROR, 0 WARNING, ≤1 NOTE

## Skip inventory (refining the remote's count)

- Tool: `grep -rn "skip(\|skip_if\|skip_on_cran" r-package/morie/tests/testthat/*.R`
- Total skip lines: **585**
- Buckets:
  - `skip_if_not_installed`: **266** (Phase 1 work — install full
    Suggests in CI, drop these calls)
  - `skip_if`: 213 (case-by-case in Phase 1)
  - `skip_on_cran`: 16 (replace with proper network-only guards)
  - bare `skip(...)`: 8 (replace with fixtures or document why)
- Self-references `skip_if_not_installed("morie")`: **17** — drop
  entirely.
- Top packages skipped on: MASS (35), jsonlite (33), survival (24),
  signal (17), morie (17), RSQLite (15), DBI (15), sodium (10),
  httr2 (10), digest (9), ggplot2 (8), randomForest (7),
  openssl (7), caret (7), withr (6).

## Core drift

`libmorie/morie_core.hpp` vs `r-package/morie/src/morie_core.h`:
identical SHA-256
`4287d6c3f04442699b2faa593a9bb6f89772cdf650ecae586c52de21f5102bd3`.

Phase 4 needs to replace the vendored copy with a build-time
`configure` step that copies the canonical header in (and refuses to
build if they ever diverge).

## Examples (`.Rd` audit)

- Total `.Rd` files: 1,451
- With `\dontrun{...}`: 146 (Phase 3 worklist)
- With `\donttest{...}`: 44
- With any `\examples{...}` block: 643 (so 808 exported names have
  NO examples at all — Phase 3 second worklist)

## Test suite

Independently verified earlier today on HEAD `6c26efcc5`:
- Total: 3,691
- Failed: 0
- Errors: 0
- Skipped: 71

The 71 skips here are the *runtime* skip count under default
environment (all Suggests installed) — most are network/offline
guards. Phase 1's reduction target is the *count of `skip_*()`
calls* in source, not the runtime skip count.
