# R-port roadmap (116 Python-only modules → R)

Generated 2026-05-22 as part of the v0.9.5.5 session.  Catalogues
every Python module under `src/morie/` that has **no R counterpart**
under `r-package/morie/R/`, sized in lines of code, grouped by
domain, with a realistic per-module effort estimate.

| | count | LOC |
|---|---:|---:|
| Python-only top-level modules | **116** | **81,393** |
| With existing R-name match | (not counted) | — |
| With existing R-alias match | (not counted) | — |

**Realistic effort to full R parity: multi-week, not "dozens of batches in
one session".**  Strategy: dispatch agents to draft, harvest dumps, write
files, smoke-test, commit per batch.  Batches are sized to keep each
agent's draft under ~800 LOC to fit in a single report fence.

## Strategic priority (highest value first)

1. **Crypto production subset** (130+151 = 281 LOC) — most-asked, smallest, real CRAN-worthy API.
2. **MRM primitives** (5 files, 626 LOC) — research-paper-relevant.
3. **Fairness toolkit** (8 files, ~2k LOC) — academic narrative.
4. **Ingest layer** (9 files, ~1.5k LOC) — needed for r-side reproducibility of OTIS/ARSAU/TPS work.
5. **Causal methodology** (did/iv/rdd/matching/effects/sensitivity, ~10k LOC) — substantial; might end up as Rcpp wrappers around existing CRAN packages.
6. **Statistical inference** (statistics/validation/survival/survey/weights, ~10k LOC) — many already have CRAN equivalents we can wrap.
7. **TPS-Toronto pipeline** (13 files, ~8k LOC) — research demo, can defer.
8. **ML/LLM bridges** (10 files, ~5k LOC) — most users won't use from R.
9. **Reporting/viz** (6 files, ~6k LOC) — Quarto + ggplot already cover most of this.
10. **Tooling (skip-or-thin)** (~25 files) — `agent`, `chat`, `tui`, `editor`, etc.
    Most of these are Python-only utilities that don't need an R analogue.
    Convert to thin `reticulate::import_from_path("morie")` wrappers if
    needed.

## Full module catalog (sized, grouped)

### Crypto (14 files, 2632 LOC)

| Module | LOC | Notes |
|---|---:|---|
| crypto/hybrid | 130 | Pure R + sodium |
| crypto/keystore | 151 | Pure R |
| crypto/_chacha (internal) | 120 | sodium wrapper |
| crypto/_kdf (internal) | 40 | openssl wrapper |
| crypto/_ecc (internal) | 200 | openssl wrapper |
| crypto/_gf2m (internal) | 175 | Pure R or Rcpp inline |
| crypto/_hashsig (internal) | 282 | Research; defer |
| crypto/_dilithium (internal) | 200 | Rcpp + liboqs |
| crypto/_lattice_core (internal) | 285 | Pure R port; slow but OK |
| crypto/_mceliece (internal) | 89 | Research; defer |
| crypto/_mlkem (internal) | 332 | Rcpp + liboqs |
| crypto/_ntru (internal) | 113 | Research; defer |
| crypto/_poly_ring (internal) | 248 | Pure R or Rcpp |

### MRM primitives (5 files, 626 LOC) — RESEARCH-CRITICAL

| Module | LOC |
|---|---:|
| mrm_primitives/gentrification | 101 |
| mrm_primitives/synthetic_exposure | 152 |
| mrm_primitives/spatial_spillover | 163 |
| mrm_primitives/score_net_residual | 198 |
| mrm_primitives/ordinal | (size TBD) |

### Fairness toolkit (7 files)

| Module | LOC |
|---|---:|
| fairness/cityprofile | 164 |
| fairness/gan | (size TBD) |
| fairness/metrics | (size TBD) |
| fairness/predpol | (size TBD) |
| fairness/simulation | (size TBD) |
| fairness/temporal | (size TBD) |
| fairness/xai | (size TBD) |

### Ingest layer (9 files)

| Module | LOC |
|---|---:|
| ingest/cihi | 72 |
| ingest/statcan | 97 |
| ingest/bigquery | 191 |
| ingest/ckan | (TBD) |
| ingest/chicago | (TBD) |
| ingest/forensics | (TBD) |
| ingest/tps | (TBD) |
| bq | (TBD) |
| siu_fetch | 201 |
| tps_fetch | 153 |

### Laniyonu modules (3 files)

| Module | LOC |
|---|---:|
| laniyonu/actuarial_risk_disparity | (TBD) |
| laniyonu/gentrification_policing | (TBD) |
| laniyonu/smi_force_disparity | (TBD) |

### Causal methodology (8 files, ~10k LOC)

| Module | LOC |
|---|---:|
| did | 2592 |
| doob_trends | (TBD) |
| effect_sizes | (TBD) |
| effects | (TBD) |
| iv | 2166 |
| rdd | 1851 |
| matching | 2210 |
| sensitivity | (TBD) |

### TPS pipeline (13 files, ~8k LOC)

| Module | LOC |
|---|---:|
| tps_all_analyze | (TBD) |
| tps_crime | (TBD) |
| tps_csi | (TBD) |
| tps_datasets | 200 |
| tps_fetch | 153 |
| tps_hawkes_advanced | (TBD) |
| tps_hawkes_jit | (TBD) |
| tps_io | (TBD) |
| tps_render | (TBD) |
| tps_spatial | (TBD) |
| tps_spatial_advanced | (TBD) |
| tps_stochastic | (TBD) |
| tps_temporal | (TBD) |

### ML / LLM bridges (10 files, ~5k LOC)

| Module | LOC |
|---|---:|
| ml | 95 |
| llm | (TBD) |
| gguf_loader | (TBD) |
| kv_cache | 212 |
| pt2gguf | (TBD) |
| tokenizer | 194 |
| quant | (TBD) |
| quant_bridge | (TBD) |
| vertex | 212 |
| agent | 1998 |

### Other top-level (50+ modules)

Full list at `/tmp/morie-staging/missing_r.txt`; sizes at
`/tmp/morie-staging/sizes.txt`.

## Dispatch strategy

- **Batch size:** 3-6 modules per agent, ~600-800 LOC total per batch.
- **Agent brief:** Read Python file via Read tool, draft idiomatic R
  equivalent, paste FULL R file content between `=== FILE DUMP: <path> ===`
  / `=== END FILE DUMP ===` markers in report (per
  `[[feedback_subagents_no_write]]`).
- **Harvest loop:** I copy each dump into a Write call, smoke-test in
  R, commit per-batch.
- **Quality bar:** Each R file MUST have AGPL header, roxygen2 docs,
  `@export` on public functions, idiomatic R (not python-translated
  prose).  Functions should call existing R / CRAN packages where
  appropriate (sodium, openssl, MatchIt, survival, etc.) rather than
  reimplementing from scratch.

## Tracking

Per-batch progress logged at `docs/r-port-progress.md` as we go.  Each
commit message lists the modules that batch ported.

**Started:** 2026-05-22 session.
**Status:** Catalog written; first batch dispatched (crypto/hybrid +
crypto/keystore + 3 mrm_primitives).
