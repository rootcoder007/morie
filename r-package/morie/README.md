# morie 森

<!-- badges: start -->
[![R-CMD-check](https://github.com/rootcoder007/morie/actions/workflows/r-cmd-check.yml/badge.svg)](https://github.com/rootcoder007/morie/actions/workflows/r-cmd-check.yml)
[![codecov](https://codecov.io/gh/rootcoder007/morie/branch/main/graph/badge.svg)](https://app.codecov.io/gh/rootcoder007/morie)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![rOpenSci review](https://img.shields.io/badge/rOpenSci-under_review_%23770-orange)](https://github.com/ropensci/software-review/issues/770)
<!-- badges: end -->

`morie` is a dual-language (R + Python) scientific computing package for
causal inference, sampling, psychometrics, point-process modeling, and
criminological accountability analysis. It expands to **Multi-domain Open
Research and Inferential Estimation**.

## Why morie?

morie is a **self-contained** research toolkit in two languages that
agree with each other: every statistical algorithm in the package —
matching (nearest-neighbour, Mahalanobis, exact, CEM, optimal, genetic,
cardinality), design-based IPW and doubly-robust estimation, double
machine learning, causal forests and meta-learners, causal DAGs with
identification and refutation, the full quasi-experimental family (DiD
incl. Callaway-Sant'Anna, event studies, synthetic control and synthetic
DiD, RDD, interrupted time series, IV), item-response theory,
geostatistics, digital signal processing, Hawkes processes,
cryptographic hashing with post-quantum primitives, and the data-access
parsers — is implemented natively in this package's R and C++, with a
Python arm (`morie.fn`, 14,000+ modules) that is executed on the same
inputs and compared key by key at twelve digits. morie does not call
MatchIt, survey, DoubleML, grf, dagitty, did, fixest, rdrobust, ivreg,
psych, gstat, spdep, signal, wavelets, digest, or openssl at runtime.
This means:

- **No upstream breakage.** A major release of any of those packages
  cannot change morie's results. A paper run in 2026 reproduces
  identically in 2030.
- **Validated, not just reimplemented.** Every native engine is
  cross-validated in `tests/cross/` against its reference package —
  to machine precision where the estimand is deterministic (see the
  table below) — and benchmarked in `inst/benchmarks/`.
- **Three arms, one answer.** The wave-3 ledger (567 modules) is
  implemented in Python, in this package and in the `rmorie` sibling;
  49 suites and 23,270 quantities agree with 0 differing, and every
  arm checks its own falsifiable anchors.
- **Composable workflows.** Matchers, ATE/CATE/DiD estimators, and
  the DAG pipeline share one class system with common `print()`,
  `summary()`, and reporting methods; the composition test runs
  DAG -> identification -> matching -> DML -> refutation ->
  publication table end to end in one suite.
- **Category-integrity guards.** `morie_safe_recode()`,
  `morie_safe_factor()`, `morie_audit_categories()`, and
  `morie_crosstab_verify()` make the silent category-mapping errors
  that have corrupted published disparity analyses structurally
  impossible in a morie workflow.

The remaining `Suggests` entries exist ONLY for the cross-validation
tests under `tests/cross/` and as optional accelerators for the
parsers (jsonlite/xml2/arrow fast paths with pure-R fallbacks); no
production statistics path requires any of them.

### Cross-validation at a glance

| Family | Replaces | Validation |
|---|---|---|
| Matching (7 methods) | MatchIt, optmatch, Matching, designmatch | pair-identical or provably better optimum |
| IPW / design-based GLM | survey | svyglm coefficients + SEs to 1e-6 |
| DML (PLR + IRM) | DoubleML/mlr3 | CI-overlap agreement; 40-60x faster |
| Causal forest / meta-learners | grf | CATE agreement; 1.5-2.9x faster |
| DAG identify/estimate/refute | dagitty, DoWhy | adjustment sets == dagitty on every graph tested |
| DiD family (TWFE/event/CS/DR/Bacon/DID-M/feTR) | fixest, did, DRDID, bacondecomp, DIDmultiplegt, TwoWayFEWeights | coefficients, SEs, and influence functions to 1e-8-1e-10 |
| Synth control + SDID | Synth/coresynth | recovers simulated truths; placebo inference built in |
| RDD (IK bw, sharp/fuzzy/kink, McCrary) | rdrobust, rdd | point estimates == rdrobust at fixed h to 1e-8 |
| IV (2SLS/LIML/GMM) + ITS | AER, ivreg, gmm | 2SLS == ivreg + sandwich HC1 to 1e-8 |
| IRT (2PL/GRM/EAP) + psychometrics | psych, mirt | KMO == psych to 1e-8; 2PL vs mirt within 0.1 |
| Geostatistics (variogram/kriging) | gstat, spdep | kriging == gstat to 1e-6; Moran variance == spdep |
| DSP (Butterworth/FIR/Welch/DWT) | signal, wavelets | butter coefficients == signal to 1e-8; DWT perfect reconstruction |
| Hawkes MLE | hawkes | exponential-kernel loglik == hawkes |
| SHA-256/HMAC/PBKDF2 + PQC | digest, openssl | NIST FIPS + RFC vectors bit-for-bit; ML-KEM/ML-DSA/SLH-DSA/HQC via liboqs |
| Parsers (JSON/XML/HTML/Parquet) | jsonlite, xml2, arrow | jsonlite-parity outputs; accelerators optional |
| Weighting family (ps/entropy/CBPS/OW/stabilized/SuperLearner) | WeightIt, CBPS | glm weights == WeightIt to 1e-8; CBPS moments < 1e-6 |
| Modern staggered DiD (Sun-Abraham/Borusyak/did2s) | did2s, didimputation | point estimates within 0.02 of did2s |
| Unified front-ends (morie_did/morie_iv_2sls/morie_rdd) | did, AER, rdrobust | CS overall == did::aggte to 0.1; 2SLS == ivreg to 1e-6 |
| Crim methods (ETAS/multivariate Hawkes/Knox/RTM) | (papers) | recovers simulated truths; Knox permutation calibrated |

## What morie is NOT

morie is not a wrapper. At runtime it does not call:

- **MatchIt / optmatch / Matching / designmatch** (matching) — replaced by `morie_matching_*`
- **WeightIt / CBPS** (propensity weighting) — replaced by `morie_weight_*`
- **survey** (design-based estimation) — replaced by the native svyglm engine behind `morie_ipw_*` / `morie_ebac_*`
- **DoubleML / mlr3** (double machine learning) — replaced by the native PLR/IRM/PLIV cross-fit engines
- **grf / EconML-style learners** (heterogeneous effects) — replaced by the native causal forest and T/S/X/DR meta-learners
- **dagitty / DoWhy** (DAGs, identification, refutation) — replaced by `morie_dag_*`
- **did / fixest / did2s / didimputation** (modern DiD) — replaced by `morie_did_*` with auto-dispatch to Callaway-Sant'Anna, Sun-Abraham, Borusyak, and Gardner two-stage
- **rdrobust** (RDD) — replaced by `morie_rdd` (IK bandwidth + McCrary + placebo cutoffs bundled)
- **Synth** (synthetic control) — replaced by `morie_synth_control` with built-in placebo inference
- **AER / ivreg** (IV) — replaced by `morie_iv_2sls` with the Staiger-Stock refusal gate
- **psych / mirt** (psychometrics) — replaced by `morie_psymet_*` and `morie_irt_*`
- **gstat / spdep** (geostatistics) — replaced by the native variogram/kriging/GWR stack
- **signal / wavelets** (DSP) — replaced by `rgfir`/`rgiir`/`rgwav` and `morie_dsp_*`
- **hawkes** (point processes) — replaced by the native C++ Hawkes kernel family + `morie_crim_etas` / `morie_crim_hawkes_multivariate`
- **digest / openssl** (hashing/KDF) — replaced by the native C++ SHA-2/HMAC/PBKDF2 + liboqs PQC
- **jsonlite / xml2 / arrow as requirements** (parsing) — replaced by `morie_fetch_*` pure-R parsers (those packages remain optional fast paths only)

Those packages appear in `Suggests` solely so `tests/cross/` can
prove that the native engines match them.

## What's in v1.1.7

- **All native-specialization modules complete** in both languages —
  the package's statistics run with zero runtime dependencies on other
  statistical packages.
- **12,600+ exported `morie_*` R functions** and a 14,000-module Python
  arm (`morie.fn`), every public callable prefixed to avoid name
  collisions with other CRAN packages.
- **SIU subsystem** — the verified 65-column corpus, the zero-wrong
  subject-official resolver and the Mixture-of-Agents reading panel,
  in R and in Python (`morie.siu`). See *SIU pipeline* below.
- **Polite-by-default HTTP fetcher** — token-bucket throttling at 4
  req/s, exponential backoff on 429/5xx, on-disk page cache.
- **Built-in datasets** through the shared SQLite store plus the
  `rmoriedata` companion package.
- **Outputs-manifest tooling**, **CPADS contract helpers**, and
  **synthetic data generators** for development and CI.
- **C/C++ computational backend** — Hawkes likelihoods, the native
  Parquet/JSON/XML/HTML parsers, the SIU text core (via
  `rmoriebricklayer`), SHA-2/HMAC/PBKDF2 and liboqs post-quantum
  primitives. See `src/`.

## Scientific guardrail

- Synthetic data is for development, testing, demos, and CI only.
- Final inferential or policy-facing results must be produced from
  approved real data with full provenance.
- Synthetic runs must be explicitly labeled as synthetic in outputs
  and reporting text.

## Install

From local source:

```r
install.packages("r-package/morie", repos = NULL, type = "source")
```

From r-universe (development snapshot):

```r
install.packages(
  "morie",
  repos = c(rootcoder007 = "https://rootcoder007.r-universe.dev",
            CRAN     = "https://cloud.r-project.org")
)
```

The assistant bridge supports a local fallback through the Python
package when no live OpenAI / Anthropic credentials are configured.

## Outputs-manifest example

```r
library(morie)

manifest <- morie_read_outputs_manifest(project_root = "/path/to/project")
audit    <- morie_audit_public_outputs(project_root = "/path/to/project",
                                       manifest     = manifest)
morie_summarize_output_audit(audit)
```

## Synthetic data example

```r
library(morie)

synthetic_path <- morie_write_synthetic_data(
  path      = "data/private/synthetic_study_data.csv",
  n         = 8000,
  seed      = 2026,
  overwrite = TRUE
)
```

## Cross-project adaptation

```r
library(morie)

name_map <- morie_default_synthetic_name_map("generic")
name_map["cannabis_use"] <- "exposure_any"
name_map["bac"]          <- "outcome_continuous"

dat <- morie_generate_synthetic_data(
  n        = 5000,
  seed     = 1,
  name_map = name_map
)
```

## SIU pipeline

morie ships the **first open-source parser and data-mining subsystem
for the Ontario Special Investigations Unit (SIU) director's-report
corpus** — created by Vansh Singh Ruhela as part of the MORIE / R-MORIE
ecosystem and the MRM (Multilevel Reconciliation Methodology)
framework. The SIU publishes the reports; there was no programmatic
pipeline to fetch, parse, and analyse them at corpus scale until this
one.

The fetcher handles both English and French templates from 2005 onward
across all three of the site's historical layout generations; the
parser extracts the report fields (police service, incident/
notification/decision dates, investigator and witness/subject-official
counts, affected-person demographics, injuries, legislation, charges
verdict, and director's decision) and is hand-rolled for correctness
under SIU's heterogeneous markup.

Since 1.1.4 the subsystem is layered on the compiled SIU core in
`rmoriebricklayer` and the **verified corpus** shipped by
`rmoriedata`: a 65-column table of 5,157 reports whose 2,182 English
entries were read and cross-audited by a multi-agent review panel
(every subject-official count verified; the mechanical resolver scores
zero wrong against it). `morie_siu_reports()` returns that corpus
verbatim and only ever fetches/parses reports newer than it;
`morie_siu_resolve_so()` answers from the verified corpus first and
falls back to the compiled rule engine; `morie_siu_panel()` runs the
same Mixture-of-Agents reading panel on new reports through any
Ollama-compatible endpoint you point it at (your models, your host —
no hardcoded default). The Python arm exposes the same three
(`morie.siu.siu_reports`, `siu_resolve_so`, `siu_panel`) with the
corpus pinned by SHA-256.

### Use the verified corpus (no re-fetching)

```r
library(morie)

df <- morie_siu_reports(update = FALSE)   # 5,157 reports x 65 columns
morie_siu_resolve_so(drid = 5038)         # verified: SO = 2
morie_siu_resolve_so(text = "SO #1 Interviewed\nSO #2 Declined")  # rules
```

```python
from morie.siu import siu_reports, siu_resolve_so, siu_panel
df = siu_reports()                         # same corpus, SHA-256 pinned
siu_resolve_so(drid=5038)                  # {'count': 2, 'reason': 'panel-reviewed corpus (verified)'}
```

### Read a new report with your own models

```r
Sys.setenv(OLLAMA_HOST = "http://your-ollama:11434")
res <- morie_siu_panel(5161, mode = 2)     # one reader + one auditor
res$fields["number_of_subject_officers"]
```

### Legacy live fetch, audits and overrides

`morie_fetch_siu()`, `morie_siu_audit_case()`, `morie_siu_anomaly_check()`,
`morie_siu_sanity_check()`, `morie_siu_audit_columns()`,
`morie_siu_record_correction()` and `morie_siu_index()` remain available
for corpus-scale re-fetches and per-case audits.

## Continuous integration

The R CMD check matrix covers six cells, all green on the
`release/v0.9.5-audit` head:

| Platform        | R version             |
| --------------- | --------------------- |
| macos-latest    | release               |
| windows-2025    | release               |
| ubuntu-latest   | release               |
| ubuntu-latest   | release + postgres-15 |
| ubuntu-latest   | oldrel-1              |
| ubuntu-latest   | devel                 |

Plus: `pkgcheck`, `covr` + Codecov upload, `lintr`, `goodpractice`, and
CodeQL.

## Citation

Run `citation("morie")` after installation. Please cite the software:

```bibtex
@Manual{ruhela_morie_2026,
  title   = {morie: Multi-domain Open Research and Inferential Estimation},
  author  = {Ruhela, Vansh Singh},
  year    = {2026},
  note    = {R package version 1.0.1},
  url     = {https://github.com/rootcoder007/morie}
}
```

The single citation above covers both the R and Python implementations
(same version, same toolkit). Methodology and empirical-applications
papers (MRM framework, criminological Hawkes process, solitary-
confinement / self-excitation / institutional churn) are in
preparation; this section will be expanded once they are publicly
available with DOIs or preprint URLs.

## License

morie is licensed under **AGPL-3.0-or-later**. See `LICENSE` for the
full text and `LICENSING.md` for the per-component breakdown.

## rOpenSci review

morie is under review at rOpenSci:
[ropensci/software-review#770](https://github.com/ropensci/software-review/issues/770).

## Bayesian priors

morie's Bayesian regression (`morie_bayes_lm`) places zero-mean Normal
priors on the regression coefficients; the `prior_sd` argument is the
prior standard deviation (the scale of plausible coefficient values).
Larger `prior_sd` is weakly informative; smaller values pull estimates
toward zero (regularisation). Example:

```r
d <- data.frame(x = rnorm(100)); d$y <- 1 + 2 * d$x + rnorm(100)
# weakly-informative prior (sd = 10) vs a tight regularising prior (sd = 0.5)
loose <- morie_bayes_lm(y ~ x, d, prior_sd = 10)
tight <- morie_bayes_lm(y ~ x, d, prior_sd = 0.5)
morie_bayes_diagnostics(loose)
```

See `vignette("bayesian-priors", package = "morie")` for the full walkthrough.
