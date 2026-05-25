# What's new in MORIE

Reverse-chronological summary of user-facing changes per release.
Per-package full changelogs:

- **R package:** [r-package/morie/NEWS.md](r-package/morie/NEWS.md)
- **Python package:** see commit history + git tags
- **Auto-generated version-stamp inventory:** [VERSION_INVENTORY.csv](VERSION_INVENTORY.csv)

---

## What's new in v0.9.5

- **SIU subsystem — first-class.** A full pipeline for the Ontario Special Investigations Unit director's-report corpus (English + French, 2005-present): `morie_fetch_siu()` with a polite token-bucket fetcher (4 req/s default, exponential backoff on 429/5xx, optional on-disk page cache), a hand-rolled C++ parser (`src/siu_parser.cpp`) that handles both 2015-2019 and 2020+ template families plus 2014 *Overview* and 2005 *Director's report* variants, 38 police-service acronyms (English + French) mapped to canonical English names, compound officer count handling, and a linear `html_to_text` state machine replacing the segfault-prone `std::regex_replace`.
- **Language-aware DRID manifest.** `inst/extdata/siu_drid_manifest.csv.gz` ships with 4,743 probed drids (en=2,531, fr=2,212, unknown=0) and a `canonical_drid` column for English-preferred dedupe. `morie_fetch_siu(lang = "en")` skips French drids — half the network round-trips. `morie_siu_index()` exposes the manifest.
- **Canonical override system — the parser learns.** `inst/extdata/siu_canonical_overrides.csv.gz` ships with 47 hand-verified corrections; `morie_siu_record_correction(case_number, field, value)` lets users add their own. Overrides are applied automatically at the end of every fetch.
- **Audit + AI tooling.** `morie_siu_audit_case()`, `morie_siu_compare()`, `morie_siu_sanity_check()`, `morie_siu_anomaly_check()`, `morie_siu_audit_columns()`, `morie_siu_translate()`, and `morie_siu_llm_extract()` with four providers — `ollama` (default, local, free), `gemini`, `claude`, `vertex` — and a `c("ollama", "gemini")` failover chain so paid APIs only fire when the local model fails. Defaults: `OLLAMA_HOST=http://localhost:11434`, `OLLAMA_MODEL=gemma3:4b`, `OLLAMA_KEEP_ALIVE=30m`. French → English translation via `translategemma:latest`.
- **559 exported `morie_*` R functions — every public callable now prefixed.** Cleared rOpenSci `pkgcheck`'s duplicated-function-names finding by renaming 352 unprefixed exports to `morie_*` across `R/`, `tests/`, `vignettes/`, `inst/`, and `data-raw/`. No aliases — the unprefixed names are gone from `NAMESPACE`.
- **TPS open-data ingestion fixes** (carried over from the original v0.9.5 plan). Corrected the Homicides and Shootings date ranges in the dataset catalog (`2004-present`, not `2014`); rewrote `morie_fetch_tps()` ArcGIS paging to follow the server's `exceededTransferLimit` flag so large layers are no longer silently truncated to the first page; daily-resolution Hawkes fits now build the occurrence date from the local-time `OCC_YEAR`/`OCC_MONTH`/`OCC_DAY` fields rather than the UTC-converted `OCC_DATE`.
- **`T_horizon` rename in the Hawkes C++ likelihood.** The time-horizon parameter was bare `T` in the auto-generated `R/RcppExports.R`, which `lintr` flags as a potential `TRUE` shadow. The C++ signature is now `T_horizon`; the math convention is preserved in C++ docstrings only.
- **rOpenSci 770 blockers cleared.** `.github/CONTRIBUTING.md` shipped, 16 `@return` docs added, 15 `@examples` added, full roxygen2 conversion (RoxygenNote 7.3.3), coverage validated ≥75% under `covr::package_coverage`, `\dontrun{}` count 72 → 0, `setwd()` replaced with `withr::local_dir()` in `R/workflow.R`.
- **Five-cell R CMD check matrix all green** on `release/v0.9.5-audit`: macOS-latest release, Windows-2025 release, Ubuntu-latest release, Ubuntu-latest release + postgres-15, Ubuntu-latest oldrel-1, Ubuntu-latest devel. Plus `pkgcheck`, `covr` + Codecov upload, `lintr`, `goodpractice`, and CodeQL.
- **Final SIU corpus stats**: 2,218 unique cases × 64 columns, 100.000% format-clean per `morie_siu_sanity_check()`.

## What's new in v0.9.4

- **CRAN source-package compliance.** The R package's vendored copy of the shared C++ core header was renamed `morie_core.hpp` → `morie_core.h`. `R CMD check --as-cran` does not recognize `.hpp` as a `src/` file extension and warned about it; the rename clears the WARNING. No behaviour change — the canonical `libmorie/morie_core.hpp` (Python/CMake side) is unchanged.

## What's new in v0.9.3

- **Docker image build fixed (completely).** v0.9.2's Dockerfile fix was incomplete — the builder stage didn't copy `LICENSE`, and `pyproject.toml`'s `license-files` declaration made scikit-build-core fail metadata generation without it. The builder now copies `LICENSE` too; the image build is verified end-to-end.
- **Homebrew tap bump fixed.** The tap-bump job raced the PyPI publish — it polled for the sdist for only ~4 minutes, but the sdist uploads *last*, after the full wheel matrix (~20 minutes). It now waits for the sdist itself, up to ~35 minutes.
- **Atomic releases.** The release pipeline now verifies the full build — the sdist *and* the Docker image — *before* the version tag is created. If any build fails, no tag is created and nothing publishes, so a half-broken release (a working PyPI package but a failed Docker image, as in v0.9.1/v0.9.2) can no longer ship.

## What's new in v0.9.2

- **Docker image build fixed for the C/C++ core.** The container build's Python stage was written for the old pure-Python package — it staged the install from a stub before copying the source. v0.9.1's compiled `libmorie` core (scikit-build-core + CMake) cannot build that way, so the published image failed to build. The builder stage now installs `cmake`/`ninja` and builds from the real `CMakeLists.txt` + `libmorie/` sources.

## What's new in v0.9.1

- **C/C++ computational backend** — the hot numerical kernels (formerly `_jit.py`) are ported to a shared C++ core (`libmorie`), exposed to Python via nanobind and to R via Rcpp. One compiled core now serves both language sides.
- **Hawkes-process engine** — a self-exciting point-process suite in the C++ core: sum-of-exponentials and complex-pole SoE engines, a matrix-pencil exponential fitter, sub-quadratic truncated Weibull / Lomax / gamma kernels, sinusoidal-baseline variants, and a hybrid gamma-tail kernel. An R-side Hawkes fitter with Poisson-degeneracy detection and multi-start restarts is included.
- **Wheels via cibuildwheel** — the PyPI wheel matrix is now built with `cibuildwheel` for the compiled extension.
- **IP / licensing cleanup** — the bundled demo dataset was replaced with public-domain Solar System data; copyrighted pop-culture quotes throughout `fn/` were replaced with public-domain ones; 85 franchise-derived function codes were renamed to neutral names and four themed categories merged into `AtomicPrimitives`.
- **OTIS data resolution fix** — `load_otis()` and the OTIS analysis modules resolve their data directory robustly (a `pyproject.toml` marker walk) instead of a hard-coded path depth.

## What's new in v0.9.0

- **`check_datasets()` dataset auditor** — probes every entry in the dataset catalogue and reports which datasets are reachable and which need attention, classified by tier.
- **More open-data sources** — new `morie.ingest.statcan` and `morie.ingest.cihi` modules add the StatCan Canadian Community Health Survey 2022 PUMF and five CIHI indicator data tables, fetched on demand.
- **16 datasets wired to verified sources** — Cannabis / Substance Use / Alcohol-and-Drugs / Student survey PUMFs got verified open.canada.ca CKAN ids; the Toronto Police crime datasets and the Ontario SIU case data now fetch through their scrapers. The catalogue went from 33 to 49 reachable datasets.
- **New-version notification + `morie update`** — `import morie` does a fail-silent, daily-cached PyPI check and warns when a newer release exists (opt out with `MORIE_NO_UPDATE_CHECK`); `morie update` upgrades in place.
- **CRAN fix** — the `morie_load_cpads` example is wrapped in `\dontrun{}`, clearing an `R CMD check --as-cran` error.

## What's new in v0.8.0

- **New: the fairness & disparity-audit subsystem (`morie.fairness`)** — a subsystem for *auditing* risk-assessment, recidivism, and predictive-policing systems for racial and other group disparities. morie measures whether an existing system encodes disparate treatment; it does not deploy one.
- **Six group-fairness metrics** — disparate impact (the four-fifths rule), demographic parity gap, equalized odds, average odds difference, Gini, and the composite Bias Amplification Score (Python + R parity).
- **Predictive-policing calibration audit** — rank areas by predicted risk vs. realised outcomes and test whether the disagreement tracks demographics; a city-agnostic `CityProfile` layer runs the audit for Chicago, New York, Toronto, or any registered city.
- **Multi-city temporal audit** — the disparity metrics per `(city, period)`, surfacing temporal instability and cross-city divergence.
- **Simulation framework** — a Noisy-OR detection model, a synthetic biased-data generator, a JAX spatial GAN, and a CTGAN-style debiaser (the optional `morie[sim]` extra — JAX, not PyTorch, to stay lean).
- **Explainability (XAI) suite** — permutation importance, partial dependence, ALE, ceteris paribus, and SHAP — model-agnostic, and wired to flag when a model leans on a protected attribute.
- Clean-room reimplementations from published methods (IBM AIF360; the SciencesPo *Predictive-policing-Chicago* project; Barman & Barman, arXiv:2603.18987; the COMPAS *XAI Stories* audit) — no third-party code copied.

## What's new in v0.7.4

- **Security fix** — resolved a regular-expression denial-of-service (ReDoS) vulnerability in the Ontario SIU scraper (`siu_fetch`), flagged by static analysis (CodeQL `py/redos`, high severity). A repeated sub-pattern could backtrack catastrophically on a maliciously crafted page; it is now linear-time, with no change to parsing of valid SIU index pages.
- **Stale `User-Agent` strings** across the data-ingestion modules aligned to the release version.

## What's new in v0.7.0

- **Licensing** — morie is licensed `AGPL-3.0-or-later` on both language sides. The two optional Linux-kernel adjuncts (`kernel-module/` and `daemon/`) stay `GPL-2.0-only` because the kernel ABI requires it; they are not part of the wheel or CRAN tarball.
- **Empirical applications paper published** — *Solitary Confinement, Self-Excitation, and Institutional Churn: Empirical Applications of MRM to Canadian Carceral and Police Data* on Zenodo at [10.5281/zenodo.20175689](https://doi.org/10.5281/zenodo.20175689). Five-paper publication set now complete.
- **`ac` / `vm` terminology locked across all 5 papers** — `ac` (alert complexity) and `vm` (volatility measure of placements, "regional-transition count" alongside) are now the canonical operational terms.
- **DOI + version propagation sweep** — empirical-paper DOI now reaches Sphinx index, `pyproject.toml [project.urls]`, `papers/README.md`, and CITATION.cff. Sphinx install snippets, Docker tag examples, and the in-tree `papers/README.md` were also un-pinned from stale versions.
- **R-package roxygen docs for fast Rcpp kernels** — `morie_mean`, `morie_var`, `morie_cor_pearson`, `morie_normal_pdf`, `morie_fast_available` ship with Rd man pages.
- **R 4.6.0 compatibility** — `DESCRIPTION` carries an explicit `Author:` field alongside `Authors@R:` so `R CMD check` passes on the strict 4.6.0 build.

## What's new in v0.6.1

- **Three replication modules from Laniyonu et al.** — `morie.laniyonu.gentrification_policing()` (Spatial Durbin replication of Laniyonu 2018 *UAR* — gentrification spillover on NYPD SQF), `morie.laniyonu.smi_force_disparity()` (Bayesian-style hierarchical neg-binomial replication of Laniyonu & Goff 2021 *BMC Psych* — police force on persons with serious mental illness), `morie.laniyonu.actuarial_risk_disparity()` (cumulative-logit replication of O'Connell & Laniyonu 2025 *Race & Justice* — Canadian federal-prison risk-assessment bias).
- **Five reusable MRM identification primitives** — `mrm.primitive.gentrification_panel`, `spatial_spillover_decomposition`, `synthetic_area_exposure`, `threshold_specific_ordinal`, `score_net_residual`. The building blocks every future module composes.
- **US + Canadian crime-data adapters** — `morie.datasets.chicago_crime()`, `nyc_stop_and_frisk()`, `bigquery()` (lazy Google-Cloud BigQuery), plus `nibrs()` (FBI Crime Data Explorer), `namus_missing_persons()`, `nist_rds()` (NIST Reference Datasets catalog).
- **Toy bundles for every new dataset** — Chicago crime (50 rows), NYC SQF (40 rows), NIBRS (30 rows), NamUs (20 rows), NIST RDS (10 rows). `offline=True` works on every loader.
- **`morie.fast` opt-in JIT acceleration surface** — drop-in JIT-compiled kernels (`normal_pdf`, `cor_pearson_jit`, `bootstrap_mean_jit`, `trimmed_ipw_weights_jit`, …) + a `jit_if_available` decorator. `pip install morie[fast]` activates Numba; without it, kernels run as pure-numpy. Numerically identical to scipy/numpy (max diff ≤5.55e-17).
- **`ci-numba-bench.yml`** nightly benchmark workflow comparing JIT vs non-JIT paths on every release.
- **Three new BibTeX entries** added to all 4 paper bibliographies: Laniyonu (2018), Laniyonu & Goff (2021), O'Connell & Laniyonu (2025).
- **Lazy-import fix** in `morie.ingest.__init__` — PEP 562 `__getattr__` for BigQuery uses `importlib.import_module` to avoid the infinite-recursion trap that `from . import bigquery` would create.

## What's new in v0.5.0

- **Any-dataset support** — bring your own column names. `morie.schema.infer_mapping(your_df, canonical=...)` fuzzy-matches your columns onto morie's canonical schema; pass the dict to `apply_mapping` and your data flows through every module without renaming. CLI users get `morie run-module ... --columns my_wt:weight,drinks_yn:alcohol_past12m`.
- **9-locale CLI** — `MORIE_LOCALE=es|de|zh|pt|ja|ar|hi morie ...` plus the existing EN + FR. Methodology docs stay English; CLI surface is multilingual.
- **No-code dataset shortcuts** — `morie pull tps-major --year 2024 --out file.csv` writes the entire Toronto Police "Major Crime" feed to disk in one line. No Python, no API URLs, no SQL. Also: `morie pull tps-shootings`, `morie pull tps-homicide`, `morie pull cpads`, `morie pull otis-a01-toy`, `morie pull siu-toy`, `morie pull tps-layers`.
- **[`TUTORIAL.md`](https://github.com/hadesllm/morie/blob/main/TUTORIAL.md)** — your first analysis, no Python knowledge required. Copy-paste five commands and you have 13 CSVs explained.
- **Python facade** — `import morie.datasets as md; df = md.tps_major_crime(year=2024)` for users who want to script.
- **Open-data adapters** — `morie ingest ckan/tps/siu` pulls feeds from CKAN portals (open.canada.ca, data.gov.uk, etc.), Toronto Police Service ArcGIS layers, and Special Investigations Unit director's-reports directly into pandas. See `morie.ingest.{ckan,tps,siu}`.
- **Synthetic CPADS bundled** — `morie run-module power-design` works on a fresh install with no manual download; emits a clear "synthetic data" warning so toy outputs aren't mistaken for real findings.
- **[`INSTALLATION.md`](https://github.com/hadesllm/morie/blob/main/INSTALLATION.md)** walkthrough covering all 5 install channels with platform-specific notes (PEP 668 on Debian, python 3.13 segfault on Raspberry Pi OS, Windows).
- **[`papers/`](https://github.com/hadesllm/morie/tree/main/papers)** allowlisted JSS paper sources in-tree (5 papers; no emails or drafts).
- Sphinx **"Edit on GitHub"** link in the sidebar so readers can suggest doc changes in one click.
- `anova_oneway` backwards-compat alias + `gibbons_chakraborti` rename (from v0.4.14, carried forward).

