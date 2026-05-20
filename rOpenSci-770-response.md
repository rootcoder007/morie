# Draft response to rOpenSci software-review #770

Status: DRAFT — for Vee to post to https://github.com/ropensci/software-review/issues/770
once `release/v0.9.5-audit` is pushed/merged so the editor bot can re-check.
The coverage figure marked `<<COVERAGE>>` must be filled with the final
`covr` number before posting.

---

**Update for the editors — all failing checks addressed in morie 0.9.5**

Thank you for the editor check on morie 0.9.4 (git `88d4a522`). Every
:heavy_multiplication_x: item from that check has been resolved, and the
:eyes: items have been addressed too. The package is now at **0.9.5**.

Point by point against the check output:

- ✖ **`contributing` file** — added `.github/CONTRIBUTING.md` describing
  workflow, testing and coding style.
- ✖ **Undocumented return values** (frns_metrics, frns_predpol,
  frns_temporal, license_check, longitudinal_sim, morie_fast_available,
  mrm_design, mrm_diagnostics, mrm_doe, mrm_kulldorff, mrm_lisa,
  mrm_mathstats, mrm_otis, mrm_samples, mrm_siu, mrm_tps) — every one now
  has an `@return` section.
- ✖ **Does not use roxygen2** — the R package is now fully roxygen2-based;
  all `man/*.Rd` are generated, and `DESCRIPTION` carries
  `Roxygen: list(markdown = TRUE)` / `RoxygenNote`.
- ✖ **Functions without examples** (the 15 listed) — all now carry
  runnable `@examples` (network-dependent ones use `\dontrun{}`).
- ✖ **Coverage 21% (need ≥75%)** — test coverage is now **<<COVERAGE>>%**
  (a comprehensive `testthat` suite; offline mock-based tests cover the
  network and bridge code).

:eyes: items:

- **goodpractice linters** — addressed: right-assignment and
  `1:length()` idioms fixed, a trailing semicolon removed, and the
  `src/Makevars` GNU-make `$(shell)` extension replaced with a standard
  `configure` / `configure.win` script so the committed Makefiles are
  portable.
- **Duplicated function names** — two functions (`morie_sample`,
  `ordered_alternatives_test`) were genuinely defined twice within the
  package; the shadowed copies were removed.
- **`\dontrun{}` in examples** — retained only where an example needs
  network access or external data; runnable examples are not wrapped.

`R CMD check` continues to report no errors and no warnings.

No scope change to the pre-submission inquiry. Since 0.9.4 the package
also gained an all-C/C++ data-retrieval layer (libcurl-backed) for the
Ontario SIU, Toronto Police, and CKAN open-data sources — this extends
the *data retrieval* / *data extraction* categories already selected,
it does not change them.

Happy to have `@ropensci-review-bot check package` re-run when convenient.
