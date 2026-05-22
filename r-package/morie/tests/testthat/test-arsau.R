# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Comprehensive coverage tests for R/arsau.R.  All tests run against
# synthetic ARSAU-shaped fixtures built under tempdir(), so coverage
# hits every branch in CI without depending on the real Ontario
# release being mounted.

# ── Helper: build a tiny ARSAU-shaped fixture ──────────────────────

.make_fixture_root <- function() {
  root <- file.path(tempdir(check = TRUE), paste0("arsau-fixture-", as.integer(Sys.time())))
  if (dir.exists(root)) unlink(root, recursive = TRUE)
  dir.create(root, recursive = TRUE)
  for (yr in c("2020-2022", "2023", "2024")) {
    dir.create(file.path(root, yr))
  }

  # 2020-2022/aggregate_summary
  write.csv(data.frame(
    SECTION = c("REPORT_SCOPE", "REPORT_SCOPE"),
    CATEGORY = c("1 to 3 Subjects - Individual Reports", "Other"),
    `UNITS OF MEASURE` = c("Number of Reports", "Number of Reports"),
    YEAR_2020 = c(100, 50),
    YEAR_2021 = c(120, 55),
    YEAR_2022 = c(110, 60),
    check.names = FALSE
  ), file.path(root, "2020-2022", "useofforce_agrregatesummarybyyear_2020-2022.csv"), row.names = FALSE)

  # 2020-2022/detailed (small synthetic version)
  set.seed(0)
  write.csv(data.frame(
    REPORTING_YEAR = sample(2020:2022, 100, replace = TRUE),
    POLICE_SERVICE = sample(c("Toronto", "OPP", "Ottawa", "Hamilton", "York"), 100, replace = TRUE),
    ASSIGNMENT_TYPE = sample(c("Patrol", "Drugs", "Other"), 100, replace = TRUE),
    REPORT_TYPE = sample(c("Individual", "Team"), 100, replace = TRUE)
  ), file.path(root, "2020-2022", "useofforce_detaileddataset_2020-2022.csv"), row.names = FALSE)

  # 2023/main
  write.csv(data.frame(
    IncidentYear = rep(2023L, 50),
    PoliceService = sample(c("Toronto", "OPP", "Halton"), 50, replace = TRUE),
    PoliceServiceType = sample(c("Municipal", "Provincial"), 50, replace = TRUE),
    OPP_PoliceService_Region = sample(c("Central", "Eastern"), 50, replace = TRUE),
    IncidentType = sample(c("Arrest", "Traffic"), 50, replace = TRUE)
  ), file.path(root, "2023", "uof_main_records.csv"), row.names = FALSE)

  # 2023/individual — note the trailing-space typo!
  df_indiv <- data.frame(
    BatchFileName = sprintf("BF-%03d", 1:60),
    Indiv_Index = 1:60,
    Race = sample(c("White", "Black", "Asian"), 60, replace = TRUE),
    Gender = sample(c("Male", "Female"), 60, replace = TRUE),
    AgeCategory = sample(c("18 - 24", "25 - 34", "35 - 64"), 60, replace = TRUE),
    `IndivInjuries_PhysicalInjuries ` = sample(c("Yes", "No"), 60, replace = TRUE),
    check.names = FALSE
  )
  write.csv(df_indiv, file.path(root, "2023", "uof_individual_records.csv"), row.names = FALSE)

  # 2023/probe
  write.csv(data.frame(
    BatchFileName = "BF-001",
    Indiv_Index = 1:5,
    CEW_CartridgeProbe_CartridgeProbeCycles_Cyc = c("Cyc1", "Cyc1,Cyc2", "", NA, "Cyc1,Cyc2,Cyc3")
  ), file.path(root, "2023", "uof_probe_cycle_records.csv"), row.names = FALSE)

  # 2023/weapon (invalid)
  write.csv(data.frame(
    BatchFileName = sprintf("BF-%03d", 1:20),
    Indiv_Index = 1:20,
    Weapon = sample(c("Firearm", "Taser", "Baton"), 20, replace = TRUE),
    Location = sample(c("Holster", "Hand"), 20, replace = TRUE)
  ), file.path(root, "2023", "uof_weapon_records_invaliddata.csv"), row.names = FALSE)

  # 2024/main (no trailing space — different from 2023)
  write.csv(data.frame(
    IncidentYear = rep(2024L, 60),
    PoliceService = sample(c("Toronto", "OPP", "York"), 60, replace = TRUE),
    PoliceServiceType = sample(c("Municipal", "Provincial"), 60, replace = TRUE),
    OPP_PoliceService_Region = sample(c("Central", "Northern"), 60, replace = TRUE),
    IncidentType = sample(c("Arrest", "Probable Cause"), 60, replace = TRUE)
  ), file.path(root, "2024", "uof_main_records.csv"), row.names = FALSE)

  # 2024/individual
  write.csv(data.frame(
    BatchFileName = sprintf("BF-%03d", 1:70),
    Indiv_Index = 1:70,
    Race = sample(c("White", "Black", "Asian"), 70, replace = TRUE),
    Gender = sample(c("Male", "Female"), 70, replace = TRUE),
    AgeCategory = sample(c("18 - 24", "25 - 34", "35 - 64"), 70, replace = TRUE),
    IndivInjuries_PhysicalInjuries = sample(c("Yes", "No"), 70, replace = TRUE)
  ), file.path(root, "2024", "uof_individual_records.csv"), row.names = FALSE)

  # 2024/probe
  write.csv(data.frame(
    BatchFileName = "BF-001",
    Indiv_Index = 1:3,
    CEW_CartridgeProbe_CartridgeProbeCycles_Cyc = c("Cyc1", "Cyc1,Cyc2", "")
  ), file.path(root, "2024", "uof_probe_cycle_records.csv"), row.names = FALSE)

  # 2024/weapon (valid)
  write.csv(data.frame(
    BatchFileName = sprintf("BF-%03d", 1:30),
    Indiv_Index = 1:30,
    Weapon = sample(c("Firearm", "Taser", "Baton"), 30, replace = TRUE),
    Location = sample(c("Holster", "Hand"), 30, replace = TRUE),
    Indiv_Weapon_Index = 1:30
  ), file.path(root, "2024", "uof_weapon_records.csv"), row.names = FALSE)

  # Tiny CKAN sidecar JSON next to 2024/main
  writeLines(jsonlite::toJSON(list(
    fields = list(
      list(id = "_id", type = "int"),
      list(id = "IncidentYear", type = "text"),
      list(id = "PoliceService", type = "text"),
      list(id = "PoliceServiceType", type = "text"),
      list(id = "OPP_PoliceService_Region", type = "text"),
      list(id = "IncidentType", type = "text")
    ),
    records = list()
  ), auto_unbox = TRUE), file.path(root, "2024", "ea9dc29c-b4f1-4426-b1f2-974ce995aca1.json"))

  root
}


# ── Path resolver branches ─────────────────────────────────────────

test_that("path resolver: data_dir argument wins", {
  withr::with_envvar(list(MORIE_ARSAU_DIR = NA, MORIE_DATA_DIR = NA), {
    fx <- .make_fixture_root()
    p <- morie:::.morie_resolve_arsau_dir(data_dir = fx)
    # On macOS, tempdir() returns /var/... while normalizePath returns
    # the canonical /private/var/... — compare normalised on both sides.
    expect_equal(normalizePath(p), normalizePath(fx))
    unlink(fx, recursive = TRUE)
  })
})

test_that("path resolver: MORIE_ARSAU_DIR env var wins when no arg", {
  fx <- .make_fixture_root()
  withr::with_envvar(list(MORIE_ARSAU_DIR = fx, MORIE_DATA_DIR = NA), {
    p <- morie:::.morie_resolve_arsau_dir()
    expect_equal(normalizePath(p), normalizePath(fx))
  })
  unlink(fx, recursive = TRUE)
})

test_that("path resolver: MORIE_DATA_DIR + /arsau falls through", {
  base <- file.path(tempdir(check = TRUE), paste0("morie-data-", as.integer(Sys.time())))
  fx <- file.path(base, "arsau")
  if (dir.exists(base)) unlink(base, recursive = TRUE)
  dir.create(fx, recursive = TRUE)
  withr::with_envvar(list(MORIE_ARSAU_DIR = NA, MORIE_DATA_DIR = base), {
    p <- morie:::.morie_resolve_arsau_dir()
    expect_equal(normalizePath(p), normalizePath(fx))
  })
  unlink(base, recursive = TRUE)
})

test_that("path resolver: error message when nothing found", {
  withr::with_envvar(list(MORIE_ARSAU_DIR = NA, MORIE_DATA_DIR = NA), {
    expect_error(
      morie:::.morie_resolve_arsau_dir(data_dir = "/nonexistent/path/xyz123"),
      "morie: could not find ARSAU data directory"
    )
  })
})

test_that("path resolver: require_exists=FALSE returns first candidate", {
  withr::with_envvar(list(MORIE_ARSAU_DIR = NA, MORIE_DATA_DIR = NA), {
    p <- morie:::.morie_resolve_arsau_dir(data_dir = "/no/such", require_exists = FALSE)
    expect_match(p, "no/such")
  })
})


# ── Registry accessors ─────────────────────────────────────────────

test_that("ARSAU_REGISTRY returns 10 entries", {
  reg <- ARSAU_REGISTRY()
  expect_equal(length(reg), 10L)
})

test_that("ARSAU_YEARS and ARSAU_KINDS are sorted unique", {
  yrs <- ARSAU_YEARS()
  expect_equal(yrs, c("2020-2022", "2023", "2024"))
  ks <- ARSAU_KINDS()
  expect_true(all(c("main_records", "individual_records", "weapon_records") %in% ks))
})


# ── Coerce key ─────────────────────────────────────────────────────

test_that(".arsau_coerce_year_key happy path + range_ok + reject bad keys", {
  expect_equal(morie:::.arsau_coerce_year_key(2024), "2024")
  expect_equal(morie:::.arsau_coerce_year_key("2020to2022", range_ok = TRUE), "2020-2022")
  expect_error(morie:::.arsau_coerce_year_key("9999"), "Unknown ARSAU year")
})


# ── Sidecar reader ─────────────────────────────────────────────────

test_that("sidecar reader handles bare shape", {
  tf <- tempfile(fileext = ".json")
  writeLines(jsonlite::toJSON(list(
    fields = list(list(id = "x", type = "int")),
    records = list()
  ), auto_unbox = TRUE), tf)
  res <- morie_arsau_read_sidecar(tf)
  expect_equal(res$fields[[1]]$id, "x")
  unlink(tf)
})

test_that("sidecar reader handles {result: {...}} wrapper", {
  tf <- tempfile(fileext = ".json")
  writeLines(jsonlite::toJSON(list(
    result = list(
      fields = list(list(id = "y", type = "text")),
      records = list()
    )
  ), auto_unbox = TRUE), tf)
  res <- morie_arsau_read_sidecar(tf)
  expect_equal(res$fields[[1]]$id, "y")
  unlink(tf)
})

test_that("sidecar reader handles empty payload gracefully", {
  tf <- tempfile(fileext = ".json")
  writeLines(jsonlite::toJSON(list(other = "data"), auto_unbox = TRUE), tf)
  res <- morie_arsau_read_sidecar(tf)
  expect_length(res$fields, 0)
  unlink(tf)
})


# ── Loaders ────────────────────────────────────────────────────────

test_that("loader: main_records happy path", {
  fx <- .make_fixture_root()
  r <- morie_arsau_load_main_records(2024, data_dir = fx)
  expect_equal(r$n_cols, 5L)
  expect_true(r$is_valid)
  unlink(fx, recursive = TRUE)
})

test_that("loader: individual_records 2023 trailing-space typo preserved", {
  fx <- .make_fixture_root()
  r <- morie_arsau_load_individual_records(2023, data_dir = fx)
  # Column with trailing space must still be in the loaded frame.
  expect_true(any(grepl("IndivInjuries_PhysicalInjuries", names(r$data))))
  unlink(fx, recursive = TRUE)
})

test_that("loader: probe_cycle happy path", {
  fx <- .make_fixture_root()
  r <- morie_arsau_load_probe_cycle_records(2024, data_dir = fx)
  expect_equal(r$kind, "probe_cycle_records")
  unlink(fx, recursive = TRUE)
})

test_that("loader: weapon_records 2023 invalid-gate errors without flag", {
  fx <- .make_fixture_root()
  expect_error(
    morie_arsau_load_weapon_records(2023, data_dir = fx),
    "flagged invalid"
  )
  unlink(fx, recursive = TRUE)
})

test_that("loader: weapon_records 2023 with allow_invalid loads + warns", {
  fx <- .make_fixture_root()
  r <- morie_arsau_load_weapon_records(2023, allow_invalid = TRUE, data_dir = fx)
  expect_false(r$is_valid)
  expect_true(length(r$warnings) >= 1)
  unlink(fx, recursive = TRUE)
})

test_that("loader: weapon_records 2024 happy", {
  fx <- .make_fixture_root()
  r <- morie_arsau_load_weapon_records(2024, data_dir = fx)
  expect_true(r$is_valid)
  unlink(fx, recursive = TRUE)
})

test_that("loader: aggregate_summary + detailed_dataset both work", {
  fx <- .make_fixture_root()
  r1 <- morie_arsau_load_aggregate_summary("2020-2022", data_dir = fx)
  r2 <- morie_arsau_load_detailed_dataset("2020-2022", data_dir = fx)
  expect_equal(r1$kind, "aggregate_summary")
  expect_equal(r2$kind, "detailed_dataset")
  unlink(fx, recursive = TRUE)
})

test_that("loader: unknown kind for year errors", {
  # 2024 is a valid year but aggregate_summary is only published for
  # 2020-2022; this hits the 'kind not published for year' branch.
  expect_error(
    morie_arsau_load_aggregate_summary("2024"),
    "aggregate_summary not published"
  )
})

test_that("loader: missing CSV file errors with remediation message", {
  empty <- file.path(tempdir(check = TRUE), paste0("empty-arsau-", as.integer(Sys.time())))
  if (dir.exists(empty)) unlink(empty, recursive = TRUE)
  dir.create(file.path(empty, "2024"), recursive = TRUE)
  expect_error(
    morie_arsau_load_main_records(2024, data_dir = empty),
    "ARSAU CSV not found"
  )
  unlink(empty, recursive = TRUE)
})

test_that("loader: French language path returns French interpretation", {
  fx <- .make_fixture_root()
  r <- morie_arsau_load_main_records(2024, language = "fr", data_dir = fx)
  expect_match(r$interpretation, "Donnees ARSAU")
  unlink(fx, recursive = TRUE)
})


# ── Discovery ──────────────────────────────────────────────────────

test_that("available_years works without data root", {
  withr::with_envvar(list(MORIE_ARSAU_DIR = NA, MORIE_DATA_DIR = NA), {
    r <- morie_arsau_available_years()
    expect_equal(r$years, c("2020-2022", "2023", "2024"))
  })
})

test_that("available_years detects present + missing", {
  fx <- .make_fixture_root()
  unlink(file.path(fx, "2024"), recursive = TRUE)
  r <- morie_arsau_available_years(data_dir = fx)
  expect_true("2023" %in% r$present)
  expect_true("2024" %in% r$missing)
  unlink(fx, recursive = TRUE)
})

test_that("available_datasets all years", {
  r <- morie_arsau_available_datasets()
  expect_equal(r$n, 10L)
})

test_that("available_datasets filtered by year", {
  r <- morie_arsau_available_datasets(year = "2024")
  expect_equal(r$n, 4L)
})

test_that("available_datasets error on unknown year", {
  expect_error(morie_arsau_available_datasets(year = "9999"), "Unknown ARSAU year")
})

test_that("describe with CSV present + sidecar present", {
  fx <- .make_fixture_root()
  r <- morie_arsau_describe("main_records", 2024, data_dir = fx)
  expect_true(r$csv_present)
  expect_true(!is.null(r$sidecar))
  unlink(fx, recursive = TRUE)
})

test_that("describe with CSV absent emits warning", {
  withr::with_envvar(list(MORIE_ARSAU_DIR = NA, MORIE_DATA_DIR = NA), {
    r <- morie_arsau_describe("weapon_records", 2024,
                                data_dir = "/no/such/dir/xyz123")
    expect_false(r$csv_present)
    expect_true(any(grepl("CSV not present", r$warnings)))
  })
})

test_that("describe French language path", {
  fx <- .make_fixture_root()
  r <- morie_arsau_describe("main_records", 2024, language = "fr", data_dir = fx)
  expect_match(r$interpretation, "ARSAU")
  unlink(fx, recursive = TRUE)
})

test_that("describe error on unknown kind", {
  expect_error(morie_arsau_describe("not_a_kind", 2024), "no .* entry for")
})


# ── Print method ───────────────────────────────────────────────────

test_that("print.morie_arsau_result emits readable output", {
  withr::with_envvar(list(MORIE_ARSAU_DIR = NA, MORIE_DATA_DIR = NA), {
    r <- morie_arsau_available_years()
    out <- capture.output(print(r))
    expect_true(any(grepl("ARSAU", out)))
  })
})
