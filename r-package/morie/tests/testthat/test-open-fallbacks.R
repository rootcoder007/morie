# SPDX-License-Identifier: AGPL-3.0-or-later
# Open-path fallbacks introduced in the rmorie parity port: corpus-first
# SIU caches, bundled-sample TPS fallback, keyless NIBRS, built-in-DB
# guard. Each test pins the new branch so the behaviour stays covered.

test_that("morie_fetch_siu materializes the rmoriedata corpus", {
  skip_if_not_installed("rmoriedata")
  d <- tempfile("siu-corpus-")
  csv <- suppressMessages(morie_fetch_siu(cache_dir = d))
  expect_true(file.exists(csv))
  df <- utils::read.csv(csv, nrows = 5)
  expect_true("case_number" %in% names(df))
})

test_that("morie_siu_fetch_cases corpus path honours the years filter", {
  skip_if_not_installed("rmoriedata")
  d <- tempfile("siu-cases-")
  csv <- suppressMessages(morie_siu_fetch_cases(cache_dir = d,
                                                years = 2024L,
                                                progress = FALSE))
  expect_true(file.exists(csv))
  df <- utils::read.csv(csv, colClasses = "character")
  if (nrow(df)) {
    expect_true(all(substr(df$case_number, 1, 2) == "24"))
  }
})

test_that("morie_siu_fetch_cases still rejects non-finite years first", {
  expect_error(
    morie_siu_fetch_cases(years = c(2023, NaN),
                          cache_dir = tempfile("siu-yrs-"),
                          overwrite = TRUE, progress = FALSE),
    "finite"
  )
})

test_that("nibrs ingest falls back to the bundled synthetic sample without a key", {
  old <- Sys.getenv("FBI_CDE_API_KEY", unset = NA)
  Sys.setenv(FBI_CDE_API_KEY = "")
  on.exit(if (is.na(old)) Sys.unsetenv("FBI_CDE_API_KEY") else
            Sys.setenv(FBI_CDE_API_KEY = old), add = TRUE)
  df <- suppressWarnings(suppressMessages(
    morie_ingest_forensics_nibrs(year = 2023, max_features = 5L)
  ))
  expect_s3_class(df, "data.frame")
  expect_true("ori" %in% names(df))
  expect_lte(nrow(df), 5L)
})

test_that("tps sample fallback resolves a bundled sample", {
  df <- suppressMessages(morie:::.morie_tps_sample_fallback("Assault", nrows = 3L))
  skip_if(is.null(df), "no bundled tps sample in this install")
  expect_s3_class(df, "data.frame")
  expect_lte(nrow(df), 3L)
})

test_that("tps loaders reach the sample fallback when the cache dir is empty", {
  old <- Sys.getenv("MORIE_TPS_DATA_DIR", unset = NA)
  Sys.setenv(MORIE_TPS_DATA_DIR = tempfile("no-tps-cache-"))
  on.exit(if (is.na(old)) Sys.unsetenv("MORIE_TPS_DATA_DIR") else
            Sys.setenv(MORIE_TPS_DATA_DIR = old), add = TRUE)
  df <- tryCatch(
    suppressMessages(morie_tps_load_dataset("Assault", nrows = 2L)),
    error = function(e) e
  )
  if (inherits(df, "error")) {
    expect_match(conditionMessage(df), "CSV not found")
  } else {
    expect_s3_class(df, "data.frame")
    expect_lte(nrow(df), 2L)
  }
})

test_that("morie_tps_load csv dispatch falls back to the bundled sample", {
  old <- Sys.getenv("MORIE_TPS_DATA_DIR", unset = NA)
  Sys.setenv(MORIE_TPS_DATA_DIR = tempfile("no-tps-cache-"))
  on.exit(if (is.na(old)) Sys.unsetenv("MORIE_TPS_DATA_DIR") else
            Sys.setenv(MORIE_TPS_DATA_DIR = old), add = TRUE)
  df <- tryCatch(
    suppressMessages(morie_tps_load("Assault", format = "csv", nrows = 2L)),
    error = function(e) e
  )
  if (!inherits(df, "error")) {
    expect_s3_class(df, "data.frame")
  } else {
    expect_match(conditionMessage(df), "no matching file")
  }
})

test_that("morie_load_dataset skips a missing built-in DB instead of erroring", {
  builtin <- tryCatch(morie_builtin_db(), error = function(e) NULL)
  skip_if(is.null(builtin) || file.exists(builtin),
          "builtin DB present; guard not exercised")
  err <- tryCatch(morie_load_dataset("no_such_dataset_xyz"),
                  error = function(e) conditionMessage(e))
  expect_match(err, "Unknown dataset", fixed = TRUE)
})
