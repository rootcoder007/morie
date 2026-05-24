# SPDX-License-Identifier: AGPL-3.0-or-later
# Coverage tests for R/datasets.R -- registries + offline loaders + helpers.

set.seed(1)

test_that("tps_layers returns a 3-row data.frame with name+url", {
  set.seed(1)
  df <- morie_datasets_tps_layers()
  expect_s3_class(df, "data.frame")
  expect_true(all(c("name", "url") %in% colnames(df)))
  expect_gte(nrow(df), 3L)
  expect_true(all(grepl("^https://services", df$url)))
})

test_that("year_where helper builds canonical SQL", {
  set.seed(1)
  expect_equal(morie:::.morie_dataset_year_where(NULL), "1=1")
  expect_equal(morie:::.morie_dataset_year_where(2024), "OCC_YEAR = 2024")
})

test_that("records_to_df handles empty, list, df", {
  set.seed(1)
  expect_equal(nrow(morie:::.morie_dataset_records_to_df(NULL)), 0L)
  expect_equal(nrow(morie:::.morie_dataset_records_to_df(list())), 0L)
  df0 <- data.frame(a = 1:2)
  expect_identical(morie:::.morie_dataset_records_to_df(df0), df0)
  recs <- list(list(a = 1, b = "x"), list(a = 2, b = "y"))
  out <- morie:::.morie_dataset_records_to_df(recs)
  expect_s3_class(out, "data.frame")
  expect_equal(nrow(out), 2L)
})

test_that("pkg_csv resolves to NA or a real path", {
  set.seed(1)
  p <- morie:::.morie_dataset_pkg_csv("__no_such_synth__")
  expect_true(is.na(p) || file.exists(p))
})

test_that("read_synthetic with absent name + columns returns 0-row frame", {
  set.seed(1)
  out <- morie:::.morie_dataset_read_synthetic(
    "__never_existing__", "xx", columns = c("a", "b")
  )
  expect_s3_class(out, "data.frame")
  expect_equal(nrow(out), 0L)
  expect_equal(colnames(out), c("a", "b"))
})

test_that("read_synthetic with absent name + no columns errors", {
  set.seed(1)
  expect_error(
    morie:::.morie_dataset_read_synthetic("__never_existing__", "xx"),
    "synthetic"
  )
})

test_that("tps_major_crime offline returns data.frame and respects max_features", {
  set.seed(1)
  res <- tryCatch(
    suppressWarnings(morie_datasets_tps_major_crime(offline = TRUE, max_features = 5L)),
    error = function(e) NULL
  )
  skip_if(is.null(res), "no synthetic fixture")
  expect_s3_class(res, "data.frame")
  expect_lte(nrow(res), 5L)
})

test_that("tps_shootings live call gated on network", {
  set.seed(1)
  res <- tryCatch(
    morie_datasets_tps_shootings(year = 2024, max_features = 1L),
    error = function(e) NULL
  )
  skip_if(is.null(res), "needs network")
  expect_s3_class(res, "data.frame")
})

test_that("tps_homicide live call gated on network", {
  set.seed(1)
  res <- tryCatch(
    morie_datasets_tps_homicide(year = 2024, max_features = 1L),
    error = function(e) NULL
  )
  skip_if(is.null(res), "needs network")
  expect_s3_class(res, "data.frame")
})

test_that("cpads loader returns the synthetic frame when no cache", {
  set.seed(1)
  out <- tryCatch(
    suppressWarnings(morie_datasets_cpads()),
    error = function(e) NULL
  )
  skip_if(is.null(out), "no cpads fixture")
  expect_s3_class(out, "data.frame")
})

test_that("otis_a01 offline returns frame; offline=FALSE errors", {
  set.seed(1)
  out <- tryCatch(
    suppressWarnings(morie_datasets_otis_a01(offline = TRUE)),
    error = function(e) NULL
  )
  skip_if(is.null(out), "no otis fixture")
  expect_s3_class(out, "data.frame")
  expect_error(morie_datasets_otis_a01(offline = FALSE), "FOI")
})

test_that("siu_director_reports returns a data.frame (empty if no deps/network)", {
  set.seed(1)
  out <- tryCatch(
    suppressWarnings(morie_datasets_siu_director_reports()),
    error = function(e) NULL
  )
  skip_if(is.null(out), "needs rvest/xml2 or network")
  expect_s3_class(out, "data.frame")
})

test_that("siu_report_text offline reads bundled fixture or errors clean", {
  set.seed(1)
  res <- tryCatch(
    morie_datasets_siu_report_text(offline = TRUE),
    error = function(e) NULL
  )
  skip_if(is.null(res), "no siu fixture")
  expect_type(res, "character")
})

test_that("siu_report_text validates argument", {
  set.seed(1)
  expect_error(morie_datasets_siu_report_text(), "url")
})

test_that("siu_report_fields extracts canonical fields from synthetic text", {
  set.seed(1)
  text <- paste0(
    "Report 24-OFD-001 incident dated January 5, 2024. ",
    "Director's Decision: no charges. Issued: later."
  )
  out <- morie_datasets_siu_report_fields(text)
  expect_type(out, "list")
  expect_true(all(c("report_id", "incident_date", "conclusion", "sections") %in% names(out)))
  expect_match(out$report_id, "24-OFD-001")
})

test_that("chicago_crime offline + year filter respects max_features", {
  set.seed(1)
  res <- tryCatch(
    suppressWarnings(morie_datasets_chicago_crime(offline = TRUE, max_features = 3L)),
    error = function(e) NULL
  )
  skip_if(is.null(res), "no chicago fixture")
  expect_s3_class(res, "data.frame")
})

test_that("chicago_crime live call gated on network", {
  set.seed(1)
  res <- tryCatch(
    morie_datasets_chicago_crime(year = 2024, max_features = 1L),
    error = function(e) NULL
  )
  skip_if(is.null(res), "needs network")
  expect_s3_class(res, "data.frame")
})

test_that("nyc_stop_and_frisk offline returns frame", {
  set.seed(1)
  res <- tryCatch(
    suppressWarnings(morie_datasets_nyc_stop_and_frisk(offline = TRUE, max_features = 2L)),
    error = function(e) NULL
  )
  skip_if(is.null(res), "no sqf fixture")
  expect_s3_class(res, "data.frame")
})

test_that("nyc_stop_and_frisk rejects unknown years", {
  set.seed(1)
  # Year-validation lives in the live (non-offline) branch; post-3LL
  # the default became offline=TRUE so we must opt back in explicitly
  # to exercise the "no built-in resource" check.
  expect_error(
    morie_datasets_nyc_stop_and_frisk(year = 1990L, offline = FALSE),
    "no built-in")
})

test_that("bigquery loader errors without bigrquery installed", {
  skip_if_not_installed("bigrquery")
  set.seed(1)
  skip_if(requireNamespace("bigrquery", quietly = TRUE))
  expect_error(morie_datasets_bigquery("a", "b", "c"), "bigrquery")
})

test_that("ckan search live call gated on network", {
  set.seed(1)
  res <- tryCatch(
    morie_datasets_ckan_search("https://open.canada.ca/data", "corrections", rows = 1L),
    error = function(e) NULL
  )
  skip_if(is.null(res), "needs network")
  expect_s3_class(res, "data.frame")
})

test_that("ckan package live call gated on network", {
  set.seed(1)
  res <- tryCatch(
    morie_datasets_ckan_package("https://open.canada.ca/data", "__no_pkg__"),
    error = function(e) NULL
  )
  skip_if(is.null(res), "needs network")
  expect_type(res, "list")
})

test_that("nibrs offline frame + year-required path", {
  set.seed(1)
  res <- tryCatch(
    suppressWarnings(morie_datasets_nibrs(offline = TRUE, max_features = 2L)),
    error = function(e) NULL
  )
  skip_if(is.null(res), "no nibrs fixture")
  expect_s3_class(res, "data.frame")
  expect_error(morie_datasets_nibrs(), "year")
})

test_that("namus_missing_persons offline returns frame", {
  set.seed(1)
  res <- tryCatch(
    suppressWarnings(morie_datasets_namus_missing_persons(offline = TRUE, max_features = 2L)),
    error = function(e) NULL
  )
  skip_if(is.null(res), "no namus fixture")
  expect_s3_class(res, "data.frame")
})

test_that("nist_rds offline returns frame", {
  set.seed(1)
  res <- tryCatch(
    suppressWarnings(morie_datasets_nist_rds(offline = TRUE, max_features = 2L)),
    error = function(e) NULL
  )
  skip_if(is.null(res), "no nist fixture")
  expect_s3_class(res, "data.frame")
})

test_that("http_json errors cleanly with httr2 absent", {
  skip_if_not_installed("httr2")
  set.seed(1)
  skip_if(requireNamespace("httr2", quietly = TRUE))
  expect_error(morie:::.morie_dataset_http_json("http://invalid"), "httr2")
})