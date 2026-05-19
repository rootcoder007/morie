test_that("list_morie_modules exposes implemented module names", {
  mods <- list_morie_modules()
  expect_true(all(c("power-design", "propensity-scores", "ebac-selection-adjustment-ipw") %in% mods$name))
})

test_that("morie_fetch_ckan pulls CPADS PUMF from the open.canada.ca datastore", {
  # CPADS 2021-2022 PUMF is a public open-data release queried through
  # the CKAN datastore_search API. Network-dependent, so skipped on CRAN
  # and offline machines per policy; runs wherever the API is reachable.
  skip_on_cran()
  testthat::skip_if_offline("open.canada.ca")
  dat <- tryCatch(
    morie_fetch_ckan(dataset_key = "cpads", limit = 1000L),
    error = function(e) NULL)
  skip_if(is.null(dat), "CKAN datastore_search API unreachable")
  expect_s3_class(dat, "data.frame")
  expect_true(nrow(dat) > 0)
  expect_false("_id" %in% names(dat))
  expect_true("SEQID" %in% names(dat))
})

test_that("morie_list_datasets shows all catalog entries", {
  skip_if_not(requireNamespace("DBI", quietly = TRUE), "DBI not installed")
  skip_if_not(requireNamespace("RSQLite", quietly = TRUE), "RSQLite not installed")
  ds <- morie_list_datasets()
  expect_true(nrow(ds) >= 20)
  expect_true("ocp21" %in% ds$key)
})

test_that("dataset catalog has expected structure", {
  cat <- morie_dataset_catalog()
  expect_true(nrow(cat) >= 20)
  expect_true(all(c("key", "name", "source", "survey", "table_name") %in% names(cat)))
})
