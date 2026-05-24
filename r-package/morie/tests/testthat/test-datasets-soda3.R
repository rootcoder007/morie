# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3QQ: SODA3 SoQL-query helper + Chicago Crimes Map view
# (ahwe-kpsy, SODA3-only because SODA2 returns [{}] empty rows on
# Socrata map/filtered views) + arbitrary-SoQL escape hatch for
# the base ijzp-q8t2 / crimes feed.

# =================================================== (1) SODA3 helper core

test_that(".morie_dataset_soda3_query(paginate=FALSE) hits SODA3 query.json with the SoQL string in ?query=", {
  calls <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_records_to_df = function(records) {
      calls[[length(calls) + 1L]] <<- records
      data.frame(stub = seq_along(records))
    },
    .package = "morie")
  # Mock httr2 layer to capture the URL.
  with_mocked_httr2_get <- function(handler) {
    if (!requireNamespace("httr2", quietly = TRUE)) skip("httr2")
    mockery_ish <- NULL
    capture <- new.env()
    capture$urls <- character()
    capture$headers <- list()
    fake_req_perform <- function(req) {
      url <- req$url
      capture$urls <- c(capture$urls, url)
      capture$headers[[length(capture$headers) + 1L]] <- req$headers
      structure(list(body_json = handler(url)),
                class = "httr2_response")
    }
    fake_resp_body_json <- function(resp, simplifyVector = TRUE) {
      resp$body_json
    }
    testthat::with_mocked_bindings(
      req_perform = fake_req_perform,
      resp_body_json = fake_resp_body_json,
      .package = "httr2",
      code = {
        out <- morie:::.morie_dataset_soda3_query(
          "ahwe-kpsy",
          soql = "SELECT * WHERE primary_type='THEFT' LIMIT 5",
          paginate = FALSE)
        list(out = out, capture = capture)
      })
  }
  result <- with_mocked_httr2_get(function(url) {
    list(list(id = "1"), list(id = "2"))
  })
  expect_equal(length(result$capture$urls), 1L)
  u <- result$capture$urls[1L]
  expect_match(u, "data\\.cityofchicago\\.org/api/v3/views/ahwe-kpsy/query\\.json")
  expect_match(u, "query=SELECT")
  expect_match(u, "primary_type")
})

test_that(".morie_dataset_soda3_query(paginate=TRUE) walks LIMIT n OFFSET m baked into the SoQL string", {
  capture <- new.env()
  capture$soqls <- character()
  # Mock the inner records-to-df function so we can pretend each
  # request returns a deterministic number of rows.
  responses <- list(
    list(list(id = "1"), list(id = "2"), list(id = "3")),  # page 0: 3 rows
    list(list(id = "4"), list(id = "5")),                  # page 1: 2 rows (short -> exhausted)
    list())                                                # never reached
  call_count <- 0L
  testthat::local_mocked_bindings(
    .morie_dataset_records_to_df = function(records) {
      if (length(records) == 0L) return(data.frame())
      as.data.frame(do.call(rbind, lapply(records, as.data.frame,
                                            stringsAsFactors = FALSE)))
    },
    .package = "morie")
  fake_req_perform <- function(req) {
    capture$soqls <- c(capture$soqls,
                        sub(".*query=([^&]*).*", "\\1", req$url))
    call_count <<- call_count + 1L
    body <- responses[[call_count]]
    structure(list(body_json = body), class = "httr2_response")
  }
  fake_resp_body_json <- function(resp, simplifyVector = TRUE) resp$body_json
  testthat::with_mocked_bindings(
    req_perform = fake_req_perform,
    resp_body_json = fake_resp_body_json,
    .package = "httr2",
    code = {
      out <- morie:::.morie_dataset_soda3_query(
        "ahwe-kpsy",
        soql = "SELECT * WHERE year=2024",
        paginate = TRUE, page_size = 3L)
      # 2 requests -- short 2nd page is the stop signal.
      expect_equal(length(capture$soqls), 2L)
      # First page: LIMIT 3 OFFSET 0.
      expect_match(URLdecode(capture$soqls[1L]),
                   "LIMIT 3 OFFSET 0$")
      # Second page: LIMIT 3 OFFSET 3.
      expect_match(URLdecode(capture$soqls[2L]),
                   "LIMIT 3 OFFSET 3$")
      # Caller's WHERE is preserved on every page.
      expect_true(all(grepl("year=2024", URLdecode(capture$soqls))))
      expect_equal(nrow(out), 5L)
    })
})

test_that(".morie_dataset_soda3_query(paginate=TRUE) strips caller-supplied LIMIT/OFFSET", {
  capture <- new.env()
  capture$soqls <- character()
  count <- 0L
  testthat::local_mocked_bindings(
    .morie_dataset_records_to_df = function(records) data.frame(),
    .package = "morie")
  fake_perf <- function(req) {
    capture$soqls <- c(capture$soqls,
                        URLdecode(sub(".*query=([^&]*).*", "\\1",
                                       req$url)))
    count <<- count + 1L
    structure(list(body_json = list()), class = "httr2_response")
  }
  testthat::with_mocked_bindings(
    req_perform = fake_perf,
    resp_body_json = function(resp, simplifyVector = TRUE) resp$body_json,
    .package = "httr2",
    code = {
      morie:::.morie_dataset_soda3_query(
        "ahwe-kpsy",
        soql = "SELECT * WHERE year=2024 LIMIT 999 OFFSET 50",
        paginate = TRUE, page_size = 100L)
    })
  expect_equal(length(capture$soqls), 1L)
  # Caller's LIMIT 999 OFFSET 50 stripped; replaced by LIMIT 100 OFFSET 0.
  expect_match(capture$soqls[1L],
               "WHERE year=2024 LIMIT 100 OFFSET 0$")
  expect_false(grepl("LIMIT 999", capture$soqls[1L]))
})

test_that(".morie_dataset_soda3_query sends app_token as X-App-Token header (not query param)", {
  capture <- new.env()
  capture$headers <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_records_to_df = function(records) data.frame(),
    .package = "morie")
  fake_perf <- function(req) {
    capture$headers[[length(capture$headers) + 1L]] <- req$headers
    structure(list(body_json = list()), class = "httr2_response")
  }
  testthat::with_mocked_bindings(
    req_perform = fake_perf,
    resp_body_json = function(resp, simplifyVector = TRUE) resp$body_json,
    .package = "httr2",
    code = {
      morie:::.morie_dataset_soda3_query(
        "ahwe-kpsy", soql = "SELECT * LIMIT 1",
        app_token = "fake-token-abc")
    })
  expect_equal(length(capture$headers), 1L)
  expect_equal(capture$headers[[1L]][["X-App-Token"]], "fake-token-abc")
})

# =================================== (2) chicago_crime_map (ahwe-kpsy)

test_that("morie_datasets_chicago_crime_map(offline=TRUE) reads bundled 39-col fixture", {
  df <- morie_datasets_chicago_crime_map(offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_equal(ncol(df), 39L)
  expect_equal(nrow(df), 5L)
  # Spot-check that we have the 4 reverse-geocoded extras and the 4
  # Socrata-metadata cols and at least one :@computed_region_*.
  for (col in c("location_address", "location_city",
                "location_state", "location_zip",
                ":id", ":version", ":created_at", ":updated_at"))
    expect_true(col %in% names(df))
  expect_true(any(grepl("^:@computed_region_", names(df))))
  expect_true(all(grepl("^JZ-MAP-SYNTH-", df$case_number)))
})

test_that("morie_datasets_chicago_crime_map(offline=TRUE) honours max_features", {
  df <- morie_datasets_chicago_crime_map(offline = TRUE,
                                           max_features = 2L)
  expect_equal(nrow(df), 2L)
})

test_that("morie_datasets_chicago_crime_map(offline=FALSE) routes to .morie_dataset_soda3_query with date-windowed SoQL", {
  seen <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_soda3_query = function(view_id, soql = "SELECT *",
                                            app_token = NULL,
                                            paginate = FALSE,
                                            page_size = 1000L,
                                            max_pages = 200L,
                                            max_features = NULL,
                                            base_url = "https://data.cityofchicago.org") {
      seen <<- list(view_id = view_id, soql = soql,
                    paginate = paginate, page_size = page_size,
                    max_features = max_features)
      data.frame(stub = 1L)
    },
    .package = "morie")
  out <- morie_datasets_chicago_crime_map(
    date_from = "2025-05-17",
    date_to = "2026-05-24",
    offline = FALSE)
  expect_equal(seen$view_id, "ahwe-kpsy")
  expect_match(seen$soql,
               "SELECT \\* WHERE `date` >= '2025-05-17' AND `date` < '2026-05-24'")
  expect_s3_class(out, "data.frame")
})

test_that("morie_datasets_chicago_crime_map(offline=FALSE) ANDs an extra `where` onto the date window", {
  seen <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_soda3_query = function(view_id, soql = "SELECT *",
                                            app_token = NULL,
                                            paginate = FALSE,
                                            page_size = 1000L,
                                            max_pages = 200L,
                                            max_features = NULL,
                                            base_url = "https://data.cityofchicago.org") {
      seen <<- list(soql = soql)
      data.frame(stub = 1L)
    },
    .package = "morie")
  morie_datasets_chicago_crime_map(
    date_from = "2025-01-01",
    date_to = "2026-01-01",
    where = "primary_type='HOMICIDE'",
    offline = FALSE)
  expect_match(seen$soql,
               "WHERE `date` >= '2025-01-01' AND `date` < '2026-01-01' AND primary_type='HOMICIDE'$")
})

test_that("morie_datasets_chicago_crime_map default date window is rolling 1 year ending today", {
  seen <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_soda3_query = function(view_id, soql = "SELECT *",
                                            app_token = NULL,
                                            paginate = FALSE,
                                            page_size = 1000L,
                                            max_pages = 200L,
                                            max_features = NULL,
                                            base_url = "https://data.cityofchicago.org") {
      seen <<- list(soql = soql)
      data.frame()
    },
    .package = "morie")
  morie_datasets_chicago_crime_map(offline = FALSE)
  today <- format(Sys.Date(), "%Y-%m-%dT00:00:00.000")
  year_ago <- format(Sys.Date() - 365L, "%Y-%m-%dT00:00:00.000")
  expect_match(seen$soql, sprintf("`date` >= '%s'", year_ago), fixed = TRUE)
  expect_match(seen$soql, sprintf("`date` < '%s'", today), fixed = TRUE)
})

test_that("morie_datasets_chicago_crime_map forwards resource_id + paginate args", {
  seen <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_soda3_query = function(view_id, soql = "SELECT *",
                                            app_token = NULL,
                                            paginate = FALSE,
                                            page_size = 1000L,
                                            max_pages = 200L,
                                            max_features = NULL,
                                            base_url = "https://data.cityofchicago.org") {
      seen <<- list(view_id = view_id, paginate = paginate,
                    page_size = page_size, max_pages = max_pages,
                    app_token = app_token)
      data.frame()
    },
    .package = "morie")
  morie_datasets_chicago_crime_map(
    offline = FALSE,
    resource_id = "override-map-xyz",
    paginate = TRUE,
    page_size = 5000L,
    max_pages = 10L,
    app_token = "tok-xyz")
  expect_equal(seen$view_id, "override-map-xyz")
  expect_true(isTRUE(seen$paginate))
  expect_equal(seen$page_size, 5000L)
  expect_equal(seen$max_pages, 10L)
  expect_equal(seen$app_token, "tok-xyz")
})

# ============================ (3) chicago_crime_soql (arbitrary WHERE)

test_that("morie_datasets_chicago_crime_soql(offline=TRUE) returns the synthetic 22-col chicago_crime fixture", {
  df <- suppressWarnings(morie_datasets_chicago_crime_soql(offline = TRUE))
  expect_s3_class(df, "data.frame")
  expect_equal(ncol(df), 22L)
  expect_true("location" %in% names(df))
})

test_that("morie_datasets_chicago_crime_soql(offline=FALSE) routes to .morie_dataset_soda3_query for ijzp-q8t2 by default", {
  seen <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_soda3_query = function(view_id, soql = "SELECT *",
                                            app_token = NULL,
                                            paginate = FALSE,
                                            page_size = 1000L,
                                            max_pages = 200L,
                                            max_features = NULL,
                                            base_url = "https://data.cityofchicago.org") {
      seen <<- list(view_id = view_id, soql = soql)
      data.frame(id = 1L)
    },
    .package = "morie")
  morie_datasets_chicago_crime_soql(
    where = "primary_type='HOMICIDE' AND year=2024",
    offline = FALSE)
  expect_equal(seen$view_id, "ijzp-q8t2")
  expect_match(seen$soql,
               "^SELECT \\* WHERE primary_type='HOMICIDE' AND year=2024$")
})

test_that("morie_datasets_chicago_crime_soql honours select + order", {
  seen <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_soda3_query = function(view_id, soql = "SELECT *",
                                            app_token = NULL,
                                            paginate = FALSE,
                                            page_size = 1000L,
                                            max_pages = 200L,
                                            max_features = NULL,
                                            base_url = "https://data.cityofchicago.org") {
      seen <<- list(soql = soql)
      data.frame()
    },
    .package = "morie")
  morie_datasets_chicago_crime_soql(
    select = "primary_type, count(*) as n",
    where = "year=2024",
    order = "n DESC",
    offline = FALSE)
  expect_equal(seen$soql,
               "SELECT primary_type, count(*) as n WHERE year=2024 ORDER BY n DESC")
})

test_that("morie_datasets_chicago_crime_soql resource_id override sends to the alias", {
  seen <- list()
  testthat::local_mocked_bindings(
    .morie_dataset_soda3_query = function(view_id, soql = "SELECT *",
                                            app_token = NULL,
                                            paginate = FALSE,
                                            page_size = 1000L,
                                            max_pages = 200L,
                                            max_features = NULL,
                                            base_url = "https://data.cityofchicago.org") {
      seen <<- list(view_id = view_id)
      data.frame()
    },
    .package = "morie")
  morie_datasets_chicago_crime_soql(offline = FALSE,
                                      resource_id = "crimes")
  expect_equal(seen$view_id, "crimes")
})

# ============================ (4) default-offline behaviour

test_that("both 3QQ live wrappers default to offline = TRUE", {
  expect_s3_class(morie_datasets_chicago_crime_map(),
                  "data.frame")
  expect_s3_class(suppressWarnings(morie_datasets_chicago_crime_soql()),
                  "data.frame")
})
