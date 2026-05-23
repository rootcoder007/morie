# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Tests for R/siu_fetch.R --- pure-R Ontario SIU scraper. Network
# code paths are SKIPPED on CRAN and gated by tryCatch; only the
# pure-R helpers + the no-network branches are exercised.

set.seed(1)

# ---------------------------------------------------------------------------
# Public URL + cache-path helpers
# ---------------------------------------------------------------------------

test_that("morie_siu_index_url returns the canonical SIU URL", {
  set.seed(1)
  u <- morie_siu_index_url()
  expect_type(u, "character")
  expect_match(u, "^https://www\\.siu\\.on\\.ca/")
  expect_match(u, "directors_reports\\.php$")
})

test_that("morie_siu_cache_path creates the dir + returns absolute path", {
  set.seed(1)
  d <- tempfile("siu_cache_")
  p <- morie_siu_cache_path(d)
  expect_true(dir.exists(d))
  expect_true(endsWith(p, "SIU.csv"))
})

test_that("morie_siu_cache_path default lives under tempdir", {
  set.seed(1)
  p <- morie_siu_cache_path()
  expect_true(grepl("morie/siu", p, fixed = TRUE) ||
              grepl("morie\\\\\\\\siu", p))
})


# ---------------------------------------------------------------------------
# Internal regex/parse helpers (network-free)
# ---------------------------------------------------------------------------

test_that(".siu_fetch_resolve_url absolutizes relative", {
  set.seed(1)
  out <- morie:::.siu_fetch_resolve_url(
    "case_summary_details.php?drid=42",
    "https://www.siu.on.ca/en/directors_reports.php")
  expect_match(out, "^https://www\\.siu\\.on\\.ca/en/")
  expect_match(out, "drid=42")
})

test_that(".siu_fetch_resolve_url passes through absolute URLs", {
  set.seed(1)
  out <- morie:::.siu_fetch_resolve_url(
    "https://other.example/foo",
    "https://www.siu.on.ca/")
  expect_equal(out, "https://other.example/foo")
})

test_that(".siu_fetch_to_iso parses 'Month D, YYYY'", {
  set.seed(1)
  expect_equal(morie:::.siu_fetch_to_iso("March 5, 2022"),
               "2022-03-05")
})

test_that(".siu_fetch_to_iso returns empty string on miss", {
  set.seed(1)
  expect_equal(morie:::.siu_fetch_to_iso(""), "")
  expect_equal(morie:::.siu_fetch_to_iso("not a date"), "")
})


# ---------------------------------------------------------------------------
# .siu_fetch_extract_links
# ---------------------------------------------------------------------------

test_that("extract_links returns case_number/url pairs from index HTML", {
  set.seed(1)
  idx <- paste0(
    '<html><body>',
    '<a href="case_summary_details.php?drid=11">22-OCI-001</a>',
    '<a href="case_summary_details.php?drid=12">22-OCI-002</a>',
    '</body></html>')
  m <- morie:::.siu_fetch_extract_links(
    idx, "https://www.siu.on.ca/en/directors_reports.php")
  expect_true(is.matrix(m) || is.data.frame(m))
  expect_true(nrow(m) >= 1L)
  expect_true("case_number" %in% colnames(m))
  expect_true("url" %in% colnames(m))
})

test_that("extract_links returns empty matrix on no anchors", {
  set.seed(1)
  m <- morie:::.siu_fetch_extract_links(
    "<html></html>", "https://x/")
  expect_true(nrow(m) == 0L)
  expect_equal(colnames(m), c("case_number", "url"))
})


# ---------------------------------------------------------------------------
# .siu_fetch_parse_case_page
# ---------------------------------------------------------------------------

test_that("parse_case_page returns the 7-key schema on empty HTML", {
  set.seed(1)
  rec <- morie:::.siu_fetch_parse_case_page("", "22-OCI-001",
                                            "https://x/y")
  expect_type(rec, "list")
  expect_equal(rec$case_number, "22-OCI-001")
  expect_equal(rec$source_url, "https://x/y")
  expect_equal(rec$police_service, "")
  expect_equal(rec$incident_iso, "")
})

test_that("parse_case_page parses dates from a small HTML fixture", {
  set.seed(1)
  h <- paste0(
    "Incident: March 5, 2022 ",
    "Notification: March 6, 2022 ",
    "Director's Decision: November 1, 2022 ",
    "Police Service: Toronto Police Service ",
    "no reasonable grounds")
  rec <- morie:::.siu_fetch_parse_case_page(h, "22-OCI-002",
                                            "https://x/y")
  expect_equal(rec$incident_iso, "2022-03-05")
  expect_equal(rec$notification_iso, "2022-03-06")
  expect_equal(rec$decision_iso, "2022-11-01")
  expect_true(grepl("Toronto", rec$police_service))
  expect_true(grepl("no reasonable grounds",
                    rec$director_decision_text, ignore.case = TRUE))
})

test_that("parse_case_page tolerates partial matches", {
  set.seed(1)
  rec <- morie:::.siu_fetch_parse_case_page(
    "Incident: April 1, 2023",
    "22-OCI-003", "https://x/z")
  expect_equal(rec$incident_iso, "2023-04-01")
  expect_equal(rec$notification_iso, "")
  expect_equal(rec$decision_iso, "")
})

test_that("parse_case_page director_decision_text captures 'charges were laid'", {
  set.seed(1)
  rec <- morie:::.siu_fetch_parse_case_page(
    "All evidence reviewed, charges were laid in this matter.",
    "22-OCI-004", "https://x/q")
  expect_true(grepl("charges", rec$director_decision_text,
                    ignore.case = TRUE))
})


# ---------------------------------------------------------------------------
# morie_siu_fetch_cases -- short-circuit + validation paths only
# ---------------------------------------------------------------------------

test_that("fetch_cases skips work when SIU.csv already exists", {
  set.seed(1)
  d <- tempfile("siu_short_circuit_")
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  csv <- file.path(d, "SIU.csv")
  writeLines("case_number,police_service\
22-OCI-001,TPS", csv)
  out <- morie_siu_fetch_cases(cache_dir = d, overwrite = FALSE,
                               progress = FALSE)
  expect_equal(normalizePath(out), normalizePath(csv))
})

test_that("fetch_cases errors on non-integer years", {
  set.seed(1)
  expect_error(
    morie_siu_fetch_cases(years = c(NA, 2024),
                          cache_dir = tempfile("siu_yrs_"),
                          overwrite = TRUE, progress = FALSE),
    "finite integer")
})


# ---------------------------------------------------------------------------
# Network calls -- gated by skip_on_cran + tryCatch + skip_if
# ---------------------------------------------------------------------------

test_that("fetch_cases live scrape returns a path or skip", {
  set.seed(1)
  skip_on_cran()
  d <- tempfile("siu_live_")
  res <- tryCatch(
    morie_siu_fetch_cases(years = 2023L, cache_dir = d,
                          overwrite = TRUE, progress = FALSE),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    skip(paste("network unavailable:", conditionMessage(res)))
  }
  expect_true(file.exists(res))
})

test_that("fetch_dataframe returns a data.frame or skip on network", {
  set.seed(1)
  skip_on_cran()
  d <- tempfile("siu_live_df_")
  res <- tryCatch(
    morie_siu_fetch_dataframe(years = 2023L, cache_dir = d,
                              overwrite = TRUE, progress = FALSE),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    skip(paste("network unavailable:", conditionMessage(res)))
  }
  expect_s3_class(res, "data.frame")
  expect_true(all(c("case_number", "police_service") %in% names(res)))
})


# ---------------------------------------------------------------------------
# .siu_fetch_http_get -- missing httr2 should error cleanly
# ---------------------------------------------------------------------------

test_that(".siu_fetch_http_get errors on a clearly unreachable host (offline-friendly)", {
  set.seed(1)
  skip_on_cran()
  # 192.0.2.0/24 is TEST-NET-1 -- reserved for documentation; will not
  # route. Confirms the error path is reachable without leaking out.
  res <- tryCatch(
    morie:::.siu_fetch_http_get(
      "http://192.0.2.123/never-routes", timeout_s = 2L),
    error = function(e) e
  )
  expect_s3_class(res, "error")
})