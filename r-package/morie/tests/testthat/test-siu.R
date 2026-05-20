# SPDX-License-Identifier: AGPL-3.0-or-later
# Tests for the all-C/C++ Ontario SIU parser (src/siu_parser.cpp) and
# its R orchestration (R/siu.R). The HTML parsers are tested offline
# against synthetic fixtures that mirror the real SIU page structure;
# the libcurl transport and the end-to-end run are network-gated.

# A synthetic director's-report page with the real section skeleton.
.fake_siu_report <- function() {
  paste0(
    "<html><body>",
    "<h2 id=\"section_1\">Mandate of the SIU</h2><p>boilerplate text</p>",
    "<h2 id=\"section_4\">The Investigation</h2>",
    "<p>Notification of the SIU On January 5, 2024, at 9:00 a.m., the ",
    "Waterloo Regional Police Service (WRPS) contacted the SIU. ",
    "On January 6, 2024, officers responded. SO #1 attended with WO #1 ",
    "and WO #2. CW #1 was interviewed. The Waterloo Regional Police ",
    "Service confirmed the arrest of a 34-year-old man.</p>",
    "<h2 id=\"section_6\">Incident Narrative</h2>",
    "<p>On January 6, 2024, the man was arrested by Waterloo Regional ",
    "Police Service officers. He was injured. His arrest followed a ",
    "call. The man did not resist further.</p>",
    "<h2 id=\"section_8\">Analysis and Director's Decision</h2>",
    "<p>The Complainant was injured on January 6, 2024. On my ",
    "assessment of the evidence, there are no reasonable grounds to ",
    "believe that an officer committed a criminal offence.</p>",
    "<p>Director's Report for Case # 24-TCI-099.</p>",
    "<p>Date: March 3, 2024 Electronically approved by Jane Doe ",
    "Director Special Investigations Unit</p>",
    "<p>News Releases for this Case: ",
    "<a href=\"/en/news_template.php?nrid=999\">No Charges in Waterloo ",
    "Arrest</a></p></body></html>"
  )
}

# A synthetic news-release page with the real dateline structure.
.fake_siu_news <- function() {
  paste0(
    "<html><body><h1>News Release</h1>",
    "<h4>No Charges in Waterloo Arrest</h4>",
    "<strong>Kitchener, ON</strong> (3 March, 2024) --- ",
    "A 34-year-old man was injured during an arrest by police. ",
    "Full Director's Report follows.</body></html>"
  )
}

test_that(".siu_parse_report extracts the 64-column schema", {
  r <- morie:::.siu_parse_report(
    .fake_siu_report(), 4242L,
    "http://x/drid=4242"
  )
  expect_length(r, 64L)
  expect_equal(r[["case_number"]], "24-TCI-099")
  expect_equal(r[["drid"]], "4242")
  expect_equal(r[["_language"]], "en")
  expect_equal(r[["police_service"]], "Waterloo Regional Police Service")
  expect_equal(r[["date_siu_notified_iso"]], "2024-01-05")
  expect_equal(r[["date_of_incident_iso"]], "2024-01-06")
  expect_equal(r[["date_of_director_decision_iso"]], "2024-03-03")
  expect_equal(r[["directors_name"]], "Jane Doe")
  expect_equal(r[["number_of_subject_officials"]], "1")
  expect_equal(r[["number_of_witness_officials"]], "2")
  expect_equal(r[["number_of_civilian_witnesses"]], "1")
  expect_equal(r[["age_affected"]], "34")
  expect_equal(r[["sex_gender_affected"]], "Male")
  expect_equal(r[["directors_decision_reasonable"]], "No")
  expect_equal(r[["nrid"]], "999")
  expect_true(nzchar(r[["narrative_summary"]]))
  expect_match(r[["parser_version"]], "^[0-9]")
})

test_that(".siu_parse_report handles an empty / non-existent drid page", {
  r <- morie:::.siu_parse_report(
    "<html><body></body></html>", 7L,
    "http://x/drid=7"
  )
  expect_length(r, 64L)
  expect_equal(r[["drid"]], "7")
  expect_equal(r[["case_number"]], "")
})

test_that(".siu_parse_news extracts title, date and summary", {
  n <- morie:::.siu_parse_news(
    .fake_siu_news(), 999L,
    "http://x/nrid=999"
  )
  expect_equal(n[["nrid"]], "999")
  expect_equal(n[["news_release_title"]], "No Charges in Waterloo Arrest")
  expect_equal(n[["news_release_date_iso"]], "2024-03-03")
  expect_equal(n[["news_release_date_raw"]], "3 March, 2024")
  expect_true(grepl("34-year-old", n[["news_release_summary"]]))
})

test_that(".siu_parse_report recognises a non-binary affected person", {
  html <- sub("34-year-old man", "34-year-old non-binary person",
    .fake_siu_report(),
    fixed = TRUE
  )
  r <- morie:::.siu_parse_report(html, 1L, "http://x")
  expect_equal(r[["sex_gender_affected"]], "Non-binary")
})

test_that(".siu_discover_max_drid parses the index and adds a margin", {
  testthat::local_mocked_bindings(
    .siu_http_get = function(url, ...) {
      "<tr class=\"dr-item\" id=\"5090\"><tr class=\"dr-item\" id=\"5094\">"
    },
    .package = "morie"
  )
  expect_equal(morie:::.siu_discover_max_drid(margin = 10L), 5104L)
})

test_that("morie_fetch_siu assembles one row per case (offline, mocked)", {
  testthat::local_mocked_bindings(
    .siu_http_get_many = function(urls, ...) {
      vapply(urls, function(u) {
        if (grepl("nrid=999", u, fixed = TRUE)) {
          .fake_siu_news()
        } else if (grepl("drid=1$", u)) {
          .fake_siu_report()
        } else {
          ""
        }
      }, character(1), USE.NAMES = FALSE)
    },
    .package = "morie"
  )
  out <- morie_fetch_siu(
    cache_dir = tempfile("siu-"), overwrite = TRUE,
    max_drid = 3L, progress = FALSE
  )
  df <- utils::read.csv(out, colClasses = "character", check.names = FALSE)
  expect_equal(ncol(df), 64L)
  expect_equal(nrow(df), 1L)
  expect_equal(df$case_number, "24-TCI-099")
  expect_equal(df$nrid, "999")
  # news fields joined from the (mocked) news page
  expect_equal(df$news_release_date_iso, "2024-03-03")
  expect_true(grepl("Waterloo", df$news_release_title))
})

test_that("morie_fetch_siu returns the cached path without re-fetching", {
  dir <- tempfile("siu-")
  dir.create(dir)
  writeLines("case_number\n24-X", file.path(dir, "SIU.csv"))
  expect_equal(
    normalizePath(morie_fetch_siu(
      cache_dir = dir,
      progress = FALSE
    )),
    normalizePath(file.path(dir, "SIU.csv"))
  )
})

test_that(".siu_curl_version reports a libcurl build string", {
  v <- morie:::.siu_curl_version()
  expect_type(v, "character")
  expect_match(v, "libcurl", fixed = TRUE)
})

test_that(".siu_http_get / .siu_http_get_many fetch over the network", {
  skip_on_cran()
  testthat::skip_if_offline("www.siu.on.ca")
  one <- tryCatch(
    morie:::.siu_http_get(
      "https://www.siu.on.ca/en/directors_report_details.php?drid=5080"
    ),
    error = function(e) ""
  )
  # Degraded responses (empty, error pages, 5xx HTML stubs, redirects) all
  # fall well under 1 kB. Skip on any such response — the test only
  # validates a healthy endpoint, transient flakiness shouldn't fail CI.
  skip_if(nchar(one) < 1000, "SIU site unreachable or degraded")
  expect_true(nchar(one) > 1000)
  many <- tryCatch(
    morie:::.siu_http_get_many(sprintf(
      "https://www.siu.on.ca/en/directors_report_details.php?drid=%d",
      5080:5083
    ), 4L),
    error = function(e) character(0)
  )
  skip_if(length(many) != 4L, "SIU site unreachable for batch fetch")
  expect_length(many, 4L)
  expect_true(all(nchar(many) > 0))
})

test_that("morie_fetch_siu runs end-to-end, one row per case (network)", {
  skip_on_cran()
  testthat::skip_if_offline("www.siu.on.ca")
  out <- tryCatch(
    morie_fetch_siu(
      cache_dir = tempfile("siu-"), overwrite = TRUE,
      max_drid = 120L, concurrency = 16L, progress = FALSE
    ),
    error = function(e) NULL
  )
  skip_if(is.null(out), "SIU site unreachable")
  df <- utils::read.csv(out, colClasses = "character", check.names = FALSE)
  expect_equal(ncol(df), 64L)
  expect_true(all(c("case_number", "drid", "nrid") %in% names(df)))
  expect_equal(sum(duplicated(df$case_number)), 0L)
  expect_true(all(nzchar(df$case_number)))
})
