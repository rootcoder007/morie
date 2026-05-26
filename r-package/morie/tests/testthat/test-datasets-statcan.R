# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3DDD3: StatCan CCJS WDS REST API loaders.

test_that("morie_datasets_statcan_ccjs_cubes returns 10-row registry", {
  c <- morie_datasets_statcan_ccjs_cubes()
  expect_s3_class(c, "data.frame")
  expect_equal(nrow(c), 10L)
  expect_setequal(names(c),
                  c("product_id", "cube_title_en",
                    "dimensions", "frequency"))
  expect_type(c$product_id, "integer")
  # All 10 product_ids are 8-digit catalogue IDs in the 35xxxxxx range
  # (CCJS subject matter code 35).
  expect_true(all(c$product_id > 35000000L &
                     c$product_id < 36000000L))
})

test_that("registered cubes include the canonical CCJS flagships", {
  c <- morie_datasets_statcan_ccjs_cubes()
  expect_true(35100177L %in% c$product_id)  # Incident-based
  expect_true(35100068L %in% c$product_id)  # Police personnel
  expect_true(35100026L %in% c$product_id)  # Homicide victims
})

test_that("morie_datasets_statcan_cube_metadata parses WDS getCubeMetadata response", {
  # Mocked, not live: StatCan WDS returned 200 OK with empty body on
  # 2026-05-26 (post-3VV libcurl backend reports this honestly as
  # "libcurl returned empty body"), and `skip_if_offline()` only
  # probes DNS/ICMP, not actual API content. Verify our parser
  # handles a canonical SUCCESS envelope instead of testing StatCan's
  # uptime.
  # WDS getCubeMetadata returns an outer ARRAY of result envelopes, not a
  # bare dict — the parser unwraps via `r[[1]]$status`. The mock must mirror
  # that shape (list of 1 inner list), not just the inner dict.
  testthat::local_mocked_bindings(
    .morie_dataset_http_post_json = function(url, body, timeout_s = 30L) {
      list(
        list(
          status = "SUCCESS",
          object = list(
            productId = 35100002L,
            cubeTitleEn = "Selected police-reported cybercrime violations",
            cubeTitleFr = "Affaires de cybercriminalite declarees par la police",
            frequencyCode = 12L
          )
        )
      )
    },
    .package = "morie"
  )
  m <- morie_datasets_statcan_cube_metadata(35100002L)
  expect_equal(m$status, "SUCCESS")
  expect_true(!is.null(m$object$cubeTitleEn))
  expect_true(grepl("cybercrime|Cybercrime",
                     m$object$cubeTitleEn))
})

test_that("morie_datasets_statcan_full_csv_url returns canonical zip URL", {
  # Mocked, same reason as the cube_metadata test above. WDS
  # getFullTableDownloadCSV returns a SUCCESS envelope whose $object
  # is the canonical .zip URL string.
  testthat::local_mocked_bindings(
    .morie_dataset_http_json = function(url) {
      list(
        status = "SUCCESS",
        object = paste0(
          "https://www150.statcan.gc.ca/n1/tbl/csv/",
          "35100002-eng.zip"
        )
      )
    },
    .package = "morie"
  )
  u <- morie_datasets_statcan_full_csv_url(35100002L)
  expect_match(u, "^https://www150\\.statcan\\.gc\\.ca/")
  expect_match(u, "35100002-eng\\.zip$")
})

test_that("morie_datasets_statcan_full_csv_url respects language=fr", {
  testthat::local_mocked_bindings(
    .morie_dataset_http_json = function(url) {
      list(
        status = "SUCCESS",
        object = paste0(
          "https://www150.statcan.gc.ca/n1/tbl/csv/",
          "35100002-fra.zip"
        )
      )
    },
    .package = "morie"
  )
  u <- morie_datasets_statcan_full_csv_url(35100002L, language = "fr")
  expect_match(u, "fra\\.zip$")
})
