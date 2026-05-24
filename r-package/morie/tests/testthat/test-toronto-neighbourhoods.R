# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3DD: Toronto neighbourhood boundary-version awareness.
#
# Covers R/toronto_neighbourhoods.R end-to-end:
#   * Offline-mode bundled-fixture loaders for 158 / 140 / NIA.
#   * Mocked CKAN datastore_search live-mode dispatch.
#   * HOOD column resolution (158 / 140) with fallback + warnings.
#   * Schema assertions + double-schema warning.
#   * Year-to-version mapping.
#   * 158<->140 crosswalk fixture loader.

# ===================================================== morie_to_neighbourhoods()

test_that("morie_to_neighbourhoods('158', offline=TRUE) returns canonical OT schema", {
  df <- morie_to_neighbourhoods("158", offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_true(nrow(df) > 0L)
  for (col in c("AREA_ID", "AREA_SHORT_CODE", "AREA_NAME",
                "CLASSIFICATION", "OBJECTID", "geometry"))
    expect_true(col %in% names(df))
  # 158-scheme codes are 3 digits, e.g. "082" for Niagara.
  expect_true("Niagara" %in% df$AREA_NAME)
})

test_that("morie_to_neighbourhoods('140', offline=TRUE) returns historical schema", {
  df <- morie_to_neighbourhoods("140", offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_true(nrow(df) > 0L)
  # Historical names carry the "(NN)" suffix per City convention.
  expect_true(any(grepl("\\(\\d+\\)$", df$AREA_NAME)))
})

test_that("morie_to_neighbourhoods('nia', offline=TRUE) returns NIA schema with DATE_EFFECTIVE", {
  df <- morie_to_neighbourhoods("nia", offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_true(nrow(df) > 0L)
  for (col in c("DATE_EFFECTIVE", "DATE_EXPIRY", "AREA_TYPE",
                "AREA_NAME"))
    expect_true(col %in% names(df))
})

test_that("morie_to_neighbourhoods errors on unknown version arg", {
  # match.arg returns the friendly error.
  expect_error(morie_to_neighbourhoods("xyz", offline = TRUE),
               regexp = "should be one of")
})

test_that("morie_to_neighbourhoods(offline=FALSE) dispatches via mocked CKAN helper", {
  # Synthetic CKAN-shaped response (matches the columns subset that
  # ckan0.cf.opendata...datastore_search returns).
  stub_df <- data.frame(
    `_id` = c(1L, 2L),
    AREA_ID = c(2502366L, 2502367L),
    AREA_SHORT_CODE = c("174", "082"),
    AREA_NAME = c("South Eglinton-Davisville", "Niagara"),
    CLASSIFICATION = c("Not an NIA or Emerging Neighbourhood",
                       "Not an NIA or Emerging Neighbourhood"),
    OBJECTID = c(17824737L, 17824738L),
    check.names = FALSE)
  testthat::local_mocked_bindings(
    .morie_to_ckan_dump_csv = function(resource_id, limit = 100000L) {
      # The default resource id for the 158-scheme should be passed.
      expect_match(resource_id,
                   "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")
      stub_df
    },
    .package = "morie")
  out <- morie_to_neighbourhoods("158", offline = FALSE)
  expect_s3_class(out, "data.frame")
  expect_equal(nrow(out), 2L)
  expect_equal(out$AREA_NAME[2], "Niagara")
})

test_that("morie_to_neighbourhoods(offline=FALSE, resource_id=X) honours override", {
  testthat::local_mocked_bindings(
    .morie_to_ckan_dump_csv = function(resource_id, limit = 100000L) {
      expect_equal(resource_id, "custom-resource-id-xyz")
      data.frame(AREA_NAME = "OverrideHit")
    },
    .package = "morie")
  out <- morie_to_neighbourhoods("nia", offline = FALSE,
                                   resource_id = "custom-resource-id-xyz")
  expect_equal(out$AREA_NAME, "OverrideHit")
})

# =================================================== morie_tps_resolve_hood_col()

test_that("morie_tps_resolve_hood_col picks HOOD_158 when both present + prefer=158", {
  df <- data.frame(HOOD_158 = "82", HOOD_140 = "82")
  expect_equal(morie_tps_resolve_hood_col(df, prefer = "158"),
               "HOOD_158")
  expect_equal(morie_tps_resolve_hood_col(df, prefer = "140"),
               "HOOD_140")
})

test_that("morie_tps_resolve_hood_col falls back with a warning", {
  df <- data.frame(HOOD_140 = "82")
  expect_warning(
    out <- morie_tps_resolve_hood_col(df, prefer = "158",
                                        fallback = TRUE),
    regexp = "falling back")
  expect_equal(out, "HOOD_140")
})

test_that("morie_tps_resolve_hood_col returns NULL without fallback", {
  df <- data.frame(HOOD_140 = "82")
  expect_warning(
    out <- morie_tps_resolve_hood_col(df, prefer = "158",
                                        fallback = FALSE),
    regexp = "no HOOD_158")
  expect_null(out)
})

test_that("morie_tps_resolve_hood_col accepts lowercase column variants", {
  df <- data.frame(hood_158 = "82")
  expect_equal(morie_tps_resolve_hood_col(df, prefer = "158"),
               "hood_158")
})

# ================================================ morie_tps_assert_hood_version()

test_that("morie_tps_assert_hood_version returns TRUE invisibly when expected schema is present", {
  df <- data.frame(HOOD_158 = "82")
  expect_true(morie_tps_assert_hood_version(df, expected = "158"))
})

test_that("morie_tps_assert_hood_version errors when expected schema is absent", {
  df <- data.frame(HOOD_140 = "82")
  expect_error(morie_tps_assert_hood_version(df, expected = "158"),
               regexp = "expected HOOD_158")
})

test_that("morie_tps_assert_hood_version warns when BOTH schemas present", {
  df <- data.frame(HOOD_158 = "82", HOOD_140 = "82")
  expect_warning(morie_tps_assert_hood_version(df, expected = "158"),
                 regexp = "BOTH HOOD_158 .* and HOOD_140")
})

# =================================================== morie_tps_year_to_hood_version()

test_that("morie_tps_year_to_hood_version maps year -> recommended scheme", {
  expect_equal(morie_tps_year_to_hood_version(2014L), "140")
  expect_equal(morie_tps_year_to_hood_version(2021L), "140")
  expect_equal(morie_tps_year_to_hood_version(2022L), "158")
  expect_equal(morie_tps_year_to_hood_version(2025L), "158")
  expect_equal(
    morie_tps_year_to_hood_version(c(2020L, 2022L, 2024L)),
    c("140", "158", "158"))
})

test_that("morie_tps_year_to_hood_version returns NA on non-numeric input", {
  expect_true(is.na(morie_tps_year_to_hood_version("not-a-year")))
})

# ==================================================== morie_to_hood_crosswalk()

test_that("morie_to_hood_crosswalk returns the bundled 158<->140 mapping", {
  cw <- morie_to_hood_crosswalk()
  expect_s3_class(cw, "data.frame")
  for (col in c("hood_140", "name_140", "hood_158", "name_158",
                "area_overlap_pct", "note"))
    expect_true(col %in% names(cw))
  # The synthetic crosswalk demonstrates a split: 140 hood 75
  # (Church-Yonge Corridor) maps to two 158 hoods (167 + 168).
  hood_75_targets <- cw$hood_158[cw$hood_140 == 75]
  expect_true(length(hood_75_targets) >= 2L)
})

test_that("morie_to_hood_crosswalk overlap percentages sum to 100 per source 140", {
  cw <- morie_to_hood_crosswalk()
  by_src <- aggregate(area_overlap_pct ~ hood_140, cw, sum)
  expect_true(all(abs(by_src$area_overlap_pct - 100) < 0.01))
})
