# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3DD + 3EE-prep: Toronto neighbourhood boundary-version
# awareness, exercised against the REAL Open Toronto attribute
# fixtures (158 + 140 + NIA + 158<->140 crosswalk) bundled under
# inst/extdata/.
#
# Network paths are exercised via testthat::local_mocked_bindings,
# so the suite never reaches the wire.

# ===================================================== morie_to_neighbourhoods()

test_that("morie_to_neighbourhoods('158', offline=TRUE) returns the full 158-row canonical layer", {
  df <- morie_to_neighbourhoods("158", offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 158L)
  for (col in c("AREA_ID", "AREA_SHORT_CODE", "AREA_NAME",
                "CLASSIFICATION", "OBJECTID"))
    expect_true(col %in% names(df))
  # Names that are stable post-2022 158-scheme (none were split or
  # renamed -- the 1:1 cohort). Pick one from the high-recall set.
  expect_true("Annex" %in% df$AREA_NAME)
  # "Niagara" (the historical 140-82 name) was SPLIT in 158 into
  # Fort York-Liberty Village + West Queen West; assert the new names
  # are there and the old single-name is not.
  expect_true("Fort York-Liberty Village" %in% df$AREA_NAME)
  expect_true("West Queen West"           %in% df$AREA_NAME)
  expect_false("Niagara" %in% df$AREA_NAME)
})

test_that("morie_to_neighbourhoods('140', offline=TRUE) returns the full 140-row historical layer", {
  df <- morie_to_neighbourhoods("140", offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 140L)
  # Historical names carry the "(NN)" suffix per City convention.
  expect_true(any(grepl("\\(\\d+\\)$", df$AREA_NAME)))
  # 140 has Niagara (82) -- still present in the historical scheme.
  expect_true(any(grepl("^Niagara \\(82\\)$", df$AREA_NAME)))
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
  expect_error(morie_to_neighbourhoods("xyz", offline = TRUE),
               regexp = "should be one of")
})

# Live-mode dispatch (mocked CKAN).

test_that("morie_to_neighbourhoods(offline=FALSE) dispatches via mocked CKAN helper", {
  stub_df <- data.frame(
    `_id` = c(1L, 2L),
    AREA_ID = c(2502366L, 2502367L),
    AREA_SHORT_CODE = c("174", "082"),
    AREA_NAME = c("South Eglinton-Davisville", "Niagara"),
    check.names = FALSE)
  testthat::local_mocked_bindings(
    .morie_to_ckan_dump_csv = function(resource_id, limit = 100000L) {
      expect_match(resource_id,
                   "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}")
      stub_df
    },
    .package = "morie")
  out <- morie_to_neighbourhoods("158", offline = FALSE)
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

# ============================================= morie_to_hood_crosswalk() (REAL)

test_that("morie_to_hood_crosswalk returns the bundled REAL 158<->140 mapping", {
  cw <- morie_to_hood_crosswalk()
  expect_s3_class(cw, "data.frame")
  for (col in c("hood_140", "name_140", "hood_158", "name_158",
                "area_overlap_pct", "relation"))
    expect_true(col %in% names(cw))
  # Per the polygon-intersection build: 159 rows covering all 140
  # historical hoods + all 158 current hoods.
  expect_equal(nrow(cw), 159L)
  expect_equal(length(unique(cw$hood_140)), 140L)
  expect_equal(length(unique(cw$hood_158)), 158L)
})

test_that("morie_to_hood_crosswalk hood codes are 3-char zero-padded strings", {
  cw <- morie_to_hood_crosswalk()
  expect_type(cw$hood_140, "character")
  expect_type(cw$hood_158, "character")
  expect_true(all(nchar(cw$hood_140) == 3L))
  expect_true(all(nchar(cw$hood_158) == 3L))
})

test_that("morie_to_hood_crosswalk per-140 overlap percentages sum to 100", {
  cw <- morie_to_hood_crosswalk()
  by_src <- aggregate(area_overlap_pct ~ hood_140, cw, sum)
  expect_true(all(abs(by_src$area_overlap_pct - 100) < 0.01))
})

test_that("morie_to_hood_crosswalk relation distribution matches the OT data", {
  cw <- morie_to_hood_crosswalk()
  rel <- table(cw$relation)
  expect_equal(unname(rel["1:1"]),         123L)
  expect_equal(unname(rel["split"]),        34L)
  expect_equal(unname(rel["merge"]),         1L)
  expect_equal(unname(rel["split+merge"]),   1L)
})

test_that("morie_to_hood_crosswalk: 140-75 Church-Yonge Corridor splits into 158-167 + 158-168", {
  cw <- morie_to_hood_crosswalk()
  row75 <- cw[cw$hood_140 == "075", ]
  expect_true(nrow(row75) >= 2L)
  expect_true(all(row75$relation %in% c("split", "split+merge")))
  expect_setequal(row75$hood_158, c("167", "168"))
})

test_that("morie_to_hood_crosswalk: 140-82 Niagara splits into Fort York-Liberty Village + West Queen West", {
  cw <- morie_to_hood_crosswalk()
  row82 <- cw[cw$hood_140 == "082", ]
  expect_equal(nrow(row82), 2L)
  expect_setequal(row82$name_158,
                  c("Fort York-Liberty Village", "West Queen West"))
})

# ====================================== morie_tps_add_hood_158_from_140() / 140-from-158

test_that("morie_tps_add_hood_158_from_140 maps a 1:1 hood unchanged", {
  df <- data.frame(EVENT_ID = 1:2, HOOD_140 = c("001", "095"))
  out <- morie_tps_add_hood_158_from_140(df)
  expect_true("HOOD_158_equiv" %in% names(out))
  # 001 (West Humber-Clairville) and 095 (Annex) are both 1:1.
  expect_equal(out$HOOD_158_equiv, c("001", "095"))
})

test_that("morie_tps_add_hood_158_from_140 maps split hoods to PRIMARY-overlap 158", {
  # 140-75 (Church-Yonge Corridor) splits into 158-167 (40.76%)
  # + 158-168 (59.24%); primary is 158-168 (Downtown Yonge East).
  df <- data.frame(HOOD_140 = "075")
  out <- morie_tps_add_hood_158_from_140(df)
  expect_equal(out$HOOD_158_equiv, "168")
})

test_that("morie_tps_add_hood_158_from_140 normalises unpadded hood codes", {
  # TPS feeds publish HOOD_140 as integer-string "82" (no padding);
  # the crosswalk keys are "082". The normaliser should bridge.
  df <- data.frame(HOOD_140 = c("82", 1L, "1"))
  out <- morie_tps_add_hood_158_from_140(df)
  # 82 -> primary 158 is 163 (Fort York-Liberty Village, 72.18%).
  # 1  -> 001 (1:1).
  expect_equal(out$HOOD_158_equiv, c("163", "001", "001"))
})

test_that("morie_tps_add_hood_158_from_140 errors when no HOOD_140 column present", {
  df <- data.frame(HOOD_158 = "82")
  expect_error(morie_tps_add_hood_158_from_140(df),
               regexp = "no HOOD_140")
})

test_that("morie_tps_add_hood_140_from_158 maps a 1:1 hood unchanged", {
  df <- data.frame(EVENT_ID = 1:2, HOOD_158 = c("001", "095"))
  out <- morie_tps_add_hood_140_from_158(df)
  expect_true("HOOD_140_equiv" %in% names(out))
  expect_equal(out$HOOD_140_equiv, c("001", "095"))
})

test_that("morie_tps_add_hood_140_from_158 maps a child-of-split back to its historical parent", {
  # 158-168 (Downtown Yonge East) is a child of 140-75 (Church-Yonge
  # Corridor) -- the only 140 it overlaps. So 168 -> 075.
  df <- data.frame(HOOD_158 = c("168", "163"))
  out <- morie_tps_add_hood_140_from_158(df)
  # 168 -> 075. 163 (Fort York-Liberty Village) -> 082 (Niagara).
  expect_equal(out$HOOD_140_equiv, c("075", "082"))
})

test_that("morie_tps_add_hood_140_from_158 honours col_in + col_out overrides", {
  df <- data.frame(my_hood = c("001", "095"))
  out <- morie_tps_add_hood_140_from_158(df, col_in = "my_hood",
                                           col_out = "old_hood")
  expect_true("old_hood" %in% names(out))
  expect_equal(out$old_hood, c("001", "095"))
})
