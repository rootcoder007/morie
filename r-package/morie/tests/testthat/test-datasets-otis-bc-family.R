# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3JJ: OTIS b01-b09 Segregation family + c01-c12 Individuals-in-
# Segregation+RC family (21 lookup-pending wrappers).

# =================================================== b01-b09 offline fixtures

test_that("OTIS b01 segregation-detailed loads canonical 18-col schema", {
  df <- morie_datasets_otis_b01_segregation_detailed(offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_equal(ncol(df), 18L)
  for (col in c("EndFiscalYear", "UniqueIndividual_ID", "Gender",
                "Region_AtTimeOfPlacement",
                "NumberConsecutiveDays_Segregation",
                "MentalHealth_Alert", "SuicideRisk_Alert",
                "Number_Of_Placements"))
    expect_true(col %in% names(df))
})

test_that("OTIS b02 segregation-total-days loads 6-col schema", {
  df <- morie_datasets_otis_b02_segregation_total_days(offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_equal(ncol(df), 6L)
  expect_true("TotalAggregatedDays_Segregation" %in% names(df))
})

test_that("OTIS b03 alerts-by-institution loads 6-col schema", {
  df <- morie_datasets_otis_b03_seg_alerts_by_institution(offline = TRUE)
  expect_equal(ncol(df), 6L)
  for (col in c("Institution_AtTimeOfPlacement", "Alert_Type",
                "Alert_Presence", "Number_SegregationPlacements"))
    expect_true(col %in% names(df))
})

test_that("OTIS b04 consecutive-by-region loads 5-col schema", {
  df <- morie_datasets_otis_b04_seg_consecutive_by_region(offline = TRUE)
  expect_equal(ncol(df), 5L)
  expect_true("Measure" %in% names(df))
})

test_that("OTIS b05 consecutive-lengths loads 3-col schema", {
  df <- morie_datasets_otis_b05_seg_consecutive_lengths(offline = TRUE)
  expect_equal(ncol(df), 3L)
  expect_true("Consecutive_Duration" %in% names(df))
})

test_that("OTIS b06 reason-by-institution loads 6-col schema", {
  df <- morie_datasets_otis_b06_seg_reason_by_institution(offline = TRUE)
  expect_equal(ncol(df), 6L)
  expect_true("Reason" %in% names(df))
})

test_that("OTIS b07 alerts-by-gender loads 5-col schema", {
  df <- morie_datasets_otis_b07_seg_alerts_by_gender(offline = TRUE)
  expect_equal(ncol(df), 5L)
  for (col in c("Number_Segregation_Placements_Without_Alert",
                "Number_Segregation_Placements_With_Alert"))
    expect_true(col %in% names(df))
})

test_that("OTIS b08 consecutive-by-institution loads 6-col schema", {
  df <- morie_datasets_otis_b08_seg_consecutive_by_institution(offline = TRUE)
  expect_equal(ncol(df), 6L)
  expect_true("Institution_AtTimeOfPlacement" %in% names(df))
})

test_that("OTIS b09 number-of-times loads 4-col schema", {
  df <- morie_datasets_otis_b09_seg_n_times(offline = TRUE)
  expect_equal(ncol(df), 4L)
  expect_true("NumberPlacements_Segregation" %in% names(df))
})

# =================================================== c01-c12 offline fixtures

test_that("OTIS c01 individuals-total loads 5-col schema with all-three counts", {
  df <- morie_datasets_otis_c01_individuals_total(offline = TRUE)
  expect_equal(ncol(df), 5L)
  for (col in c("NumberIndividuals_InCustody",
                "NumberIndividuals_RestrictiveConfinement",
                "NumberIndividuals_Segregation"))
    expect_true(col %in% names(df))
})

test_that("OTIS c02 by-institution loads 6-col schema", {
  df <- morie_datasets_otis_c02_individuals_by_institution(offline = TRUE)
  expect_equal(ncol(df), 6L)
  expect_true("Institution_MostRecentPlacement" %in% names(df))
})

test_that("OTIS c03 race-by-gender loads 6-col schema", {
  df <- morie_datasets_otis_c03_individuals_race_by_gender(offline = TRUE)
  expect_equal(ncol(df), 6L)
  for (col in c("Race", "Gender")) expect_true(col %in% names(df))
})

test_that("OTIS c04 race-by-region loads 5-col schema", {
  df <- morie_datasets_otis_c04_individuals_race_by_region(offline = TRUE)
  expect_equal(ncol(df), 5L)
})

test_that("OTIS c05 religion-by-region loads 5-col schema", {
  df <- morie_datasets_otis_c05_individuals_religion_by_region(offline = TRUE)
  expect_equal(ncol(df), 5L)
  expect_true("Religion" %in% names(df))
})

test_that("OTIS c06 age-by-region loads 5-col schema", {
  df <- morie_datasets_otis_c06_individuals_age_by_region(offline = TRUE)
  expect_equal(ncol(df), 5L)
  expect_true("Age_Category" %in% names(df))
})

test_that("OTIS c07 alerts loads 6-col schema", {
  df <- morie_datasets_otis_c07_individuals_alerts(offline = TRUE)
  expect_equal(ncol(df), 6L)
  expect_true("Alert_Type" %in% names(df))
})

test_that("OTIS c08 religion-by-gender loads 6-col schema", {
  df <- morie_datasets_otis_c08_individuals_religion_by_gender(offline = TRUE)
  expect_equal(ncol(df), 6L)
})

test_that("OTIS c09 age-by-gender loads 6-col schema", {
  df <- morie_datasets_otis_c09_individuals_age_by_gender(offline = TRUE)
  expect_equal(ncol(df), 6L)
})

test_that("OTIS c10 aggregate-by-institution loads 7-col schema", {
  df <- morie_datasets_otis_c10_aggregate_durations_by_institution(
    offline = TRUE)
  expect_equal(ncol(df), 7L)
  for (col in c("TotalAggregatedDays_RestrictiveConfinement",
                "TotalAggregatedDays_Segregation"))
    expect_true(col %in% names(df))
})

test_that("OTIS c11 aggregate-lengths loads 4-col schema", {
  df <- morie_datasets_otis_c11_aggregate_lengths(offline = TRUE)
  expect_equal(ncol(df), 4L)
  expect_true("Aggregate_Duration" %in% names(df))
})

test_that("OTIS c12 aggregate-by-region loads 6-col schema", {
  df <- morie_datasets_otis_c12_aggregate_durations_by_region(offline = TRUE)
  expect_equal(ncol(df), 6L)
})

# =================================================== live-mode "lookup pending"

test_that("every b/c wrapper errors with 'lookup pending' when offline=FALSE + no resource_id", {
  wrappers <- list(
    morie_datasets_otis_b01_segregation_detailed,
    morie_datasets_otis_b02_segregation_total_days,
    morie_datasets_otis_b03_seg_alerts_by_institution,
    morie_datasets_otis_b04_seg_consecutive_by_region,
    morie_datasets_otis_b05_seg_consecutive_lengths,
    morie_datasets_otis_b06_seg_reason_by_institution,
    morie_datasets_otis_b07_seg_alerts_by_gender,
    morie_datasets_otis_b08_seg_consecutive_by_institution,
    morie_datasets_otis_b09_seg_n_times,
    morie_datasets_otis_c01_individuals_total,
    morie_datasets_otis_c02_individuals_by_institution,
    morie_datasets_otis_c03_individuals_race_by_gender,
    morie_datasets_otis_c04_individuals_race_by_region,
    morie_datasets_otis_c05_individuals_religion_by_region,
    morie_datasets_otis_c06_individuals_age_by_region,
    morie_datasets_otis_c07_individuals_alerts,
    morie_datasets_otis_c08_individuals_religion_by_gender,
    morie_datasets_otis_c09_individuals_age_by_gender,
    morie_datasets_otis_c10_aggregate_durations_by_institution,
    morie_datasets_otis_c11_aggregate_lengths,
    morie_datasets_otis_c12_aggregate_durations_by_region)
  for (fn in wrappers) {
    expect_error(fn(offline = FALSE), regexp = "lookup pending")
  }
})

# =================================================== mocked live dispatch (3 reps)

test_that("OTIS b01 dispatches via mock when resource_id is supplied", {
  out <- testthat::with_mocked_bindings(
    .morie_ontario_ckan_dump_csv = function(resource_id,
                                              limit = 200000L) {
      expect_equal(resource_id, "future-b01-id")
      data.frame(EndFiscalYear = 2024L,
                  UniqueIndividual_ID = "LIVE-B01-1")
    },
    .package = "morie",
    code = morie_datasets_otis_b01_segregation_detailed(
      offline = FALSE, resource_id = "future-b01-id"))
  expect_equal(out$UniqueIndividual_ID, "LIVE-B01-1")
})

test_that("OTIS c01 dispatches via mock when resource_id is supplied", {
  out <- testthat::with_mocked_bindings(
    .morie_ontario_ckan_dump_csv = function(resource_id,
                                              limit = 200000L) {
      expect_equal(resource_id, "future-c01-id")
      data.frame(EndFiscalYear = 2024L,
                  NumberIndividuals_InCustody = 7777L)
    },
    .package = "morie",
    code = morie_datasets_otis_c01_individuals_total(
      offline = FALSE, resource_id = "future-c01-id"))
  expect_equal(out$NumberIndividuals_InCustody, 7777L)
})

test_that("OTIS c11 dispatches via mock when resource_id is supplied", {
  out <- testthat::with_mocked_bindings(
    .morie_ontario_ckan_dump_csv = function(resource_id,
                                              limit = 200000L) {
      expect_equal(resource_id, "future-c11-id")
      data.frame(EndFiscalYear = 2024L,
                  Aggregate_Duration = "Over 90 days",
                  NumberIndividuals_RestrictiveConfinement = 9L,
                  NumberIndividuals_Segregation = 5L)
    },
    .package = "morie",
    code = morie_datasets_otis_c11_aggregate_lengths(
      offline = FALSE, resource_id = "future-c11-id"))
  expect_equal(out$NumberIndividuals_Segregation, 5L)
})

# =================================================== registry integration

test_that("morie_datasets_ontario_ckan_layers now includes all 21 lookup-pending b/c entries", {
  reg <- morie_datasets_ontario_ckan_layers()
  # 9 ARSAU + 1 OTIS d01 wired + 7 OTIS lookup-pending (3HH) + 21 new = 38.
  expect_true(nrow(reg) >= 38L)
  b_keys <- sprintf("otis_b%02d_%s", 1:9,
                     c("segregation_detailed",
                       "segregation_total_days",
                       "seg_alerts_by_institution",
                       "seg_consecutive_by_region",
                       "seg_consecutive_lengths",
                       "seg_reason_by_institution",
                       "seg_alerts_by_gender",
                       "seg_consecutive_by_institution",
                       "seg_n_times"))
  c_keys <- sprintf("otis_c%02d_%s", 1:12,
                     c("individuals_total",
                       "individuals_by_institution",
                       "individuals_race_by_gender",
                       "individuals_race_by_region",
                       "individuals_religion_by_region",
                       "individuals_age_by_region",
                       "individuals_alerts",
                       "individuals_religion_by_gender",
                       "individuals_age_by_gender",
                       "aggregate_durations_by_institution",
                       "aggregate_lengths",
                       "aggregate_durations_by_region"))
  for (k in c(b_keys, c_keys))
    expect_true(k %in% reg$dataset_key)
  # Combined lookup-pending OTIS entries: 7 + 21 = 28.
  na_entries <- reg[is.na(reg$resource_id), ]
  expect_equal(nrow(na_entries), 28L)
  expect_true(all(na_entries$family == "otis"))
})

test_that("morie_datasets_ontario_ckan_by_key('otis_c03_individuals_race_by_gender', offline=TRUE) reads bundled fixture", {
  df <- morie_datasets_ontario_ckan_by_key(
    "otis_c03_individuals_race_by_gender", offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_true(all(c("Race", "Gender") %in% names(df)))
})
