# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3HH: OTIS a01 Restrictive Confinement + d02-d07 Deaths-in-
# Custody wrappers. Lookup-pending live mode (CKAN resource_ids not
# yet wired in; offline fixtures fully exercised + live errors with
# clear "lookup pending" message; resource_id= override works).

# ====================================================== a01 restrictive confinement

test_that("morie_datasets_otis_a01_restrictive_confinement(offline=TRUE) reads 10-col schema", {
  df <- morie_datasets_otis_a01_restrictive_confinement(offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 5L)
  expect_equal(ncol(df), 10L)
  for (col in c("EndFiscalYear", "UniqueIndividual_ID",
                "Region_AtTimeOfPlacement",
                "Region_MostRecentPlacement", "Gender",
                "Age_Category", "MentalHealth_Alert",
                "SuicideRisk_Alert", "SuicideWatch_Alert",
                "Number_Of_Placements"))
    expect_true(col %in% names(df))
  expect_true(all(grepl("-SYNTH-", df$UniqueIndividual_ID)))
})

test_that("morie_datasets_otis_a01_restrictive_confinement(offline=FALSE) errors when no resource_id", {
  expect_error(
    morie_datasets_otis_a01_restrictive_confinement(offline = FALSE),
    regexp = "lookup pending")
})

test_that("morie_datasets_otis_a01_restrictive_confinement(offline=FALSE) dispatches via mock when resource_id given", {
  testthat::local_mocked_bindings(
    .morie_ontario_ckan_dump_csv = function(resource_id, limit = 200000L) {
      expect_equal(resource_id, "future-a01-id")
      data.frame(EndFiscalYear = 2024L,
                  UniqueIndividual_ID = "LIVE-A01-1")
    },
    .package = "morie")
  out <- morie_datasets_otis_a01_restrictive_confinement(
    offline = FALSE, resource_id = "future-a01-id")
  expect_equal(out$UniqueIndividual_ID, "LIVE-A01-1")
})

# ====================================================== d02-d05 (3-col schema)

test_that("morie_datasets_otis_d02_deaths_by_gender(offline=TRUE) reads 3-col schema", {
  df <- morie_datasets_otis_d02_deaths_by_gender(offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_setequal(names(df),
                  c("Year", "Gender", "Number_CustodialDeaths"))
})

test_that("morie_datasets_otis_d03_deaths_by_race(offline=TRUE) reads 3-col schema", {
  df <- morie_datasets_otis_d03_deaths_by_race(offline = TRUE)
  expect_setequal(names(df),
                  c("Year", "Race", "Number_CustodialDeaths"))
})

test_that("morie_datasets_otis_d04_deaths_by_religion(offline=TRUE) reads 3-col schema", {
  df <- morie_datasets_otis_d04_deaths_by_religion(offline = TRUE)
  expect_setequal(names(df),
                  c("Year", "Religion", "Number_CustodialDeaths"))
})

test_that("morie_datasets_otis_d05_deaths_by_age_category(offline=TRUE) reads 3-col schema", {
  df <- morie_datasets_otis_d05_deaths_by_age_category(offline = TRUE)
  expect_setequal(names(df),
                  c("Year", "Age_Category", "Number_CustodialDeaths"))
})

# ====================================================== d06-d07 (4-col schema)

test_that("morie_datasets_otis_d06_cause_by_alert(offline=TRUE) reads 4-col cause-by-alert schema", {
  df <- morie_datasets_otis_d06_cause_by_alert(offline = TRUE)
  expect_setequal(names(df),
                  c("Year", "Alert_Type", "MedicalCauseOfDeath",
                    "Number_CustodialDeaths"))
})

test_that("morie_datasets_otis_d07_alerts_by_housing_unit(offline=TRUE) reads 4-col alerts-by-housing schema", {
  df <- morie_datasets_otis_d07_alerts_by_housing_unit(offline = TRUE)
  expect_setequal(names(df),
                  c("Year", "Alert_Type", "Housing_Type",
                    "Number_CustodialDeaths"))
})

# ====================================================== live-mode error contract

test_that("every d02-d07 + a01 wrapper errors with 'lookup pending' when offline=FALSE + no resource_id", {
  wrappers <- list(
    morie_datasets_otis_a01_restrictive_confinement,
    morie_datasets_otis_d02_deaths_by_gender,
    morie_datasets_otis_d03_deaths_by_race,
    morie_datasets_otis_d04_deaths_by_religion,
    morie_datasets_otis_d05_deaths_by_age_category,
    morie_datasets_otis_d06_cause_by_alert,
    morie_datasets_otis_d07_alerts_by_housing_unit)
  for (fn in wrappers) {
    expect_error(fn(offline = FALSE),
                 regexp = "lookup pending")
  }
})

test_that("every d02-d07 wrapper dispatches via mock when resource_id is supplied", {
  reps <- list(
    list(fn = morie_datasets_otis_d02_deaths_by_gender,        rid = "id-d02"),
    list(fn = morie_datasets_otis_d03_deaths_by_race,          rid = "id-d03"),
    list(fn = morie_datasets_otis_d04_deaths_by_religion,      rid = "id-d04"),
    list(fn = morie_datasets_otis_d05_deaths_by_age_category,  rid = "id-d05"),
    list(fn = morie_datasets_otis_d06_cause_by_alert,          rid = "id-d06"),
    list(fn = morie_datasets_otis_d07_alerts_by_housing_unit,  rid = "id-d07"))
  for (R in reps) {
    out <- testthat::with_mocked_bindings(
      .morie_ontario_ckan_dump_csv = function(resource_id,
                                                limit = 200000L) {
        expect_equal(resource_id, R$rid)
        data.frame(Year = 2024L, Number_CustodialDeaths = 1L)
      },
      .package = "morie",
      code = R$fn(offline = FALSE, resource_id = R$rid))
    expect_s3_class(out, "data.frame")
  }
})

# ====================================================== registry integration

test_that("morie_datasets_ontario_ckan_layers now includes all 7 lookup-pending OTIS entries", {
  reg <- morie_datasets_ontario_ckan_layers()
  # 9 ARSAU + 1 OTIS d01 (already had) + 7 new = 17 total.
  expect_true(nrow(reg) >= 17L)
  expected_otis_keys <- c("otis_d01_deaths_in_custody",
                           "otis_a01_restrictive_confinement",
                           "otis_d02_deaths_by_gender",
                           "otis_d03_deaths_by_race",
                           "otis_d04_deaths_by_religion",
                           "otis_d05_deaths_by_age_category",
                           "otis_d06_cause_by_alert",
                           "otis_d07_alerts_by_housing_unit")
  for (k in expected_otis_keys)
    expect_true(k %in% reg$dataset_key)
  # The 7 new entries have NA resource_id (lookup pending).
  na_entries <- reg[is.na(reg$resource_id), ]
  expect_equal(nrow(na_entries), 7L)
  expect_true(all(na_entries$family == "otis"))
})

test_that("morie_datasets_ontario_ckan_by_key('otis_d03_deaths_by_race', offline=TRUE) reads bundled fixture", {
  df <- morie_datasets_ontario_ckan_by_key(
    "otis_d03_deaths_by_race", offline = TRUE)
  expect_s3_class(df, "data.frame")
  expect_true("Race" %in% names(df))
})

test_that("morie_datasets_ontario_ckan_by_key honours resource_id override on lookup-pending entries", {
  testthat::local_mocked_bindings(
    .morie_ontario_ckan_dump_csv = function(resource_id, limit = 200000L) {
      expect_equal(resource_id, "live-id-for-d06")
      data.frame(Year = 2024L, Alert_Type = "LIVE")
    },
    .package = "morie")
  out <- morie_datasets_ontario_ckan_by_key(
    "otis_d06_cause_by_alert",
    offline = FALSE, resource_id = "live-id-for-d06")
  expect_equal(out$Alert_Type, "LIVE")
})
