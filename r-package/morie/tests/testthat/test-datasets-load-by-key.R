# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3EEE4: unified morie_datasets_load_by_key() dispatcher.

test_that("targeted-fixture dispatch hits VPD bundled sample (550 rows)", {
  df <- morie_datasets_load_by_key("vpd_crime")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 550L)
})

test_that("targeted-fixture dispatch hits NYPD arrests YTD (5 rows)", {
  df <- morie_datasets_load_by_key("nypd_arrests_ytd")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 5L)
})

test_that("targeted-fixture dispatch hits TPS PSDP layer (assault)", {
  df <- morie_datasets_load_by_key("assault")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 5L)
})

test_that("targeted-fixture dispatch hits MTL SIM interventions", {
  df <- morie_datasets_load_by_key(
    "interventions-service-securite-incendie-montreal")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 349L)
})

test_that("targeted-fixture dispatch hits TO ambulance stations", {
  df <- morie_datasets_load_by_key("ambulance-station-locations")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 46L)
})

test_that("targeted-fixture dispatch hits TO ASR misc", {
  df <- morie_datasets_load_by_key(
    "police-annual-statistical-report-miscellaneous-data")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 40L)
})

test_that("targeted-fixture dispatch hits Vancouver graffiti sample", {
  df <- morie_datasets_load_by_key("graffiti")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 100L)
})

test_that("targeted-fixture dispatch hits Vancouver fire halls", {
  df <- morie_datasets_load_by_key("fire-halls")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 20L)
})

test_that("targeted-fixture dispatch hits NYC borough boundaries", {
  df <- morie_datasets_load_by_key("borough")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 5L)
})

test_that("targeted-fixture dispatch hits NYC police precincts", {
  df <- morie_datasets_load_by_key("police_precinct")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 78L)
})

test_that("targeted-fixture dispatch hits NYC ZCTAs", {
  df <- morie_datasets_load_by_key("zcta")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 221L)
})

test_that("max_features cap is propagated", {
  df <- morie_datasets_load_by_key("vpd_crime", max_features = 7L)
  expect_equal(nrow(df), 7L)
})

test_that("unknown dataset_key raises clear error", {
  expect_error(
    morie_datasets_load_by_key("nonexistent-key-9999"),
    regexp = "unknown dataset_key")
})

test_that("MTL CKAN generic dispatch surfaces helpful unwired error", {
  # communique-presse is in the MTL catalog but not a targeted fixture.
  expect_error(
    morie_datasets_load_by_key("communique-presse"),
    regexp = "not yet wired")
})

test_that("TO CKAN generic dispatch surfaces helpful unwired error", {
  # Use a TO package known to not have a bundled fixture.
  expect_error(
    morie_datasets_load_by_key("311-service-requests-customer-initiated"),
    regexp = "not yet wired")
})
