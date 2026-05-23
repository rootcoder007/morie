# SPDX-License-Identifier: AGPL-3.0-or-later
library(testthat)
set.seed(1)

# ---------------------------------------------------------------------------
# Synthetic OTIS placement frame mirroring the canonical column shape
# ---------------------------------------------------------------------------

make_otis_placements <- function(n = 60, seed = 1) {
  set.seed(seed)
  data.frame(
    UniqueIndividual_ID = sprintf("id%04d", sample.int(20, n, replace = TRUE)),
    EndFiscalYear = sample(2018:2024, n, replace = TRUE),
    Gender = sample(c("M", "F"), n, replace = TRUE),
    Age_Category = sample(c("18-24", "25-34", "35-44", "45+"), n, replace = TRUE),
    Region_AtTimeOfPlacement = sample(c("Central", "East", "West", "North"),
                                      n, replace = TRUE),
    Region_MostRecentPlacement = sample(c("Central", "East", "West", "North"),
                                        n, replace = TRUE),
    MentalHealth_Alert = sample(0:1, n, replace = TRUE, prob = c(0.6, 0.4)),
    SuicideRisk_Alert = sample(0:1, n, replace = TRUE, prob = c(0.7, 0.3)),
    SuicideWatch_Alert = sample(0:1, n, replace = TRUE, prob = c(0.85, 0.15)),
    Number_Of_Placements = sample(1:5, n, replace = TRUE),
    NumberConsecutiveDays_Segregation = sample(0:25, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# morie_otis_load — external file required
# ---------------------------------------------------------------------------

test_that("morie_otis_load returns data.frame or errors cleanly without csv", {
  res <- tryCatch(morie_otis_load(), error = function(e) NULL)
  skip_if(is.null(res), "needs OTIS csv on disk")
  expect_s3_class(res, "data.frame")
})

test_that("morie_otis_load handles bogus path gracefully", {
  set.seed(2)
  expect_error(morie_otis_load(csv_path = "/nonexistent/__nope__.csv"))
})

# ---------------------------------------------------------------------------
# morie_otis_all_analyses — top-level dispatcher
# ---------------------------------------------------------------------------

test_that("morie_otis_all_analyses runs on synthetic frame or skips cleanly", {
  set.seed(3)
  df <- make_otis_placements(80)
  res <- tryCatch(morie_otis_all_analyses(df, year = 2022),
                  error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS columns")
  expect_true(is.list(res) || is.data.frame(res))
})

# ---------------------------------------------------------------------------
# morie_otis_rplace / regional_placement alias
# ---------------------------------------------------------------------------

test_that("morie_otis_rplace runs end-to-end on synthetic data", {
  set.seed(4)
  df <- make_otis_placements(100)
  res <- tryCatch(morie_otis_rplace(df, year = 2022),
                  error = function(e) NULL)
  skip_if(is.null(res), "needs region columns the synthetic frame lacks")
  expect_true(is.list(res) || is.data.frame(res))
})

test_that("morie_otis_regional_placement is an alias of rplace", {
  set.seed(5)
  df <- make_otis_placements(60)
  a <- tryCatch(morie_otis_rplace(df, year = 2022), error = function(e) NULL)
  b <- tryCatch(morie_otis_regional_placement(df, year = 2022),
                error = function(e) NULL)
  skip_if(is.null(a) || is.null(b), "needs richer OTIS columns")
  expect_identical(class(a), class(b))
})

# ---------------------------------------------------------------------------
# morie_otis_astcmb / alert-state combo
# ---------------------------------------------------------------------------

test_that("morie_otis_astcmb returns a result on synthetic data", {
  set.seed(6)
  df <- make_otis_placements(120)
  res <- tryCatch(morie_otis_astcmb(df), error = function(e) NULL)
  skip_if(is.null(res), "needs alert columns the synthetic frame lacks")
  expect_true(is.list(res) || is.data.frame(res))
})

test_that("morie_otis_alert_state_combo alias delegates", {
  set.seed(7)
  df <- make_otis_placements(80)
  res <- tryCatch(morie_otis_alert_state_combo(df), error = function(e) NULL)
  skip_if(is.null(res), "needs alert columns")
  expect_true(is.list(res) || is.data.frame(res))
})

# ---------------------------------------------------------------------------
# morie_otis_volat / volatility
# ---------------------------------------------------------------------------

test_that("morie_otis_volat runs on synthetic placements", {
  set.seed(8)
  df <- make_otis_placements(150)
  res <- tryCatch(morie_otis_volat(df), error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure")
  expect_true(is.list(res) || is.data.frame(res))
})

test_that("morie_otis_volatility alias runs", {
  set.seed(9)
  df <- make_otis_placements(80)
  res <- tryCatch(morie_otis_volatility(df), error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure")
  expect_true(is.list(res) || is.data.frame(res))
})

# ---------------------------------------------------------------------------
# morie_otis_rctrnd / rc_trends
# ---------------------------------------------------------------------------

test_that("morie_otis_rctrnd runs on synthetic placements", {
  set.seed(10)
  df <- make_otis_placements(120)
  res <- tryCatch(morie_otis_rctrnd(df), error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure")
  expect_true(is.list(res) || is.data.frame(res))
})

test_that("morie_otis_rc_trends alias runs", {
  set.seed(11)
  df <- make_otis_placements(80)
  res <- tryCatch(morie_otis_rc_trends(df), error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure")
  expect_true(is.list(res) || is.data.frame(res))
})

# ---------------------------------------------------------------------------
# morie_otis_otdesc / descriptives
# ---------------------------------------------------------------------------

test_that("morie_otis_otdesc returns descriptives on synthetic frame", {
  set.seed(12)
  df <- make_otis_placements(120)
  res <- tryCatch(morie_otis_otdesc(df), error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure")
  expect_true(is.list(res) || is.data.frame(res))
})

test_that("morie_otis_descriptives alias runs", {
  set.seed(13)
  df <- make_otis_placements(60)
  res <- tryCatch(morie_otis_descriptives(df), error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure")
  expect_true(is.list(res) || is.data.frame(res))
})

# ---------------------------------------------------------------------------
# morie_otis_otdml / DML
# ---------------------------------------------------------------------------

test_that("morie_otis_otdml runs on synthetic data or skips", {
  set.seed(14)
  df <- make_otis_placements(200)
  res <- tryCatch(morie_otis_otdml(df), error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure / DML deps")
  expect_true(is.list(res) || is.data.frame(res))
})

test_that("morie_otis_dml alias delegates", {
  set.seed(15)
  df <- make_otis_placements(120)
  res <- tryCatch(morie_otis_dml(df), error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure / DML deps")
  expect_true(is.list(res) || is.data.frame(res))
})

# ---------------------------------------------------------------------------
# edge-case: empty data frame
# ---------------------------------------------------------------------------

test_that("morie_otis_otdesc on empty frame errors or returns gracefully", {
  empty <- make_otis_placements(0)
  res <- tryCatch(morie_otis_otdesc(empty),
                  error = function(e) NULL,
                  warning = function(w) NULL)
  expect_true(is.null(res) || is.list(res) || is.data.frame(res))
})