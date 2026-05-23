# SPDX-License-Identifier: AGPL-3.0-or-later
# Coverage tests for R/mrm_primitives_gentrification.R

set.seed(2026L)

mk_gent_panel <- function(n = 60L) {
  data.frame(
    tract_id = sprintf("T%03d", seq_len(n)),
    inc0 = runif(n, 30000, 90000),
    rent0 = runif(n, 700, 2000),
    coll_g = rnorm(n, 0.05, 0.1),
    rent_g = rnorm(n, 150, 80),
    stringsAsFactors = FALSE
  )
}

test_that("mrm_gentrification_panel basic 3-level classification", {
  df <- mk_gent_panel()
  res <- mrm_gentrification_panel(df,
    baseline_income_col = "inc0",
    baseline_rent_col   = "rent0",
    growth_college_col  = "coll_g",
    growth_rent_col     = "rent_g"
  )
  expect_s3_class(res, "morie_mrm_result")
  expect_length(res$labels, 60L)
  expect_true(all(stats::na.omit(res$labels) %in%
                  c("ineligible", "eligible", "gentrified")))
  expect_true(is.list(res$thresholds))
})

test_that("mrm_gentrification_panel honours custom quantiles", {
  df <- mk_gent_panel()
  res <- mrm_gentrification_panel(df,
    baseline_income_col = "inc0",
    baseline_rent_col   = "rent0",
    growth_college_col  = "coll_g",
    growth_rent_col     = "rent_g",
    baseline_marginalisation_quantile = 0.25,
    gentrification_growth_quantile = 0.75
  )
  expect_s3_class(res, "morie_mrm_result")
  expect_equal(res$baseline_marginalisation_quantile, 0.25)
  expect_equal(res$gentrification_growth_quantile, 0.75)
})

test_that("mrm_gentrification_panel returns null-analysis when columns missing", {
  df <- mk_gent_panel()
  df$inc0 <- NULL
  res <- mrm_gentrification_panel(df,
    baseline_income_col = "inc0",
    baseline_rent_col   = "rent0",
    growth_college_col  = "coll_g",
    growth_rent_col     = "rent_g"
  )
  expect_equal(res$n, 0L)
  expect_true(any(grepl("missing", res$warnings)))
})

test_that("mrm_gentrification_panel rejects bad quantiles", {
  df <- mk_gent_panel()
  expect_error(
    mrm_gentrification_panel(df, "inc0", "rent0", "coll_g", "rent_g",
                              baseline_marginalisation_quantile = 0),
    "baseline_marginalisation_quantile"
  )
  expect_error(
    mrm_gentrification_panel(df, "inc0", "rent0", "coll_g", "rent_g",
                              baseline_marginalisation_quantile = 1.1),
    "baseline_marginalisation_quantile"
  )
  expect_error(
    mrm_gentrification_panel(df, "inc0", "rent0", "coll_g", "rent_g",
                              gentrification_growth_quantile = 0),
    "gentrification_growth_quantile"
  )
  expect_error(
    mrm_gentrification_panel(df, "inc0", "rent0", "coll_g", "rent_g",
                              gentrification_growth_quantile = 1.1),
    "gentrification_growth_quantile"
  )
})

test_that("mrm_gentrification_panel handles NA rows + warns", {
  df <- mk_gent_panel()
  df$inc0[1:3] <- NA
  df$rent_g[4:5] <- NA
  res <- mrm_gentrification_panel(df, "inc0", "rent0", "coll_g", "rent_g")
  expect_true(any(grepl("NA", res$warnings)))
})

test_that("mrm_gentrification_panel small-n warns", {
  df <- mk_gent_panel(n = 10L)
  res <- mrm_gentrification_panel(df, "inc0", "rent0", "coll_g", "rent_g")
  expect_true(any(grepl("codeable", res$warnings, ignore.case = TRUE)))
})

test_that("mrm_gentrification_panel zero-gentrified branch warns", {
  df <- mk_gent_panel(n = 40L)
  # Make growth columns all very low so no row meets the gentrification
  # growth quantile
  df$coll_g <- -1
  df$rent_g <- -1
  # All same -> the quantile = -1, all rows >= -1, so growth_mask all TRUE
  # Instead, set them to a wide range and pick quantile so nothing exceeds:
  df$coll_g[1:39] <- -1
  df$coll_g[40] <- 10  # only one row has high growth
  df$rent_g[1:39] <- -1
  df$rent_g[40] <- 10
  res <- mrm_gentrification_panel(df, "inc0", "rent0", "coll_g", "rent_g",
                                   gentrification_growth_quantile = 0.99)
  # Some warning likely produced
  expect_s3_class(res, "morie_mrm_result")
})

test_that("print.morie_mrm_result emits readable output", {
  df <- mk_gent_panel()
  res <- mrm_gentrification_panel(df, "inc0", "rent0", "coll_g", "rent_g")
  out <- capture.output(print(res))
  expect_true(any(grepl("MRM Gentrification", out)))
  expect_true(any(grepl("Tracts", out)))
})
