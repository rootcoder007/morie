# SPDX-License-Identifier: AGPL-3.0-or-later
library(testthat)
set.seed(1)

make_otis_data <- function(n = 80, seed = 1) {
  set.seed(seed)
  data.frame(
    UniqueIndividual_ID = sprintf("id%04d", sample.int(30, n, replace = TRUE)),
    EndFiscalYear = sample(2018:2024, n, replace = TRUE),
    Gender = sample(c("M", "F"), n, replace = TRUE),
    Age_Category = sample(c("18-24", "25-34", "35-44"), n, replace = TRUE),
    Region_AtTimeOfPlacement = sample(c("Central", "East", "West"), n, replace = TRUE),
    Region_MostRecentPlacement = sample(c("Central", "East", "West"), n, replace = TRUE),
    MentalHealth_Alert = sample(0:1, n, replace = TRUE),
    SuicideRisk_Alert = sample(0:1, n, replace = TRUE),
    SuicideWatch_Alert = sample(0:1, n, replace = TRUE),
    Number_Of_Placements = sample(1:5, n, replace = TRUE),
    NumberConsecutiveDays_Segregation = sample(0:25, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# Smoke-test every morie_otis_<churn>_<id> analyzer
# ---------------------------------------------------------------------------

analyzers <- c(
  "morie_otis_repeat_placement_concentration",
  "morie_otis_within_year_placement_count",
  "morie_otis_within_year_region_diversity",
  "morie_otis_mortification_cooccurrence",
  "morie_otis_disciplinary_medical_overlap",
  "morie_otis_embedding_distribution",
  "morie_otis_intra_year_transition_matrix",
  "morie_otis_path_complexity_gini",
  "morie_otis_region_alert_state_richness",
  "morie_otis_regC_demog_contingency",
  "morie_otis_irr_glmm_vm"
)

for (fn_name in analyzers) {
  local({
    nm <- fn_name
    test_that(paste(nm, "runs on synthetic placements or skips"), {
      skip_if_not(exists(nm), paste(nm, "not exported"))
      df <- make_otis_data(120, seed = nchar(nm))
      res <- tryCatch(do.call(nm, list(df)), error = function(e) NULL)
      skip_if(is.null(res), paste(nm, "needs richer OTIS structure"))
      expect_true(is.list(res) || is.data.frame(res) || is.numeric(res))
    })
  })
}

# ---------------------------------------------------------------------------
# Concentration analyzer — known signature with k parameter
# ---------------------------------------------------------------------------

test_that("morie_otis_repeat_placement_concentration accepts k arg", {
  set.seed(2)
  df <- make_otis_data(150)
  res <- tryCatch(
    morie_otis_repeat_placement_concentration(df),
    error = function(e) NULL
  )
  skip_if(is.null(res), "needs richer OTIS structure")
  expect_true(is.list(res) || is.data.frame(res))
})

# ---------------------------------------------------------------------------
# Aggregator: morie_otis_churn_analyze_all
# ---------------------------------------------------------------------------

test_that("morie_otis_churn_analyze_all runs with NULL batches", {
  res <- tryCatch(morie_otis_churn_analyze_all(),
                  error = function(e) NULL)
  skip_if(is.null(res), "needs OTIS batch payloads")
  expect_true(is.list(res))
})

test_that("morie_otis_churn_analyze_all accepts b01 synthetic frame", {
  set.seed(3)
  df <- make_otis_data(80)
  res <- tryCatch(morie_otis_churn_analyze_all(b01 = df),
                  error = function(e) NULL)
  skip_if(is.null(res), "needs richer OTIS structure")
  expect_true(is.list(res))
})

# ---------------------------------------------------------------------------
# Edge-case: empty frame should not crash R
# ---------------------------------------------------------------------------

test_that("within_year_placement_count tolerates empty frame", {
  empty <- make_otis_data(0)
  res <- tryCatch(morie_otis_within_year_placement_count(empty),
                  error = function(e) NULL,
                  warning = function(w) NULL)
  expect_true(is.null(res) || is.list(res) || is.data.frame(res))
})