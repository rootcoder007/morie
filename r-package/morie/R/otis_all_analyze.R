# SPDX-License-Identifier: AGPL-3.0-or-later
#' Comprehensive per-dataset analyses for ALL 28 OTIS public-release files
#'
#' R port of \code{morie.otis_all_analyze}. Pairs with the OTIS loaders
#' (see \code{?morie_otis} and the b01/c-series/d-series CSV files
#' under \code{data/datasets/OTIS/}) and chains the existing MRM-OTIS
#' callables in \code{mrm_otis.R} the same way
#' \code{morie_arsau_analyze_*} (in \code{mrm_arsau.R}) chains the
#' generic MRM-UoF callables.
#'
#' For every dataset id (\code{b01}..\code{d07}) this module exposes
#' \code{morie_otis_analyze_<id>(data)}. Each analyzer returns a named
#' \code{list} with class \code{c("morie_otis_analysis_result",
#' "morie_rich_result", "list")} containing
#' \code{title} / \code{summary_lines} / \code{tables} /
#' \code{interpretation} / \code{warnings} / \code{payload}, mirroring
#' the Python \code{RichResult} shape used in
#' \code{src/morie/otis_all_analyze.py}.
#'
#' Cross-year invariant: \code{UniqueIndividual_ID} is reassigned
#' every fiscal year (see \code{variable_taxonomy.R}). Every analyzer
#' that touches that column is within-year only.
#'
#' @name morie_otis_all_analyze
NULL


# ---------------------------------------------------------------------------
# Internal helpers (mirror _summary_lines, _crosstab, _year_trend, _to_int)
# ---------------------------------------------------------------------------

.otis_year_col <- function(df) {
  for (c in c("EndFiscalYear", "Year")) {
    if (c %in% names(df)) return(c)
  }
  NULL
}

.otis_to_int <- function(x) {
  v <- suppressWarnings(as.integer(x))
  if (length(v) == 0L || is.na(v)) 0L else v
}

.otis_is_truthy <- function(x) {
  if (is.logical(x)) return(as.integer(x))
  if (is.numeric(x)) return(as.integer(x == 1))
  as.integer(tolower(trimws(as.character(x))) %in% c("yes", "true", "1"))
}

.otis_summary_lines <- function(df, ds_id, description = NULL,
                                  series = NULL, primary_metric = NULL) {
  yc <- .otis_year_col(df)
  out <- list()
  out[["Dataset"]] <- if (!is.null(description)) {
    sprintf("%s -- %s", ds_id, description)
  } else ds_id
  if (!is.null(series)) out[["Series"]] <- series
  out[["Rows"]] <- nrow(df)
  out[["Columns"]] <- ncol(df)
  if (!is.null(yc) && nrow(df) > 0L) {
    yrs <- suppressWarnings(as.numeric(df[[yc]]))
    yrs <- yrs[is.finite(yrs)]
    if (length(yrs) > 0L) {
      out[["Years covered"]] <- sprintf("%d-%d", as.integer(min(yrs)),
                                          as.integer(max(yrs)))
    }
  }
  if (!is.null(primary_metric) && primary_metric %in% names(df)) {
    v <- suppressWarnings(as.numeric(df[[primary_metric]]))
    out[[sprintf("Total %s", primary_metric)]] <-
      as.integer(sum(v, na.rm = TRUE))
  }
  out
}

.otis_crosstab <- function(df, row, col, value,
                            aggfunc = c("sum", "max"),
                            top_rows = 20L) {
  aggfunc <- match.arg(aggfunc)
  if (!all(c(row, col, value) %in% names(df))) return(NULL)
  v <- suppressWarnings(as.numeric(df[[value]]))
  ok <- !is.na(v)
  if (!any(ok)) return(NULL)
  agg <- if (aggfunc == "max") max else sum
  pivot <- stats::xtabs(
    stats::as.formula(sprintf("v ~ %s + %s", row, col)),
    data = data.frame(df[ok, c(row, col)], v = v[ok]),
    addNA = FALSE
  )
  m <- as.matrix(pivot)
  totals <- rowSums(m)
  ord <- order(-totals)[seq_len(min(top_rows, nrow(m)))]
  m <- m[ord, , drop = FALSE]
  totals <- totals[ord]
  list(
    title = sprintf("%s by %s x %s:", value, row, col),
    headers = c(row, colnames(m), "TOTAL"),
    rows = lapply(seq_len(nrow(m)), function(i) {
      c(rownames(m)[i], as.integer(m[i, ]), as.integer(totals[i]))
    })
  )
}

.otis_year_trend <- function(df, value, year_col = NULL) {
  yc <- year_col %||% .otis_year_col(df)
  if (is.null(yc) || !(value %in% names(df))) return(NULL)
  v <- suppressWarnings(as.numeric(df[[value]]))
  g <- stats::aggregate(v, by = list(year = df[[yc]]),
                          FUN = sum, na.rm = TRUE)
  g <- g[order(g$year), , drop = FALSE]
  list(
    title = sprintf("%s by %s:", value, yc),
    headers = c(yc, value),
    rows = lapply(seq_len(nrow(g)), function(i) {
      c(as.integer(g$year[i]), as.integer(g$x[i]))
    })
  )
}

# Local `%||%` (R < 4.4)
`%||%` <- function(a, b) if (is.null(a)) b else a


.otis_wrap <- function(title, summary_lines, tables = list(),
                        interpretation = NULL, warnings = character(0),
                        payload = NULL) {
  tables <- Filter(Negate(is.null), tables)
  out <- list(
    title = title,
    summary_lines = summary_lines,
    tables = tables,
    warnings = warnings,
    interpretation = interpretation %||% "",
    payload = payload
  )
  class(out) <- c("morie_otis_analysis_result", "morie_rich_result",
                  "list")
  out
}


# ---------------------------------------------------------------------------
# b-series analyses (placements / durations / alerts)
# ---------------------------------------------------------------------------

#' Person-level segregation-placement analysis (b01)
#'
#' @param data b01 data.frame (76,934 rows in the public release).
#' @return A \code{morie_otis_analysis_result} list with reason / alert /
#'   year-trend tables. Within-year only -- \code{UniqueIndividual_ID}
#'   is not cross-year-safe.
#' @export
morie_otis_analyze_b01 <- function(data) {
  stopifnot(is.data.frame(data))
  s <- .otis_summary_lines(data, "b01",
    description = "Segregation placements (person-level detail)")
  if ("UniqueIndividual_ID" %in% names(data)) {
    s[["Unique individuals"]] <-
      length(unique(data[["UniqueIndividual_ID"]]))
  }
  dur <- "NumberConsecutiveDays_Segregation"
  if (dur %in% names(data)) {
    d <- suppressWarnings(as.numeric(data[[dur]]))
    s[["Mean consecutive days"]] <- round(mean(d, na.rm = TRUE), 2)
    s[["Median consecutive days"]] <- stats::median(d, na.rm = TRUE)
    s[["Max consecutive days"]] <- as.integer(max(d, na.rm = TRUE))
  }

  reason_cols <- grep("^SegReason_", names(data), value = TRUE)
  reason_rows <- lapply(reason_cols, function(r) {
    n_yes <- sum(.otis_is_truthy(data[[r]]))
    if (n_yes == 0L) return(NULL)
    list(reason = sub("^SegReason_", "", r),
         count = n_yes,
         pct = sprintf("%.1f%%", 100 * n_yes / nrow(data)))
  })
  reason_rows <- Filter(Negate(is.null), reason_rows)
  reason_rows <- reason_rows[order(-vapply(reason_rows,
    function(r) r$count, integer(1)))]

  alert_rows <- lapply(c("MentalHealth_Alert", "SuicideRisk_Alert",
                          "SuicideWatch_Alert"), function(a) {
    if (!(a %in% names(data))) return(NULL)
    n_yes <- sum(.otis_is_truthy(data[[a]]))
    list(alert = a, count = n_yes,
         pct = sprintf("%.1f%%", 100 * n_yes / nrow(data)))
  })
  alert_rows <- Filter(Negate(is.null), alert_rows)

  tables <- list(
    list(title = "Reasons for placement (count, % of rows):",
         headers = c("Reason", "Count", "Percent"),
         rows = lapply(reason_rows, function(r)
           c(r$reason, r$count, r$pct))),
    list(title = "Alert flags on placements:",
         headers = c("Alert", "Count", "Percent"),
         rows = lapply(alert_rows, function(r)
           c(r$alert, r$count, r$pct))),
    .otis_year_trend(data, "Number_Of_Placements")
  )

  .otis_wrap(
    title = "b01 -- Segregation placements (person-level detail)",
    summary_lines = s,
    tables = tables,
    payload = list(
      n_rows = nrow(data),
      n_individuals = if ("UniqueIndividual_ID" %in% names(data))
        length(unique(data[["UniqueIndividual_ID"]])) else NA_integer_
    )
  )
}

#' Aggregate segregation days per person per year (b02)
#' @param data b02 data.frame.
#' @export
morie_otis_analyze_b02 <- function(data) {
  s <- .otis_summary_lines(data, "b02",
    description = "Segregation total days per person per fiscal year")
  daycol <- "TotalAggregatedDays_Segregation"
  if (daycol %in% names(data)) {
    d <- suppressWarnings(as.numeric(data[[daycol]]))
    s[["Mean total days"]] <- round(mean(d, na.rm = TRUE), 2)
    s[["Median total days"]] <- stats::median(d, na.rm = TRUE)
    s[["Max total days"]] <- as.integer(max(d, na.rm = TRUE))
  }
  .otis_wrap(
    "b02 -- Segregation total days per person per fiscal year",
    s,
    list(
      .otis_year_trend(data, daycol),
      .otis_crosstab(data, "Gender", "Region_MostRecentPlacement",
                     daycol)
    )
  )
}

#' Segregation placements by alert x institution (b03)
#' @param data b03 data.frame.
#' @export
morie_otis_analyze_b03 <- function(data) {
  s <- .otis_summary_lines(data, "b03",
    description = "Placements by alert/hold flag x institution")
  .otis_wrap(
    "b03 -- Segregation placements by alert/hold flag x institution",
    s,
    list(
      .otis_crosstab(data, "Alert_Type", "Alert_Presence",
                     "Number_SegregationPlacements"),
      .otis_crosstab(data, "Region_AtTimeOfPlacement", "Alert_Type",
                     "Number_SegregationPlacements")
    )
  )
}

#' Placement durations by region & gender (b04)
#' @param data b04 data.frame.
#' @export
morie_otis_analyze_b04 <- function(data) {
  s <- .otis_summary_lines(data, "b04",
    description = "Placement durations (max/median/mode) by region & gender")
  .otis_wrap(
    "b04 -- Placement durations by region & gender",
    s,
    list(.otis_crosstab(data, "Region_AtTimeOfPlacement", "Measure",
                        "NumberConsecutiveDays_Segregation",
                        aggfunc = "max"))
  )
}

#' Distribution of placements by binned duration (b05)
#' @param data b05 data.frame.
#' @export
morie_otis_analyze_b05 <- function(data) {
  s <- .otis_summary_lines(data, "b05",
    description = "Distribution by binned duration")
  .otis_wrap(
    "b05 -- Distribution of placements by binned duration",
    s,
    list(.otis_crosstab(data, "Consecutive_Duration", "EndFiscalYear",
                        "Number_SegregationPlacements"))
  )
}

#' Reasons for placement x institution x gender (b06)
#' @param data b06 data.frame.
#' @export
morie_otis_analyze_b06 <- function(data) {
  s <- .otis_summary_lines(data, "b06",
    description = "Reasons for placement by institution & gender")
  .otis_wrap(
    "b06 -- Reasons for placement by institution & gender",
    s,
    list(
      .otis_crosstab(data, "Reason", "EndFiscalYear",
                     "Number_SegregationPlacements"),
      .otis_crosstab(data, "Reason", "Gender",
                     "Number_SegregationPlacements")
    )
  )
}

#' Alerts x gender (b07)
#' @param data b07 data.frame.
#' @export
morie_otis_analyze_b07 <- function(data) {
  s <- .otis_summary_lines(data, "b07",
    description = "Placements with/without alert x gender")
  with_col <- "Number_Segregation_Placements_With_Alert"
  wo_col   <- "Number_Segregation_Placements_Without_Alert"
  rows <- list()
  if (all(c(with_col, wo_col, "EndFiscalYear", "Alert_Type",
            "Gender") %in% names(data))) {
    rows <- lapply(seq_len(nrow(data)), function(i) {
      w  <- .otis_to_int(data[[with_col]][i])
      wo <- .otis_to_int(data[[wo_col]][i])
      tot <- w + wo
      rate <- if (tot > 0L) 100 * w / tot else 0
      c(.otis_to_int(data[["EndFiscalYear"]][i]),
        as.character(data[["Alert_Type"]][i]),
        as.character(data[["Gender"]][i]),
        w, wo, sprintf("%.1f%%", rate))
    })
  }
  .otis_wrap(
    "b07 -- Segregation placements with/without alert x gender",
    s,
    list(list(title = "By alert x gender x year:",
              headers = c("Year", "Alert_Type", "Gender",
                          "With_Alert", "Without_Alert",
                          "% with alert"),
              rows = rows))
  )
}

#' Durations by institution & gender (b08)
#' @param data b08 data.frame.
#' @export
morie_otis_analyze_b08 <- function(data) {
  s <- .otis_summary_lines(data, "b08",
    description = "Placement durations by institution & gender")
  .otis_wrap(
    "b08 -- Placement durations by institution & gender",
    s,
    list(.otis_crosstab(data, "Institution_AtTimeOfPlacement", "Measure",
                        "NumberConsecutiveDays_Segregation",
                        aggfunc = "max", top_rows = 15L))
  )
}

#' Individuals by number of placements x gender (b09)
#' @param data b09 data.frame.
#' @export
morie_otis_analyze_b09 <- function(data) {
  s <- .otis_summary_lines(data, "b09",
    description = "Individuals by number of placements x gender")
  .otis_wrap(
    "b09 -- Individuals by number of placements x gender",
    s,
    list(.otis_crosstab(data, "NumberPlacements_Segregation", "Gender",
                        "NumberIndividuals_Segregation"))
  )
}


# ---------------------------------------------------------------------------
# c-series analyses (custody / RC / seg counts + demographics)
# ---------------------------------------------------------------------------

#' Total individuals x custody/RC/seg x gender (c01)
#' @param data c01 data.frame.
#' @export
morie_otis_analyze_c01 <- function(data) {
  s <- .otis_summary_lines(data, "c01",
    description = "Total individuals x custody/RC/seg x gender")
  rate_rows <- lapply(seq_len(nrow(data)), function(i) {
    cust <- .otis_to_int(data[["NumberIndividuals_InCustody"]][i])
    rc   <- .otis_to_int(data[["NumberIndividuals_RestrictiveConfinement"]][i])
    seg  <- .otis_to_int(data[["NumberIndividuals_Segregation"]][i])
    c(.otis_to_int(data[["EndFiscalYear"]][i]),
      as.character(data[["Gender"]][i]),
      cust, rc, seg,
      sprintf("%.1f%%", if (cust > 0) 100 * rc / cust else 0),
      sprintf("%.1f%%", if (cust > 0) 100 * seg / cust else 0))
  })
  .otis_wrap(
    "c01 -- Total individuals x custody/RC/seg x gender",
    s,
    list(list(title = "Cohort sizes + ratios:",
              headers = c("Year", "Gender", "Custody", "RC", "Seg",
                          "RC/custody", "Seg/custody"),
              rows = rate_rows))
  )
}

#' Individuals in RC/seg by institution (c02)
#' @param data c02 data.frame.
#' @export
morie_otis_analyze_c02 <- function(data) {
  s <- .otis_summary_lines(data, "c02",
    description = "Individuals in RC/seg by institution x region x gender")
  .otis_wrap(
    "c02 -- Individuals in RC/seg by institution x region x gender",
    s,
    list(.otis_crosstab(data, "Institution_MostRecentPlacement",
                        "EndFiscalYear",
                        "NumberIndividuals_RestrictiveConfinement",
                        top_rows = 15L))
  )
}

#' Individuals x race x gender (c03)
#' @param data c03 data.frame.
#' @export
morie_otis_analyze_c03 <- function(data) {
  s <- .otis_summary_lines(data, "c03",
    description = "Individuals x race x gender")
  cols <- c("NumberIndividuals_InCustody",
            "NumberIndividuals_RestrictiveConfinement",
            "NumberIndividuals_Segregation")
  rows <- list()
  if (all(c("Race", cols) %in% names(data))) {
    agg <- stats::aggregate(
      data[, cols, drop = FALSE],
      by = list(Race = data[["Race"]]),
      FUN = function(x) sum(as.numeric(x), na.rm = TRUE)
    )
    agg <- agg[order(-agg[[cols[1]]]), , drop = FALSE]
    rows <- lapply(seq_len(nrow(agg)), function(i) {
      cust <- .otis_to_int(agg[[cols[1]]][i])
      rc   <- .otis_to_int(agg[[cols[2]]][i])
      seg  <- .otis_to_int(agg[[cols[3]]][i])
      c(as.character(agg$Race[i]), cust, rc, seg,
        sprintf("%.1f%%", if (cust > 0) 100 * rc / cust else 0),
        sprintf("%.1f%%", if (cust > 0) 100 * seg / cust else 0))
    })
  }
  .otis_wrap(
    "c03 -- Individuals x race x gender",
    s,
    list(list(title = "By race (totals across years/genders):",
              headers = c("Race", "Custody", "RC", "Seg",
                          "RC/custody", "Seg/custody"),
              rows = rows)),
    interpretation = paste(
      "Race x confinement disparities are visible in the RC/custody",
      "and Seg/custody ratios. Compare across race categories --",
      "the gap between Indigenous and White rates is a documented",
      "Ontario-corrections finding."
    )
  )
}

.otis_c_simple <- function(ds_id, description, row_col,
                            col_col = "Region_MostRecentPlacement",
                            value_col = "NumberIndividuals_RestrictiveConfinement") {
  function(data) {
    s <- .otis_summary_lines(data, ds_id, description = description)
    .otis_wrap(
      sprintf("%s -- %s", ds_id, description),
      s,
      list(.otis_crosstab(data, row_col, col_col, value_col))
    )
  }
}

#' @export
morie_otis_analyze_c04 <- .otis_c_simple(
  "c04", "Individuals in RC/seg x race x region", "Race")
#' @export
morie_otis_analyze_c05 <- .otis_c_simple(
  "c05", "Individuals in RC/seg x religion x region", "Religion")
#' @export
morie_otis_analyze_c06 <- .otis_c_simple(
  "c06", "Individuals in RC/seg x age category x region", "Age_Category")

#' Individuals x alerts x gender (c07)
#' @param data c07 data.frame.
#' @export
morie_otis_analyze_c07 <- function(data) {
  s <- .otis_summary_lines(data, "c07",
    description = "Individuals in custody/RC/seg x alert type x gender")
  .otis_wrap(
    "c07 -- Individuals in custody/RC/seg x alert type x gender",
    s,
    list(
      .otis_crosstab(data, "Alert_Type", "Gender",
                     "NumberIndividuals_RestrictiveConfinement"),
      .otis_crosstab(data, "Alert_Type", "EndFiscalYear",
                     "NumberIndividuals_Segregation")
    )
  )
}

#' @export
morie_otis_analyze_c08 <- .otis_c_simple(
  "c08", "Individuals x religion x gender", "Religion", "Gender")
#' @export
morie_otis_analyze_c09 <- .otis_c_simple(
  "c09", "Individuals x age category x gender", "Age_Category", "Gender")

#' RC/seg aggregate durations by institution (c10)
#' @param data c10 data.frame.
#' @export
morie_otis_analyze_c10 <- function(data) {
  s <- .otis_summary_lines(data, "c10",
    description = "RC/seg aggregate durations by institution")
  .otis_wrap(
    "c10 -- RC/seg aggregate durations by institution",
    s,
    list(.otis_crosstab(data, "Institution_MostRecentPlacement", "Measure",
                        "TotalAggregatedDays_RestrictiveConfinement",
                        aggfunc = "max", top_rows = 15L))
  )
}

#' Individuals by aggregate-duration bin (c11)
#' @param data c11 data.frame.
#' @export
morie_otis_analyze_c11 <- function(data) {
  s <- .otis_summary_lines(data, "c11",
    description = "Individuals by binned aggregate duration")
  .otis_wrap(
    "c11 -- Individuals by binned aggregate duration",
    s,
    list(.otis_crosstab(data, "Aggregate_Duration", "EndFiscalYear",
                        "NumberIndividuals_RestrictiveConfinement"))
  )
}

#' RC/seg aggregate durations by region & gender (c12)
#' @param data c12 data.frame.
#' @export
morie_otis_analyze_c12 <- function(data) {
  s <- .otis_summary_lines(data, "c12",
    description = "RC/seg aggregate durations by region & gender")
  .otis_wrap(
    "c12 -- RC/seg aggregate durations by region & gender",
    s,
    list(.otis_crosstab(data, "Region_MostRecentPlacement", "Measure",
                        "TotalAggregatedDays_RestrictiveConfinement",
                        aggfunc = "max"))
  )
}


# ---------------------------------------------------------------------------
# d-series analyses (custodial deaths)
# ---------------------------------------------------------------------------

#' Person-level custodial deaths (d01)
#' @param data d01 data.frame.
#' @export
morie_otis_analyze_d01 <- function(data) {
  s <- .otis_summary_lines(data, "d01",
    description = "Custodial deaths (person-level)")
  if ("UniqueIndividual_ID" %in% names(data)) {
    s[["Distinct individuals"]] <-
      length(unique(data[["UniqueIndividual_ID"]]))
  }
  cause_col <- if ("MedicalCauseofDeath" %in% names(data))
    "MedicalCauseofDeath" else "MedicalCauseOfDeath"
  means_col <- if ("MeansofDeath" %in% names(data))
    "MeansofDeath" else "MeansOfDeath"
  .tab <- function(col, title, hkey) {
    if (!(col %in% names(data))) return(NULL)
    tab <- sort(table(data[[col]]), decreasing = TRUE)
    list(title = title, headers = c(hkey, "Deaths"),
         rows = lapply(seq_along(tab),
           function(i) c(names(tab)[i], as.integer(tab[i]))))
  }
  .otis_wrap(
    "d01 -- Custodial deaths (person-level)", s,
    list(.tab("Region_AtTimeOfDeath", "By region:", "Region"),
         .tab("HousingUnit_Type", "By housing unit type:",
              "HousingUnit"),
         .tab(cause_col, "By medical cause:", "Cause"),
         .tab(means_col, "By means of death:", "Means"))
  )
}

.otis_d_simple <- function(ds_id, description, by) {
  function(data) {
    s <- .otis_summary_lines(data, ds_id, description = description)
    .otis_wrap(
      sprintf("%s -- %s", ds_id, description),
      s,
      list(.otis_year_trend(data, "Number_CustodialDeaths"),
           .otis_crosstab(data, by, "Year", "Number_CustodialDeaths"))
    )
  }
}

#' @export
morie_otis_analyze_d02 <- .otis_d_simple(
  "d02", "Custodial deaths x gender", "Gender")
#' @export
morie_otis_analyze_d03 <- .otis_d_simple(
  "d03", "Custodial deaths x race", "Race")
#' @export
morie_otis_analyze_d04 <- .otis_d_simple(
  "d04", "Custodial deaths x religion", "Religion")
#' @export
morie_otis_analyze_d05 <- .otis_d_simple(
  "d05", "Custodial deaths x age category", "Age_Category")

#' @export
morie_otis_analyze_d06 <- function(data) {
  s <- .otis_summary_lines(data, "d06",
    description = "Custodial deaths x alert x medical cause")
  .otis_wrap(
    "d06 -- Custodial deaths x alert x medical cause", s,
    list(.otis_crosstab(data, "MedicalCauseOfDeath", "Alert_Type",
                        "Number_CustodialDeaths"))
  )
}

#' @export
morie_otis_analyze_d07 <- function(data) {
  s <- .otis_summary_lines(data, "d07",
    description = "Custodial deaths x alert x housing unit")
  .otis_wrap(
    "d07 -- Custodial deaths x alert x housing unit", s,
    list(.otis_crosstab(data, "HousingUnit_Type", "Alert_Type",
                        "Number_CustodialDeaths"))
  )
}


# ---------------------------------------------------------------------------
# Master driver
# ---------------------------------------------------------------------------

#' Registry of dataset-id -> analyzer
#'
#' Mirrors \code{_ANALYSES} in
#' \code{src/morie/otis_all_analyze.py}.
#' @export
morie_otis_analyzers <- function() {
  list(
    b01 = morie_otis_analyze_b01, b02 = morie_otis_analyze_b02,
    b03 = morie_otis_analyze_b03, b04 = morie_otis_analyze_b04,
    b05 = morie_otis_analyze_b05, b06 = morie_otis_analyze_b06,
    b07 = morie_otis_analyze_b07, b08 = morie_otis_analyze_b08,
    b09 = morie_otis_analyze_b09,
    c01 = morie_otis_analyze_c01, c02 = morie_otis_analyze_c02,
    c03 = morie_otis_analyze_c03, c04 = morie_otis_analyze_c04,
    c05 = morie_otis_analyze_c05, c06 = morie_otis_analyze_c06,
    c07 = morie_otis_analyze_c07, c08 = morie_otis_analyze_c08,
    c09 = morie_otis_analyze_c09, c10 = morie_otis_analyze_c10,
    c11 = morie_otis_analyze_c11, c12 = morie_otis_analyze_c12,
    d01 = morie_otis_analyze_d01, d02 = morie_otis_analyze_d02,
    d03 = morie_otis_analyze_d03, d04 = morie_otis_analyze_d04,
    d05 = morie_otis_analyze_d05, d06 = morie_otis_analyze_d06,
    d07 = morie_otis_analyze_d07
  )
}

#' Run every OTIS analyzer against a named list of datasets
#'
#' Counterpart to Python \code{analyze_all()}.
#'
#' @param datasets Named list \code{list(b01 = <df>, c01 = <df>, ...)}.
#'   IDs absent from the list are skipped silently.
#' @param out_dir Optional directory to write per-dataset
#'   \code{overlay_<id>.rds} files. \code{NULL} means in-memory only.
#' @return Named list of \code{morie_otis_analysis_result}s.
#' @export
morie_otis_analyze_all <- function(datasets, out_dir = NULL) {
  stopifnot(is.list(datasets))
  out <- list()
  for (id in names(datasets)) {
    fn <- morie_otis_analyzers()[[id]]
    if (is.null(fn)) next
    out[[id]] <- tryCatch(
      fn(datasets[[id]]),
      error = function(e) .otis_wrap(
        title = sprintf("%s (failed)", id),
        summary_lines = list(error = conditionMessage(e)),
        warnings = conditionMessage(e)
      )
    )
    if (!is.null(out_dir)) {
      dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
      saveRDS(out[[id]], file.path(out_dir,
                                    sprintf("otis_%s.rds", id)))
    }
  }
  out
}

#' @export
print.morie_otis_analysis_result <- function(x, ...) {
  cat(x$title, "\
", strrep("=", nchar(x$title)), "\
", sep = "")
  if (length(x$summary_lines) > 0L) {
    nms <- names(x$summary_lines)
    lw <- max(nchar(nms))
    for (i in seq_along(x$summary_lines)) {
      cat(sprintf("  %-*s  %s\
", lw, nms[i],
                  format(x$summary_lines[[i]])))
    }
    cat("\
")
  }
  for (w in x$warnings) cat("Warning:", w, "\
")
  if (nzchar(x$interpretation)) cat(x$interpretation, "\
")
  invisible(x)
}
