# SPDX-License-Identifier: AGPL-3.0-or-later
#
# NYC NYPD criminal-justice Socrata wrappers.
#
# Eight canonical NYPD datasets published under the NYC OpenData
# program (data.cityofnewyork.us). Resource ids verified live via
# https://api.us.socrata.com/api/catalog/v1?q=NYPD&domains=data.cityofnewyork.us
# during phase 3NN. Each loader follows the offline-default pattern
# established in phase 3LL:
#
#   offline = TRUE  (default) -> read bundled inst/extdata/*.csv
#   offline = FALSE            -> hit SODA2 endpoint via the mockable
#                                  .morie_dataset_socrata_fetch helper
#
# Live mode honours an explicit resource_id override; otherwise the
# canonical id from .MORIE_NYC_NYPD_REGISTRY is used.

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

.MORIE_NYC_NYPD_REGISTRY <- list(
  nypd_arrests_historic = list(
    resource_id = "8h9b-rp9u",
    label = "NYPD Arrests Data (Historic)",
    fixture = "nypd_arrests_historic_sample.csv"),
  nypd_arrests_ytd = list(
    resource_id = "uip8-fykc",
    label = "NYPD Arrest Data (Year to Date)",
    fixture = "nypd_arrests_ytd_sample.csv"),
  nypd_complaint_historic = list(
    resource_id = "qgea-i56i",
    label = "NYPD Complaint Data Historic",
    fixture = "nypd_complaint_historic_sample.csv"),
  nypd_complaint_ytd = list(
    resource_id = "5uac-w243",
    label = "NYPD Complaint Data Current (Year To Date)",
    fixture = "nypd_complaint_ytd_sample.csv"),
  nypd_hate_crimes = list(
    resource_id = "bqiq-cu78",
    label = "NYPD Hate Crimes",
    fixture = "nypd_hate_crimes_sample.csv"),
  nypd_uof_incidents = list(
    resource_id = "f4tj-796d",
    label = "NYPD Use of Force Incidents",
    fixture = "nypd_uof_incidents_sample.csv"),
  nypd_uof_subjects = list(
    resource_id = "dufe-vxb7",
    label = "NYPD Use of Force: Subjects",
    fixture = "nypd_uof_subjects_sample.csv"),
  nypd_vehicle_stops = list(
    resource_id = "hn9i-dwpr",
    label = "NYPD Vehicle Stop Reports",
    fixture = "nypd_vehicle_stops_sample.csv"))

#' List the NYPD criminal-justice Socrata datasets wrapped by morie
#'
#' @return A `data.frame` with columns `dataset_key`, `label`,
#'   `resource_id`, `resource_url`, `fixture`.
#' @export
morie_datasets_nyc_nypd_layers <- function() {
  rows <- lapply(names(.MORIE_NYC_NYPD_REGISTRY), function(k) {
    e <- .MORIE_NYC_NYPD_REGISTRY[[k]]
    data.frame(
      dataset_key = k, label = e$label,
      resource_id = e$resource_id,
      resource_url = sprintf(
        "https://data.cityofnewyork.us/resource/%s.json",
        e$resource_id),
      fixture = e$fixture,
      stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

.morie_nyc_nypd_dispatch <- function(dataset_key, year, max_features,
                                       offline, resource_id) {
  if (!(dataset_key %in% names(.MORIE_NYC_NYPD_REGISTRY))) {
    stop(sprintf(paste0(
      "unknown NYC NYPD dataset_key '%s'. Available: %s"),
      dataset_key,
      paste(names(.MORIE_NYC_NYPD_REGISTRY), collapse = ", ")),
      call. = FALSE)
  }
  entry <- .MORIE_NYC_NYPD_REGISTRY[[dataset_key]]
  if (isTRUE(offline)) {
    path <- system.file("extdata", entry$fixture, package = "morie")
    if (!nzchar(path)) {
      stop(sprintf("bundled NYC NYPD fixture %s missing",
                   entry$fixture), call. = FALSE)
    }
    df <- utils::read.csv(path, stringsAsFactors = FALSE,
                           check.names = FALSE)
    if (!is.null(max_features)) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  if (is.null(resource_id)) resource_id <- entry$resource_id
  url <- sprintf("https://data.cityofnewyork.us/resource/%s.json",
                 resource_id)
  # Honour an OCC-year-style filter via SoQL $where when supplied.
  where <- NULL
  if (!is.null(year)) {
    # Use the first year-like column we know about per dataset_key.
    year_col <- switch(dataset_key,
      "nypd_arrests_historic"   = "arrest_date",
      "nypd_arrests_ytd"        = "arrest_date",
      "nypd_complaint_historic" = "cmplnt_fr_dt",
      "nypd_complaint_ytd"      = "cmplnt_fr_dt",
      "nypd_hate_crimes"        = "complaint_year_number",
      "nypd_uof_incidents"      = "occurrence_date",
      "nypd_uof_subjects"       = NULL,
      "nypd_vehicle_stops"      = "occur_dt",
      NULL)
    if (!is.null(year_col)) {
      if (year_col == "complaint_year_number") {
        where <- sprintf("%s = %d", year_col, as.integer(year))
      } else {
        where <- sprintf("date_extract_y(%s) = %d",
                          year_col, as.integer(year))
      }
    }
  }
  .morie_dataset_socrata_fetch(url, where = where,
                                max_features = max_features)
}

#' Generic NYC NYPD dataset loader by registry key
#'
#' @param dataset_key One of the keys in
#'   [morie_datasets_nyc_nypd_layers()].
#' @param year Optional year filter (server-side SoQL).
#' @param max_features Optional row cap.
#' @param offline If `TRUE` (default), read the bundled fixture.
#' @param resource_id Optional Socrata resource id override.
#' @return A `data.frame`.
#' @export
morie_datasets_nyc_nypd_by_key <- function(dataset_key,
                                             year = NULL,
                                             max_features = NULL,
                                             offline = TRUE,
                                             resource_id = NULL) {
  .morie_nyc_nypd_dispatch(dataset_key, year, max_features,
                             offline, resource_id)
}

# ---------------------------------------------------------------------------
# Eight per-dataset wrappers
# ---------------------------------------------------------------------------

#' NYPD Arrests Data (Historic)
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_arrests_historic <- function(year = NULL,
                                                       max_features = NULL,
                                                       offline = TRUE,
                                                       resource_id = NULL) {
  .morie_nyc_nypd_dispatch("nypd_arrests_historic", year,
                             max_features, offline, resource_id)
}

#' NYPD Arrest Data (Year to Date)
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_arrests_ytd <- function(year = NULL,
                                                  max_features = NULL,
                                                  offline = TRUE,
                                                  resource_id = NULL) {
  .morie_nyc_nypd_dispatch("nypd_arrests_ytd", year, max_features,
                             offline, resource_id)
}

#' NYPD Complaint Data Historic
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_complaint_historic <- function(year = NULL,
                                                         max_features = NULL,
                                                         offline = TRUE,
                                                         resource_id = NULL) {
  .morie_nyc_nypd_dispatch("nypd_complaint_historic", year,
                             max_features, offline, resource_id)
}

#' NYPD Complaint Data Current (Year To Date)
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_complaint_ytd <- function(year = NULL,
                                                    max_features = NULL,
                                                    offline = TRUE,
                                                    resource_id = NULL) {
  .morie_nyc_nypd_dispatch("nypd_complaint_ytd", year,
                             max_features, offline, resource_id)
}

#' NYPD Hate Crimes
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_hate_crimes <- function(year = NULL,
                                                  max_features = NULL,
                                                  offline = TRUE,
                                                  resource_id = NULL) {
  .morie_nyc_nypd_dispatch("nypd_hate_crimes", year, max_features,
                             offline, resource_id)
}

#' NYPD Use of Force Incidents
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_uof_incidents <- function(year = NULL,
                                                    max_features = NULL,
                                                    offline = TRUE,
                                                    resource_id = NULL) {
  .morie_nyc_nypd_dispatch("nypd_uof_incidents", year, max_features,
                             offline, resource_id)
}

#' NYPD Use of Force: Subjects
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_uof_subjects <- function(year = NULL,
                                                   max_features = NULL,
                                                   offline = TRUE,
                                                   resource_id = NULL) {
  .morie_nyc_nypd_dispatch("nypd_uof_subjects", year, max_features,
                             offline, resource_id)
}

#' NYPD Vehicle Stop Reports
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_vehicle_stops <- function(year = NULL,
                                                    max_features = NULL,
                                                    offline = TRUE,
                                                    resource_id = NULL) {
  .morie_nyc_nypd_dispatch("nypd_vehicle_stops", year, max_features,
                             offline, resource_id)
}
