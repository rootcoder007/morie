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
    fixture = "nypd_arrests_historic_sample.csv",
    permalink = "https://data.cityofnewyork.us/d/8h9b-rp9u",
    data_dictionary_url = NA_character_,
    footnotes_url = NA_character_),
  nypd_arrests_ytd = list(
    resource_id = "uip8-fykc",
    label = "NYPD Arrest Data (Year to Date)",
    fixture = "nypd_arrests_ytd_sample.csv",
    permalink = "https://data.cityofnewyork.us/d/uip8-fykc",
    data_dictionary_url = paste0(
      "https://data.cityofnewyork.us/api/views/uip8-fykc/files/",
      "f0dbff24-5794-4034-a52d-b091e8dd61a8?download=true",
      "&filename=NYPD_Arrest_YTD_DataDictionary.xlsx"),
    footnotes_url = paste0(
      "https://data.cityofnewyork.us/api/views/uip8-fykc/files/",
      "62a746df-66ca-4603-aae4-46c02bac2972?download=true",
      "&filename=NYPD_Arrest_Incident_Level_Data_Footnotes.pdf")),
  nypd_complaint_historic = list(
    resource_id = "qgea-i56i",
    label = "NYPD Complaint Data Historic",
    fixture = "nypd_complaint_historic_sample.csv",
    permalink = "https://data.cityofnewyork.us/d/qgea-i56i",
    data_dictionary_url = NA_character_,
    footnotes_url = NA_character_),
  nypd_complaint_ytd = list(
    resource_id = "5uac-w243",
    label = "NYPD Complaint Data Current (Year To Date)",
    fixture = "nypd_complaint_ytd_sample.csv",
    permalink = "https://data.cityofnewyork.us/d/5uac-w243",
    data_dictionary_url = NA_character_,
    footnotes_url = NA_character_),
  nypd_hate_crimes = list(
    resource_id = "bqiq-cu78",
    label = "NYPD Hate Crimes",
    fixture = "nypd_hate_crimes_sample.csv",
    permalink = "https://data.cityofnewyork.us/d/bqiq-cu78",
    data_dictionary_url = NA_character_,
    footnotes_url = NA_character_),
  nypd_uof_incidents = list(
    resource_id = "f4tj-796d",
    label = "NYPD Use of Force Incidents",
    fixture = "nypd_uof_incidents_sample.csv",
    permalink = "https://data.cityofnewyork.us/d/f4tj-796d",
    data_dictionary_url = NA_character_,
    footnotes_url = NA_character_),
  nypd_uof_subjects = list(
    resource_id = "dufe-vxb7",
    label = "NYPD Use of Force: Subjects",
    fixture = "nypd_uof_subjects_sample.csv",
    permalink = "https://data.cityofnewyork.us/d/dufe-vxb7",
    data_dictionary_url = NA_character_,
    footnotes_url = NA_character_),
  nypd_vehicle_stops = list(
    resource_id = "hn9i-dwpr",
    label = "NYPD Vehicle Stop Reports",
    fixture = "nypd_vehicle_stops_sample.csv",
    permalink = "https://data.cityofnewyork.us/d/hn9i-dwpr",
    data_dictionary_url = NA_character_,
    footnotes_url = NA_character_))

#' List the NYPD criminal-justice Socrata datasets wrapped by morie
#'
#' @return A `data.frame` with 8 columns:
#'   `dataset_key`, `label`, `resource_id`, `resource_url`,
#'   `permalink` (`data.cityofnewyork.us/d/<id>` stable redirect),
#'   `data_dictionary_url` (XLSX, when published as a dataset
#'   attachment; `NA_character_` otherwise),
#'   `footnotes_url` (PDF, when published; `NA_character_` otherwise),
#'   `fixture` (bundled-fixture filename).
#'
#' Currently only `nypd_arrests_ytd` carries the
#' canonical NYC OpenData attachment URLs (XLSX dictionary + PDF
#' footnotes). The other 7 entries leave those slots `NA`; PRs
#' welcome to fill them in when the asset UUIDs are looked up at
#' the dataset's landing page.
#'
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
      permalink = e$permalink,
      data_dictionary_url = e$data_dictionary_url,
      footnotes_url = e$footnotes_url,
      fixture = e$fixture,
      stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' Socrata default-API-cap note + pagination wiring
#'
#' All NYC OpenData SODA2 endpoints apply a default cap of 1,000 rows
#' per request unless an explicit `$limit` (or `$$app_token` for
#' authenticated requests) is supplied. For the NYPD CJ datasets
#' wrapped here that means:
#'
#'   * `morie_datasets_nyc_nypd_arrests_ytd(offline = FALSE)` returns
#'     **only 1,000 rows** by default, even though the live feed
#'     carries ~69,300 rows.
#'   * Pass `max_features = N` to lift the single-request cap to N
#'     rows (Socrata enforces a hard server-side cap of 50,000 rows
#'     per request).
#'   * **Pagination (wired in 3OO).** For full pulls over the cap,
#'     pass `paginate = TRUE`. morie walks SODA2 `$offset` in
#'     `page_size`-row chunks until the server returns a short page
#'     (exhausted) or `max_features` is reached. Without an app_token
#'     the per-request ceiling is 1,000 rows so `page_size = 1000` is
#'     the default; with `page_size = 50000` + `app_token` you can
#'     pull the full ~69K-row arrests_ytd feed in two requests.
#'     `max_pages` (default 200) is a safety net against runaway pulls.
#'
#' Worked example:
#'
#' ```r
#' # Full live pull of the YTD arrests feed (~69K rows over ~70 pages).
#' df <- morie_datasets_nyc_nypd_arrests_ytd(
#'   offline = FALSE, paginate = TRUE)
#'
#' # First 5,000 rows only (5 paged requests of 1,000 each).
#' df <- morie_datasets_nyc_nypd_arrests_ytd(
#'   offline = FALSE, paginate = TRUE, max_features = 5000L)
#' ```
#'
#' The bundled fixtures (offline mode) are unaffected -- they ship 5
#' rows each as deterministic sample data, and `max_features` simply
#' truncates the fixture.
#'
#' @name morie_nyc_nypd_socrata_cap_note
NULL

# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

.morie_nyc_nypd_dispatch <- function(dataset_key, year, max_features,
                                       offline, resource_id,
                                       paginate = FALSE,
                                       page_size = 1000L,
                                       max_pages = 200L) {
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
                                max_features = max_features,
                                paginate = paginate,
                                page_size = page_size,
                                max_pages = max_pages)
}

#' Generic NYC NYPD dataset loader by registry key
#'
#' @param dataset_key One of the keys in
#'   [morie_datasets_nyc_nypd_layers()].
#' @param year Optional year filter (server-side SoQL).
#' @param max_features Optional row cap. When `paginate = TRUE` this
#'   is the total cap across walked pages.
#' @param offline If `TRUE` (default), read the bundled fixture.
#' @param resource_id Optional Socrata resource id override.
#' @param paginate Logical; if `TRUE` and `offline = FALSE`, walk
#'   SODA2 `$offset` in `page_size` chunks until exhausted or
#'   `max_features` is reached. Default `FALSE` (single 1,000-row
#'   request, matching the historical pre-3OO behaviour).
#' @param page_size Per-page row count when paginating (default 1,000,
#'   the unauthenticated SODA2 ceiling).
#' @param max_pages Safety net on paginated walks (default 200 ->
#'   up to 200,000 rows without an app_token).
#' @return A `data.frame`.
#' @export
morie_datasets_nyc_nypd_by_key <- function(dataset_key,
                                             year = NULL,
                                             max_features = NULL,
                                             offline = TRUE,
                                             resource_id = NULL,
                                             paginate = FALSE,
                                             page_size = 1000L,
                                             max_pages = 200L) {
  .morie_nyc_nypd_dispatch(dataset_key, year, max_features,
                             offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}

# ---------------------------------------------------------------------------
# Eight per-dataset wrappers
# ---------------------------------------------------------------------------
#
# All 8 share an identical signature -- year + max_features + offline +
# resource_id (existing) plus the 3-arg pagination tail wired in 3OO
# (paginate + page_size + max_pages). They forward verbatim to
# .morie_nyc_nypd_dispatch().

#' NYPD Arrests Data (Historic)
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_arrests_historic <- function(year = NULL,
                                                       max_features = NULL,
                                                       offline = TRUE,
                                                       resource_id = NULL,
                                                       paginate = FALSE,
                                                       page_size = 1000L,
                                                       max_pages = 200L) {
  .morie_nyc_nypd_dispatch("nypd_arrests_historic", year,
                             max_features, offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}

#' NYPD Arrest Data (Year to Date)
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_arrests_ytd <- function(year = NULL,
                                                  max_features = NULL,
                                                  offline = TRUE,
                                                  resource_id = NULL,
                                                  paginate = FALSE,
                                                  page_size = 1000L,
                                                  max_pages = 200L) {
  .morie_nyc_nypd_dispatch("nypd_arrests_ytd", year, max_features,
                             offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}

#' NYPD Complaint Data Historic
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_complaint_historic <- function(year = NULL,
                                                         max_features = NULL,
                                                         offline = TRUE,
                                                         resource_id = NULL,
                                                         paginate = FALSE,
                                                         page_size = 1000L,
                                                         max_pages = 200L) {
  .morie_nyc_nypd_dispatch("nypd_complaint_historic", year,
                             max_features, offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}

#' NYPD Complaint Data Current (Year To Date)
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_complaint_ytd <- function(year = NULL,
                                                    max_features = NULL,
                                                    offline = TRUE,
                                                    resource_id = NULL,
                                                    paginate = FALSE,
                                                    page_size = 1000L,
                                                    max_pages = 200L) {
  .morie_nyc_nypd_dispatch("nypd_complaint_ytd", year,
                             max_features, offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}

#' NYPD Hate Crimes
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_hate_crimes <- function(year = NULL,
                                                  max_features = NULL,
                                                  offline = TRUE,
                                                  resource_id = NULL,
                                                  paginate = FALSE,
                                                  page_size = 1000L,
                                                  max_pages = 200L) {
  .morie_nyc_nypd_dispatch("nypd_hate_crimes", year, max_features,
                             offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}

#' NYPD Use of Force Incidents
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_uof_incidents <- function(year = NULL,
                                                    max_features = NULL,
                                                    offline = TRUE,
                                                    resource_id = NULL,
                                                    paginate = FALSE,
                                                    page_size = 1000L,
                                                    max_pages = 200L) {
  .morie_nyc_nypd_dispatch("nypd_uof_incidents", year, max_features,
                             offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}

#' NYPD Use of Force: Subjects
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_uof_subjects <- function(year = NULL,
                                                   max_features = NULL,
                                                   offline = TRUE,
                                                   resource_id = NULL,
                                                   paginate = FALSE,
                                                   page_size = 1000L,
                                                   max_pages = 200L) {
  .morie_nyc_nypd_dispatch("nypd_uof_subjects", year, max_features,
                             offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}

#' NYPD Vehicle Stop Reports
#' @inheritParams morie_datasets_nyc_nypd_by_key
#' @export
morie_datasets_nyc_nypd_vehicle_stops <- function(year = NULL,
                                                    max_features = NULL,
                                                    offline = TRUE,
                                                    resource_id = NULL,
                                                    paginate = FALSE,
                                                    page_size = 1000L,
                                                    max_pages = 200L) {
  .morie_nyc_nypd_dispatch("nypd_vehicle_stops", year, max_features,
                             offline, resource_id,
                             paginate = paginate,
                             page_size = page_size,
                             max_pages = max_pages)
}
