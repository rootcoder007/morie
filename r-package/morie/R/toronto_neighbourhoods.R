# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Toronto neighbourhood boundary-version awareness (HOOD_158 vs HOOD_140)
# plus Neighbourhood Improvement Area (NIA) loader and 158<->140 crosswalk.
#
# Why this module exists
# ----------------------
# The City of Toronto publishes TWO parallel neighbourhood schemas:
#
#   * 158 social-planning neighbourhoods (CURRENT, adopted 2022) -- the
#     column `HOOD_158` / `NEIGHBOURHOOD_158` on TPS PSDP feeds.
#   * 140 social-planning neighbourhoods (HISTORICAL, in force
#     ~2014-2021) -- the column `HOOD_140` / `NEIGHBOURHOOD_140`.
#
# The polygons are NOT identical: some 140s were split, some merged,
# and a handful were renamed. TPS crime feeds back-fill BOTH columns
# onto historical records via lat/lon re-geocoding, so a naive analysis
# silently mixes the two schemas across years.
#
# This module gives morie callers explicit version awareness:
#   * morie_to_neighbourhoods()         -- load polygons for 158 / 140 / NIA
#   * morie_tps_resolve_hood_col()      -- pick the right hood column
#   * morie_tps_assert_hood_version()   -- error / warn on version drift
#   * morie_tps_year_to_hood_version()  -- recommended schema per year
#   * morie_to_hood_crosswalk()         -- bundled 140<->158 mapping
#
# Upstream sources
# ----------------
#   City of Toronto Open Data (Open Government Licence -- Toronto):
#     - https://open.toronto.ca/dataset/neighbourhoods/
#     - https://open.toronto.ca/dataset/neighbourhood-improvement-areas/
#     - https://open.toronto.ca/dataset/neighbourhood-crime-rates/
#   CKAN datastore dump (live mode):
#     - https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/
#         datastore_search?resource_id=<id>
#
# Bundled small synthetic fixtures live in inst/extdata/:
#   to_neighbourhoods_158.csv
#   to_neighbourhoods_140.csv
#   to_neighbourhood_improvement_areas.csv
#   to_hood_158_140_crosswalk.csv

#' Toronto neighbourhood boundary versions
#'
#' Wraps loading of the City of Toronto neighbourhood polygon layers
#' for the CURRENT 158-neighbourhood scheme (`HOOD_158`), the
#' HISTORICAL 140-neighbourhood scheme (`HOOD_140`), and the
#' Neighbourhood Improvement Area (NIA) layer. Also provides
#' version-resolution helpers so downstream analyses do not silently
#' mix the two schemes across years.
#'
#' @name morie_toronto_neighbourhoods
NULL

# Map a "version" string to the bundled fixture filename.
.morie_to_fixture_name <- function(version) {
  switch(version,
    "158" = "to_neighbourhoods_158.csv",
    "140" = "to_neighbourhoods_140.csv",
    "nia" = "to_neighbourhood_improvement_areas.csv",
    stop("unknown version: ", version, call. = FALSE))
}

# Read a bundled neighbourhood fixture from inst/extdata.
.morie_to_neighbourhoods_fixture <- function(version) {
  fname <- .morie_to_fixture_name(version)
  path <- system.file("extdata", fname, package = "morie")
  if (!nzchar(path)) {
    stop(sprintf(paste0(
      "Bundled fixture %s not found in the installed morie package. ",
      "Re-install morie, or pass offline = FALSE to fetch from ",
      "open.toronto.ca CKAN."), fname), call. = FALSE)
  }
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

# Internal: live CKAN datastore_search fetcher. Mockable via
# testthat::local_mocked_bindings(.morie_to_ckan_dump_csv = ...).
.morie_to_ckan_dump_csv <- function(resource_id, limit = 100000L) {
  if (!requireNamespace("httr2", quietly = TRUE) ||
      !requireNamespace("jsonlite", quietly = TRUE)) {
    stop(paste0(
      "Open Toronto CKAN dump fetch needs httr2 + jsonlite. ",
      "install.packages(c('httr2', 'jsonlite'))"),
      call. = FALSE)
  }
  url <- sprintf(paste0(
    "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/",
    "datastore_search?resource_id=%s&limit=%d"),
    resource_id, as.integer(limit))
  req <- httr2::request(url)
  req <- httr2::req_timeout(req, 60)
  resp <- httr2::req_perform(req)
  body <- jsonlite::fromJSON(httr2::resp_body_string(resp),
                              simplifyVector = TRUE)
  recs <- body$result$records
  if (is.null(recs)) data.frame() else as.data.frame(recs)
}

# Default CKAN resource ids for the live mode (current as of 2026-05).
# Pass `resource_id =` to morie_to_neighbourhoods() to override.
.MORIE_TO_DEFAULT_RESOURCE_IDS <- list(
  "158" = "4def3f65-2a65-4a4f-83c4-867e4cca287a",
  "140" = "0c39c4d8-c75c-49ba-9b00-4a3c0c0b5d6e",
  "nia" = "fec0e35a-5da9-4ab3-b69e-66e5a98c8d99")

#' Load a Toronto neighbourhood polygon layer
#'
#' Returns a data.frame with the canonical City of Toronto Open Data
#' schema (`_id`, `AREA_ID`, `AREA_ATTR_ID`, `PARENT_AREA_ID`,
#' `AREA_SHORT_CODE`, `AREA_LONG_CODE`, `AREA_NAME`, `AREA_DESC`,
#' `CLASSIFICATION`, `CLASSIFICATION_CODE`, `OBJECTID`, `geometry`)
#' for the requested version.
#'
#' @param version One of `"158"` (current City scheme), `"140"`
#'   (historical 2014--2021 scheme), or `"nia"` (Neighbourhood
#'   Improvement Areas).
#' @param offline If `TRUE` (default), read the small bundled synthetic
#'   fixture from `inst/extdata/`. If `FALSE`, hit the live City of
#'   Toronto CKAN `datastore_search` endpoint via httr2.
#' @param resource_id Optional CKAN resource id override. Used only
#'   when `offline = FALSE`.
#' @return A `data.frame`.
#' @references City of Toronto Open Data, "Neighbourhoods" dataset
#'   (\url{https://open.toronto.ca/dataset/neighbourhoods/});
#'   "Neighbourhood Improvement Areas"
#'   (\url{https://open.toronto.ca/dataset/neighbourhood-improvement-areas/});
#'   licensed under the Open Government Licence -- Toronto.
#' @examples
#' df <- morie_to_neighbourhoods("158", offline = TRUE)
#' head(df[, c("AREA_SHORT_CODE", "AREA_NAME")])
#' @export
morie_to_neighbourhoods <- function(version = c("158", "140", "nia"),
                                     offline = TRUE,
                                     resource_id = NULL) {
  version <- match.arg(version)
  if (isTRUE(offline)) {
    return(.morie_to_neighbourhoods_fixture(version))
  }
  if (is.null(resource_id)) {
    resource_id <- .MORIE_TO_DEFAULT_RESOURCE_IDS[[version]]
  }
  .morie_to_ckan_dump_csv(resource_id)
}

# Column-name candidates for each version (TPS feeds use HOOD_*;
# Open Toronto layers use NEIGHBOURHOOD_*; some legacy exports use
# lower-case).
.MORIE_TPS_HOOD_VERSIONS <- list(
  "158" = c("HOOD_158", "hood_158", "NEIGHBOURHOOD_158",
            "neighbourhood_158"),
  "140" = c("HOOD_140", "hood_140", "NEIGHBOURHOOD_140",
            "neighbourhood_140"))

#' Resolve which HOOD_* column to use on a TPS crime data.frame
#'
#' Many TPS PSDP crime layers carry BOTH `HOOD_158` (current) and
#' `HOOD_140` (historical 2014--2021) columns. Pick the version your
#' analysis needs explicitly so the two schemes are not silently mixed
#' across years.
#'
#' @param df A TPS crime `data.frame`.
#' @param prefer Either `"158"` (current) or `"140"` (historical).
#' @param fallback If `TRUE` (default), accept the other version when
#'   the preferred one is absent (with a warning). If `FALSE`, return
#'   `NULL` when the preferred version is missing.
#' @return Character scalar (the chosen column name), or `NULL` if no
#'   suitable column is present.
#' @examples
#' df <- data.frame(OCC_YEAR = 2024L, HOOD_158 = "82", HOOD_140 = "82")
#' morie_tps_resolve_hood_col(df, prefer = "158")
#' @export
morie_tps_resolve_hood_col <- function(df, prefer = c("158", "140"),
                                        fallback = TRUE) {
  prefer <- match.arg(prefer)
  other  <- setdiff(c("158", "140"), prefer)
  for (cand in .MORIE_TPS_HOOD_VERSIONS[[prefer]]) {
    if (cand %in% names(df)) return(cand)
  }
  if (isTRUE(fallback)) {
    for (cand in .MORIE_TPS_HOOD_VERSIONS[[other]]) {
      if (cand %in% names(df)) {
        warning(sprintf(
          "preferred HOOD_%s not present; falling back to %s",
          prefer, cand), call. = FALSE)
        return(cand)
      }
    }
  }
  warning("no HOOD_158 / HOOD_140 column found in df", call. = FALSE)
  NULL
}

#' Assert the HOOD_* schema version of a TPS data.frame
#'
#' Errors when the expected schema's column is absent. Warns when
#' BOTH schemas are present (downstream code MAY accidentally use the
#' wrong one).
#'
#' @param df A TPS crime `data.frame`.
#' @param expected Either `"158"` or `"140"`.
#' @return Invisibly `TRUE` on success.
#' @export
morie_tps_assert_hood_version <- function(df,
                                            expected = c("158", "140")) {
  expected <- match.arg(expected)
  cands <- .MORIE_TPS_HOOD_VERSIONS[[expected]]
  if (length(intersect(cands, names(df))) == 0L) {
    stop(sprintf(
      "expected HOOD_%s schema but no %s column found in df",
      expected, paste(cands, collapse = " / ")),
      call. = FALSE)
  }
  other <- setdiff(c("158", "140"), expected)
  other_present <- intersect(.MORIE_TPS_HOOD_VERSIONS[[other]], names(df))
  if (length(other_present) > 0L) {
    warning(sprintf(paste0(
      "df carries BOTH HOOD_%s (expected) and HOOD_%s columns; pick ",
      "one explicitly via morie_tps_resolve_hood_col() to avoid silent ",
      "mismatches between the 158 (current) and 140 (historical) ",
      "neighbourhood schemes."),
      expected, other), call. = FALSE)
  }
  invisible(TRUE)
}

#' Recommended HOOD_* schema version for a given OCC_YEAR
#'
#' The City of Toronto adopted the 158-neighbourhood scheme in 2022.
#' Pre-2022 TPS crime records are most faithfully analysed in the
#' historical 140-neighbourhood scheme; 2022-onwards records align
#' with the 158-scheme. TPS often back-fills both columns onto the
#' same record via lat/lon re-geocoding, but the polygon boundaries
#' do not match.
#'
#' @param year Integer year (or vector of years).
#' @return Character vector of `"158"` / `"140"` recommendations,
#'   parallel to `year`.
#' @export
morie_tps_year_to_hood_version <- function(year) {
  y <- suppressWarnings(as.integer(year))
  ifelse(is.na(y), NA_character_, ifelse(y >= 2022L, "158", "140"))
}

#' Load the bundled 158 <-> 140 neighbourhood crosswalk
#'
#' The City of Toronto does not publish a single authoritative
#' 158<->140 crosswalk because the two schemes are NOT a clean
#' refinement of each other (some 140s split, some merged, a few
#' renamed). The bundled fixture
#' `inst/extdata/to_hood_158_140_crosswalk.csv` is a SYNTHETIC
#' starter mapping with columns `hood_140`, `name_140`, `hood_158`,
#' `name_158`, `area_overlap_pct` (synthetic) -- intended to
#' demonstrate the API. Replace with your own polygon-intersection
#' crosswalk derived from
#' `morie_to_neighbourhoods("158")` and `morie_to_neighbourhoods("140")`
#' for production use.
#'
#' @return A `data.frame` with the columns above.
#' @export
morie_to_hood_crosswalk <- function() {
  path <- system.file("extdata", "to_hood_158_140_crosswalk.csv",
                      package = "morie")
  if (!nzchar(path)) {
    stop(paste0(
      "Bundled 158<->140 crosswalk fixture missing from the installed ",
      "morie package."), call. = FALSE)
  }
  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}
