# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3CCC4: Vancouver Open Data (Opendatasoft v2.1) loader.
#
# opendata.vancouver.ca runs Opendatasoft's Explore API v2.1 (a
# different stack than Socrata's SODA2/SODA3 or ArcGIS Hub). API
# shape:
#
#   * Catalog list:
#       /api/explore/v2.1/catalog/datasets?limit=100&offset=0
#   * Per-dataset records:
#       /api/explore/v2.1/catalog/datasets/<id>/records?limit=N
#   * Per-dataset exports (CSV/GeoJSON):
#       /api/explore/v2.1/catalog/datasets/<id>/exports/csv
#       /api/explore/v2.1/catalog/datasets/<id>/exports/geojson
#
# NOTE: as of 2026-05-24, opendata.vancouver.ca does NOT host VPD
# crime/incident data. VPD publishes that separately on the
# vpd-crimedata Open Data Portal at geodash.vpd.ca. This loader
# covers the City-of-Vancouver civic datasets (190 datasets:
# property, transit, parks, infrastructure, budgets, etc.) which
# are useful for context joins on the morie geographic side.

.MORIE_VANCOUVER_OPENDATA_BASE <-
  "https://opendata.vancouver.ca/api/explore/v2.1"

#' Vancouver Open Data full dataset catalog (Opendatasoft v2.1)
#'
#' Phase 3CCC4. Bundled snapshot of every City-of-Vancouver dataset
#' published on opendata.vancouver.ca (190 datasets as of
#' 2026-05-24). Each row identifies a dataset by its Opendatasoft
#' `dataset_id` slug (used as the URL path segment for records /
#' exports endpoints).
#'
#' @param offline If `TRUE` (default), reads the bundled CSV; if
#'   `FALSE`, paginates the live catalog endpoint.
#' @param max_features Optional row cap.
#' @return A `data.frame` with `dataset_id`, `title`, `publisher`,
#'   `records_count`.
#' @references Opendatasoft Explore API v2.1,
#'   \url{https://opendata.vancouver.ca/api-console/explore/v2.1/}.
#' @examples
#' cat_df <- morie_datasets_vancouver_opendata_layers(offline = TRUE)
#' nrow(cat_df)  # 190
#' head(cat_df$title)
#' @export
morie_datasets_vancouver_opendata_layers <- function(offline = TRUE,
                                                       max_features = NULL) {
  if (isTRUE(offline)) {
    path <- system.file("extdata", "vancouver_opendata_catalog.csv",
                        package = "morie")
    if (!nzchar(path))
      stop("bundled Vancouver Open Data catalog fixture missing",
           call. = FALSE)
    df <- utils::read.csv(path, stringsAsFactors = FALSE,
                           check.names = FALSE)
  } else {
    # Paginate via offset (Opendatasoft max limit=100/req).
    all_rows <- list()
    offset <- 0L
    repeat {
      url <- sprintf(paste0("%s/catalog/datasets?limit=100",
                              "&select=dataset_id,title,publisher,records_count",
                              "&offset=%d&order_by=dataset_id"),
                       .MORIE_VANCOUVER_OPENDATA_BASE, offset)
      r <- .morie_dataset_http_json(url)
      if (is.null(r$results) || length(r$results) == 0L) break
      all_rows[[length(all_rows) + 1L]] <- r$results
      offset <- offset + nrow(r$results)
      if (!is.null(r$total_count) && offset >= r$total_count) break
    }
    df <- if (length(all_rows)) do.call(rbind, all_rows) else data.frame()
  }
  if (!is.null(max_features))
    df <- utils::head(df, as.integer(max_features))
  df
}

#' Fetch records from a Vancouver Open Data dataset by ID
#'
#' Phase 3CCC4. Hits the Opendatasoft v2.1 `/records` endpoint for
#' an arbitrary Vancouver dataset slug. Returns the `results` array
#' as a `data.frame`. For larger pulls, use `format = "csv"` to hit
#' the unrestricted `/exports/csv` endpoint instead.
#'
#' @param dataset_id Opendatasoft dataset slug (from
#'   [morie_datasets_vancouver_opendata_layers()]).
#' @param limit Page size (max 100 for `/records`).
#' @param format One of `"json"` (default, `/records` endpoint) or
#'   `"csv"` (`/exports/csv` endpoint, no row limit).
#' @return A `data.frame` of records.
#' @examples
#' \dontrun{
#' df <- morie_datasets_vancouver_opendata_by_id("non-market-housing",
#'                                                  limit = 50)
#' nrow(df)
#' }
#' @export
morie_datasets_vancouver_opendata_by_id <- function(dataset_id,
                                                      limit = 100L,
                                                      format = c("json", "csv")) {
  format <- match.arg(format)
  if (format == "json") {
    url <- sprintf("%s/catalog/datasets/%s/records?limit=%d",
                    .MORIE_VANCOUVER_OPENDATA_BASE,
                    dataset_id, as.integer(limit))
    r <- .morie_dataset_http_json(url)
    if (is.null(r$results)) return(data.frame())
    return(r$results)
  }
  # CSV export
  url <- sprintf("%s/catalog/datasets/%s/exports/csv",
                  .MORIE_VANCOUVER_OPENDATA_BASE, dataset_id)
  text <- .morie_dataset_http_text(url)
  utils::read.csv(text = text, sep = ";", stringsAsFactors = FALSE,
                  check.names = FALSE)
}
