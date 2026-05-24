# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Toronto Police Service ArcGIS Hub catalog (data.tps.ca).
#
# The TPS Hub publishes 71 canonical datasets (verified live via
# `https://data.tps.ca/api/search/v1/collections/dataset/items?limit=100`
# returning `numberMatched: 71`, all owned by TorontoPoliceService).
# Every dataset is exposed as an ArcGIS "Feature Service" reachable
# at a stable FeatureServer URL on `services.arcgis.com/S9th0jAJ7bqgIRjw`.
#
# Each Hub item resolves through three identifiers:
#
#   * `hub_id`             -- 32-char hex GUID used in
#                              data.tps.ca/maps/<hub_id> URLs and the
#                              ArcGIS Online items API
#                              (`/sharing/rest/content/items/<id>?f=json`).
#
#   * `feature_server_url` -- the underlying ArcGIS Feature Service
#                              endpoint
#                              (`services.arcgis.com/.../FeatureServer`).
#                              `/0/query?where=1=1&outFields=*&f=json` for
#                              tabular data; `?f=geojson` for GeoJSON.
#
#   * `Hub download URL`   -- on-the-fly CSV / GeoJSON / Shapefile /
#                              File Geodatabase exports at
#                              `https://hub.arcgis.com/api/v3/datasets/<hub_id>_0/downloads/data?format=<fmt>`.
#
# 3SS bundles the full 71-row catalog as a CSV fixture so discovery
# works offline. By-id loading dispatches to FeatureServer (JSON /
# GeoJSON) or the Hub downloads API (CSV / Shapefile / FGDB) based on
# `format`.

# ---------------------------------------------------------------------------
# Catalog discovery
# ---------------------------------------------------------------------------

#' List the Toronto Police Service ArcGIS Hub datasets wrapped by morie
#'
#' Sibling discovery helper to [morie_datasets_external_socrata_layers()]
#' and [morie_datasets_ontario_ckan_layers()], covering all 71 datasets
#' currently published on `data.tps.ca`.
#'
#' Verified live against the canonical TPS Hub catalog API
#' (`https://data.tps.ca/api/search/v1/collections/dataset/items?limit=100`)
#' on 2026-05-24 -- returned `numberMatched: 71`, all owned by
#' `TorontoPoliceService`.
#'
#' Coverage spans nine families:
#'
#' \describe{
#'   \item{MCI Open Data}{Assault, Auto Theft, Bicycle Thefts,
#'     Break and Enter, Hate Crimes, Homicides, Intimate Partner +
#'     Family Violence, Robbery, Shooting + Firearm Discharges,
#'     Theft From Motor Vehicle, Theft Over.}
#'   \item{Use of Force (RBDC-UOF)}{Six per-axis breakdown tables
#'     plus the Types-and-Perceived-Weapons crosstab.}
#'   \item{Annual Statistical Report (ASR)}{Administrative,
#'     Calls for Service, Enforcement, Firearms, Misc., Personnel,
#'     Public Complaints, Reported Crimes, Regulated Interactions,
#'     Search of Persons, Traffic, Victims of Crime.}
#'   \item{Killed and Seriously Injured (KSI)}{Main + per-mode
#'     (Automobile / Cyclist / Fatals / Motorcyclist / Passenger /
#'     Pedestrian).}
#'   \item{Budget}{Annual figures 2020 through 2026 plus
#'     `Budget_by_Command`.}
#'   \item{Personnel + Staffing}{ASR-PB family + `Staffing_by_Command`.}
#'   \item{Historical FIRS}{Annual files 2008 through 2013.}
#'   \item{Geographic boundaries}{Facilities, Patrol Zone,
#'     Police Divisions.}
#'   \item{Other}{Community Safety Indicators, Mental Health Act
#'     Apprehensions, Neighbourhood Crime Rates, Persons in Crisis
#'     Calls for Service Attended.}
#' }
#'
#' @param offline If `TRUE` (default), read the bundled 71-row
#'   catalog fixture (`inst/extdata/tps_arcgis_hub_catalog.csv`).
#'   Live mode hits the TPS Hub search API.
#' @return A `data.frame` with columns `hub_id`, `title`, `type`,
#'   `feature_server_url`, `owner`, `tags`, `snippet`.
#' @references TPS Public Safety Data Portal,
#'   \url{https://data.tps.ca/search?collection=dataset}.
#' @examples
#' cat <- morie_datasets_tps_arcgis_hub_layers()
#' nrow(cat)        # 71
#' head(cat$title)
#' @export
morie_datasets_tps_arcgis_hub_layers <- function(offline = TRUE) {
  if (isTRUE(offline)) {
    path <- system.file("extdata", "tps_arcgis_hub_catalog.csv",
                        package = "morie")
    if (!nzchar(path)) {
      stop("bundled TPS ArcGIS Hub catalog fixture missing",
           call. = FALSE)
    }
    df <- utils::read.csv(path, stringsAsFactors = FALSE,
                           check.names = FALSE)
    return(df)
  }
  # Live mode -- hit the TPS Hub OGC-API-Features search endpoint.
  body <- .morie_dataset_http_json(
    "https://data.tps.ca/api/search/v1/collections/dataset/items",
    query = list(limit = 100L))
  feats <- body$features
  if (is.null(feats) || length(feats) == 0L) return(data.frame())
  rows <- lapply(feats, function(f) {
    p <- f$properties
    data.frame(
      hub_id = f$id,
      title  = p$title %||% "",
      type   = p$type  %||% "Feature Service",
      feature_server_url = p$url %||% "",
      owner  = p$owner %||% "TorontoPoliceService",
      tags   = paste(unlist(p$tags) %||% character(), collapse = "; "),
      snippet = substr(p$snippet %||% "", 1L, 300L),
      stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  out[order(out$title), , drop = FALSE]
}

# ---------------------------------------------------------------------------
# Hub-id -> FeatureServer URL resolver
# ---------------------------------------------------------------------------

#' Resolve a TPS Hub item id to its underlying ArcGIS FeatureServer URL.
#' Offline mode looks up the cached catalog; live mode hits the ArcGIS
#' Online items API (`/sharing/rest/content/items/<id>?f=json`).
#' @keywords internal
#' @noRd
.morie_dataset_tps_hub_resolve <- function(hub_id, offline = TRUE) {
  if (!grepl("^[a-f0-9]{32}$", hub_id)) {
    stop(sprintf(paste0("morie TPS hub_id must be a 32-char hex GUID; ",
                         "got '%s'"), hub_id), call. = FALSE)
  }
  if (isTRUE(offline)) {
    cat <- morie_datasets_tps_arcgis_hub_layers(offline = TRUE)
    hit <- cat[cat$hub_id == hub_id, , drop = FALSE]
    if (nrow(hit) == 0L) {
      stop(sprintf(paste0(
        "morie TPS hub_id '%s' not in the bundled catalog ",
        "(71 datasets verified 2026-05-24). Pass offline = FALSE ",
        "to resolve via the live ArcGIS Online items API."),
        hub_id), call. = FALSE)
    }
    return(hit$feature_server_url[[1L]])
  }
  body <- .morie_dataset_http_json(
    sprintf("https://www.arcgis.com/sharing/rest/content/items/%s",
            hub_id),
    query = list(f = "json"))
  if (is.null(body$url) || !nzchar(body$url)) {
    stop(sprintf(paste0(
      "morie TPS hub_id '%s' did not return a FeatureServer url ",
      "from the ArcGIS Online items API."), hub_id), call. = FALSE)
  }
  body$url
}

# ---------------------------------------------------------------------------
# Generic by-id loader (multi-format aware)
# ---------------------------------------------------------------------------

.MORIE_TPS_HUB_FORMATS <- c("json", "geojson", "csv",
                             "shapefile", "fgdb")

#' Generic TPS ArcGIS Hub dataset loader by hub_id
#'
#' Single entry point for the 71 datasets listed by
#' [morie_datasets_tps_arcgis_hub_layers()]. Supports five export
#' formats:
#'
#' \describe{
#'   \item{`"json"` (default)}{Hits the FeatureServer
#'     `/0/query?where=...&outFields=*&f=json` endpoint and parses
#'     attributes into a tidy `data.frame`. Same path the existing
#'     TPS PSDP loaders (3FF) use; honours an arbitrary `where`
#'     clause and `max_features` cap.}
#'   \item{`"geojson"`}{Hits `?f=geojson` and returns the raw GeoJSON
#'     as a parsed R list. Caller can pass to `sf::st_read()`.}
#'   \item{`"csv"`}{Hits the Hub CSV exporter at
#'     `hub.arcgis.com/api/v3/datasets/<hub_id>_0/downloads/data?format=csv`
#'     and parses the returned CSV into a `data.frame`.}
#'   \item{`"shapefile"` / `"fgdb"`}{Downloads the binary archive
#'     (Esri Shapefile zip / File Geodatabase zip) to `dest`
#'     (default `tempfile()`) and returns the file path. Caller can
#'     `sf::st_read()` the unzipped contents.}
#' }
#'
#' For boundary layers (Police Divisions, Patrol Zone, Facilities)
#' you'll typically want `format = "geojson"` to get the polygon
#' geometry. For tabular outputs (UoF tables, KSI counts, ASR
#' tables, budgets) `format = "json"` is sufficient and lightest.
#'
#' @param hub_id 32-char hex GUID. See
#'   [morie_datasets_tps_arcgis_hub_layers()] for the canonical 71.
#' @param format One of `"json"` (default), `"geojson"`, `"csv"`,
#'   `"shapefile"`, `"fgdb"`.
#' @param where SoQL-style `WHERE` filter passed to FeatureServer
#'   `?where=` (default `"1=1"`).
#' @param max_features Optional cap on returned rows. Passed as
#'   `resultRecordCount` to FeatureServer; ignored for binary formats.
#' @param layer_idx Integer index of the FeatureServer layer to pull
#'   (default `0L`, the first layer).
#' @param offline Logical; if `TRUE`, the hub_id is resolved via the
#'   bundled catalog (no network needed for the resolution step).
#'   Default `TRUE` -- you can run this against the 71 catalog
#'   entries without network. Live data fetches always hit the
#'   network regardless of this argument; "offline" here only
#'   affects the hub_id -> FeatureServer URL lookup.
#' @param dest Optional path for binary downloads
#'   (`format %in% c("shapefile", "fgdb")`); defaults to `tempfile()`.
#' @return A `data.frame` (json / csv), a parsed GeoJSON list, or a
#'   file path (binary).
#' @export
morie_datasets_tps_arcgis_hub_by_id <- function(hub_id,
                                                  format = "json",
                                                  where = "1=1",
                                                  max_features = NULL,
                                                  layer_idx = 0L,
                                                  offline = TRUE,
                                                  dest = NULL) {
  format <- match.arg(format, choices = .MORIE_TPS_HUB_FORMATS)
  fs_url <- .morie_dataset_tps_hub_resolve(hub_id, offline = offline)
  if (format == "json") {
    layer_url <- sprintf("%s/%d/query", fs_url, as.integer(layer_idx))
    query <- list(where = where, outFields = "*", f = "json",
                  returnGeometry = "false")
    if (!is.null(max_features)) {
      query$resultRecordCount <- as.integer(max_features)
    }
    body <- .morie_dataset_http_json(layer_url, query = query)
    feats <- body$features
    if (is.null(feats) || length(feats) == 0L) return(data.frame())
    attrs <- lapply(feats, function(f) f$attributes)
    return(.morie_dataset_records_to_df(attrs))
  }
  if (format == "geojson") {
    layer_url <- sprintf("%s/%d/query", fs_url, as.integer(layer_idx))
    query <- list(where = where, outFields = "*", f = "geojson")
    if (!is.null(max_features)) {
      query$resultRecordCount <- as.integer(max_features)
    }
    return(.morie_dataset_http_json(layer_url, query = query))
  }
  if (format == "csv") {
    csv_url <- sprintf(paste0(
      "https://hub.arcgis.com/api/v3/datasets/%s_%d/",
      "downloads/data"),
      hub_id, as.integer(layer_idx))
    raw <- .morie_dataset_http_text(csv_url,
                                     query = list(format = "csv"))
    return(utils::read.csv(text = raw, stringsAsFactors = FALSE,
                            check.names = FALSE))
  }
  # Binary formats: download to dest.
  fmt_token <- switch(format,
                      shapefile = "shp",
                      fgdb      = "fgdb")
  bin_url <- sprintf(paste0(
    "https://hub.arcgis.com/api/v3/datasets/%s_%d/",
    "downloads/data?format=%s"),
    hub_id, as.integer(layer_idx), fmt_token)
  if (is.null(dest)) {
    suffix <- switch(format, shapefile = ".shp.zip", fgdb = ".fgdb.zip")
    dest <- tempfile(fileext = suffix)
  }
  if (!requireNamespace("httr2", quietly = TRUE)) {
    stop("morie TPS hub binary download requires the 'httr2' package.",
         call. = FALSE)
  }
  req <- httr2::request(bin_url)
  resp <- httr2::req_perform(req)
  bytes <- httr2::resp_body_raw(resp)
  writeBin(bytes, dest)
  dest
}

#' Direct multi-format downloader (binary or text) for a TPS Hub item
#'
#' Thin wrapper that just hits the Hub downloads endpoint without the
#' FeatureServer `/query` round-trip. Use this when you want the
#' canonical CSV / GeoJSON / Shapefile / FGDB exactly as the Hub UI
#' serves them (including any column-name and projection differences
#' that on-the-fly exports introduce vs the FeatureServer source).
#'
#' @param hub_id 32-char hex GUID.
#' @param format One of `"csv"`, `"geojson"`, `"shapefile"`, `"fgdb"`.
#' @param layer_idx Integer layer index (default `0L`).
#' @param dest Optional destination path; defaults to `tempfile()`.
#' @return Path to the downloaded file.
#' @export
morie_datasets_tps_arcgis_hub_download <- function(hub_id,
                                                     format = "csv",
                                                     layer_idx = 0L,
                                                     dest = NULL) {
  format <- match.arg(format,
                       choices = c("csv", "geojson",
                                    "shapefile", "fgdb"))
  fmt_token <- switch(format,
                      csv = "csv",
                      geojson = "geojson",
                      shapefile = "shp",
                      fgdb = "fgdb")
  suffix <- switch(format,
                   csv = ".csv",
                   geojson = ".geojson",
                   shapefile = ".shp.zip",
                   fgdb = ".fgdb.zip")
  url <- sprintf(paste0(
    "https://hub.arcgis.com/api/v3/datasets/%s_%d/",
    "downloads/data?format=%s"),
    hub_id, as.integer(layer_idx), fmt_token)
  if (is.null(dest)) dest <- tempfile(fileext = suffix)
  if (!requireNamespace("httr2", quietly = TRUE)) {
    stop("morie TPS hub download requires the 'httr2' package.",
         call. = FALSE)
  }
  req <- httr2::request(url)
  resp <- httr2::req_perform(req)
  bytes <- httr2::resp_body_raw(resp)
  writeBin(bytes, dest)
  dest
}

# (`%||%` is provided by other modules; no local re-definition needed.)
