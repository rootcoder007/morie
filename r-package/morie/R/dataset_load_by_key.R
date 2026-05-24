# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3EEE4: unified dispatcher -- load any dataset by its
# canonical cross-portal catalog key without users having to know
# which loader function to call.

#' Load a dataset by its cross-portal catalog `dataset_key`
#'
#' Phase 3EEE4. Single entry point that dispatches to the right
#' loader function for any of the ~550 datasets in
#' [morie_dataset_portal_catalog()]. Lets callers say
#' `morie_datasets_load_by_key("vpd_crime")` or
#' `morie_datasets_load_by_key("hub:b4d0...assault")` without
#' remembering whether the relevant loader is
#' `morie_datasets_vpd_crime()` or
#' `morie_datasets_tps_arcgis_hub_by_id()`.
#'
#' Resolution rules:
#'
#' \describe{
#'   \item{Bundled-fixture loaders}{If the catalog's `loader`
#'     column names a function that takes no required arguments
#'     beyond optionally `offline`/`max_features`, that function is
#'     called directly with `offline = offline` + `max_features`.}
#'   \item{Generic CKAN dispatchers}{For Ontario / Montreal /
#'     Toronto CKAN catalog entries whose loader is the generic
#'     `morie_datasets_*_ckan_resource()`, the `id` (CKAN
#'     package_name slug) is resolved to a primary resource via
#'     `package_show` + first CSV resource, then fetched.}
#'   \item{Generic ArcGIS Hub dispatcher}{For TPS Hub entries
#'     (`source == "tps_arcgis_hub"`), the bare `hub_id` is passed
#'     to [morie_datasets_tps_arcgis_hub_by_id()].}
#'   \item{StatCan WDS}{For statcan_ccjs entries, returns the cube
#'     metadata via [morie_datasets_statcan_cube_metadata()].}
#'   \item{Vancouver Opendatasoft}{For vancouver_opendata entries
#'     beyond the 9 bundled fixtures, dispatches to
#'     [morie_datasets_vancouver_opendata_by_id()].}
#' }
#'
#' For datasets where the catalog only knows a key + portal
#' (no row-level fixture, no targeted wrapper), `live = FALSE`
#' raises a clear error pointing at the right live-mode dispatcher.
#'
#' @param dataset_key A `dataset_key` from
#'   [morie_dataset_portal_catalog()].
#' @param offline If `TRUE` (default), prefer bundled fixtures.
#'   `FALSE` forces live mode for loaders that support it.
#' @param max_features Optional row cap forwarded to the underlying
#'   loader.
#' @param mode One of `"auto"` (default), `"soda2"`, `"soda3"`,
#'   `"odata"`. Only honoured for Socrata-backed sources
#'   (chicago / nyc_nypd / nyc_opendata) that have a per-wrapper
#'   `mode` argument. `"auto"` lets each underlying loader pick
#'   its native default (soda2 today). 3FFF2.
#' @param app_token Optional Socrata application token forwarded
#'   to SODA3-capable loaders. 3FFF2.
#' @return A `data.frame` (or, for StatCan, the WDS metadata list).
#' @examples
#' # VPD bundled sample
#' df1 <- morie_datasets_load_by_key("vpd_crime")
#' nrow(df1)
#'
#' # NYC NYPD arrests YTD bundled sample
#' df2 <- morie_datasets_load_by_key("nypd_arrests_ytd")
#' nrow(df2)
#'
#' # TPS PSDP layer
#' df3 <- morie_datasets_load_by_key("assault")
#' nrow(df3)
#' @export
morie_datasets_load_by_key <- function(dataset_key,
                                         offline = TRUE,
                                         max_features = NULL,
                                         mode = c("auto", "soda2",
                                                   "soda3", "odata"),
                                         app_token = NULL) {
  mode <- match.arg(mode)
  cat_df <- morie_dataset_portal_catalog()
  row <- cat_df[cat_df$dataset_key == dataset_key, , drop = FALSE]
  if (nrow(row) == 0L) {
    stop(sprintf(
      "unknown dataset_key '%s'. Try morie_datasets_browse(keyword = ...) ",
      dataset_key), call. = FALSE)
  }
  if (nrow(row) > 1L) {
    # Same key across multiple sources -- prefer the first.
    row <- row[1L, , drop = FALSE]
  }
  src    <- row$source
  id     <- row$id
  loader <- row$loader

  # --- Targeted (bundled-fixture) wrappers --------------------
  # The catalog loader names a real morie function. If it does NOT
  # contain "by_id" / "by_key" / "ckan_resource" / "cube_metadata"
  # / "vancouver_opendata_by_id", call it directly.
  is_targeted <- !grepl("(_by_id|_by_key|_ckan_resource|_cube_metadata)$",
                          loader)
  if (is_targeted &&
      exists(loader, mode = "function", envir = asNamespace("morie"))) {
    fn <- get(loader, envir = asNamespace("morie"))
    args <- formals(fn)
    call_args <- list()
    if ("offline" %in% names(args)) call_args$offline <- offline
    if ("max_features" %in% names(args))
      call_args$max_features <- max_features
    # 3FFF2: thread mode + app_token to Socrata-backed wrappers
    # that support them.
    if ("mode" %in% names(args) && mode != "auto")
      call_args$mode <- mode
    if ("app_token" %in% names(args) && !is.null(app_token))
      call_args$app_token <- app_token
    return(do.call(fn, call_args))
  }

  # --- Per-source generic dispatchers --------------------------
  switch(src,
    "tps_arcgis_hub" = {
      morie_datasets_tps_arcgis_hub_by_id(id, max_features = max_features)
    },
    "ontario_ckan" = {
      morie_datasets_ontario_ckan_by_key(id,
                                           max_features = max_features,
                                           offline = offline)
    },
    "montreal_opendata" = {
      rid <- .morie_ckan_resolve_first_csv(
        package_name = id, ckan_base = .MORIE_MONTREAL_CKAN_BASE)
      morie_datasets_montreal_ckan_resource(
        resource_id = rid,
        limit = if (is.null(max_features)) 100L
                else as.integer(max_features))
    },
    "toronto_opendata" = {
      rid <- .morie_ckan_resolve_first_csv(
        package_name = id, ckan_base = .MORIE_TORONTO_CKAN_BASE)
      morie_datasets_toronto_open_ckan_resource(
        resource_id = rid,
        limit = if (is.null(max_features)) 100L
                else as.integer(max_features))
    },
    "vancouver_opendata" = {
      morie_datasets_vancouver_opendata_by_id(id)
    },
    "calgary_opendata" = {
      morie_datasets_calgary_socrata_by_id(id)
    },
    "edmonton_opendata" = {
      morie_datasets_edmonton_socrata_by_id(id)
    },
    "ottawa_opendata" = {
      morie_datasets_tps_arcgis_hub_by_id(id, max_features = max_features)
    },
    "vpd_geodash" = {
      morie_datasets_vpd_crime(offline = offline,
                                max_features = max_features)
    },
    "statcan_ccjs" = {
      morie_datasets_statcan_cube_metadata(as.integer(id))
    },
    "nyc_nypd" = {
      args <- list(dataset_key = dataset_key,
                    max_features = max_features,
                    offline = offline)
      if (mode != "auto") args$mode <- mode
      if (!is.null(app_token)) args$app_token <- app_token
      do.call(morie_datasets_nyc_nypd_by_key, args)
    },
    stop(sprintf(
      "morie_datasets_load_by_key('%s'): unhandled source '%s'.",
      dataset_key, src), call. = FALSE)
  )
}

# ---------------------------------------------------------------------------
# CKAN package_name -> first-CSV resource_id resolver (3FFF1)
# ---------------------------------------------------------------------------

#' Resolve a CKAN `package_name` to its first-CSV resource UUID
#'
#' Phase 3FFF1. Internal helper used by
#' [morie_datasets_load_by_key()] to look up an arbitrary CKAN
#' package's primary CSV resource without requiring the caller to
#' know the UUID. Hits `/action/package_show?id=<pkg>` then walks
#' the `resources` list for the first row whose `format` (or
#' `mimetype`) matches `"CSV"` (case-insensitive).
#'
#' Falls back to the bare first resource if no CSV is present and
#' raises a clear error if the package has no resources at all.
#'
#' @param package_name CKAN package slug.
#' @param ckan_base CKAN host base URL (e.g.,
#'   `.MORIE_MONTREAL_CKAN_BASE` or `.MORIE_TORONTO_CKAN_BASE`).
#' @return Character resource UUID.
#' @keywords internal
#' @noRd
.morie_ckan_resolve_first_csv <- function(package_name, ckan_base) {
  url <- sprintf("%s/action/package_show?id=%s", ckan_base, package_name)
  r <- .morie_dataset_http_json(url)
  if (!isTRUE(r$success))
    stop(sprintf("CKAN package_show failed for '%s'", package_name),
          call. = FALSE)
  res <- r$result$resources
  if (is.null(res) || (is.data.frame(res) && nrow(res) == 0L))
    stop(sprintf("CKAN package '%s' has no resources", package_name),
          call. = FALSE)
  fmt <- if ("format" %in% names(res)) res$format else NULL
  if (!is.null(fmt)) {
    csv_idx <- which(toupper(fmt) == "CSV")
    if (length(csv_idx) > 0L) return(res$id[csv_idx[1]])
  }
  res$id[1]
}
