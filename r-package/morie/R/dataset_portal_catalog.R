# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Phase 3CCC4: Cross-portal dataset catalog.
#
# Unifies the per-portal registries into a single canonical view
# so callers can ask "what datasets ship with morie?" across:
#
#   * NYC NYPD CJ datasets (Socrata, .MORIE_NYC_NYPD_REGISTRY)
#   * External Socrata datasets (Chicago + NYPD SQF,
#     morie_datasets_external_socrata_layers())
#   * TPS PSDP layers (ArcGIS, morie_tps_psdp_layers())
#   * TPS ArcGIS Hub catalog (71 items,
#     morie_datasets_tps_arcgis_hub_layers())
#   * Ontario CKAN (OTIS family, morie_datasets_ontario_ckan_layers())
#   * NYC OpenData boundary loaders (7 fixtures incl. borough +
#     precinct, morie_datasets_nyc_boundaries_catalog())
#   * Vancouver Open Data civic catalog (Opendatasoft v2.1, 190
#     datasets, morie_datasets_vancouver_opendata_layers())
#
# Each entry surfaces a uniform schema:
#
#   dataset_key | source        | id          | api_modes
#               | loader        | dict_url    | n_rows_bundled

#' Unified cross-portal catalog of every morie-shippable dataset
#'
#' Phase 3CCC4. Single-view aggregator over morie's per-portal
#' registries. Returns one row per (portal, dataset) pair with a
#' uniform schema so callers can answer questions like "which TPS
#' datasets does morie ship?" or "what fixtures are bundled for
#' Chicago?" without knowing each portal's registry shape.
#'
#' API mode column legend (`api_modes`):
#' \itemize{
#'   \item \strong{soda2}        -- Socrata SODA2 `/resource/<id>.json` (open).
#'   \item \strong{soda2_csv}    -- Socrata SODA2 `/resource/<id>.csv` (open).
#'   \item \strong{soda2_geojson}-- Socrata SODA2 `/resource/<id>.geojson` (open).
#'   \item \strong{soda3}        -- Socrata SODA3
#'     `/api/v3/views/<id>/query.json` or `.csv` (requires
#'     authentication; see Socrata support ref below).
#'   \item \strong{odata}        -- Socrata OData v4
#'     `/api/odata/v4/<id>` (open, but `$filter` is broken on many
#'     Socrata datasets due to Edm typing -- prefer soda3 for filters).
#'   \item \strong{arcgis_rest}  -- ArcGIS FeatureServer REST query
#'     endpoint (open).
#'   \item \strong{arcgis_hub}   -- TPS Hub OGC-API-Features (open).
#'   \item \strong{ckan}         -- CKAN datastore (open).
#'   \item \strong{opendatasoft_v21} -- Opendatasoft Explore API v2.1
#'     `/api/explore/v2.1/catalog/datasets/<id>/records` (open).
#'   \item \strong{manual_download} -- Data is gated behind manual
#'     T&C acceptance via a publisher web UI; no automation API.
#'     Caller downloads manually and passes the local path to the
#'     morie loader. Used by VPD GeoDASH.
#'   \item \strong{statcan_wds} -- Statistics Canada Web Data Service
#'     REST API (open; POST endpoints for metadata + vector data).
#' }
#'
#' @note SODA3 (`/api/v3/views/<id>/query.{json,csv}`) requires
#'   authentication per Socrata's API documentation
#'   (\url{https://support.socrata.com/hc/en-us/articles/34730618169623-SODA3-API}).
#'   morie's `mode = "soda3"` accepts an `app_token` argument on
#'   every wrapper that supports it.
#'
#' @param portal Optional character filter: `"chicago"`, `"nyc_nypd"`,
#'   `"nyc_opendata"`, `"tps_arcgis_hub"`, `"tps_psdp"`,
#'   `"ontario_ckan"`, `"vancouver_opendata"`, `"vpd_geodash"`,
#'   `"statcan_ccjs"`, `"montreal_opendata"`, `"toronto_opendata"`.
#'   `NULL` (default) returns all portals.
#' @return A `data.frame` with one row per dataset. Columns:
#'   `dataset_key`, `source`, `id`, `api_modes`, `loader`,
#'   `dict_url`, `n_rows_bundled`.
#' @examples
#' cat_df <- morie_dataset_portal_catalog()
#' nrow(cat_df)
#' table(cat_df$source)
#'
#' tps <- morie_dataset_portal_catalog(portal = "tps_arcgis_hub")
#' head(tps$dataset_key)
#' @export
morie_dataset_portal_catalog <- function(portal = NULL) {
  rows <- list()
  push <- function(r) rows[[length(rows) + 1L]] <<- r

  socrata_modes <- "soda2,soda2_csv,soda2_geojson,soda3,odata"

  # --- NYC NYPD CJ (9 datasets) ---------------------------------
  reg <- .MORIE_NYC_NYPD_REGISTRY
  for (k in names(reg)) {
    e <- reg[[k]]
    push(data.frame(
      dataset_key = k,
      source = "nyc_nypd",
      id = e$resource_id,
      api_modes = socrata_modes,
      loader = sprintf("morie_datasets_%s", k),
      dict_url = if (!is.null(e$data_dictionary_url) &&
                       !is.na(e$data_dictionary_url))
                    e$data_dictionary_url else NA_character_,
      n_rows_bundled = .morie_portal_fixture_rows(e$fixture),
      stringsAsFactors = FALSE))
  }

  # --- External Socrata (Chicago + NYPD SQF) --------------------
  ext <- morie_datasets_external_socrata_layers()
  for (i in seq_len(nrow(ext))) {
    rid <- sub(".*resource/([^.]+)\\.json.*", "\\1",
                ext$resource_url[i])
    src <- if (grepl("cityofchicago", ext$portal[i])) "chicago"
           else if (grepl("cityofnewyork", ext$portal[i])) "nyc_opendata"
           else ext$portal[i]
    push(data.frame(
      dataset_key = ext$dataset_key[i],
      source = src,
      id = rid,
      api_modes = socrata_modes,
      loader = sprintf("morie_datasets_%s", ext$dataset_key[i]),
      dict_url = NA_character_,
      n_rows_bundled = .morie_portal_fixture_rows(ext$fixture[i]),
      stringsAsFactors = FALSE))
  }

  # --- TPS PSDP (11 layers) -------------------------------------
  psdp <- morie_tps_psdp_layers()
  for (i in seq_len(nrow(psdp))) {
    push(data.frame(
      dataset_key = psdp$layer_key[i],
      source = "tps_psdp",
      id = psdp$hub_id[i],
      api_modes = "arcgis_rest,arcgis_hub",
      loader = sprintf("morie_datasets_tps_%s", psdp$layer_key[i]),
      dict_url = NA_character_,
      n_rows_bundled = .morie_portal_fixture_rows(psdp$fixture[i]),
      stringsAsFactors = FALSE))
  }

  # --- TPS ArcGIS Hub (71 catalog items, by hub_id) -------------
  hub <- morie_datasets_tps_arcgis_hub_layers(offline = TRUE)
  for (i in seq_len(nrow(hub))) {
    push(data.frame(
      dataset_key = sprintf("hub:%s", hub$hub_id[i]),
      source = "tps_arcgis_hub",
      id = hub$hub_id[i],
      api_modes = "arcgis_rest,arcgis_hub",
      loader = "morie_datasets_tps_arcgis_hub_by_id",
      dict_url = NA_character_,
      n_rows_bundled = NA_integer_,
      stringsAsFactors = FALSE))
  }

  # --- Ontario CKAN (OTIS family) -------------------------------
  ock <- morie_datasets_ontario_ckan_layers()
  for (i in seq_len(nrow(ock))) {
    push(data.frame(
      dataset_key = ock$dataset_key[i],
      source = "ontario_ckan",
      id = ock$resource_id[i],
      api_modes = "ckan",
      loader = "morie_datasets_ontario_ckan_by_key",
      dict_url = NA_character_,
      n_rows_bundled = .morie_portal_fixture_rows(ock$fixture[i]),
      stringsAsFactors = FALSE))
  }

  # --- NYC OpenData boundary loaders (7 fixtures from 3CCC2) ---
  bnd <- morie_datasets_nyc_boundaries_catalog()
  for (i in seq_len(nrow(bnd))) {
    push(data.frame(
      dataset_key = bnd$boundary[i],
      source = "nyc_opendata",
      id = bnd$soda_id[i],
      api_modes = socrata_modes,
      loader = bnd$loader[i],
      dict_url = NA_character_,
      n_rows_bundled = bnd$n_rows[i],
      stringsAsFactors = FALSE))
  }

  # --- Vancouver Open Data (Opendatasoft v2.1) -----------------
  van <- morie_datasets_vancouver_opendata_layers(offline = TRUE)
  for (i in seq_len(nrow(van))) {
    push(data.frame(
      dataset_key = van$dataset_id[i],
      source = "vancouver_opendata",
      id = van$dataset_id[i],
      api_modes = "opendatasoft_v21",
      loader = "morie_datasets_vancouver_opendata_by_id",
      dict_url = NA_character_,
      n_rows_bundled = NA_integer_,
      stringsAsFactors = FALSE))
  }

  # --- Toronto Open Data CKAN crime-adjacent (3EEE2) -----------
  tor <- morie_datasets_toronto_open_crime_adjacent_layers(offline = TRUE)
  for (i in seq_len(nrow(tor))) {
    push(data.frame(
      dataset_key = tor$package_name[i],
      source = "toronto_opendata",
      id = tor$package_name[i],
      api_modes = "ckan",
      loader = if (tor$package_name[i] == "ambulance-station-locations")
                 "morie_datasets_toronto_ambulance_stations"
              else if (tor$package_name[i] ==
                         "police-annual-statistical-report-miscellaneous-data")
                 "morie_datasets_toronto_asr_miscellaneous"
              else "morie_datasets_toronto_open_ckan_resource",
      dict_url = sprintf("https://open.toronto.ca/dataset/%s",
                          tor$package_name[i]),
      n_rows_bundled = if (tor$package_name[i] == "ambulance-station-locations")
                         46L
                       else if (tor$package_name[i] ==
                                  "police-annual-statistical-report-miscellaneous-data")
                         40L
                       else NA_integer_,
      stringsAsFactors = FALSE))
  }

  # --- Montreal Open Data CKAN (Loi/Justice/Securite group) ----
  mtl <- morie_datasets_montreal_justice_safety_layers(offline = TRUE)
  for (i in seq_len(nrow(mtl))) {
    push(data.frame(
      dataset_key = mtl$package_name[i],
      source = "montreal_opendata",
      id = mtl$package_name[i],
      api_modes = "ckan",
      loader = if (mtl$package_name[i] ==
                     "interventions-service-securite-incendie-montreal")
                "morie_datasets_montreal_sim_interventions"
              else "morie_datasets_montreal_ckan_resource",
      dict_url = sprintf("https://donnees.montreal.ca/dataset/%s",
                          mtl$package_name[i]),
      n_rows_bundled = if (mtl$package_name[i] ==
                              "interventions-service-securite-incendie-montreal")
                         349L else NA_integer_,
      stringsAsFactors = FALSE))
  }

  # --- StatCan CCJS WDS REST cubes (10 curated) ----------------
  sc <- morie_datasets_statcan_ccjs_cubes()
  for (i in seq_len(nrow(sc))) {
    push(data.frame(
      dataset_key = sprintf("statcan_%d", sc$product_id[i]),
      source = "statcan_ccjs",
      id = as.character(sc$product_id[i]),
      api_modes = "statcan_wds",
      loader = "morie_datasets_statcan_cube_metadata",
      dict_url = sprintf("https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=%d",
                          sc$product_id[i]),
      n_rows_bundled = NA_integer_,
      stringsAsFactors = FALSE))
  }

  # --- VPD GeoDASH (manual download, sample bundled) -----------
  push(data.frame(
    dataset_key = "vpd_crime",
    source = "vpd_geodash",
    id = "crimedata_csv_AllNeighbourhoods_AllYears",
    api_modes = "manual_download",
    loader = "morie_datasets_vpd_crime",
    dict_url = "https://geodash.vpd.ca/opendata/",
    n_rows_bundled = .morie_portal_fixture_rows("vpd_crime_sample.csv"),
    stringsAsFactors = FALSE))

  out <- do.call(rbind, rows)
  rownames(out) <- NULL

  if (!is.null(portal)) {
    portal <- match.arg(portal,
                         choices = c("chicago", "nyc_nypd",
                                      "nyc_opendata",
                                      "tps_arcgis_hub", "tps_psdp",
                                      "ontario_ckan",
                                      "vancouver_opendata",
                                      "vpd_geodash",
                                      "statcan_ccjs",
                                      "montreal_opendata",
                                      "toronto_opendata"))
    out <- out[out$source == portal, , drop = FALSE]
    rownames(out) <- NULL
  }
  out
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

.morie_portal_fixture_rows <- function(fname) {
  if (is.null(fname) || is.na(fname) || !nzchar(fname))
    return(NA_integer_)
  path <- system.file("extdata", fname, package = "morie")
  if (!nzchar(path)) return(NA_integer_)
  # Lightweight row count via tally of newlines minus the header.
  n_lines <- length(readLines(path, warn = FALSE))
  max(0L, n_lines - 1L)
}
