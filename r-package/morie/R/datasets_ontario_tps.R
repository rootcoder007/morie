# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Three high-level dataset loaders that mirror the canonical
# upstream feeds for our three highest-stakes data sources:
#
#   1. ARSAU UoF main_records  -- Ontario Police Use of Force,
#        Race-Based Data Strategy
#        https://data.ontario.ca/dataset/police-use-of-force-race-based-data
#        Resource:
#          ea9dc29c-b4f1-4426-b1f2-974ce995aca1
#        Direct CSV:
#          .../resource/ea9dc29c-b4f1-4426-b1f2-974ce995aca1/download/
#            uof_main_records.csv
#        Datastore dump JSON:
#          https://data.ontario.ca/datastore/dump/
#            ea9dc29c-b4f1-4426-b1f2-974ce995aca1?format=json
#
#   2. OTIS d01 Deaths-in-Custody  -- Ministry of the Solicitor General
#        Data on Inmates in Ontario
#        https://data.ontario.ca/dataset/data-on-inmates-in-ontario
#        Resource:
#          89e3b63f-5679-4fa4-b98a-fdd2dc486f29
#        Direct CSV:
#          .../resource/89e3b63f-5679-4fa4-b98a-fdd2dc486f29/download/
#            d01_deaths_in_custody_detailed_dataset.csv
#        Datastore dump JSON:
#          https://data.ontario.ca/datastore/dump/
#            89e3b63f-5679-4fa4-b98a-fdd2dc486f29?format=json
#
#   3. TPS Mental Health Act Apprehensions  -- TPS Public Safety
#        Data Portal (PSDP)
#        https://data.tps.ca/datasets/333c4e1c96314741a83425045b6a7642_0/
#          explore
#        Published as GeoPackage / Shapefile / GeoJSON / CSV / KML /
#        File Geodatabase / Feature Collection / Excel /
#        SQLite Geodatabase. We default to the ArcGIS REST
#        FeatureServer JSON for the live-mode path.
#
# Each loader supports offline = TRUE (read a small bundled synthetic
# fixture from inst/extdata/) and offline = FALSE (hit the live
# upstream via a mockable internal helper:
#
#   .morie_ontario_ckan_dump_csv()    -- for the two Ontario CKAN feeds
#   .morie_tps_psdp_feature_query()   -- for the TPS PSDP ArcGIS feed
#
# Tests stub these helpers via testthat::local_mocked_bindings(.package
# = "morie") so the suite never reaches the wire.

# ---------------------------------------------------------------------------
# Internal http helpers (mockable)
# ---------------------------------------------------------------------------

# Default Ontario CKAN portal endpoint (data.ontario.ca).
.MORIE_ONTARIO_CKAN_BASE <- "https://data.ontario.ca"

# Internal: GET the Ontario CKAN datastore_dump JSON endpoint for a
# given resource_id; return a parsed data.frame. Mock this in tests.
.morie_ontario_ckan_dump_csv <- function(resource_id, limit = 200000L) {
  if (!requireNamespace("httr2", quietly = TRUE) ||
      !requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Ontario CKAN dump fetch needs httr2 + jsonlite. ",
         "install.packages(c('httr2', 'jsonlite'))",
         call. = FALSE)
  }
  url <- sprintf("%s/datastore/dump/%s?format=json",
                 .MORIE_ONTARIO_CKAN_BASE, resource_id)
  req <- httr2::request(url)
  req <- httr2::req_timeout(req, 120)
  resp <- httr2::req_perform(req)
  body <- jsonlite::fromJSON(httr2::resp_body_string(resp),
                              simplifyVector = TRUE)
  # CKAN datastore_dump returns either {"records":[...]} or a bare array
  # depending on portal version; handle both.
  if (is.data.frame(body)) return(body)
  if (!is.null(body$records)) return(as.data.frame(body$records))
  if (!is.null(body$result$records))
    return(as.data.frame(body$result$records))
  stop("unrecognised Ontario CKAN datastore_dump payload shape",
       call. = FALSE)
}

# Default TPS Public Safety Data Portal ArcGIS host.
.MORIE_TPS_PSDP_BASE <- "https://services.arcgis.com"

# Internal: GET a TPS PSDP ArcGIS FeatureServer layer; return a
# data.frame of attributes. Mock this in tests.
.morie_tps_psdp_feature_query <- function(layer_url, where = "1=1",
                                            max_features = NULL,
                                            return_geometry = FALSE) {
  # Delegate to the existing TPS ingest helper so this stays a thin
  # wrapper that callers (and mocks) can target cleanly.
  if (exists(".morie_dataset_tps_fetch",
             envir = asNamespace("morie"), inherits = FALSE)) {
    return(.morie_dataset_tps_fetch(
      layer_url, where = where, max_features = max_features,
      return_geometry = return_geometry))
  }
  stop(".morie_dataset_tps_fetch unavailable; cannot dispatch TPS PSDP query",
       call. = FALSE)
}

# ---------------------------------------------------------------------------
# ARSAU UoF main_records
# ---------------------------------------------------------------------------

# Canonical Ontario CKAN resource ids for the ARSAU UoF main_records
# resource, by reporting year.
.MORIE_ARSAU_UOF_MAIN_RECORDS_RESOURCE_IDS <- list(
  "2023" = "94f303a2-963e-4fd1-958d-6681b310cb6d",
  "2024" = "ea9dc29c-b4f1-4426-b1f2-974ce995aca1")

#' Ontario Use-of-Force main records (one row per incident)
#'
#' Wraps the Ontario Police Use-of-Force Race-Based Data Strategy
#' resource. Offline mode reads a small bundled synthetic fixture
#' from `inst/extdata/arsau_uof_main_records_sample.csv` (5 rows in
#' the canonical 23-column subset of the 65-column upstream schema,
#' clearly stamped `SYNTHETIC-FIXTURE-XXX`). Live mode hits the
#' Ontario CKAN datastore-dump JSON endpoint for the requested
#' reporting year.
#'
#' @param year Reporting year (`"2023"` or `"2024"`). Honoured only
#'   when `offline = FALSE`.
#' @param offline If `TRUE` (default), read the bundled fixture. If
#'   `FALSE`, hit the live CKAN endpoint via httr2.
#' @param resource_id Optional CKAN resource id override.
#' @return A `data.frame`.
#' @references Ontario Open Data Catalogue, "Police Use of Force"
#'   (\url{https://data.ontario.ca/dataset/police-use-of-force-race-based-data});
#'   Open Government Licence -- Ontario.
#' @examples
#' df <- morie_datasets_arsau_uof_main_records(offline = TRUE)
#' head(df[, c("IncidentYear", "PoliceService", "IncidentType")])
#' @export
morie_datasets_arsau_uof_main_records <- function(year = "2024",
                                                    offline = TRUE,
                                                    resource_id = NULL) {
  if (isTRUE(offline)) {
    path <- system.file("extdata", "arsau_uof_main_records_sample.csv",
                        package = "morie")
    if (!nzchar(path)) {
      stop("bundled ARSAU UoF main_records fixture missing",
           call. = FALSE)
    }
    return(utils::read.csv(path, stringsAsFactors = FALSE,
                            check.names = FALSE))
  }
  if (is.null(resource_id)) {
    resource_id <- .MORIE_ARSAU_UOF_MAIN_RECORDS_RESOURCE_IDS[[
      as.character(year)]]
    if (is.null(resource_id)) {
      stop(sprintf(paste0(
        "no canonical Ontario CKAN resource_id for ARSAU UoF ",
        "main_records year %s; pass resource_id= explicitly"),
        year), call. = FALSE)
    }
  }
  .morie_ontario_ckan_dump_csv(resource_id)
}

# ---------------------------------------------------------------------------
# OTIS d01 Deaths in Custody
# ---------------------------------------------------------------------------

.MORIE_OTIS_D01_RESOURCE_ID <- "89e3b63f-5679-4fa4-b98a-fdd2dc486f29"

#' OTIS Deaths-in-Custody detailed dataset (d01)
#'
#' Wraps the Ontario "Data on Inmates in Ontario" d01 resource.
#' Schema: `Year, UniqueIndividual_ID, Region_AtTimeOfDeath,
#' HousingUnit_Type, MedicalCauseofDeath, MeansofDeath`.
#'
#' @param offline If `TRUE` (default), read the bundled synthetic
#'   fixture from `inst/extdata/otis_d01_deaths_in_custody_sample.csv`
#'   (5 rows). If `FALSE`, hit the live CKAN datastore-dump JSON
#'   endpoint for resource id `89e3b63f-5679-4fa4-b98a-fdd2dc486f29`.
#' @param resource_id Optional CKAN resource id override.
#' @return A `data.frame`.
#' @references Ontario Open Data Catalogue, "Data on Inmates in
#'   Ontario"
#'   (\url{https://data.ontario.ca/dataset/data-on-inmates-in-ontario});
#'   Open Government Licence -- Ontario.
#' @examples
#' df <- morie_datasets_otis_d01_deaths_in_custody(offline = TRUE)
#' table(df$Region_AtTimeOfDeath)
#' @export
morie_datasets_otis_d01_deaths_in_custody <- function(offline = TRUE,
                                                       resource_id = NULL) {
  if (isTRUE(offline)) {
    path <- system.file("extdata",
                        "otis_d01_deaths_in_custody_sample.csv",
                        package = "morie")
    if (!nzchar(path)) {
      stop("bundled OTIS d01 fixture missing", call. = FALSE)
    }
    return(utils::read.csv(path, stringsAsFactors = FALSE,
                            check.names = FALSE))
  }
  if (is.null(resource_id)) {
    resource_id <- .MORIE_OTIS_D01_RESOURCE_ID
  }
  .morie_ontario_ckan_dump_csv(resource_id)
}

# ---------------------------------------------------------------------------
# TPS Mental Health Act Apprehensions (PSDP)
# ---------------------------------------------------------------------------

# Canonical ArcGIS FeatureServer URL for the TPS PSDP MHA layer
# (333c4e1c96314741a83425045b6a7642_0).
.MORIE_TPS_PSDP_MHA_LAYER_URL <- paste0(
  "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/",
  "Mental_Health_Act_Apprehensions_Open_Data/FeatureServer/0")

#' TPS Mental Health Act Apprehensions (PSDP)
#'
#' Wraps the TPS Public Safety Data Portal "Mental Health Act
#' Apprehensions" layer (one row per police-attended MHA event).
#' Carries both HOOD_158 and HOOD_140 columns -- callers should pick
#' a version via [morie_tps_resolve_hood_col()] before downstream
#' analysis.
#'
#' @param year Optional reporting year filter (applies an
#'   `OCC_YEAR = <year>` WHERE clause when `offline = FALSE`).
#' @param max_features Optional cap on returned rows
#'   (`offline = FALSE` only).
#' @param offline If `TRUE` (default), read the bundled synthetic
#'   fixture from `inst/extdata/tps_mha_apprehensions_sample.csv`
#'   (5 rows in the canonical TPS PSDP 22-column schema). If
#'   `FALSE`, hit the TPS PSDP ArcGIS FeatureServer.
#' @param layer_url Optional ArcGIS layer URL override.
#' @return A `data.frame`.
#' @references TPS Public Safety Data Portal, "Mental Health Act
#'   Apprehensions Open Data"
#'   (\url{https://data.tps.ca/datasets/333c4e1c96314741a83425045b6a7642_0/explore}).
#' @examples
#' df <- morie_datasets_tps_mha_apprehensions(offline = TRUE)
#' table(df$APPREHENSION_TYPE)
#' @export
morie_datasets_tps_mha_apprehensions <- function(year = NULL,
                                                   max_features = NULL,
                                                   offline = TRUE,
                                                   layer_url = NULL) {
  if (isTRUE(offline)) {
    path <- system.file("extdata", "tps_mha_apprehensions_sample.csv",
                        package = "morie")
    if (!nzchar(path)) {
      stop("bundled TPS MHA fixture missing", call. = FALSE)
    }
    df <- utils::read.csv(path, stringsAsFactors = FALSE,
                           check.names = FALSE)
    if (!is.null(year) && "OCC_YEAR" %in% names(df)) {
      df <- df[df$OCC_YEAR == as.integer(year), , drop = FALSE]
      rownames(df) <- NULL
    }
    if (!is.null(max_features)) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  if (is.null(layer_url)) layer_url <- .MORIE_TPS_PSDP_MHA_LAYER_URL
  where <- if (is.null(year)) "1=1" else sprintf("OCC_YEAR = %d",
                                                  as.integer(year))
  .morie_tps_psdp_feature_query(layer_url, where = where,
                                  max_features = max_features,
                                  return_geometry = FALSE)
}

# ---------------------------------------------------------------------------
# ARSAU UoF -- the four remaining year x record-type loaders
# ---------------------------------------------------------------------------
#
# Canonical Ontario CKAN resource ids by year x kind (extracted from
# the ARSAU sidecar JSONs the morie repo already ships). 2023|
# weapon_records is INVALID per the ministry's technical report and
# has no published resource; we leave the slot NA.

.MORIE_ARSAU_UOF_RESOURCE_IDS <- list(
  individual_records = list(
    "2023" = "133c73fa-9d8e-435e-8c6d-7d1e14d1e88d",
    "2024" = "690d4c5e-095e-49a0-bbab-b7fc680f3c6b"),
  probe_cycle_records = list(
    "2023" = "339b9e63-9521-44a6-8719-c2cb9aa39a8a",
    "2024" = "76875b6a-4352-4722-a3f6-997cc220dc4f"),
  weapon_records = list(
    "2023" = NA_character_,
    "2024" = "2c1ab494-d636-4c17-9699-3819112982a5"),
  # 5-year aggregate (2020-2022) -- single resource per kind.
  aggregate_summary  = list(
    "2020-2022" = "7560d405-444c-4340-95c4-f73849015501"),
  detailed_dataset   = list(
    "2020-2022" = "2150ac23-4e55-474a-b61f-81baf6850851"))

# Internal: shared offline+live dispatch for ARSAU UoF wrappers.
.morie_arsau_uof_dispatch <- function(kind, year, offline,
                                        resource_id, fixture_name) {
  if (isTRUE(offline)) {
    path <- system.file("extdata", fixture_name, package = "morie")
    if (!nzchar(path)) {
      stop(sprintf("bundled ARSAU UoF fixture %s missing",
                   fixture_name), call. = FALSE)
    }
    return(utils::read.csv(path, stringsAsFactors = FALSE,
                            check.names = FALSE))
  }
  if (is.null(resource_id)) {
    rid <- .MORIE_ARSAU_UOF_RESOURCE_IDS[[kind]][[as.character(year)]]
    if (is.null(rid) || is.na(rid)) {
      stop(sprintf(paste0(
        "no canonical Ontario CKAN resource_id for ARSAU UoF ",
        "%s year %s (NULL or marked INVALID); pass resource_id= ",
        "explicitly"), kind, year), call. = FALSE)
    }
    resource_id <- rid
  }
  .morie_ontario_ckan_dump_csv(resource_id)
}

#' Ontario Use-of-Force individual records (one row per
#' individual-in-incident)
#' @param year Reporting year (`"2023"` or `"2024"`).
#' @inheritParams morie_datasets_arsau_uof_main_records
#' @return A `data.frame`.
#' @export
morie_datasets_arsau_uof_individual_records <- function(year = "2024",
                                                          offline = TRUE,
                                                          resource_id = NULL) {
  .morie_arsau_uof_dispatch("individual_records", year, offline,
                              resource_id,
                              "arsau_uof_individual_records_sample.csv")
}

#' Ontario Use-of-Force probe-cycle records (one row per CEW cartridge
#' probe per individual-in-incident)
#' @inheritParams morie_datasets_arsau_uof_individual_records
#' @return A `data.frame`.
#' @export
morie_datasets_arsau_uof_probe_cycle_records <- function(year = "2024",
                                                           offline = TRUE,
                                                           resource_id = NULL) {
  .morie_arsau_uof_dispatch("probe_cycle_records", year, offline,
                              resource_id,
                              "arsau_uof_probe_cycle_records_sample.csv")
}

#' Ontario Use-of-Force weapon records (one row per weapon per
#' individual-in-incident)
#'
#' Year 2023 weapon_records is marked INVALID by the Ontario ministry's
#' technical report and has no published CKAN resource; passing
#' `year = "2023"` with `offline = FALSE` raises.
#'
#' @inheritParams morie_datasets_arsau_uof_individual_records
#' @return A `data.frame`.
#' @export
morie_datasets_arsau_uof_weapon_records <- function(year = "2024",
                                                      offline = TRUE,
                                                      resource_id = NULL) {
  .morie_arsau_uof_dispatch("weapon_records", year, offline,
                              resource_id,
                              "arsau_uof_weapon_records_sample.csv")
}

#' Ontario Use-of-Force aggregate summary (5-year 2020-2022, pre-RBDS
#' rollup)
#'
#' @param offline If `TRUE` (default), read the bundled synthetic
#'   fixture. If `FALSE`, hit Ontario CKAN.
#' @param resource_id Optional override.
#' @return A `data.frame`.
#' @export
morie_datasets_arsau_aggregate_summary <- function(offline = TRUE,
                                                     resource_id = NULL) {
  .morie_arsau_uof_dispatch(
    "aggregate_summary", "2020-2022", offline, resource_id,
    "arsau_uof_aggregate_summary_2020_2022_sample.csv")
}

#' Ontario Use-of-Force detailed dataset (5-year 2020-2022, pre-RBDS)
#' @inheritParams morie_datasets_arsau_aggregate_summary
#' @return A `data.frame`.
#' @export
morie_datasets_arsau_detailed_dataset <- function(offline = TRUE,
                                                    resource_id = NULL) {
  .morie_arsau_uof_dispatch(
    "detailed_dataset", "2020-2022", offline, resource_id,
    "arsau_uof_detailed_dataset_2020_2022_sample.csv")
}

# ---------------------------------------------------------------------------
# Consolidated Ontario CKAN registry (discovery)
# ---------------------------------------------------------------------------

.MORIE_ONTARIO_CKAN_REGISTRY <- list(
  arsau_uof_main_records_2023 = list(
    resource_id = "94f303a2-963e-4fd1-958d-6681b310cb6d",
    label = "ARSAU UoF main records (2023)",
    fixture = "arsau_uof_main_records_sample.csv",
    family = "arsau", year = "2023"),
  arsau_uof_main_records_2024 = list(
    resource_id = "ea9dc29c-b4f1-4426-b1f2-974ce995aca1",
    label = "ARSAU UoF main records (2024)",
    fixture = "arsau_uof_main_records_sample.csv",
    family = "arsau", year = "2024"),
  arsau_uof_individual_records_2023 = list(
    resource_id = "133c73fa-9d8e-435e-8c6d-7d1e14d1e88d",
    label = "ARSAU UoF individual records (2023)",
    fixture = "arsau_uof_individual_records_sample.csv",
    family = "arsau", year = "2023"),
  arsau_uof_individual_records_2024 = list(
    resource_id = "690d4c5e-095e-49a0-bbab-b7fc680f3c6b",
    label = "ARSAU UoF individual records (2024)",
    fixture = "arsau_uof_individual_records_sample.csv",
    family = "arsau", year = "2024"),
  arsau_uof_probe_cycle_records_2023 = list(
    resource_id = "339b9e63-9521-44a6-8719-c2cb9aa39a8a",
    label = "ARSAU UoF probe-cycle records (2023)",
    fixture = "arsau_uof_probe_cycle_records_sample.csv",
    family = "arsau", year = "2023"),
  arsau_uof_probe_cycle_records_2024 = list(
    resource_id = "76875b6a-4352-4722-a3f6-997cc220dc4f",
    label = "ARSAU UoF probe-cycle records (2024)",
    fixture = "arsau_uof_probe_cycle_records_sample.csv",
    family = "arsau", year = "2024"),
  arsau_uof_weapon_records_2024 = list(
    resource_id = "2c1ab494-d636-4c17-9699-3819112982a5",
    label = "ARSAU UoF weapon records (2024)",
    fixture = "arsau_uof_weapon_records_sample.csv",
    family = "arsau", year = "2024"),
  arsau_uof_aggregate_summary_2020_2022 = list(
    resource_id = "7560d405-444c-4340-95c4-f73849015501",
    label = "ARSAU UoF aggregate summary 2020-2022",
    fixture = "arsau_uof_aggregate_summary_2020_2022_sample.csv",
    family = "arsau", year = "2020-2022"),
  arsau_uof_detailed_dataset_2020_2022 = list(
    resource_id = "2150ac23-4e55-474a-b61f-81baf6850851",
    label = "ARSAU UoF detailed dataset 2020-2022",
    fixture = "arsau_uof_detailed_dataset_2020_2022_sample.csv",
    family = "arsau", year = "2020-2022"),
  otis_d01_deaths_in_custody = list(
    resource_id = "89e3b63f-5679-4fa4-b98a-fdd2dc486f29",
    label = "OTIS d01 Deaths-in-Custody detailed",
    fixture = "otis_d01_deaths_in_custody_sample.csv",
    family = "otis", year = "all"))

#' List the Ontario CKAN datasets wrapped by morie
#'
#' Returns the consolidated registry of every Ontario Open Data feed
#' morie ships an offline-mode fixture + mocked live-mode dispatch for.
#' Pair with [morie_datasets_ontario_ckan_by_key()] for generic factory
#' access by `dataset_key`.
#'
#' Coverage as of this release:
#'   * ARSAU UoF: 2 main + 2 individual + 2 probe_cycle + 1 weapon (2024)
#'     + 2 5-year aggregates (aggregate_summary + detailed_dataset).
#'   * OTIS: d01 Deaths-in-Custody.
#'   * d02-d07 OTIS deaths variants + OTIS a01/b/c families: known but
#'     resource_ids not yet wired in (PRs welcome).
#'
#' @return A `data.frame` with columns `dataset_key`, `label`,
#'   `resource_id`, `family`, `year`, `fixture`.
#' @export
morie_datasets_ontario_ckan_layers <- function() {
  rows <- lapply(names(.MORIE_ONTARIO_CKAN_REGISTRY), function(k) {
    e <- .MORIE_ONTARIO_CKAN_REGISTRY[[k]]
    data.frame(dataset_key = k, label = e$label,
                resource_id = e$resource_id, family = e$family,
                year = e$year, fixture = e$fixture,
                stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' Generic Ontario CKAN dataset loader (by registry key)
#'
#' @param dataset_key One of the keys in
#'   [morie_datasets_ontario_ckan_layers()] (e.g.
#'   `"arsau_uof_main_records_2024"`).
#' @param offline If `TRUE` (default), read the bundled synthetic
#'   fixture. If `FALSE`, hit the live Ontario CKAN datastore-dump
#'   endpoint.
#' @param resource_id Optional CKAN resource_id override.
#' @return A `data.frame`.
#' @export
morie_datasets_ontario_ckan_by_key <- function(dataset_key,
                                                  offline = TRUE,
                                                  resource_id = NULL) {
  if (!(dataset_key %in% names(.MORIE_ONTARIO_CKAN_REGISTRY))) {
    stop(sprintf(paste0(
      "unknown Ontario CKAN dataset_key '%s'. Available: %s"),
      dataset_key,
      paste(names(.MORIE_ONTARIO_CKAN_REGISTRY), collapse = ", ")),
      call. = FALSE)
  }
  entry <- .MORIE_ONTARIO_CKAN_REGISTRY[[dataset_key]]
  if (isTRUE(offline)) {
    path <- system.file("extdata", entry$fixture, package = "morie")
    if (!nzchar(path)) {
      stop(sprintf("bundled Ontario CKAN fixture %s missing",
                   entry$fixture), call. = FALSE)
    }
    return(utils::read.csv(path, stringsAsFactors = FALSE,
                            check.names = FALSE))
  }
  if (is.null(resource_id)) resource_id <- entry$resource_id
  .morie_ontario_ckan_dump_csv(resource_id)
}
