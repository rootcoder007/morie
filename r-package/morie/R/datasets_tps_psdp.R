# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Toronto Police Service Public Safety Data Portal (PSDP) -- shared
# factory for the 11 open-data crime layers. Mirrors the canonical
# TPS PSDP layer registry and wraps each layer with:
#
#   * offline = TRUE  -- read a small bundled synthetic fixture from
#                         inst/extdata/tps_psdp_<key>_sample.csv
#   * offline = FALSE -- hit the registered ArcGIS FeatureServer layer
#                         via .morie_tps_psdp_feature_query() (mockable)
#
# Eleven layers across six distinct schema clusters:
#
#   Cluster A: standard 31-col crime feed (OFFENCE + CSI_CATEGORY +
#     UCR_CODE/EXT + REPORT_/OCC_* + HOOD_158/140 + WGS84) --
#     assault, autotheft, breakandenter, robbery,
#     theft_from_motor_vehicle, theft_over.
#   Cluster B: BicycleTheft 35-col (PRIMARY_OFFENCE instead of
#     OFFENCE + BIKE_MAKE/MODEL/TYPE/SPEED/COLOUR/COST + STATUS).
#   Cluster C: HateCrimes 25-col (OCCURRENCE_/REPORTED_* dates +
#     AGE_BIAS/MENTAL_OR_PHYSICAL_DISABILITY/RACE_BIAS/ETHNICITY_BIAS/
#     LANGUAGE_BIAS/RELIGION_BIAS/SEXUAL_ORIENTATION_BIAS/GENDER_BIAS/
#     MULTIPLE_BIAS + PRIMARY_OFFENCE + ARREST_MADE).
#   Cluster D: Homicides 18-col (HOMICIDE_TYPE + minimal date triple).
#   Cluster E: IntimatePartnerAndFamilyViolence 15-col (INDEX +
#     HISTORICAL + FAMILY_VIOLENCE_FLAG/RELATION + COUNT).
#   Cluster F: ShootingAndFirearmDischarges 22-col (OCC_TIME_RANGE +
#     DEATH + INJURIES + EVENT_TYPE).
#
# All eleven carry BOTH HOOD_158 and HOOD_140 columns so the bundled
# 158<->140 crosswalk + the cake-cutting helpers (see
# R/toronto_neighbourhoods.R) work end-to-end on any TPS PSDP layer.

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

.MORIE_TPS_PSDP_BASE_ARCGIS <-
  "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services"

#
# Each entry carries:
#   arcgis_url - direct FeatureServer layer URL (the pre-3TT+ live
#                  path; still honoured when the caller passes
#                  layer_url = ... as an override).
#   fixture    - bundled offline sample CSV.
#   label      - human-readable label for the discovery table.
#   hub_id     - 32-char hex GUID matching the canonical TPS Hub
#                  catalog entry; the new 3TT+ default live path
#                  routes through morie_datasets_tps_arcgis_hub_by_id
#                  using this id (verified live against
#                  inst/extdata/tps_arcgis_hub_catalog.csv during 3TT+).

.MORIE_TPS_PSDP_REGISTRY <- list(
  assault = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Assault_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_assault_sample.csv",
    label = "Assault",
    hub_id = "b4d0398d37eb4aa184065ed625ddb922"),
  autotheft = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Auto_Theft_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_autotheft_sample.csv",
    label = "AutoTheft",
    hub_id = "95ab41aee16847dba8453bf1688249d6"),
  bicycletheft = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Bicycle_Thefts_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_bicycletheft_sample.csv",
    label = "BicycleTheft",
    hub_id = "a89d10d5e28444ceb0c8d1d4c0ee39cc"),
  breakandenter = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Break_and_Enter_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_breakandenter_sample.csv",
    label = "BreakAndEnter",
    hub_id = "040ead448df2412da252cfbb532e77ac"),
  hatecrimes = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Hate_Crimes_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_hatecrimes_sample.csv",
    label = "HateCrimes",
    hub_id = "3dc9a8fae28b42c7aaf8fc62c7fbfdaa"),
  homicides = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Homicides_Open_Data_ASR_RC_TBL_002/FeatureServer/0"),
    fixture = "tps_psdp_homicides_sample.csv",
    label = "Homicides",
    hub_id = "d96bf5b67c1c49879f354dad51cf81f9"),
  intimate_partner_family_violence = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Intimate_Partner_and_Family_Violence_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_intimate_partner_family_violence_sample.csv",
    label = "IntimatePartnerAndFamilyViolence",
    hub_id = "724113c886ee4df2b917dcc047f82d26"),
  robbery = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Robbery_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_robbery_sample.csv",
    label = "Robbery",
    hub_id = "d0e1e98de5f945faa2fe635dee3f4062"),
  shooting_firearm_discharges = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Shooting_and_Firearm_Discharges_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_shooting_firearm_discharges_sample.csv",
    label = "ShootingAndFirearmDischarges",
    hub_id = "64ddeca12da34403869968ec725e23c4"),
  theft_from_motor_vehicle = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Theft_From_Motor_Vehicle_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_theft_from_motor_vehicle_sample.csv",
    label = "TheftFromMotorVehicle",
    hub_id = "d9303bc20f8a4351b7744a8703eecb80"),
  theft_over = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Theft_Over_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_theft_over_sample.csv",
    label = "TheftOver",
    hub_id = "7530d9b637c340059ccb81a782481c04")
)

#' List the TPS PSDP layers wrapped by morie
#'
#' @return A `data.frame` with columns `layer_key`, `label`,
#'   `arcgis_url`, `fixture`, `hub_id` (3TT+ canonical id matching
#'   the TPS Hub catalog).
#' @export
morie_tps_psdp_layers <- function() {
  rows <- lapply(names(.MORIE_TPS_PSDP_REGISTRY), function(k) {
    e <- .MORIE_TPS_PSDP_REGISTRY[[k]]
    data.frame(layer_key = k, label = e$label,
                arcgis_url = e$arcgis_url,
                fixture = e$fixture,
                hub_id = e$hub_id %||% NA_character_,
                stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

# ---------------------------------------------------------------------------
# Factory (3TT+: live default routes through the TPS Hub canonical
# hub_id loader; layer_url override remains as the backward-compat
# escape hatch for callers who want a non-canonical FeatureServer URL).
# ---------------------------------------------------------------------------

.morie_tps_psdp_dispatch <- function(layer_key, year, max_features,
                                      offline, layer_url) {
  if (!(layer_key %in% names(.MORIE_TPS_PSDP_REGISTRY))) {
    stop(sprintf(paste0(
      "unknown TPS PSDP layer_key '%s'. Available: %s"),
      layer_key,
      paste(names(.MORIE_TPS_PSDP_REGISTRY), collapse = ", ")),
      call. = FALSE)
  }
  entry <- .MORIE_TPS_PSDP_REGISTRY[[layer_key]]
  if (isTRUE(offline)) {
    path <- system.file("extdata", entry$fixture, package = "morie")
    if (!nzchar(path)) {
      stop(sprintf("bundled TPS PSDP fixture %s missing",
                   entry$fixture), call. = FALSE)
    }
    df <- utils::read.csv(path, stringsAsFactors = FALSE,
                           check.names = FALSE)
    # The OCC_YEAR column is canonical across most clusters; for the
    # HateCrimes (OCCURRENCE_YEAR) + IPFV (REPORT_YEAR) variants we
    # honour year filtering on the right column if present.
    year_col <- intersect(c("OCC_YEAR", "OCCURRENCE_YEAR",
                             "REPORT_YEAR"), names(df))[1]
    if (!is.null(year) && !is.na(year_col)) {
      df <- df[df[[year_col]] == as.integer(year), , drop = FALSE]
      rownames(df) <- NULL
    }
    if (!is.null(max_features)) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  where <- if (is.null(year)) "1=1" else sprintf("OCC_YEAR = %d",
                                                  as.integer(year))
  # Backward-compat escape hatch: if the caller passes a layer_url
  # override, hit it directly via the pre-3TT+ FeatureServer query
  # helper. This preserves the historical behaviour for callers
  # depending on non-canonical mirrors.
  if (!is.null(layer_url)) {
    return(.morie_tps_psdp_feature_query(
      layer_url, where = where,
      max_features = max_features,
      return_geometry = FALSE))
  }
  # 3TT+ canonical path: route through the TPS Hub catalog by
  # hub_id. Same underlying FeatureServer, but the resolution +
  # query goes through morie_datasets_tps_arcgis_hub_by_id so all
  # 71 TPS Hub items share a single code path.
  morie_datasets_tps_arcgis_hub_by_id(
    entry$hub_id,
    format = "json",
    where = where,
    max_features = max_features,
    layer_idx = 0L,
    offline = TRUE)  # offline=TRUE here means "resolve the
                      # FeatureServer URL via the bundled catalog
                      # (no network for the resolve step)"; the
                      # data query itself always hits the network.
}

# ---------------------------------------------------------------------------
# 11 thin per-layer wrappers
# ---------------------------------------------------------------------------

#' TPS PSDP -- Assault
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_assault <- function(year = NULL,
                                         max_features = NULL,
                                         offline = TRUE,
                                         layer_url = NULL) {
  .morie_tps_psdp_dispatch("assault", year, max_features, offline,
                            layer_url)
}

#' TPS PSDP -- Auto Theft
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_autotheft <- function(year = NULL,
                                           max_features = NULL,
                                           offline = TRUE,
                                           layer_url = NULL) {
  .morie_tps_psdp_dispatch("autotheft", year, max_features, offline,
                            layer_url)
}

#' TPS PSDP -- Bicycle Theft
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_bicycletheft <- function(year = NULL,
                                              max_features = NULL,
                                              offline = TRUE,
                                              layer_url = NULL) {
  .morie_tps_psdp_dispatch("bicycletheft", year, max_features,
                            offline, layer_url)
}

#' TPS PSDP -- Break and Enter
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_breakandenter <- function(year = NULL,
                                               max_features = NULL,
                                               offline = TRUE,
                                               layer_url = NULL) {
  .morie_tps_psdp_dispatch("breakandenter", year, max_features,
                            offline, layer_url)
}

#' TPS PSDP -- Hate Crimes
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_hatecrimes <- function(year = NULL,
                                            max_features = NULL,
                                            offline = TRUE,
                                            layer_url = NULL) {
  .morie_tps_psdp_dispatch("hatecrimes", year, max_features, offline,
                            layer_url)
}

#' TPS PSDP -- Homicides
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_homicides <- function(year = NULL,
                                           max_features = NULL,
                                           offline = TRUE,
                                           layer_url = NULL) {
  .morie_tps_psdp_dispatch("homicides", year, max_features, offline,
                            layer_url)
}

#' TPS PSDP -- Intimate Partner and Family Violence
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_intimate_partner_family_violence <- function(
  year = NULL, max_features = NULL,
  offline = TRUE, layer_url = NULL) {
  .morie_tps_psdp_dispatch("intimate_partner_family_violence",
                            year, max_features, offline, layer_url)
}

#' TPS PSDP -- Robbery
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_robbery <- function(year = NULL,
                                         max_features = NULL,
                                         offline = TRUE,
                                         layer_url = NULL) {
  .morie_tps_psdp_dispatch("robbery", year, max_features, offline,
                            layer_url)
}

#' TPS PSDP -- Shooting and Firearm Discharges
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_shooting_firearm_discharges <- function(
  year = NULL, max_features = NULL,
  offline = TRUE, layer_url = NULL) {
  .morie_tps_psdp_dispatch("shooting_firearm_discharges",
                            year, max_features, offline, layer_url)
}

#' TPS PSDP -- Theft From Motor Vehicle
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_theft_from_motor_vehicle <- function(
  year = NULL, max_features = NULL,
  offline = TRUE, layer_url = NULL) {
  .morie_tps_psdp_dispatch("theft_from_motor_vehicle",
                            year, max_features, offline, layer_url)
}

#' TPS PSDP -- Theft Over
#' @inheritParams morie_datasets_tps_mha_apprehensions
#' @export
morie_datasets_tps_theft_over <- function(year = NULL,
                                            max_features = NULL,
                                            offline = TRUE,
                                            layer_url = NULL) {
  .morie_tps_psdp_dispatch("theft_over", year, max_features, offline,
                            layer_url)
}
