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

.MORIE_TPS_PSDP_REGISTRY <- list(
  assault = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Assault_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_assault_sample.csv",
    label = "Assault"),
  autotheft = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Auto_Theft_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_autotheft_sample.csv",
    label = "AutoTheft"),
  bicycletheft = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Bicycle_Thefts_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_bicycletheft_sample.csv",
    label = "BicycleTheft"),
  breakandenter = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Break_and_Enter_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_breakandenter_sample.csv",
    label = "BreakAndEnter"),
  hatecrimes = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Hate_Crimes_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_hatecrimes_sample.csv",
    label = "HateCrimes"),
  homicides = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Homicides_Open_Data_ASR_RC_TBL_002/FeatureServer/0"),
    fixture = "tps_psdp_homicides_sample.csv",
    label = "Homicides"),
  intimate_partner_family_violence = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Intimate_Partner_and_Family_Violence_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_intimate_partner_family_violence_sample.csv",
    label = "IntimatePartnerAndFamilyViolence"),
  robbery = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Robbery_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_robbery_sample.csv",
    label = "Robbery"),
  shooting_firearm_discharges = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Shooting_and_Firearm_Discharges_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_shooting_firearm_discharges_sample.csv",
    label = "ShootingAndFirearmDischarges"),
  theft_from_motor_vehicle = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Theft_From_Motor_Vehicle_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_theft_from_motor_vehicle_sample.csv",
    label = "TheftFromMotorVehicle"),
  theft_over = list(
    arcgis_url = paste0(.MORIE_TPS_PSDP_BASE_ARCGIS,
                        "/Theft_Over_Open_Data/FeatureServer/0"),
    fixture = "tps_psdp_theft_over_sample.csv",
    label = "TheftOver")
)

#' List the TPS PSDP layers wrapped by morie
#'
#' @return A `data.frame` with columns `layer_key`, `label`,
#'   `arcgis_url`, `fixture`.
#' @export
morie_tps_psdp_layers <- function() {
  rows <- lapply(names(.MORIE_TPS_PSDP_REGISTRY), function(k) {
    e <- .MORIE_TPS_PSDP_REGISTRY[[k]]
    data.frame(layer_key = k, label = e$label,
                arcgis_url = e$arcgis_url, fixture = e$fixture,
                stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

# ---------------------------------------------------------------------------
# Factory
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
  if (is.null(layer_url)) layer_url <- entry$arcgis_url
  where <- if (is.null(year)) "1=1" else sprintf("OCC_YEAR = %d",
                                                  as.integer(year))
  .morie_tps_psdp_feature_query(layer_url, where = where,
                                  max_features = max_features,
                                  return_geometry = FALSE)
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
