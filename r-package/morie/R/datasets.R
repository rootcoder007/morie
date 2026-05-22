# SPDX-License-Identifier: AGPL-3.0-or-later
#
# datasets.R -- one-call dataset loaders for common Canadian / US
# sociolegal open data.
#
# Ported from src/morie/datasets.py.  Each function is a thin, stable
# wrapper that resolves to either a live HTTP fetch (httr2) or a
# bundled-in-package synthetic frame.  Synthetic frames are CSVs under
# `inst/extdata/` of the morie R package; absence of a synthetic CSV is
# surfaced as a clean `FileNotFoundError`-style condition.

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#' Resolve a bundled synthetic CSV path from the morie R package.
#' @keywords internal
#' @noRd
.morie_dataset_pkg_csv <- function(name) {
  path <- system.file("extdata", paste0(name, ".csv"), package = "morie")
  if (!nzchar(path)) {
    return(NA_character_)
  }
  path
}

#' Read a bundled synthetic frame, warning the user it's a toy dataset.
#' @keywords internal
#' @noRd
.morie_dataset_read_synthetic <- function(name, kind, columns = NULL) {
  path <- .morie_dataset_pkg_csv(name)
  if (is.na(path)) {
    if (!is.null(columns)) {
      empty <- as.data.frame(
        stats::setNames(replicate(length(columns), character(0), simplify = FALSE), columns),
        stringsAsFactors = FALSE
      )
      return(empty)
    }
    stop(sprintf(
      "morie_datasets_%s(offline=TRUE): no bundled synthetic frame at inst/extdata/%s.csv.",
      kind, name
    ))
  }
  warning(sprintf(
    paste0(
      "morie_datasets_%s(offline=TRUE): using the bundled synthetic %s frame. ",
      "This is a toy dataset with the documented schema but random data; ",
      "do not interpret outputs as findings about the real population."
    ),
    kind, kind
  ), call. = FALSE)
  utils::read.csv(path, stringsAsFactors = FALSE)
}

#' Build a SQL-ish `WHERE` clause for an "OCC_YEAR = ?" or "1=1" filter.
#' @keywords internal
#' @noRd
.morie_dataset_year_where <- function(year) {
  if (is.null(year)) {
    return("1=1")
  }
  sprintf("OCC_YEAR = %d", as.integer(year))
}

#' Minimal httr2 GET that returns a parsed JSON body or raises.
#' @keywords internal
#' @noRd
.morie_dataset_http_json <- function(url, query = NULL) {
  if (!requireNamespace("httr2", quietly = TRUE)) {
    stop("morie datasets HTTP fetch requires the 'httr2' package.")
  }
  req <- httr2::request(url)
  if (!is.null(query)) {
    req <- httr2::req_url_query(req, !!!query)
  }
  resp <- httr2::req_perform(req)
  httr2::resp_body_json(resp, simplifyVector = TRUE)
}

#' Convert a list-of-records / data.frame response into a clean data.frame.
#' @keywords internal
#' @noRd
.morie_dataset_records_to_df <- function(records) {
  if (is.data.frame(records)) {
    return(records)
  }
  if (is.null(records) || length(records) == 0L) {
    return(data.frame())
  }
  do.call(rbind, lapply(records, function(r) {
    as.data.frame(lapply(r, function(v) if (is.null(v)) NA else v),
                  stringsAsFactors = FALSE)
  }))
}

# ---------------------------------------------------------------------------
# TPS -- Toronto Police Service ArcGIS
# ---------------------------------------------------------------------------

#' Default TPS ArcGIS layer registry (verified 2026-05).
#' @keywords internal
#' @noRd
.MORIE_TPS_LAYER_REGISTRY <- list(
  `major-crime` = paste0(
    "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/",
    "Major_Crime_Indicators_Open_Data/FeatureServer/0"
  ),
  `shooting-firearms` = paste0(
    "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/",
    "Shooting_and_Firearm_Discharges_Open_Data/FeatureServer/0"
  ),
  homicide = paste0(
    "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/",
    "Homicides_Open_Data_ASR_RC_TBL_002/FeatureServer/0"
  )
)

#' Fetch a TPS ArcGIS FeatureServer layer as a data frame.
#' @keywords internal
#' @noRd
.morie_dataset_tps_fetch <- function(layer_url, where = "1=1",
                                     max_features = NULL,
                                     return_geometry = FALSE) {
  query <- list(
    where = where,
    outFields = "*",
    f = "json",
    returnGeometry = if (isTRUE(return_geometry)) "true" else "false"
  )
  if (!is.null(max_features)) {
    query$resultRecordCount <- as.integer(max_features)
  }
  body <- .morie_dataset_http_json(paste0(layer_url, "/query"), query = query)
  features <- body$features
  if (is.null(features) || length(features) == 0L) {
    return(data.frame())
  }
  attrs <- lapply(features, function(f) f$attributes)
  df <- .morie_dataset_records_to_df(attrs)
  if (isTRUE(return_geometry)) {
    geoms <- lapply(features, function(f) f$geometry)
    df$geom_x <- vapply(geoms, function(g) if (is.null(g$x)) NA_real_ else g$x,
                       numeric(1))
    df$geom_y <- vapply(geoms, function(g) if (is.null(g$y)) NA_real_ else g$y,
                       numeric(1))
  }
  df
}

#' TPS Major Crime Indicators feed.
#'
#' @param year Integer or `NULL`.  If set, filter to `OCC_YEAR == year`
#'   server-side.
#' @param max_features Integer or `NULL`; cap on returned rows.
#' @param include_geometry Logical; include `geom_x` / `geom_y`.
#' @param offline Logical; return the bundled synthetic frame instead
#'   of hitting the live ArcGIS endpoint.
#' @return A `data.frame` with the documented TPS schema.
#' @export
morie_datasets_tps_major_crime <- function(year = NULL,
                                           max_features = NULL,
                                           include_geometry = FALSE,
                                           offline = FALSE) {
  if (isTRUE(offline)) {
    df <- .morie_dataset_read_synthetic("tps_major_crime", "tps_major_crime")
    if (!is.null(year) && "OCC_YEAR" %in% colnames(df)) {
      df <- df[df$OCC_YEAR == year, , drop = FALSE]
      rownames(df) <- NULL
    }
    if (!is.null(max_features)) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  .morie_dataset_tps_fetch(
    .MORIE_TPS_LAYER_REGISTRY[["major-crime"]],
    where = .morie_dataset_year_where(year),
    max_features = max_features,
    return_geometry = include_geometry
  )
}

#' TPS Shootings and Firearm Discharges feed.
#'
#' @inheritParams morie_datasets_tps_major_crime
#' @return A `data.frame`.
#' @export
morie_datasets_tps_shootings <- function(year = NULL, max_features = NULL) {
  .morie_dataset_tps_fetch(
    .MORIE_TPS_LAYER_REGISTRY[["shooting-firearms"]],
    where = .morie_dataset_year_where(year),
    max_features = max_features
  )
}

#' TPS Homicides feed.
#'
#' @inheritParams morie_datasets_tps_major_crime
#' @return A `data.frame`.
#' @export
morie_datasets_tps_homicide <- function(year = NULL, max_features = NULL) {
  .morie_dataset_tps_fetch(
    .MORIE_TPS_LAYER_REGISTRY[["homicide"]],
    where = .morie_dataset_year_where(year),
    max_features = max_features
  )
}

#' List the TPS open-data layers bundled with morie.
#'
#' @return A `data.frame` with columns `name` and `url`.
#' @export
morie_datasets_tps_layers <- function() {
  data.frame(
    name = names(.MORIE_TPS_LAYER_REGISTRY),
    url = unlist(unname(.MORIE_TPS_LAYER_REGISTRY)),
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# CPADS / OTIS
# ---------------------------------------------------------------------------

#' Load the morie CPADS analysis frame.
#'
#' Resolves in order: (1) a real Statistics Canada PUMF on disk at the
#' contract path, or (2) the bundled 1,200-row synthetic frame with the
#' canonical schema but random values.  Warns when the synthetic frame
#' is used.
#'
#' @return A `data.frame` with morie's canonical CPADS analysis columns.
#' @export
morie_datasets_cpads <- function() {
  contract <- morie_cpads_contract()
  if (file.exists(contract$expected_wrangled_path)) {
    frame <- readRDS(contract$expected_wrangled_path)
    return(morie_cpads_canonicalize_frame(frame))
  }
  .morie_dataset_read_synthetic("cpads_pumf_synthetic", "cpads")
}

#' Load the morie OTIS A01-RCDD restrictive-confinement frame.
#'
#' Real OTIS data is FOI-only and cannot be shipped publicly; this
#' function returns the bundled 800-row synthetic frame by default.
#'
#' @param offline Logical; if `TRUE` (default), return the synthetic
#'   frame.  Passing `FALSE` raises -- morie cannot fetch real OTIS.
#' @return A `data.frame`.
#' @export
morie_datasets_otis_a01 <- function(offline = TRUE) {
  if (!isTRUE(offline)) {
    stop(
      "morie_datasets_otis_a01(offline=FALSE): real OTIS data is FOI-only ",
      "and morie cannot fetch it for you.  Pass offline=TRUE for the ",
      "synthetic frame, or load your own copy with read.csv()."
    )
  }
  .morie_dataset_read_synthetic("otis_a01_synthetic", "otis_a01")
}

# ---------------------------------------------------------------------------
# SIU -- Special Investigations Unit director's reports
# ---------------------------------------------------------------------------

#' SIU director's-reports index (legacy PDF anchors).
#'
#' The SIU re-launched their site in 2025 with a JS-rendered case list;
#' this returns the legacy-pattern anchor frame which may be empty.
#'
#' @return A `data.frame` with columns `case_number`, `url`, `posted_date`.
#' @export
morie_datasets_siu_director_reports <- function() {
  if (!requireNamespace("rvest", quietly = TRUE) ||
      !requireNamespace("xml2", quietly = TRUE)) {
    warning("morie_datasets_siu_director_reports: 'rvest' + 'xml2' required for live scrape.",
            call. = FALSE)
    return(data.frame(case_number = character(),
                      url = character(),
                      posted_date = as.Date(character()),
                      stringsAsFactors = FALSE))
  }
  page <- xml2::read_html("https://www.siu.on.ca/en/directors_reports.php")
  anchors <- rvest::html_elements(page, "a[href$='.pdf']")
  urls <- rvest::html_attr(anchors, "href")
  if (length(urls) == 0L) {
    return(data.frame(case_number = character(),
                      url = character(),
                      stringsAsFactors = FALSE))
  }
  case_re <- "([0-9]{2}-[A-Z]{2,4}-[0-9]{3,4})"
  m <- regmatches(urls, regexpr(case_re, urls))
  data.frame(
    case_number = ifelse(nchar(m) > 0L, m, NA_character_),
    url = urls,
    stringsAsFactors = FALSE
  )
}

#' Download an SIU director's-report PDF and return its plain text.
#'
#' @param url Character; direct PDF URL.  Required unless `offline = TRUE`.
#' @param offline Logical; if `TRUE`, return the bundled synthetic
#'   `24-OFD-001` report text instead of hitting the SIU site.
#' @return Character scalar (the plain text).
#' @export
morie_datasets_siu_report_text <- function(url = NULL, offline = FALSE) {
  if (isTRUE(offline)) {
    path <- system.file("extdata", "siu_24-OFD-001_synthetic.txt",
                        package = "morie")
    if (!nzchar(path)) {
      stop("morie_datasets_siu_report_text(offline=TRUE): bundled synthetic missing.")
    }
    return(paste(readLines(path, warn = FALSE), collapse = "\
"))
  }
  if (is.null(url)) {
    stop("morie_datasets_siu_report_text: provide url=... or offline=TRUE")
  }
  if (!requireNamespace("pdftools", quietly = TRUE)) {
    stop("morie_datasets_siu_report_text: 'pdftools' is required to extract PDF text.")
  }
  if (!requireNamespace("httr2", quietly = TRUE)) {
    stop("morie_datasets_siu_report_text: 'httr2' is required to fetch PDFs.")
  }
  tmp <- tempfile(fileext = ".pdf")
  on.exit(unlink(tmp), add = TRUE)
  resp <- httr2::req_perform(httr2::request(url))
  writeBin(httr2::resp_body_raw(resp), tmp)
  paste(pdftools::pdf_text(tmp), collapse = "\
")
}

#' Extract structured fields from an SIU director's-report text or URL.
#'
#' @param text_or_url Character scalar; either the report text (re-used)
#'   or a PDF URL (fetched and parsed first).
#' @return A named list with fields `report_id`, `incident_date`,
#'   `conclusion`, `sections`.
#' @export
morie_datasets_siu_report_fields <- function(text_or_url) {
  text <- text_or_url
  if (grepl("^https?://", text_or_url)) {
    text <- morie_datasets_siu_report_text(text_or_url)
  }
  report_id_m <- regmatches(
    text, regexpr("[0-9]{2}-[A-Z]{2,4}-[0-9]{3,4}", text)
  )
  date_m <- regmatches(
    text,
    regexpr(
      "(January|February|March|April|May|June|July|August|September|October|November|December)\\s+[0-9]{1,2},\\s*[0-9]{4}",
      text
    )
  )
  conclusion_m <- regmatches(
    text,
    regexpr("(?si)Director'?s\\s+Decision.*?(?=(Issued|Dated|$))", text, perl = TRUE)
  )
  list(
    report_id = if (length(report_id_m)) report_id_m else NA_character_,
    incident_date = if (length(date_m)) date_m else NA_character_,
    conclusion = if (length(conclusion_m)) conclusion_m else NA_character_,
    sections = strsplit(text, "\
\\s*\
")[[1]]
  )
}

# ---------------------------------------------------------------------------
# Socrata -- Chicago + NYC OpenData
# ---------------------------------------------------------------------------

#' Documented Socrata column schema for Chicago Crimes (ijzp-q8t2).
#' @keywords internal
#' @noRd
.MORIE_CHICAGO_CRIME_COLUMNS <- c(
  "id", "case_number", "date", "block", "iucr", "primary_type",
  "description", "location_description", "arrest", "domestic",
  "beat", "district", "ward", "community_area", "fbi_code",
  "x_coordinate", "y_coordinate", "year", "updated_on",
  "latitude", "longitude"
)

#' Generic Socrata JSON fetch (handles `$where`, `$limit`, `$$app_token`).
#' @keywords internal
#' @noRd
.morie_dataset_socrata_fetch <- function(url, where = NULL,
                                         max_features = NULL,
                                         app_token = NULL) {
  query <- list()
  if (!is.null(where)) query$`$where` <- where
  if (!is.null(max_features)) query$`$limit` <- as.integer(max_features)
  if (!is.null(app_token)) query$`$$app_token` <- app_token
  records <- .morie_dataset_http_json(url, query = query)
  .morie_dataset_records_to_df(records)
}

#' City of Chicago "Crimes -- 2001 to Present" feed.
#'
#' @param year Integer or `NULL`; server-side year filter.
#' @param max_features Integer or `NULL`; cap on returned rows.
#' @param offline Logical; if `TRUE`, return the bundled synthetic frame.
#' @return A `data.frame` with the documented Socrata schema.
#' @export
morie_datasets_chicago_crime <- function(year = NULL,
                                         max_features = NULL,
                                         offline = FALSE) {
  if (isTRUE(offline)) {
    df <- .morie_dataset_read_synthetic(
      "chicago_crime_synthetic", "chicago_crime",
      columns = .MORIE_CHICAGO_CRIME_COLUMNS
    )
    if (!is.null(year) && "year" %in% colnames(df) && nrow(df) > 0L) {
      df <- df[df$year == year, , drop = FALSE]
      rownames(df) <- NULL
    }
    if (!is.null(max_features)) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  where <- if (is.null(year)) NULL else sprintf("year=%d", as.integer(year))
  .morie_dataset_socrata_fetch(
    "https://data.cityofchicago.org/resource/ijzp-q8t2.json",
    where = where,
    max_features = max_features
  )
}

#' NYC OpenData SQF resource map (verified 2026-05).
#' @keywords internal
#' @noRd
.MORIE_NYC_SQF_RESOURCES <- list(
  `2024` = "https://data.cityofnewyork.us/resource/7v9w-k82r.json",
  `2023` = "https://data.cityofnewyork.us/resource/rbed-zzin.json",
  `2022` = "https://data.cityofnewyork.us/resource/e4yi-bvqr.json"
)

#' NYPD Stop, Question and Frisk (SQF) microdata via NYC OpenData.
#'
#' @param year Integer or `NULL`; release year (one of 2022, 2023, 2024).
#'   `NULL` defaults to the most-recent registered year.
#' @param max_features Integer or `NULL`; cap on returned rows.
#' @param offline Logical; if `TRUE`, return the bundled synthetic frame.
#' @return A `data.frame`.  Schema is NOT normalised across years.
#' @export
morie_datasets_nyc_stop_and_frisk <- function(year = NULL,
                                              max_features = NULL,
                                              offline = FALSE) {
  if (isTRUE(offline)) {
    df <- .morie_dataset_read_synthetic("nyc_sqf_synthetic", "nyc_stop_and_frisk")
    if (!is.null(max_features) && nrow(df) > 0L) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  known_years <- as.integer(names(.MORIE_NYC_SQF_RESOURCES))
  chosen_year <- if (is.null(year)) max(known_years) else as.integer(year)
  if (!(chosen_year %in% known_years)) {
    stop(sprintf(
      paste0(
        "morie_datasets_nyc_stop_and_frisk: no built-in NYC OpenData ",
        "resource for year=%d.  Known years: %s."
      ),
      chosen_year, paste(sort(known_years), collapse = ", ")
    ))
  }
  .morie_dataset_socrata_fetch(
    .MORIE_NYC_SQF_RESOURCES[[as.character(chosen_year)]],
    max_features = max_features
  )
}

# ---------------------------------------------------------------------------
# BigQuery -- thin wrapper (optional dep: bigrquery)
# ---------------------------------------------------------------------------

#' Pull a BigQuery table (or filtered slice) as a `data.frame`.
#'
#' Requires the `bigrquery` package and Application Default Credentials.
#'
#' @param project Source project (e.g. `"bigquery-public-data"`).
#' @param dataset Source dataset (e.g. `"chicago_crime"`).
#' @param table Source table (e.g. `"crime"`).
#' @param where Raw SQL WHERE clause (without leading `WHERE`).
#' @param limit Optional integer `LIMIT`.
#' @param select Projection list; defaults to `"*"`.
#' @param billing_project GCP project to bill; `NULL` uses ADC-discovered.
#' @return A `data.frame`.
#' @export
morie_datasets_bigquery <- function(project, dataset, table,
                                    where = NULL, limit = NULL,
                                    select = "*", billing_project = NULL) {
  if (!requireNamespace("bigrquery", quietly = TRUE)) {
    stop("morie_datasets_bigquery: install 'bigrquery' to use this loader.")
  }
  sql <- sprintf("SELECT %s FROM `%s.%s.%s`", select, project, dataset, table)
  if (!is.null(where)) sql <- paste(sql, "WHERE", where)
  if (!is.null(limit)) sql <- paste(sql, "LIMIT", as.integer(limit))
  bill <- if (is.null(billing_project)) project else billing_project
  bigrquery::bq_table_download(bigrquery::bq_project_query(bill, sql))
}

# ---------------------------------------------------------------------------
# CKAN -- generic open-data portal helpers
# ---------------------------------------------------------------------------

#' Search a CKAN open-data portal by free-text query.
#'
#' Examples: `"https://open.canada.ca/data"`, `"https://data.ontario.ca"`,
#' `"https://data.gov.uk"`, `"https://data.europa.eu"`.
#'
#' @param portal Character; base URL of the CKAN portal.
#' @param query Character; free-text search.
#' @param rows Integer; max packages to return (default 50).
#' @return A `data.frame` of package metadata.
#' @export
morie_datasets_ckan_search <- function(portal, query, rows = 50L) {
  url <- paste0(sub("/$", "", portal), "/api/3/action/package_search")
  body <- .morie_dataset_http_json(url, query = list(q = query, rows = as.integer(rows)))
  results <- body$result$results
  if (is.null(results) || length(results) == 0L) {
    return(data.frame())
  }
  if (is.data.frame(results)) {
    return(results)
  }
  .morie_dataset_records_to_df(results)
}

#' Pull every CSV resource of a CKAN package as a list of data frames.
#'
#' @param portal Character; CKAN portal base URL.
#' @param package_id Character; CKAN package id or slug.
#' @return Named list mapping `resource_name -> data.frame`.
#' @export
morie_datasets_ckan_package <- function(portal, package_id) {
  url <- paste0(sub("/$", "", portal), "/api/3/action/package_show")
  body <- .morie_dataset_http_json(url, query = list(id = package_id))
  resources <- body$result$resources
  out <- list()
  for (i in seq_along(resources)) {
    res <- resources[[i]]
    fmt <- tolower(as.character(res$format %||% ""))
    if (identical(fmt, "csv") && nzchar(res$url %||% "")) {
      name <- as.character(res$name %||% paste0("resource_", i))
      out[[name]] <- tryCatch(
        utils::read.csv(res$url, stringsAsFactors = FALSE),
        error = function(e) NULL
      )
    }
  }
  out
}

# Null-coalescing helper (avoids dep on rlang).
#' @keywords internal
#' @noRd
`%||%` <- function(a, b) if (is.null(a)) b else a

# ---------------------------------------------------------------------------
# US forensics endpoints -- NIBRS, NamUs, NIST RDS
# ---------------------------------------------------------------------------

#' FBI NIBRS offence-event records via the Crime Data Explorer API.
#'
#' Requires an API key (`api_key=` or `FBI_CDE_API_KEY` env var).
#'
#' @param year Integer; reporting year (required unless `offline = TRUE`).
#' @param max_features Integer or `NULL`; cap on returned rows.
#' @param state Character; two-letter US state code, or `NULL` for national.
#' @param offense Character; NIBRS offence slug, or `NULL` for all.
#' @param api_key Character; FBI CDE API key (or `NULL` -> env var).
#' @param offline Logical; if `TRUE`, return a bundled synthetic frame.
#' @return A `data.frame`.
#' @export
morie_datasets_nibrs <- function(year = NULL, max_features = NULL,
                                 state = NULL, offense = NULL,
                                 api_key = NULL, offline = FALSE) {
  if (isTRUE(offline)) {
    df <- .morie_dataset_read_synthetic("nibrs_synthetic", "nibrs")
    if (!is.null(max_features) && nrow(df) > 0L) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  if (is.null(year)) {
    stop("morie_datasets_nibrs: year=... is required unless offline=TRUE")
  }
  api_key <- api_key %||% Sys.getenv("FBI_CDE_API_KEY", unset = NA)
  if (is.na(api_key) || !nzchar(api_key)) {
    stop("morie_datasets_nibrs: provide api_key=... or set FBI_CDE_API_KEY env var.")
  }
  base <- "https://api.usa.gov/crime/fbi/cde/nibrs"
  path <- paste(c(base, state, offense, year), collapse = "/")
  body <- .morie_dataset_http_json(path, query = list(API_KEY = api_key))
  df <- .morie_dataset_records_to_df(body$results %||% body)
  if (!is.null(max_features) && nrow(df) > 0L) {
    df <- utils::head(df, as.integer(max_features))
  }
  df
}

#' NamUs missing-persons case metadata.
#'
#' @param state Character; two-letter US state code or `NULL` (national).
#' @param max_features Integer or `NULL`; cap on returned rows.
#' @param offline Logical; if `TRUE`, return a bundled synthetic frame.
#' @return A `data.frame`.
#' @export
morie_datasets_namus_missing_persons <- function(state = NULL,
                                                 max_features = NULL,
                                                 offline = FALSE) {
  if (isTRUE(offline)) {
    df <- .morie_dataset_read_synthetic(
      "namus_missing_persons_synthetic", "namus_missing_persons"
    )
    if (!is.null(state) && "state" %in% colnames(df)) {
      df <- df[toupper(as.character(df$state)) == toupper(state), , drop = FALSE]
      rownames(df) <- NULL
    }
    if (!is.null(max_features) && nrow(df) > 0L) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  url <- "https://www.namus.gov/api/CaseSets/NamUs/MissingPersons/Search"
  body <- .morie_dataset_http_json(
    url, query = c(list(take = max_features %||% 200L),
                   if (!is.null(state)) list(state = state) else NULL)
  )
  .morie_dataset_records_to_df(body$results %||% body)
}

#' NIST Reference Datasets (RDS) catalog metadata.
#'
#' @param dataset_id Character or `NULL`; specific NIST RDS id.
#' @param query Character or `NULL`; free-text search.
#' @param max_features Integer or `NULL`; cap on returned rows.
#' @param offline Logical; if `TRUE`, return a bundled synthetic frame.
#' @return A `data.frame` with the NIST RDS catalog schema.
#' @export
morie_datasets_nist_rds <- function(dataset_id = NULL, query = NULL,
                                    max_features = NULL, offline = FALSE) {
  if (isTRUE(offline)) {
    df <- .morie_dataset_read_synthetic("nist_rds_synthetic", "nist_rds")
    if (!is.null(dataset_id) && "dataset_id" %in% colnames(df)) {
      df <- df[as.character(df$dataset_id) == dataset_id, , drop = FALSE]
      rownames(df) <- NULL
    }
    if (!is.null(max_features) && nrow(df) > 0L) {
      df <- utils::head(df, as.integer(max_features))
    }
    return(df)
  }
  url <- "https://data.nist.gov/rmm/records"
  q <- list()
  if (!is.null(dataset_id)) q$`@id` <- dataset_id
  if (!is.null(query)) q$searchphrase <- query
  if (!is.null(max_features)) q$size <- as.integer(max_features)
  body <- .morie_dataset_http_json(url, query = q)
  .morie_dataset_records_to_df(body$ResultData %||% body)
}
