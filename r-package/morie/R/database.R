# database.R -- DBI/RSQLite data layer for MORIE
#
# Built-in database: inst/extdata/morie.db ships with the package.
# User cache: morie.db under the per-user cache directory.
# Both R (DBI/RSQLite) and Python (sqlite3) share the same SQLite files.

# Per-user cache directory -- portable, always user-writable, and the
# same location the Python package uses, so the SQLite cache is shared.
# Honours XDG_CACHE_HOME; otherwise ~/.cache/morie.
morie_cache_dir <- function() {
  base <- Sys.getenv("XDG_CACHE_HOME", "")
  if (!nzchar(base)) base <- file.path(path.expand("~"), ".cache")
  file.path(base, "morie")
}

#' Get path to the built-in MORIE datasets database
#'
#' Returns the path to \code{morie.db} that ships with the package
#' (\code{inst/extdata/morie.db}). This database contains all CPADS,
#' CCS, CSADS, CSUS, HealthInfobase, and CIHI datasets pre-loaded as
#' SQLite tables.
#'
#' @return File path string.
#' @examples
#' morie_builtin_db()
#' @export
morie_builtin_db <- function() {
  db <- system.file("extdata", "morie.db", package = "morie")
  if (nzchar(db)) {
    return(db)
  }
  # Source-checkout / dev fallback: the per-user cache copy.
  file.path(morie_cache_dir(), "morie.db")
}

#' Connect to the MORIE SQLite cache database
#'
#' Opens (or creates) the shared cache at \code{morie.db} in the
#' per-user cache directory (\code{~/.cache/morie} or
#' \code{XDG_CACHE_HOME}). Both R (DBI/RSQLite) and Python (sqlite3)
#' read/write this same file.
#'
#' @param db_path Path to the SQLite file. Defaults to the
#'   \code{MORIE_CACHE_DB} env var, else \code{morie.db} in the
#'   per-user cache directory.
#' @return A DBI connection object.
#' @examples
#' \donttest{
#' if (requireNamespace("DBI", quietly = TRUE) &&
#'   requireNamespace("RSQLite", quietly = TRUE)) {
#'   tmp <- tempfile(fileext = ".db")
#'   con <- morie_db_connect(db_path = tmp)
#'   DBI::dbListTables(con)
#'   DBI::dbDisconnect(con)
#'   file.remove(tmp)
#' }
#' }
#' @export
morie_db_connect <- function(db_path = NULL) {
  if (!requireNamespace("DBI", quietly = TRUE) ||
    !requireNamespace("RSQLite", quietly = TRUE)) {
    stop("Packages 'DBI' and 'RSQLite' are required. Install with:\n",
      "  install.packages(c('DBI', 'RSQLite'))",
      call. = FALSE
    )
  }
  if (is.null(db_path)) {
    db_path <- Sys.getenv("MORIE_CACHE_DB", "")
    if (!nzchar(db_path)) {
      db_path <- file.path(morie_cache_dir(), "morie.db")
    }
  }
  dir.create(dirname(db_path), recursive = TRUE, showWarnings = FALSE)
  con <- DBI::dbConnect(RSQLite::SQLite(), dbname = db_path)
  DBI::dbExecute(con, "PRAGMA journal_mode=WAL")
  con
}

#' Store a data frame in the MORIE cache
#'
#' Writes (or replaces) a table in the shared SQLite cache.
#'
#' @param data A data.frame to cache.
#' @param table_name Name of the SQLite table.
#' @param db_path Optional override for the database path.
#' @return Number of rows written (invisible).
#' @examples
#' \dontrun{
#' # See the package vignettes for usage examples:
#' #   vignette(package = "morie")
#' }
#' @export
morie_cache_store <- function(data, table_name, db_path = NULL) {
  con <- morie_db_connect(db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)
  DBI::dbWriteTable(con, table_name, data, overwrite = TRUE)
  invisible(nrow(data))
}

#' Load a table from the MORIE cache
#'
#' @param table_name Name of the SQLite table.
#' @param db_path Optional override for the database path.
#' @return A data.frame, or \code{NULL} if the table does not exist.
#' @examples
#' \dontrun{
#' # See the package vignettes for usage examples:
#' #   vignette(package = "morie")
#' }
#' @export
morie_cache_load <- function(table_name, db_path = NULL) {
  con <- morie_db_connect(db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)
  if (!DBI::dbExistsTable(con, table_name)) {
    return(NULL)
  }
  DBI::dbReadTable(con, table_name)
}

#' List all tables in the MORIE cache
#'
#' @param db_path Optional override for the database path.
#' @return A data.frame with columns \code{table} and \code{rows}.
#' @examples
#' morie_cache_list()
#' @export
morie_cache_list <- function(db_path = NULL) {
  con <- morie_db_connect(db_path)
  on.exit(DBI::dbDisconnect(con), add = TRUE)
  tables <- DBI::dbListTables(con)
  if (length(tables) == 0L) {
    return(data.frame(table = character(), rows = integer()))
  }
  counts <- vapply(tables, function(t) {
    DBI::dbGetQuery(con, sprintf("SELECT COUNT(*) AS n FROM [%s]", t))$n
  }, integer(1))
  data.frame(table = tables, rows = counts, stringsAsFactors = FALSE)
}

#' Cache local RDS/CSV data into the SQLite database
#'
#' Reads a local file and writes it to the cache so that CI and Docker
#' environments (which may lack the original files) can still run tests.
#'
#' @param path Path to a CSV or RDS file.
#' @param table_name Name for the cached table.
#' @param db_path Optional override for the database path.
#' @return Number of rows cached (invisible).
#' @examples
#' \dontrun{
#' # See the package vignettes for usage examples:
#' #   vignette(package = "morie")
#' }
#' @export
morie_cache_file <- function(path, table_name, db_path = NULL) {
  ext <- tolower(tools::file_ext(path))
  data <- if (ext == "rds") {
    readRDS(path)
  } else if (ext == "csv") {
    utils::read.csv(path, stringsAsFactors = FALSE)
  } else {
    stop("Unsupported format: ", ext, call. = FALSE)
  }
  morie_cache_store(data, table_name, db_path)
}

#' Load CPADS data: local files -> cache -> CKAN API
#'
#' Resolution order:
#' \enumerate{
#'   \item Local RDS/CSV files in standard project locations
#'   \item SQLite cache (\code{data/cache/morie.db})
#'   \item CKAN API fetch (requires internet)
#' }
#'
#' @param db_path Optional override for the database path.
#' @param use_ckan Logical; if TRUE and data not found locally or in cache,
#'   attempt to fetch from the CKAN API.
#' @return A data.frame with canonical CPADS columns.
#' @examples
#' \dontrun{
#' # Needs the CPADS PUMF (local file, cache, or a live CKAN fetch),
#' # so it cannot run inside an offline R CMD check.
#' cpads <- morie_load_cpads(use_ckan = TRUE)
#' if (!is.null(cpads)) head(cpads)
#' }
#' @export
morie_load_cpads <- function(db_path = NULL, use_ckan = TRUE) {
  # 1. Local files.
  local_paths <- c(
    "data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv",
    "data/cache/cpads_pumf_wrangled.rds"
  )
  for (p in local_paths) {
    if (file.exists(p)) {
      message("Loading CPADS from local: ", p)
      ext <- tolower(tools::file_ext(p))
      data <- if (ext == "rds") readRDS(p) else utils::read.csv(p, stringsAsFactors = FALSE)
      # Cache it for next time.
      tryCatch(morie_cache_store(data, "cpads_canonical", db_path), error = function(e) NULL)
      return(data)
    }
  }

  # 2. SQLite cache.
  cached <- morie_cache_load("cpads_canonical", db_path)
  if (!is.null(cached)) {
    message("Loading CPADS from cache (", nrow(cached), " rows)")
    return(cached)
  }

  # 3. CKAN API.
  if (use_ckan) {
    message("Fetching CPADS from CKAN API...")
    data <- morie_fetch_ckan("cpads", db_path = db_path)
    return(data)
  }

  stop("CPADS data not found locally, in cache, or via CKAN.", call. = FALSE)
}

#' Fetch data from the CKAN API and cache it
#'
#' @param dataset_key One of \code{"cpads"}, \code{"csads"}, \code{"csus"}.
#' @param limit Maximum records to fetch. The CKAN datastore caps a
#'   single request at 32000 rows, so larger resources are paged through
#'   with `offset`; the default reads the entire resource.
#' @param db_path Optional override for the database path.
#' @param resource_id Optional CKAN datastore resource id. When supplied
#'   (e.g. from \code{morie_dataset_catalog()$ckan_resource_id}) it is used
#'   directly, so any catalogued dataset can be fetched without a built-in
#'   database; \code{dataset_key} then only labels the cache table.
#' @return A data.frame.
#' @examples
#' \dontrun{
#' # Requires network access. Fetches the first 5000 rows of the
#' # Canadian Postsecondary Alcohol and Drug Use Survey from the
#' # Government of Canada CKAN datastore:
#' cpads <- morie_fetch_ckan(dataset_key = "cpads", limit = 5000L)
#' nrow(cpads)
#' }
#' @export
morie_fetch_ckan <- function(dataset_key = "cpads", limit = Inf,
                             db_path = NULL, resource_id = NULL) {
  ckan_base <- "https://open.canada.ca/data/en/api/3/action/datastore_search"

  resource_ids <- list(
    cpads = "d2639429-c304-45a6-90b3-770562f4d46d",
    csads = NULL,
    csus  = NULL
  )

  metadata_urls <- list(
    cpads = "https://open.canada.ca/data/api/action/package_show?id=736fa9b2-62e4-4e31-aea4-51869605b363",
    csads = "https://open.canada.ca/data/api/action/package_show?id=1f15ca45-8bfd-4f9c-9ec6-2c0c440e69c2",
    csus  = "https://open.canada.ca/data/api/action/package_show?id=65e2d45e-efc6-4c29-9a9b-db59bc96aa0e"
  )

  # A catalog-supplied resource id is used directly; otherwise fall back
  # to the survey-keyed lookup, then to package-metadata resolution.
  rid <- if (!is.null(resource_id) && nzchar(resource_id)) {
    resource_id
  } else {
    resource_ids[[dataset_key]]
  }
  if (is.null(rid) || !nzchar(rid)) {
    # Resolve from package metadata.
    meta_url <- metadata_urls[[dataset_key]]
    if (is.null(meta_url)) {
      stop("Unknown dataset / no CKAN resource id: ", dataset_key, call. = FALSE)
    }
    meta_raw <- readLines(url(meta_url), warn = FALSE)
    meta <- jsonlite::fromJSON(paste(meta_raw, collapse = ""))
    resources <- meta$result$resources
    csv_idx <- which(toupper(resources$format) == "CSV")
    rid <- if (length(csv_idx) > 0) resources$id[csv_idx[1]] else resources$id[1]
  }

  # CKAN datastore_search caps a single request at 32000 rows, so page
  # through with `offset` until the whole resource (or `limit`) is read.
  cap <- as.integer(min(limit, .Machine$integer.max))
  page <- min(cap, 32000L)
  message("Fetching from CKAN datastore: resource_id=", rid)
  pages <- list()
  fetched <- 0L
  total <- NA_real_
  repeat {
    api_url <- sprintf(
      "%s?resource_id=%s&limit=%d&offset=%d",
      ckan_base, rid, page, fetched
    )
    raw <- readLines(url(api_url), warn = FALSE)
    payload <- jsonlite::fromJSON(paste(raw, collapse = ""))
    recs <- payload$result$records
    if (is.null(recs) || NROW(recs) == 0L) break
    pages[[length(pages) + 1L]] <- recs
    fetched <- fetched + NROW(recs)
    if (is.na(total)) {
      total <- if (!is.null(payload$result$total)) {
        as.numeric(payload$result$total)
      } else {
        fetched
      }
    }
    if (fetched >= total || fetched >= cap) break
  }
  records <- if (length(pages) == 0L) {
    NULL
  } else if (length(pages) == 1L) {
    pages[[1L]]
  } else {
    do.call(rbind, pages)
  }

  if (is.null(records) || NROW(records) == 0L) {
    stop("CKAN returned 0 records for ", dataset_key, call. = FALSE)
  }

  # Drop CKAN internal column.
  records[["_id"]] <- NULL

  # Cache.
  table_name <- paste0(dataset_key, "_raw")
  tryCatch(morie_cache_store(records, table_name, db_path), error = function(e) {
    message("Warning: could not cache: ", conditionMessage(e))
  })

  records
}


# ---------------------------------------------------------------------------
# Unified load interface
# ---------------------------------------------------------------------------

.fuzzy_match_key <- function(key) {
  catalog <- morie_dataset_catalog()
  key_lower <- tolower(gsub("-", "_", key))
  # Exact match on new short keys.
  idx <- which(catalog$key == key_lower)
  if (length(idx) == 1L) {
    return(catalog$key[idx])
  }
  # Backward-compat: resolve old long keys to new short keys.
  if (key_lower %in% names(.OLD_TO_SHORT)) {
    short <- .OLD_TO_SHORT[[key_lower]]
    idx <- which(catalog$key == short)
    if (length(idx) == 1L) {
      return(catalog$key[idx])
    }
  }
  # Substring match on keys.
  idx <- which(grepl(key_lower, catalog$key, fixed = TRUE))
  if (length(idx) >= 1L) {
    return(catalog$key[idx[1L]])
  }
  # Substring match on dataset names.
  idx <- which(grepl(key_lower, tolower(catalog$name), fixed = TRUE))
  if (length(idx) >= 1L) {
    return(catalog$key[idx[1L]])
  }
  NULL
}

#' Load a dataset by catalog key
#'
#' Resolution tiers, tried in order: built-in DB -> user cache -> local
#' file -> CKAN datastore -> direct download URL -> ArcGIS layer ->
#' error. Supports fuzzy matching: \code{morie_load_dataset("cpads_2021")}
#' resolves to \code{ocp21}.
#'
#' @param key Dataset catalog key (or fuzzy match).
#' @param db_path Optional override for the database path.
#' @param refresh If \code{TRUE}, bypass the built-in database and the
#'   user cache (and, for remotely-backed datasets, the local file) and
#'   re-fetch from the remote source, overwriting the cached copy. Use
#'   this to pick up time-to-time updates to a dataset.
#' @return A data.frame.
#' @examples
#' \dontrun{
#' df <- morie_load_dataset("ocp21") # CPADS 2021-2022
#' df <- morie_load_dataset("ocp21", refresh = TRUE) # force re-fetch
#' }
#' @seealso \code{\link{morie_fetch}}, \code{\link{morie_ckan_search}}
#' @export
morie_load_dataset <- function(key, db_path = NULL, refresh = FALSE) {
  matched <- .fuzzy_match_key(key)
  if (is.null(matched)) {
    stop("Unknown dataset key: '", key, "'. See morie_dataset_catalog().", call. = FALSE)
  }
  catalog <- morie_dataset_catalog()
  entry <- catalog[catalog$key == matched, ]
  has <- function(col) col %in% names(entry) && nzchar(entry[[col]])
  has_remote <- has("ckan_resource_id") || has("download_url") ||
    has("arcgis_url")

  if (!refresh) {
    # 1. Built-in database (ships with package).
    builtin_path <- tryCatch(morie_builtin_db(), error = function(e) NULL)
    if (!is.null(builtin_path) && requireNamespace("DBI", quietly = TRUE) &&
      requireNamespace("RSQLite", quietly = TRUE)) {
      bcon <- DBI::dbConnect(RSQLite::SQLite(), dbname = builtin_path)
      on.exit(DBI::dbDisconnect(bcon), add = TRUE)
      if (DBI::dbExistsTable(bcon, entry$table_name)) {
        data <- DBI::dbReadTable(bcon, entry$table_name)
        message(
          "Loaded ", matched, " from built-in DB (", nrow(data),
          " rows)"
        )
        return(data)
      }
    }

    # 2. User cache.
    cached <- morie_cache_load(entry$table_name, db_path)
    if (!is.null(cached)) {
      message("Loaded ", matched, " from cache (", nrow(cached), " rows)")
      return(cached)
    }
  }

  # 3. Local file. Skipped on refresh when a remote source exists, so a
  #    refresh re-pulls from the authoritative remote rather than a stale
  #    on-disk copy; for local-only datasets the file remains the source.
  if (file.exists(entry$local_path) && !(refresh && has_remote)) {
    message("Ingesting ", matched, " from local: ", entry$local_path)
    ext <- tolower(tools::file_ext(entry$local_path))
    data <- if (ext == "csv") {
      utils::read.csv(entry$local_path, stringsAsFactors = FALSE)
    } else if (ext %in% c("xlsx", "xls")) {
      if (!requireNamespace("readxl", quietly = TRUE)) stop("readxl required")
      as.data.frame(readxl::read_excel(entry$local_path))
    } else if (ext == "rds") {
      readRDS(entry$local_path)
    } else {
      stop("Unsupported format: ", ext, call. = FALSE)
    }
    morie_cache_store(data, entry$table_name, db_path)
    return(data)
  }

  # 4. CKAN datastore -- resolved directly from the catalog resource id,
  #    matching the Python load_dataset() design (no built-in DB needed).
  if (has("ckan_resource_id")) {
    message("Fetching ", matched, " from the CKAN datastore ...")
    data <- morie_fetch_ckan(
      dataset_key = matched,
      resource_id = entry$ckan_resource_id,
      db_path = db_path
    )
    morie_cache_store(data, entry$table_name, db_path)
    return(data)
  }

  # 5. Direct download URL -- open-data files not exposed through the CKAN
  #    datastore (direct CSV/XLSX, or a file bundled inside a .zip archive).
  if (has("download_url")) {
    message("Downloading ", matched, " from ", entry$download_url, " ...")
    zm <- if ("zip_member" %in% names(entry)) entry$zip_member else ""
    is_zip <- grepl("\\.zip$", entry$download_url, ignore.case = TRUE)
    data <- morie_fetch(entry$download_url,
      format = if (is_zip) "zip" else "auto",
      zip_member = zm
    )
    morie_cache_store(data, entry$table_name, db_path)
    return(data)
  }

  # 6. ArcGIS FeatureServer / MapServer layer (e.g. TPS crime open data).
  if (has("arcgis_url")) {
    message("Querying ", matched, " from the ArcGIS layer ...")
    data <- morie_fetch_arcgis(entry$arcgis_url)
    morie_cache_store(data, entry$table_name, db_path)
    return(data)
  }

  stop("Dataset '", matched, "' not found locally, in cache, via CKAN, ",
    "via a direct download URL, or via an ArcGIS layer.\n",
    "Run: Rscript data-raw/ingest_datasets.R --only ", matched,
    call. = FALSE
  )
}

#' List all datasets with cache status
#'
#' @param db_path Optional override for the database path.
#' @return A data.frame with columns: key, name, source, survey, year, type,
#'   cached (logical), rows (integer or NA).
#' @examples
#' morie_list_datasets()
#' @export
morie_list_datasets <- function(db_path = NULL) {
  catalog <- morie_dataset_catalog()
  cached_tables <- tryCatch(
    {
      cl <- morie_cache_list(db_path)
      stats::setNames(cl$rows, cl$table)
    },
    error = function(e) stats::setNames(integer(0), character(0))
  )

  catalog$cached <- catalog$table_name %in% names(cached_tables)
  catalog$rows <- as.integer(cached_tables[catalog$table_name])
  catalog[, c("key", "name", "source", "survey", "year", "type", "cached", "rows")]
}

#' Get metadata for a single dataset
#'
#' @param key Dataset catalog key (or fuzzy match).
#' @return A named list with dataset metadata.
#' @examples
#' # Use a real catalog key (run `morie_dataset_catalog()$key` to list them):
#' info <- morie_dataset_info("ocp21")
#' info$source
#' info$year
#' # Fuzzy match works for partial / forgiving keys:
#' morie_dataset_info("cpads")$key
#' @export
morie_dataset_info <- function(key) {
  matched <- .fuzzy_match_key(key)
  if (is.null(matched)) stop("Unknown dataset key: '", key, "'", call. = FALSE)
  catalog <- morie_dataset_catalog()
  entry <- catalog[catalog$key == matched, ]
  as.list(entry)
}

#' Get path to an MORIE userguide
#'
#' Lists or retrieves bundled userguide PDF files. These are the official
#' PUMF codebooks and user guides from Health Canada / Statistics Canada.
#'
#' @param name Filename (e.g., \code{"20212022-cpads-pumf-user-guide.pdf"}).
#'   If \code{NULL}, lists all available userguides.
#' @return File path string, or character vector of filenames.
#' @examples
#' morie_userguide()
#' @export
morie_userguide <- function(name = NULL) {
  if (is.null(name)) {
    dir(system.file("extdata", "userguides", package = "morie"))
  } else {
    system.file("extdata", "userguides", name, package = "morie", mustWork = TRUE)
  }
}


#' Download bootstrap weight files from CKAN API
#'
#' Downloads large bootstrap weight CSVs that are too big to ship with the
#' package. Data is cached in the user cache database for future use.
#'
#' @param survey One of \code{"csads_2021"}, \code{"csads_2023"},
#'   \code{"csus_2019"}, \code{"csus_2023"}, or \code{"all"} (default).
#' @param limit Max records per CKAN request (default 32000).
#' @param db_path Optional override for cache database path.
#' @return Invisibly, the number of CSV files successfully downloaded.
#' @examples
#' \dontrun{
#' # See the package vignettes for usage examples:
#' #   vignette(package = "morie")
#' }
#' @export
morie_download_bootstrap <- function(survey = "all", limit = 32000L, db_path = NULL) {
  # Current short catalog keys (see morie_dataset_catalog()); the older
  # oc_<survey>_<year>_bootstrap long keys are no longer in the catalog.
  bootstrap_keys <- list(
    csads_2021 = "ocs22bt",
    csads_2023 = "ocs24bt",
    csus_2019  = "cu20bt",
    csus_2023  = "cu23bt"
  )
  if (survey == "all") {
    targets <- unlist(bootstrap_keys, use.names = FALSE)
  } else {
    targets <- bootstrap_keys[[survey]]
    if (is.null(targets)) stop("Unknown survey: ", survey, call. = FALSE)
  }

  catalog <- morie_dataset_catalog()
  for (key in targets) {
    entry <- catalog[catalog$key == key, ]
    if (nrow(entry) == 0L) {
      message("Unknown key: ", key)
      next
    }

    # Try local file first.
    if (file.exists(entry$local_path)) {
      message("Ingesting ", key, " from local file: ", entry$local_path)
      morie_cache_file(entry$local_path, entry$table_name, db_path)
      message("  OK: cached ", entry$table_name)
      next
    }

    # Try CKAN.
    if (nzchar(entry$ckan_resource_id)) {
      message("Downloading ", key, " from CKAN (limit=", limit, ")...")
      tryCatch(
        {
          data <- morie_fetch_ckan(entry$survey, limit = limit, db_path = db_path)
          morie_cache_store(data, entry$table_name, db_path)
          message("  OK: ", nrow(data), " rows cached as ", entry$table_name)
        },
        error = function(e) {
          message("  ERROR: ", conditionMessage(e))
        }
      )
    } else {
      message("  ", key, ": no CKAN resource ID. Download CSV manually to ", entry$local_path)
    }
  }
  invisible(NULL)
}
