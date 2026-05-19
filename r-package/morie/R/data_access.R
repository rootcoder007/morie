# SPDX-License-Identifier: AGPL-3.0-or-later
#
# data_access.R -- generic open-data access layer for MORIE.
#
# Three public entry points let users reach data sources beyond the
# built-in catalog:
#   * morie_fetch()        -- universal URL fetcher, auto-detects format
#   * morie_ckan_search()  -- discover datasets on any CKAN portal
#   * morie_fetch_arcgis() -- query an ArcGIS FeatureServer/MapServer layer
#
# Everything is automatic by default (format is detected, pagination is
# handled) but every step can be overridden by the caller.

# --- internal helpers ------------------------------------------------------

# Append a named list of query parameters to a URL, URL-encoding values.
.morie_url_with_params <- function(url, params = NULL) {
  if (is.null(params) || length(params) == 0L) return(url)
  params <- params[!vapply(params, is.null, logical(1))]
  if (length(params) == 0L) return(url)
  kv <- vapply(seq_along(params), function(i) {
    paste0(utils::URLencode(names(params)[i], reserved = TRUE), "=",
           utils::URLencode(as.character(params[[i]]), reserved = TRUE))
  }, character(1))
  sep <- if (grepl("?", url, fixed = TRUE)) "&" else "?"
  paste0(url, sep, paste(kv, collapse = "&"))
}

# Known CKAN portals. A caller may also pass a full base URL directly.
.MORIE_CKAN_PORTALS <- c(
  "open.canada.ca"  = "https://open.canada.ca/data/en",
  "data.ontario.ca" = "https://data.ontario.ca",
  "open.toronto.ca" =
    "https://ckan0.cf.opendata.inter.prod-toronto.ca"
)

.morie_ckan_portal <- function(portal) {
  if (grepl("^https?://", portal)) return(sub("/+$", "", portal))
  if (portal %in% names(.MORIE_CKAN_PORTALS))
    return(.MORIE_CKAN_PORTALS[[portal]])
  stop("Unknown CKAN portal '", portal, "'. Known portals: ",
       paste(names(.MORIE_CKAN_PORTALS), collapse = ", "),
       " -- or pass a full https:// base URL.", call. = FALSE)
}

# Read text from a URL (used for JSON/XML/HTML API responses).
.morie_read_text <- function(url) {
  con <- url(url)
  on.exit(close(con), add = TRUE)
  paste(readLines(con, warn = FALSE), collapse = "\n")
}

# Download a URL to a temp file, returning the local path.
.morie_download <- function(url, ext = "") {
  if (!nzchar(ext)) ext <- tools::file_ext(sub("\\?.*$", "", url))
  tmp <- tempfile(fileext = if (nzchar(ext)) paste0(".", ext) else "")
  utils::download.file(url, tmp, mode = "wb", quiet = TRUE)
  tmp
}

# Detect the format of a URL from its HTTP Content-Type header, falling
# back to the URL file extension. Returns one of the morie_fetch formats.
.morie_detect_format <- function(url) {
  ct <- tryCatch({
    h <- suppressWarnings(curlGetHeaders(url))
    line <- grep("^content-type:", tolower(h), value = TRUE)
    if (length(line)) sub("^content-type:\\s*", "", line[length(line)]) else ""
  }, error = function(e) "")
  ct <- tolower(ct)
  if (grepl("zip", ct))                                   return("zip")
  if (grepl("json", ct))                                  return("json")
  if (grepl("csv", ct))                                   return("csv")
  if (grepl("tab-separated", ct))                         return("tsv")
  if (grepl("spreadsheet|ms-excel|officedocument", ct))   return("xlsx")
  if (grepl("xml", ct))                                   return("xml")
  if (grepl("html", ct))                                  return("html")
  ext <- tolower(tools::file_ext(sub("\\?.*$", "", url)))
  switch(ext,
         zip = "zip", json = "json", csv = "csv", tsv = "tsv",
         txt = "csv", xlsx = "xlsx", xls = "xlsx", xml = "xml",
         html = "html", htm = "html",
         "csv")  # last-resort default
}

# Parse a downloaded local file according to a known format.
.morie_parse_file <- function(path, format, simplify, ...) {
  if (format %in% c("xlsx")) {
    if (!requireNamespace("readxl", quietly = TRUE))
      stop("Package 'readxl' is required to read xlsx data.", call. = FALSE)
    return(as.data.frame(readxl::read_excel(path, ...)))
  }
  if (format == "tsv")
    return(utils::read.delim(path, stringsAsFactors = FALSE,
                             check.names = FALSE, ...))
  if (format == "csv")
    return(utils::read.csv(path, stringsAsFactors = FALSE,
                           check.names = FALSE, ...))
  if (format == "json") {
    if (!requireNamespace("jsonlite", quietly = TRUE))
      stop("Package 'jsonlite' is required to read JSON data.", call. = FALSE)
    return(jsonlite::fromJSON(path, simplifyVector = simplify))
  }
  if (format == "xml") {
    if (!requireNamespace("xml2", quietly = TRUE))
      stop("Package 'xml2' is required to read XML data.", call. = FALSE)
    doc <- xml2::read_xml(path)
    return(if (simplify) xml2::as_list(doc) else doc)
  }
  if (format == "html") {
    if (!requireNamespace("xml2", quietly = TRUE))
      stop("Package 'xml2' is required to read HTML data.", call. = FALSE)
    doc <- xml2::read_html(path)
    if (simplify && requireNamespace("rvest", quietly = TRUE)) {
      tbls <- rvest::html_table(doc)
      if (length(tbls) == 1L) return(as.data.frame(tbls[[1L]]))
      if (length(tbls) > 1L)  return(lapply(tbls, as.data.frame))
    }
    return(doc)
  }
  stop("Unsupported parse format: ", format, call. = FALSE)
}

# --- morie_fetch -----------------------------------------------------------

#' Fetch a dataset from any URL, with automatic format detection
#'
#' A universal data-access entry point. Given a URL, MORIE detects the
#' format from the HTTP \code{Content-Type} header (falling back to the
#' URL extension), downloads the resource, and parses it into an R
#' object. The behaviour is automatic by default but every step is
#' controllable: pass an explicit \code{format}, extra query
#' \code{params}, a \code{zip_member} to extract, or reader arguments
#' via \code{...}.
#'
#' Supported formats: \code{csv}, \code{tsv}, \code{json}, \code{xml},
#' \code{html}, \code{xlsx}, \code{zip} (extract one member), and
#' \code{arcgis} (delegates to \code{\link{morie_fetch_arcgis}}).
#'
#' @param url The resource URL.
#' @param format One of \code{"auto"} (default), \code{"csv"},
#'   \code{"tsv"}, \code{"json"}, \code{"xml"}, \code{"html"},
#'   \code{"xlsx"}, \code{"zip"}, \code{"arcgis"}.
#' @param params Optional named list appended to \code{url} as a
#'   URL-encoded query string.
#' @param zip_member For \code{zip} downloads, the archive member to
#'   extract (matched by basename, then by substring).
#' @param simplify For \code{json}/\code{xml}/\code{html}, whether to
#'   simplify into a data.frame where possible (default \code{TRUE}).
#' @param ... Passed to the underlying reader (e.g. \code{\link{read.csv}}
#'   arguments, or \code{\link{morie_fetch_arcgis}} arguments).
#' @return A data.frame for tabular formats; a list or document object
#'   for non-tabular \code{json}/\code{xml}/\code{html}.
#' @examples
#' \dontrun{
#'   # Auto-detected CSV:
#'   df <- morie_fetch("https://example.org/data.csv")
#'   # Force JSON parsing and pass an API parameter:
#'   js <- morie_fetch("https://api.example.org/records",
#'                     format = "json", params = list(limit = 100))
#' }
#' @seealso \code{\link{morie_ckan_search}}, \code{\link{morie_fetch_arcgis}}
#' @export
morie_fetch <- function(url,
                        format = c("auto", "csv", "tsv", "json", "xml",
                                   "html", "xlsx", "zip", "arcgis"),
                        params = NULL, zip_member = "", simplify = TRUE,
                        ...) {
  format <- match.arg(format)
  full_url <- .morie_url_with_params(url, params)

  if (format == "arcgis")
    return(morie_fetch_arcgis(url, params = params, ...))

  if (format == "auto") format <- .morie_detect_format(full_url)

  if (format == "zip") {
    if (!nzchar(zip_member))
      stop("A 'zip_member' is required to extract from a .zip resource.",
           call. = FALSE)
    zpath <- .morie_download(full_url, ext = "zip")
    on.exit(unlink(zpath), add = TRUE)
    exdir <- tempfile("morie-unzip-")
    dir.create(exdir)
    on.exit(unlink(exdir, recursive = TRUE), add = TRUE)
    members <- utils::unzip(zpath, list = TRUE)$Name
    hit <- members[basename(members) == zip_member]
    if (length(hit) == 0L)
      hit <- members[grepl(zip_member, members, fixed = TRUE)]
    if (length(hit) == 0L)
      stop("zip member '", zip_member, "' not found in ", url, call. = FALSE)
    utils::unzip(zpath, files = hit[1L], exdir = exdir, junkpaths = TRUE)
    inner <- file.path(exdir, basename(hit[1L]))
    inner_fmt <- .morie_detect_format(inner)
    return(.morie_parse_file(inner, inner_fmt, simplify, ...))
  }

  if (format %in% c("csv", "tsv", "xlsx")) {
    path <- .morie_download(full_url, ext = format)
    on.exit(unlink(path), add = TRUE)
    return(.morie_parse_file(path, format, simplify, ...))
  }

  # json / xml / html: read the response text into a temp file.
  txt <- .morie_read_text(full_url)
  path <- tempfile(fileext = paste0(".", format))
  on.exit(unlink(path), add = TRUE)
  writeLines(txt, path, useBytes = TRUE)
  .morie_parse_file(path, format, simplify, ...)
}

# --- morie_ckan_search -----------------------------------------------------

#' Search any CKAN open-data portal for datasets
#'
#' Wraps the CKAN \code{package_search} action so users can discover
#' datasets that are not in the built-in MORIE catalog and fetch them
#' through \code{\link{morie_fetch_ckan}} or \code{\link{morie_fetch}}.
#'
#' @param query Free-text search string.
#' @param portal A known portal name (\code{"open.canada.ca"},
#'   \code{"data.ontario.ca"}, \code{"open.toronto.ca"}) or a full
#'   CKAN base URL (e.g. \code{"https://catalogue.example.org"}).
#' @param rows Maximum number of datasets to return (default 25).
#' @param ... Extra named CKAN \code{package_search} parameters
#'   (e.g. \code{fq = "res_format:CSV"}, \code{sort = "metadata_modified desc"}).
#' @return A data.frame with one row per resource, columns:
#'   \code{dataset_title}, \code{dataset_id}, \code{resource_id},
#'   \code{resource_name}, \code{format}, \code{datastore_active},
#'   \code{url}. Feed \code{resource_id} into
#'   \code{morie_fetch_ckan(resource_id = ...)}.
#' @examples
#' \dontrun{
#'   hits <- morie_ckan_search("cannabis survey", portal = "open.canada.ca")
#'   head(hits[, c("dataset_title", "resource_id", "format")])
#' }
#' @seealso \code{\link{morie_fetch_ckan}}, \code{\link{morie_fetch}}
#' @export
morie_ckan_search <- function(query, portal = "open.canada.ca",
                              rows = 25L, ...) {
  if (!requireNamespace("jsonlite", quietly = TRUE))
    stop("Package 'jsonlite' is required for morie_ckan_search().",
         call. = FALSE)
  base <- .morie_ckan_portal(portal)
  api  <- paste0(base, "/api/3/action/package_search")
  url  <- .morie_url_with_params(
    api, c(list(q = query, rows = as.integer(rows)), list(...)))
  payload <- jsonlite::fromJSON(.morie_read_text(url), simplifyVector = FALSE)
  if (!isTRUE(payload$success))
    stop("CKAN package_search failed on portal ", base, call. = FALSE)
  results <- payload$result$results
  if (length(results) == 0L) {
    return(data.frame(
      dataset_title = character(0), dataset_id = character(0),
      resource_id = character(0), resource_name = character(0),
      format = character(0), datastore_active = logical(0),
      url = character(0), stringsAsFactors = FALSE))
  }
  rows_out <- list()
  for (ds in results) {
    res <- ds$resources
    if (length(res) == 0L) next
    for (r in res) {
      rows_out[[length(rows_out) + 1L]] <- data.frame(
        dataset_title = .nz(ds$title, ds$name),
        dataset_id    = .nz(ds$id),
        resource_id   = .nz(r$id),
        resource_name = .nz(r$name),
        format        = toupper(.nz(r$format)),
        datastore_active = isTRUE(r$datastore_active),
        url           = .nz(r$url),
        stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows_out)
}

# Small helper: first non-empty scalar, else "".
.nz <- function(...) {
  for (x in list(...)) {
    if (!is.null(x) && length(x) >= 1L && !is.na(x[[1L]]) &&
        nzchar(as.character(x[[1L]]))) return(as.character(x[[1L]]))
  }
  ""
}

# --- morie_fetch_arcgis ----------------------------------------------------

#' Query an ArcGIS FeatureServer / MapServer layer
#'
#' Pulls attribute records from an ArcGIS REST layer, paginating through
#' the server transfer limit automatically (ArcGIS caps a single query
#' at \code{maxRecordCount} features, typically 1000-2000).
#'
#' @param layer_url The layer URL, ending in \code{/FeatureServer/<n>}
#'   or \code{/MapServer/<n>}.
#' @param where SQL-style WHERE filter (default \code{"1=1"}, all rows).
#' @param out_fields Comma-separated field list (default \code{"*"}).
#' @param params Optional named list of extra query parameters.
#' @param page_size Records requested per page (default 2000).
#' @param max_records Cap on the total number of records (default
#'   \code{Inf} -- fetch the whole layer).
#' @return A data.frame of feature attributes (geometry is dropped).
#' @examples
#' \dontrun{
#'   layer <- paste0("https://services.arcgis.com/ORG/arcgis/rest/",
#'                   "services/Assault/FeatureServer/0")
#'   df <- morie_fetch_arcgis(layer)
#' }
#' @seealso \code{\link{morie_fetch}}
#' @export
morie_fetch_arcgis <- function(layer_url, where = "1=1", out_fields = "*",
                               params = NULL, page_size = 2000L,
                               max_records = Inf) {
  if (!requireNamespace("jsonlite", quietly = TRUE))
    stop("Package 'jsonlite' is required for morie_fetch_arcgis().",
         call. = FALSE)
  layer_url <- sub("/+$", "", layer_url)
  query_url <- paste0(layer_url, "/query")
  offset  <- 0L
  fetched <- 0L
  pages   <- list()
  repeat {
    this_page <- min(page_size, max_records - fetched)
    if (this_page <= 0L) break
    p <- c(list(where = where, outFields = out_fields,
                returnGeometry = "false", f = "json",
                resultOffset = offset,
                resultRecordCount = as.integer(this_page)), params)
    payload <- jsonlite::fromJSON(
      .morie_read_text(.morie_url_with_params(query_url, p)),
      simplifyVector = TRUE)
    if (!is.null(payload$error))
      stop("ArcGIS query error: ",
           .nz(payload$error$message, "unknown"), call. = FALSE)
    feats <- payload$features
    attrs <- if (is.null(feats) || NROW(feats) == 0L) NULL
             else if (is.data.frame(feats) && !is.null(feats$attributes))
               feats$attributes
             else feats
    if (is.null(attrs) || NROW(attrs) == 0L) break
    attrs <- as.data.frame(attrs, stringsAsFactors = FALSE)
    pages[[length(pages) + 1L]] <- attrs
    fetched <- fetched + NROW(attrs)
    if (!isTRUE(payload$exceededTransferLimit) || fetched >= max_records)
      break
    offset <- offset + NROW(attrs)
  }
  if (length(pages) == 0L) return(data.frame())
  if (length(pages) == 1L) return(pages[[1L]])
  do.call(rbind, pages)
}

# --- morie_siu_directors_reports -------------------------------------------------------

# Parse a block of SIU director's-report table rows into a data.frame.
# The SIU AJAX endpoint returns bare <tr class="dr-item"> fragments.
.morie_parse_siu_rows <- function(html) {
  empty <- data.frame(drid = integer(0), case_number = character(0),
                       incident_type = character(0),
                       date_signed = character(0),
                       report_url = character(0), stringsAsFactors = FALSE)
  blocks <- regmatches(
    html, gregexpr("(?s)<tr class=\"dr-item\".*?</tr>", html,
                   perl = TRUE))[[1L]]
  if (length(blocks) == 0L) return(empty)
  cap <- function(block, pat) {
    m <- regmatches(block, regexec(pat, block, perl = TRUE))[[1L]]
    if (length(m) >= 2L) trimws(m[2L]) else NA_character_
  }
  drid <- as.integer(vapply(blocks, cap, character(1), 'id="(\\d+)"'))
  case <- vapply(blocks, cap, character(1), "<nobr>([^<]+)</nobr>")
  date <- vapply(blocks, cap, character(1),
                 "<nobr>[^<]+</nobr>\\s*</td>\\s*<td[^>]*>([^<]+)</td>")
  href <- vapply(blocks, cap, character(1), 'href="([^"]+)"')
  itype <- toupper(vapply(strsplit(case, "-"), function(p)
    if (length(p) >= 2L) p[2L] else NA_character_, character(1)))
  data.frame(drid = drid, case_number = case, incident_type = itype,
             date_signed = date, report_url = href,
             stringsAsFactors = FALSE)
}

# Fetch one director's report page and return its plain-text body.
.morie_siu_report_text <- function(url) {
  if (!requireNamespace("xml2", quietly = TRUE))
    stop("Package 'xml2' is required to fetch SIU report text.",
         call. = FALSE)
  txt <- xml2::xml_text(xml2::read_html(url))
  trimws(gsub("[ \t]*\n[ \t\n]*", "\n", txt))
}

#' Harvest the Ontario SIU director's-reports index
#'
#' Mines the Special Investigations Unit (SIU) director's-reports
#' catalogue at \code{siu.on.ca}. The SIU exposes no public API; this
#' function drives the site's incremental AJAX endpoint to collect every
#' report's case number, signing date, incident-type code, and a link to
#' the full HTML report. Optionally it also downloads each report's text.
#'
#' This is a live web harvester of a third-party site: it is inherently
#' best-effort and may break if SIU changes its page structure. For a
#' stable (but historical, 2005-2018) structured copy, see the
#' \dQuote{Special Investigations Unit director's reports} dataset on
#' \code{data.ontario.ca}, fetchable with \code{\link{morie_ckan_search}}.
#'
#' @param max_reports Maximum number of reports to collect (default
#'   \code{Inf}, the full catalogue of ~2200+ reports).
#' @param lang Report language, \code{"en"} (default) or \code{"fr"}.
#' @param fetch_text If \code{TRUE}, also download each report's full
#'   HTML text into a \code{report_text} column. This issues one HTTP
#'   request per report and is slow; default \code{FALSE} (index only).
#' @param base_url Base URL of the SIU site (override for testing or if
#'   the site moves).
#' @return A data.frame with columns \code{drid} (report id),
#'   \code{case_number}, \code{incident_type} (the case-number middle
#'   code, e.g. \code{TCI}), \code{date_signed}, \code{report_url}, and
#'   \code{report_text} when \code{fetch_text = TRUE}.
#' @examples
#' \dontrun{
#'   idx <- morie_siu_directors_reports(max_reports = 60)
#'   table(idx$incident_type)
#'   one <- morie_siu_directors_reports(max_reports = 1, fetch_text = TRUE)
#' }
#' @seealso \code{\link{morie_fetch}}, \code{\link{morie_ckan_search}}.
#'   \code{morie_fetch_siu()} is a related helper that delegates to a
#'   Python scraper to build the full \code{SIU.csv} corpus.
#' @export
morie_siu_directors_reports <- function(max_reports = Inf, lang = c("en", "fr"),
                            fetch_text = FALSE,
                            base_url = "https://www.siu.on.ca") {
  lang <- match.arg(lang)
  base_url <- sub("/+$", "", base_url)
  ajax <- paste0(base_url, "/ssi/get_more_drs.php")
  collected <- list()
  last <- 0L
  repeat {
    html <- tryCatch(
      .morie_read_text(.morie_url_with_params(
        ajax, list(lang = lang, lastCount = last))),
      error = function(e) "")
    chunk <- .morie_parse_siu_rows(html)
    if (nrow(chunk) == 0L) break
    collected[[length(collected) + 1L]] <- chunk
    last <- last + nrow(chunk)
    if (last >= max_reports) break
    if (nrow(chunk) < 15L) break  # short page -> end of catalogue
  }
  out <- if (length(collected)) do.call(rbind, collected) else
    .morie_parse_siu_rows("")
  if (is.finite(max_reports) && nrow(out) > max_reports)
    out <- out[seq_len(max_reports), , drop = FALSE]
  if (nrow(out)) {
    out$report_url <- ifelse(
      grepl("^https?://", out$report_url), out$report_url,
      paste0(base_url, out$report_url))
  }
  if (fetch_text && nrow(out)) {
    message("Fetching full text for ", nrow(out), " SIU reports ...")
    out$report_text <- vapply(out$report_url, function(u)
      tryCatch(.morie_siu_report_text(u),
               error = function(e) NA_character_), character(1))
  }
  out
}
