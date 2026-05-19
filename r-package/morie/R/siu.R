# SPDX-License-Identifier: AGPL-3.0-or-later
#
# siu.R -- orchestration for the all-C/C++ Ontario SIU parser.
#
# The HTTP transport (libcurl) and the 64-column HTML parsing both live
# in src/siu_parser.cpp. This file drives them: discover the published
# director's-report id range, concurrently fetch and parse every
# report page, fetch and parse the news releases they link, join the
# two, and write the 64-column SIU.csv.

# Highest director's-report id to fetch. Discovered from the SIU site's
# incremental index endpoint (its newest row is the highest indexed id),
# plus a margin: a report finalised after the index was last built can
# sit at a drid just above the newest indexed one, so iterating a little
# past it guarantees those are captured. Ids with no report parse to
# blank rows that are dropped, so the margin is free.
.siu_discover_max_drid <- function(default = 5300L, margin = 150L) {
  html <- tryCatch(
    .siu_http_get(paste0("https://www.siu.on.ca/ssi/get_more_drs.php",
                         "?lang=en&lastCount=0")),
    error = function(e) "")
  hits <- regmatches(html, gregexpr('id="[0-9]+"', html))[[1L]]
  ids <- suppressWarnings(as.integer(gsub("\\D", "", hits)))
  ids <- ids[is.finite(ids)]
  if (length(ids)) max(ids) + as.integer(margin) else as.integer(default)
}

#' Fetch the Ontario SIU corpus into a 64-column SIU.csv
#'
#' Fetches and parses the Ontario Special Investigations Unit
#' (police-oversight) corpus -- every director's report and the news
#' releases they link -- into a single CSV with the canonical
#' 64-column schema, one row per case.
#'
#' The parser is implemented entirely in C/C++ (\code{src/siu_parser.cpp}):
#' libcurl drives the HTTP transport and a concurrent \code{curl_multi}
#' pool fetches the ~9,000+ pages, while the 64-field extraction is C++
#' \code{std::regex} parsing. There is no Python dependency.
#'
#' This is the \emph{Ontario} Special Investigations Unit -- distinct
#' from the federal Structured Intervention Units and from OTIS. The
#' parsed corpus is not shipped with the package; each user runs the
#' parser themselves, which is fair use of public oversight reports.
#'
#' @param cache_dir Output directory (default \code{"~/.cache/morie/siu"}).
#' @param overwrite Logical; if \code{FALSE} and \code{SIU.csv} already
#'   exists in \code{cache_dir}, its path is returned without reparsing.
#' @param max_drid Highest director's-report id to fetch. \code{NULL}
#'   (default) discovers the current maximum from the SIU site.
#' @param concurrency Maximum simultaneous HTTP transfers.
#' @param progress Logical; print progress messages.
#' @return Path to the written \code{SIU.csv}.
#' @examples
#' \dontrun{
#'   # Network: parses the full Ontario SIU corpus (~10-15 min).
#'   csv <- morie_fetch_siu(cache_dir = tempdir())
#'   siu <- utils::read.csv(csv)
#'   nrow(siu)
#' }
#' @export
morie_fetch_siu <- function(cache_dir = "~/.cache/morie/siu",
                            overwrite = FALSE, max_drid = NULL,
                            concurrency = 24L, progress = TRUE) {
  cache_dir <- path.expand(cache_dir)
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(cache_dir, "SIU.csv")
  if (file.exists(out_path) && !overwrite) return(out_path)

  if (is.null(max_drid)) max_drid <- .siu_discover_max_drid()
  max_drid <- as.integer(max_drid)
  base_r <- "https://www.siu.on.ca/en/directors_report_details.php?drid="
  base_n <- "https://www.siu.on.ca/en/news_template.php?nrid="

  # -- 1. fetch and parse every director's-report page --
  drids <- seq_len(max_drid)
  if (progress)
    message("SIU: fetching ", length(drids), " report pages ...")
  rep_html <- .siu_http_get_many(paste0(base_r, drids),
                                 as.integer(concurrency))
  rep_rows <- lapply(drids, function(i)
    .siu_parse_report(rep_html[[i]], i, paste0(base_r, i)))
  df <- as.data.frame(do.call(rbind, lapply(rep_rows, as.character)),
                      stringsAsFactors = FALSE)
  names(df) <- names(rep_rows[[1L]])

  # -- 2. fetch and parse the linked news-release pages --
  nrids <- unique(df$nrid[nzchar(df$nrid)])
  if (length(nrids)) {
    if (progress)
      message("SIU: fetching ", length(nrids), " news-release pages ...")
    news_html <- .siu_http_get_many(paste0(base_n, nrids),
                                    as.integer(concurrency))
    news_rows <- lapply(seq_along(nrids), function(i)
      .siu_parse_news(news_html[[i]], as.integer(nrids[i]),
                      paste0(base_n, nrids[i])))
    news_df <- as.data.frame(do.call(rbind, lapply(news_rows, as.character)),
                             stringsAsFactors = FALSE)
    names(news_df) <- names(news_rows[[1L]])

    # -- 3. join the news fields onto the report rows by nrid --
    j <- match(df$nrid, news_df$nrid)
    hit <- !is.na(j)
    for (col in c("source_url_news", "news_release_title",
                  "news_release_date_iso", "news_release_date_raw",
                  "news_release_summary")) {
      df[[col]][hit] <- news_df[[col]][j[hit]]
    }
  }

  # One row per case. Drop pages with no case number (non-existent or
  # draft drids), then collapse the English and French copies of a case
  # to a single row -- keeping the English one and its drid / nrid, so
  # every emitted row is one case identified by one case_number.
  df <- df[nzchar(df$case_number), , drop = FALSE]
  if (nrow(df) > 0L) {
    df <- df[order(df$case_number, df[["_language"]] != "en"), ,
             drop = FALSE]
    df <- df[!duplicated(df$case_number), , drop = FALSE]
    rownames(df) <- NULL
  }

  # SIU pages (notably the French releases) are UTF-8.
  for (col in names(df)) Encoding(df[[col]]) <- "UTF-8"

  utils::write.csv(df, out_path, row.names = FALSE, na = "")
  if (progress)
    message("SIU: wrote ", nrow(df), " rows (",
            sum(nzchar(df$case_number)), " with a case number) to ",
            out_path)
  out_path
}
