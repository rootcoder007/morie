# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Verified-corpus SIU layer: the rmoriedata corpus is the source of truth,
# new reports are fetched/parsed through the rmoriebricklayer core, and the
# reading panel (siu_panel.R) handles what the corpus does not cover.
# Ported verbatim from the rmorie arm on 2026-09-01 (three-way parity with
# morie.siu in Python).

#' Parse one SIU director's report into the full 16-field schema
#'
#' Delegates to the compiled parser in the ecosystem's C/C++ foundation
#' package, \pkg{rmoriebricklayer} (>= 0.3.5): the sixteen panel-reviewed
#' schema fields (dates, investigator counts, witness/subject-official
#' counts, injuries, legislation, ...) plus \code{_language}, extracted
#' deterministically and offline. This supersedes the conservative
#' six-field regex parse used by \code{morie_siu_fetch_cases()}, which
#' itself now routes through the same compiled parser when available.
#'
#' @param html A length-1 character vector of raw report HTML, or the
#'   path to a saved report file.
#' @return A named character vector: the 16 schema fields plus
#'   \code{_language}.
#' @examplesIf requireNamespace("rmoriebricklayer", quietly = TRUE) && exists("bricklayer_parse_siu", envir = asNamespace("rmoriebricklayer"))
#' f <- morie_siu_parse_report(system.file("extdata",
#'   "siu_synthetic_report.html", package = "rmoriebricklayer"))
#' f[["number_of_subject_officers"]]
#' @export
morie_siu_parse_report <- function(html) {
  if (!requireNamespace("rmoriebricklayer", quietly = TRUE) ||
      !exists("bricklayer_parse_siu",
              envir = asNamespace("rmoriebricklayer"))) {
    stop("morie_siu_parse_report() needs rmoriebricklayer >= 0.3.5 ",
         "(the compiled SIU parser). Install/update it with:\n",
         "  install.packages(\"rmoriebricklayer\", ",
         "repos = \"https://rootcoder007.r-universe.dev\")")
  }
  rmoriebricklayer::bricklayer_parse_siu(html)
}

#' SIU director's-reports corpus: reviewed data first, fetch only what's new
#'
#' The right way to get SIU data in the morie ecosystem. Loads the
#' panel-reviewed 65-column corpus bundled in \pkg{rmoriedata} (2,182
#' English reports, subject-official coverage 100 percent, built by a
#' multi-model reading panel plus deterministic residual resolution) --
#' nothing is re-fetched or re-parsed for reports already reviewed. With
#' \code{update = TRUE} it then discovers reports published AFTER the
#' corpus snapshot (report ids above the bundled maximum), fetches only
#' those few (through \pkg{rmoriebricklayer}'s live-plus-Wayback engine
#' when installed), fills the mechanical schema fields with the compiled
#' parser, and appends them flagged \code{panel_reviewed = FALSE} with the
#' judgment columns left \code{NA} -- run the reading panel (the
#' \code{siu} command-line tool against your own 'Ollama' server) to fill
#' those, exactly as the reviewed corpus was built.
#'
#' @param update Also fetch + parse reports newer than the bundled corpus
#'   (default \code{FALSE}: fully offline).
#' @param max_new Ceiling on how many new reports to fetch per call
#'   (default 25; a normal refresh sees 0-15).
#' @param quiet Suppress progress messages.
#' @return A data.frame in the 65-column reviewed-corpus schema. New rows
#'   (if any) carry \code{panel_reviewed = FALSE}.
#' @examplesIf requireNamespace("rmoriedata", quietly = TRUE)
#' df <- morie_siu_reports()
#' nrow(df)
#' table(df$panel_reviewed)
#' @export
morie_siu_reports <- function(update = FALSE, max_new = 25L, quiet = FALSE) {
  if (!requireNamespace("rmoriedata", quietly = TRUE)) {
    stop("morie_siu_reports() needs the rmoriedata package (it carries ",
         "the reviewed corpus). install.packages(\"rmoriedata\", ",
         "repos = \"https://rootcoder007.r-universe.dev\")")
  }
  corpus <- rmoriedata::load_siu_reports()
  if (!isTRUE(update)) return(corpus)

  max_drid <- suppressWarnings(max(as.integer(corpus$drid), na.rm = TRUE))
  latest <- tryCatch(.siu_discover_max_drid(default = max_drid),
                     error = function(e) max_drid)
  if (!is.finite(latest) || latest <= max_drid) {
    if (!quiet) message("siu: corpus is current (max drid ", max_drid, ")")
    return(corpus)
  }
  new_ids <- seq.int(max_drid + 1L, latest)
  if (length(new_ids) > max_new) new_ids <- new_ids[seq_len(max_new)]
  if (!quiet) {
    message("siu: ", length(new_ids), " report id(s) newer than the ",
            "reviewed corpus; fetching only those")
  }

  fetch_one <- function(id) {
    # bricklayer's live+Wayback fetch engine when available, else plain R.
    if (requireNamespace("rmoriebricklayer", quietly = TRUE)) {
      dest <- tempfile(fileext = ".html")
      on.exit(unlink(dest), add = TRUE)
      ok <- tryCatch(
        rmoriebricklayer::bricklayer_fetch_siu(id, dest),
        error = function(e) NULL)
      if (!is.null(ok) && file.exists(dest) && file.size(dest) > 0) {
        return(paste(readLines(dest, warn = FALSE, encoding = "UTF-8"),
                     collapse = "\n"))
      }
      return(NULL)
    }
    url <- sprintf(
      "https://www.siu.on.ca/en/directors_report_details.php?drid=%d", id)
    tryCatch(.siu_fetch_http_get(url), error = function(e) NULL)
  }

  # parser field -> corpus column (the mechanical tier of the schema).
  fmap <- c(police_service = "police_service",
            date_of_incident_iso = "date_of_incident_iso",
            date_siu_notified_iso = "date_siu_notified_iso",
            date_of_director_decision_iso = "date_of_director_decision_iso",
            siu_investigators = "siu_investigators",
            siu_forensics_investigators = "siu_forensics_investigators",
            number_of_witness_officials = "number_of_witness_officials",
            number_of_civilian_witnesses = "number_of_civilian_witnesses",
            number_of_subject_officers = "number_of_subject_officials",
            age_affected = "age_affected",
            sex_gender_affected = "sex_gender_affected",
            charges_recommended = "charges_recommended",
            directors_name = "directors_name",
            location_of_call = "location_of_call",
            specific_injuries = "specific_injuries",
            relevant_legislation = "relevant_legislation",
            "_language" = "X_language")
  fmap <- fmap[fmap %in% names(corpus)]

  new_rows <- list()
  for (id in new_ids) {
    html <- fetch_one(id)
    if (is.null(html) || !nzchar(html)) next
    fields <- tryCatch(morie_siu_parse_report(html), error = function(e) NULL)
    if (is.null(fields)) next
    # A dead drid returns a page with no report body; require a date or a
    # service before treating it as a real report.
    if (!nzchar(fields[["police_service"]]) &&
        !nzchar(fields[["date_of_incident_iso"]])) next
    row <- corpus[NA_integer_, , drop = FALSE][1, ]
    row$drid <- as.character(id)
    row$source_url_report <- sprintf(
      "https://www.siu.on.ca/en/directors_report_details.php?drid=%d", id)
    for (k in names(fmap)) {
      v <- fields[[k]]
      if (!is.null(v) && nzchar(v)) row[[fmap[[k]]]] <- v
    }
    row$panel_reviewed <- FALSE
    new_rows[[length(new_rows) + 1L]] <- row
    if (!quiet) message("siu: added drid ", id, " (panel_reviewed = FALSE)")
  }
  if (!length(new_rows)) return(corpus)
  out <- rbind(corpus, do.call(rbind, new_rows))
  if (!quiet) {
    message("siu: ", length(new_rows), " new report(s) appended -- ",
            "mechanical fields parsed; run the reading panel (siu CLI + ",
            "your Ollama server) to fill the judgment columns")
  }
  out
}

#' Resolve a subject-official count: verified corpus first, rules second
#'
#' The correct order for the ecosystem: a report already in the
#' panel-reviewed corpus (\pkg{rmoriedata}) returns its VERIFIED count --
#' nothing re-derives an established answer. Only a report outside the
#' corpus falls through to the deterministic rule set compiled in
#' \pkg{rmoriebricklayer} (the foundation layer), whose rules were proven
#' zero-wrong against all 2,182 reviewed reports; where even the rules
#' cannot answer, the reading panel ([morie_siu_panel()]) decides.
#'
#' @param text Plain report text (needed only for unreviewed reports).
#' @param drid Report id; supply whenever known.
#' @return A list with `count` (integer, `NA` only when both corpus and
#'   rules are silent -- run the panel) and `reason`.
#' @examplesIf requireNamespace("rmoriedata", quietly = TRUE)
#' morie_siu_resolve_so(drid = 5038)
#' @export
morie_siu_resolve_so <- function(text = NULL, drid = NULL) {
  if (!is.null(drid) && requireNamespace("rmoriedata", quietly = TRUE)) {
    corpus <- tryCatch(rmoriedata::load_siu_reports(),
                       error = function(e) NULL)
    if (!is.null(corpus)) {
      hit <- corpus[corpus$drid == as.character(drid), , drop = FALSE]
      if (nrow(hit) == 1L) {
        n <- suppressWarnings(as.integer(hit$number_of_subject_officials))
        if (!is.na(n)) {
          return(list(count = n,
                      reason = "panel-reviewed corpus (verified)"))
        }
      }
    }
  }
  if (is.null(text)) {
    stop("report not in the reviewed corpus; supply `text` for the ",
         "rule-based resolution (rmoriebricklayer)")
  }
  if (!requireNamespace("rmoriebricklayer", quietly = TRUE)) {
    stop("rule-based resolution needs rmoriebricklayer")
  }
  rmoriebricklayer::bricklayer_siu_resolve_so(text)
}
