# SPDX-License-Identifier: AGPL-3.0-or-later
#
# llm.R -- Provider chain for MORIE's LLM integration layer.
#
# R port of src/morie/llm.py. Implements the same provider priority:
#   1. Ollama (local, no key)
#   2. Gemini (Google AI Studio, OpenAI-compatible endpoint)
#   3. Generic OpenAI-compatible endpoint (LLM_API_BASE_URL/LLM_API_KEY)
#   4. Official OpenAI API
#   5. Local fallback help text (no network)
#
# OllamaFreeAPI (Python SDK) is intentionally NOT ported -- R has no
# equivalent client. HTTP providers use `httr2` + `jsonlite`. All
# public functions are prefixed `morie_llm_*` and exported. Streaming
# is not supported in this R port (always returns the full string).

DEFAULT_OLLAMA_BASE_URL <- "http://localhost:11434"
DEFAULT_GEMINI_MODEL    <- "gemini-2.5-flash"
DEFAULT_API_MODEL       <- "google/gemma-3-27b-it"
DEFAULT_OPENAI_MODEL    <- "gpt-4o-mini"
OPENAI_BASE_URL <- "https://api.openai.com"
GEMINI_BASE_URL <- "https://generativelanguage.googleapis.com/v1beta/openai"

`%||%` <- function(a, b) if (is.null(a) || identical(a, "")) b else a

.morie_llm_env <- function(name, default = "") {
  v <- trimws(Sys.getenv(name, unset = ""))
  if (nzchar(v)) v else default
}
.morie_llm_ollama_base <- function() {
  sub("/+$", "", .morie_llm_env("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
}
.morie_llm_gemini_key  <- function() { v <- .morie_llm_env("GEMINI_API_KEY"); if (nzchar(v)) v else NULL }
.morie_llm_openai_key  <- function() { v <- .morie_llm_env("OPENAI_API_KEY"); if (nzchar(v)) v else NULL }
.morie_llm_api_base    <- function() { v <- .morie_llm_env("LLM_API_BASE_URL"); if (nzchar(v)) sub("/+$", "", v) else NULL }
.morie_llm_api_key     <- function() { v <- .morie_llm_env("LLM_API_KEY"); if (nzchar(v)) v else NULL }
.morie_llm_gemini_model <- function() .morie_llm_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

#' Probe a local Ollama instance
#' @param timeout Probe timeout in seconds.
#' @return Logical scalar -- TRUE when reachable.
#' @export
morie_llm_probe_ollama <- function(timeout = 2) {
  if (!requireNamespace("httr2", quietly = TRUE)) return(FALSE)
  cache <- getOption("morie.llm.ollama_cached", default = NULL)
  if (!is.null(cache)) return(cache)
  out <- tryCatch({
    req <- httr2::request(paste0(.morie_llm_ollama_base(), "/api/tags"))
    req <- httr2::req_timeout(req, timeout)
    resp <- httr2::req_perform(req)
    httr2::resp_status(resp) < 400
  }, error = function(e) FALSE)
  options(morie.llm.ollama_cached = out)
  out
}

#' Detect the active LLM provider
#' @return Character scalar provider key: ollama / gemini / api / openai / local.
#' @export
morie_llm_detect_provider <- function() {
  if (morie_llm_probe_ollama())                                 return("ollama")
  if (!is.null(.morie_llm_gemini_key()))                        return("gemini")
  if (!is.null(.morie_llm_api_base()) && !is.null(.morie_llm_api_key()))
                                                                return("api")
  if (!is.null(.morie_llm_openai_key()))                        return("openai")
  "local"
}

.morie_llm_system_prompt <- function(context_block = "") {
  paste0(
    "You are the MORIE agent for methods for observational inference and ",
    "robust analysis of interventions in sociolegal studies.\
\
",
    "MORIE is a Python+R toolkit for Canadian public-health and carceral ",
    "data analysis, causal inference, and reproducible research.\
\
",
    context_block
  )
}

.morie_llm_messages <- function(prompt, context = NULL, system_prompt = NULL) {
  if (is.null(system_prompt)) {
    ctx_block <- if (is.null(context)) "" else paste(
      vapply(names(context),
             function(k) paste0(k, ": ", as.character(context[[k]])),
             character(1)),
      collapse = "\
")
    system_prompt <- .morie_llm_system_prompt(ctx_block)
  }
  list(
    list(role = "system", content = system_prompt),
    list(role = "user",   content = prompt)
  )
}

#' POST a chat-completion request to an OpenAI-compatible endpoint
#' @param base_url Provider base URL.
#' @param model Model identifier.
#' @param messages List of role/content lists.
#' @param api_key Optional bearer token (NULL for local Ollama).
#' @param timeout Seconds. Default 120.
#' @return Parsed JSON list (the response body).
#' @export
morie_llm_request_completion <- function(base_url, model, messages,
                                         api_key = NULL, timeout = 120) {
  if (!requireNamespace("httr2", quietly = TRUE) ||
      !requireNamespace("jsonlite", quietly = TRUE)) {
    stop("morie_llm_request_completion requires httr2 and jsonlite.")
  }
  url <- paste0(base_url, "/v1/chat/completions")
  payload <- list(model = model, messages = messages, stream = FALSE)
  if (grepl("localhost|127\\.0\\.0\\.1", base_url)) {
    payload$max_tokens <- 4096L
    timeout <- max(timeout, 300)
  }
  req <- httr2::request(url)
  req <- httr2::req_headers(req, `Content-Type` = "application/json")
  if (!is.null(api_key)) {
    req <- httr2::req_headers(req, Authorization = paste("Bearer", api_key))
  }
  req <- httr2::req_body_raw(req,
    jsonlite::toJSON(payload, auto_unbox = TRUE),
    type = "application/json")
  req <- httr2::req_timeout(req, timeout)
  resp <- httr2::req_perform(req)
  jsonlite::fromJSON(httr2::resp_body_string(resp), simplifyVector = FALSE)
}

.morie_llm_extract_text <- function(data) {
  choices <- data$choices
  if (length(choices) == 0L) return("")
  msg <- choices[[1]]$message
  if (is.null(msg)) "" else (msg$content %||% "")
}

.morie_llm_local_fallback <- function(prompt) {
  paste0(
    "MORIE is running in local-only mode (no LLM provider detected).\
\
",
    "Available capabilities without an LLM:\
",
    "  - morie list-modules        List analysis modules\
",
    "  - morie run-module <name>   Run a specific module\
",
    "  - morie pipeline --all -y   Run the full analysis pipeline\
\
",
    "Enable an LLM by setting one of GEMINI_API_KEY, ",
    "LLM_API_BASE_URL + LLM_API_KEY, or OPENAI_API_KEY, ",
    "or by running a local Ollama instance."
  )
}

#' Send a prompt to the best available LLM provider
#'
#' R port of `morie.llm.ask`. Tries each provider in priority order; on
#' HTTP/timeout failure falls through to the next, and finally to a
#' static local help string.
#'
#' @param prompt User question or instruction.
#' @param context Optional named list injected as text into the system prompt.
#' @param model Optional model override.
#' @param provider Optional provider override (ollama/gemini/api/openai/local).
#'   NULL = auto-detect.
#' @param system_prompt Optional full system-prompt override.
#' @param timeout HTTP timeout in seconds. Default 120.
#' @return Character scalar response text, or local-fallback text when all
#'   providers fail.
#' @export
morie_llm_ask <- function(prompt, context = NULL, model = NULL,
                          provider = NULL, system_prompt = NULL,
                          timeout = 120) {
  if (is.null(provider)) provider <- morie_llm_detect_provider()
  if (identical(provider, "local")) return(.morie_llm_local_fallback(prompt))

  messages <- .morie_llm_messages(prompt, context = context,
                                  system_prompt = system_prompt)
  attempts <- list()
  add <- function(base, mdl, key) attempts[[length(attempts) + 1L]] <<-
    list(base = base, model = mdl, key = key)
  if (provider == "ollama") {
    add(.morie_llm_ollama_base(), model %||% "llama3", NULL)
  }
  if (provider %in% c("ollama", "gemini") && !is.null(.morie_llm_gemini_key())) {
    add(GEMINI_BASE_URL, model %||% .morie_llm_gemini_model(),
        .morie_llm_gemini_key())
  }
  if (!is.null(.morie_llm_api_base()) && !is.null(.morie_llm_api_key())) {
    add(.morie_llm_api_base(), model %||% DEFAULT_API_MODEL,
        .morie_llm_api_key())
  }
  if (!is.null(.morie_llm_openai_key())) {
    add(OPENAI_BASE_URL, model %||% DEFAULT_OPENAI_MODEL,
        .morie_llm_openai_key())
  }
  if (length(attempts) == 0L) return(.morie_llm_local_fallback(prompt))

  for (a in attempts) {
    out <- tryCatch(
      .morie_llm_extract_text(
        morie_llm_request_completion(a$base, a$model, messages,
                                     api_key = a$key, timeout = timeout)),
      error = function(e) NULL)
    if (!is.null(out) && nzchar(out)) return(out)
  }
  .morie_llm_local_fallback(prompt)
}

#' Return TRUE when at least one live LLM provider is available
#' @return Logical scalar.
#' @export
morie_llm_agent_available <- function() {
  morie_llm_detect_provider() != "local"
}
