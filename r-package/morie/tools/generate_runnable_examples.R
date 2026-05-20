# SPDX-License-Identifier: AGPL-3.0-or-later
# tools/generate_runnable_examples.R
#
# rOpenSci #770 :eyes: 8 — replace 250 boilerplate `\dontrun{<vignette pointer>}`
# example blocks with auto-generated runnable examples driven by formals().
#
# Strategy:
#   1. For every exported function whose @examples is the literal boilerplate,
#      examine formals(fn) and craft a candidate call using arg-name heuristics.
#   2. Execute the candidate in a subprocess with a 5-s timeout.
#   3. Functions whose candidate runs without error become "runnable";
#      the boilerplate \dontrun is replaced with the candidate.
#   4. Functions whose candidate errors are left untouched.
#
# Run with:
#   Rscript tools/generate_runnable_examples.R [--apply | --dry]

suppressMessages({
  library(morie)
  library(callr)
})

ARGS <- commandArgs(trailingOnly = TRUE)
APPLY <- "--apply" %in% ARGS

# ---- value-guess heuristics (by formal argument name) --------------------
guess <- function(name) {
  switch(name,
    # common univariate / matrix data
    x = quote(rnorm(50)), y = quote(rnorm(50)), z = quote(rnorm(50)),
    X = quote(matrix(rnorm(150), 50, 3)),
    Y = quote(matrix(rnorm(100), 50, 2)),
    Z = quote(matrix(rnorm(100), 50, 2)),
    # scalars
    n = 50L, N = 50L, k = 2L, K = 2L, p = 3L, P = 3L,
    m = 3L, M = 3L, J = 3L, d = 2L, D = 2L,
    alpha = 0.5, beta = 0.5, gamma = 0.5, lambda = 0.5,
    h = 1.0, bandwidth = 1.0, tol = 1e-6,
    fs = 100, order = 2L, low = 0.1, high = 0.4,
    seed = 1L, verbose = FALSE,
    # ts/event/data
    times = quote(sort(cumsum(rexp(50)))),
    event_times = quote(sort(cumsum(rexp(50)))),
    data = quote(data.frame(x = rnorm(50), y = rnorm(50))),
    df = quote(data.frame(x = rnorm(50), y = rnorm(50))),
    # spatial
    coords = quote(matrix(runif(100), 50, 2)),
    location = quote(matrix(runif(100), 50, 2)),
    # functional / categorical
    f = quote(function(z) sum(z^2)),
    formula = quote(stats::as.formula("y ~ x")),
    grid = quote(seq(-2, 2, length.out = 20)),
    party = quote(sample(c("A", "B"), 50, TRUE)),
    group = quote(sample(c("A", "B"), 50, TRUE)),
    treatment = quote(rbinom(50, 1, 0.5)),
    outcome = quote(rnorm(50)),
    # named-config defaults
    options = quote(NULL),
    setter_ideal = 0.5, reversion = 0.5,
    NULL # fallback signals "no heuristic"
  )
}

craft_args <- function(fname, fn) {
  fmls <- as.list(formals(fn))
  out <- list()
  for (a in names(fmls)) {
    # Detect "no default" — the formal slot holds an empty name. Comparing
    # via identical() against quote(expr=) is fragile; use deparse() == ""
    # which is robust to all-of: empty symbol, "..." dots, normal symbols.
    if (a == "...") next # variadic: leave alone
    deparsed <- deparse(fmls[[a]], control = NULL)
    has_default <- !(length(deparsed) == 1 && deparsed == "")
    if (has_default) next # function fills it
    g <- guess(a)
    if (is.null(g)) return(NULL) # bail — heuristic doesn't cover this arg
    out[[a]] <- g
  }
  out
}

# ---- find candidate functions --------------------------------------------
exports <- ls("package:morie")
fn_files <- list.files("R", pattern = "\\.R$", full.names = TRUE)

# Map function name → source file
fn_to_file <- list()
def_rx <- "^([A-Za-z_.][A-Za-z0-9_.]*)\\s*<-\\s*function"
for (f in fn_files) {
  s <- readLines(f, warn = FALSE)
  for (line in s) {
    m <- regmatches(line, regexec(def_rx, line))[[1]]
    if (length(m) >= 2 && m[2] %in% exports) {
      fn_to_file[[m[2]]] <- f
    }
  }
}

boilerplate_rx <- "See the package vignettes for usage examples"
boilerplate_fns <- character()
for (nm in names(fn_to_file)) {
  txt <- paste(readLines(fn_to_file[[nm]], warn = FALSE), collapse = "\n")
  # Look for any roxygen \dontrun block in the file containing the boilerplate
  if (grepl(boilerplate_rx, txt)) {
    boilerplate_fns <- c(boilerplate_fns, nm)
  }
}
cat(sprintf("found %d exported functions with boilerplate-dontrun in their source file\n",
            length(boilerplate_fns)))

# ---- probe each candidate ------------------------------------------------
results <- list()
for (i in seq_along(boilerplate_fns)) {
  nm <- boilerplate_fns[i]
  fn <- tryCatch(get(nm, envir = asNamespace("morie")), error = function(e) NULL)
  if (is.null(fn) || !is.function(fn)) {
    results[[nm]] <- list(status = "no-fn", call = NULL)
    next
  }
  args <- craft_args(nm, fn)
  if (is.null(args)) {
    results[[nm]] <- list(status = "no-heuristic", call = NULL)
    next
  }
  # build a deparse'd call for the example
  call_expr <- as.call(c(list(as.symbol(nm)), args))
  call_str <- paste(deparse(call_expr), collapse = " ")
  # probe in a fresh subprocess
  ok <- tryCatch({
    callr::r(
      function(nm, args) {
        suppressMessages(library(morie))
        set.seed(1)
        do.call(nm, args)
        TRUE
      },
      args = list(nm = nm, args = args),
      timeout = 5
    )
    "ok"
  }, error = function(e) {
    msg <- conditionMessage(e)
    if (grepl("timeout|reached elapsed time limit", msg, ignore.case = TRUE)) "timeout" else "error"
  })
  results[[nm]] <- list(status = ok, call = call_str)
  if (i %% 20 == 0) cat(sprintf("  probed %d/%d (%s last)\n", i, length(boilerplate_fns), ok))
}

# ---- report --------------------------------------------------------------
status_counts <- table(vapply(results, function(r) r$status, character(1)))
cat("\n=== probe results ===\n")
print(status_counts)

ok_fns <- names(results)[vapply(results, function(r) r$status == "ok", logical(1))]
cat(sprintf("\nRunnable examples generated for %d functions.\n", length(ok_fns)))

# write a JSON map of fn -> example call for downstream apply step
out_path <- "tools/runnable_examples.json"
if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(
    lapply(results, function(r) list(status = r$status, call = r$call %||% "")),
    out_path, auto_unbox = TRUE, pretty = TRUE
  )
  cat(sprintf("wrote %s\n", out_path))
}

# ---- apply --------------------------------------------------------------
if (APPLY) {
  cat("\n=== applying to source ===\n")
  applied <- 0
  for (nm in ok_fns) {
    fpath <- fn_to_file[[nm]]
    src <- readLines(fpath, warn = FALSE)
    # find the @examples block for `nm` and rewrite it
    # naive approach: rewrite EVERY boilerplate-dontrun in the file (file-level)
    new <- character()
    i <- 1L
    while (i <= length(src)) {
      ln <- src[i]
      if (grepl("^#' \\\\dontrun\\{\\s*$", ln) && i + 2 <= length(src) &&
          grepl(boilerplate_rx, src[i + 1])) {
        # boilerplate hit — replace 4 lines with the runnable call
        new <- c(new, sprintf("#' %s", results[[nm]]$call))
        i <- i + 4L
        applied <- applied + 1L
      } else {
        new <- c(new, ln); i <- i + 1L
      }
    }
    writeLines(new, fpath)
  }
  cat(sprintf("applied %d replacements\n", applied))
}
