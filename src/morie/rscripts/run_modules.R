# SPDX-License-Identifier: AGPL-3.0-or-later
#
# run_modules.R -- CLI shim for the Python->R bridge in morie/modules.py.
#
# It is shipped INSIDE the Python package (morie/rscripts/run_modules.R) so
# the bridge resolves it from `__file__` and never depends on a fragile
# install-layout / project-root guess. It parses the three flags the bridge
# passes and dispatches to the R `morie` package's exported entrypoints, which
# write each module's output CSV/TXT files into --output-dir for the Python
# side to read back.
#
# Flags (all `--key=value`):
#   --modules=a,b,c     comma-separated module names (default: all implemented)
#   --cpads-csv=PATH    CPADS CSV (default: the R package's bundled sample)
#   --output-dir=PATH   REQUIRED: where module outputs are written

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag) {
  hit <- grep(paste0("^--", flag, "="), args, value = TRUE)
  if (length(hit) == 0L) {
    return(NULL)
  }
  sub(paste0("^--", flag, "="), "", hit[[1L]])
}

modules_raw <- get_arg("modules")
cpads_csv <- get_arg("cpads-csv")
output_dir <- get_arg("output-dir")

if (is.null(output_dir) || !nzchar(output_dir)) {
  stop("run_modules.R: --output-dir is required.", call. = FALSE)
}

if (!requireNamespace("morie", quietly = TRUE)) {
  stop(
    "The R 'morie' package is required for R-backed modules but is not ",
    "installed. Install it from the repository's r-package/morie ",
    "(`R CMD INSTALL r-package/morie`) or from r-universe ",
    "(`install.packages('morie', repos = 'https://rootcoder007.r-universe.dev')`).",
    call. = FALSE
  )
}

modules <- if (is.null(modules_raw) || !nzchar(modules_raw)) {
  morie::morie_list_morie_modules()$name
} else {
  trimws(strsplit(modules_raw, ",", fixed = TRUE)[[1L]])
}

call_args <- list(modules = modules, output_dir = output_dir)
if (!is.null(cpads_csv) && nzchar(cpads_csv)) {
  call_args$cpads_csv <- cpads_csv
}

invisible(do.call(morie::morie_run_morie_modules, call_args))
