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
  # requireNamespace() returns FALSE for two very different situations:
  # the package is absent, or it is installed but cannot be loaded (most
  # often a shared library its DLL links against is not on the runtime
  # loader path). Reporting both as "not installed" sends the user off to
  # reinstall something they already have, so tell them apart and surface
  # the real loader error.
  installed_at <- tryCatch(find.package("morie"), error = function(e) NULL)

  if (is.null(installed_at)) {
    stop(
      "The R 'morie' package is required for R-backed modules but is not ",
      "installed. Install it from the repository's r-package/morie ",
      "(`R CMD INSTALL r-package/morie`) or from r-universe ",
      "(`install.packages('morie', repos = 'https://rootcoder007.r-universe.dev')`).",
      call. = FALSE
    )
  }

  load_err <- tryCatch({
    loadNamespace("morie")
    NULL
  }, error = function(e) conditionMessage(e))

  stop(
    "The R 'morie' package IS installed at ", installed_at, " but could not ",
    "be loaded, so R-backed modules cannot run. This is an environment ",
    "problem, not a missing install -- do NOT reinstall.\n\n",
    "  Loader error: ", if (is.null(load_err)) "(unknown)" else load_err, "\n\n",
    "If the message above names a shared library (for example ",
    "'liboqs.so.N: cannot open shared object file'), that library is either ",
    "absent or not on the runtime loader path. Locate it and export the ",
    "directory before re-running:\n",
    "  find / -name '<library>*' 2>/dev/null\n",
    "  export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH\n",
    "Note that a DIFFERENT soversion being present (say liboqs.so.7 when ",
    "liboqs.so.9 is required) does not satisfy the link -- the exact ",
    "soversion morie was compiled against must be findable.",
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
