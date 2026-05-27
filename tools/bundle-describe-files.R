#!/usr/bin/env Rscript
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Bundle the 36,000+ describe_<name>.md files shipped under
# src/morie/fn/ into a single compressed Rds for the R package.
#
# Why: shipping 36k individual files in r-package/morie/inst/extdata/
# would create ~145 MB of inode-overhead on R CMD INSTALL even though
# the content is only ~5.6 MB. A single xz-compressed Rds named-
# character vector is ~1-2 MB on disk, loads in milliseconds, and
# fits the R-native idiom.
#
# Run from the repo root:
#   Rscript tools/bundle-describe-files.R
#
# Re-run whenever src/morie/fn/describe_*.md changes; commit the
# regenerated r-package/morie/inst/extdata/describe_corpus.Rds.

suppressPackageStartupMessages({
  # base R only; no Suggests/Imports load.
})

# Assume CWD is repo root (script invoked from there). Sanity-check
# by looking for the src/morie/fn/ tree.
repo_root <- normalizePath(getwd())

fn_dir   <- file.path(repo_root, "src", "morie", "fn")
out_path <- file.path(repo_root, "r-package", "morie", "inst",
                      "extdata", "describe_corpus.Rds")

if (!dir.exists(fn_dir)) {
  stop("src/morie/fn/ not found at: ", fn_dir,
       "\nRun this script from the morie repo root.")
}

files <- list.files(fn_dir, pattern = "^describe_.+\\.md$",
                    full.names = TRUE)
n_files <- length(files)
if (n_files == 0L) {
  stop("No describe_*.md files found under ", fn_dir)
}
message("Found ", n_files, " describe_*.md files under ", fn_dir)

t0 <- Sys.time()
corpus <- setNames(
  vapply(files,
         function(f) paste(readLines(f, warn = FALSE), collapse = "\n"),
         character(1L)),
  sub("^describe_(.+)\\.md$", "\\1", basename(files))
)
t_read <- Sys.time() - t0

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
t1 <- Sys.time()
saveRDS(corpus, out_path, compress = "xz", version = 2L)
t_write <- Sys.time() - t1

info <- file.info(out_path)
message(sprintf("Read   %d files in %.2f s", n_files, as.numeric(t_read)))
message(sprintf("Wrote  %s (%.2f MB) in %.2f s",
                out_path, info$size / 1024^2, as.numeric(t_write)))
message(sprintf("Corpus: %d named entries, %.2f MB in memory",
                length(corpus),
                sum(nchar(corpus, type = "bytes")) / 1024^2))

# Re-load smoke test
loaded <- readRDS(out_path)
stopifnot(
  identical(length(loaded), length(corpus)),
  identical(names(loaded)[1L], names(corpus)[1L])
)
message("Re-load smoke test: OK")
