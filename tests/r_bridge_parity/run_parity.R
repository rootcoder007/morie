# SPDX-License-Identifier: AGPL-3.0-or-later
# Check the R bridge's shipped native adapters against the reference packages they replace.
#   Rscript run_parity.R <group>_adapters.R <bridge_natives.R> <morie R package name>
# One line per command: name, status (OK, OK-NOT-SHIPPED, MISMATCH, DECLARED, NO-NATIVE, SKIP, ERROR),
# max rel diff. OK-NOT-SHIPPED: a native that matches but the bridge does not use yet (ship = FALSE).
# DECLARED marks a difference the adapter documents (mismatch = TRUE: a different estimator or
# specification, explained in its note); SKIP a reference package that is not installed here.
args <- commandArgs(trailingOnly = TRUE)
group_file <- args[1]; natives_file <- args[2]; pkg <- args[3]
suppressPackageStartupMessages(library(pkg, character.only = TRUE))
M <- asNamespace(pkg)
# reference formulas name Surv(); the natives read Surv(time, status) without the survival package
if (requireNamespace("survival", quietly = TRUE)) suppressPackageStartupMessages(library(survival))
group <- new.env(parent = globalenv())
sys.source(group_file, envir = group)
shipped <- new.env(parent = globalenv())
sys.source(natives_file, envir = shipped)
for (nm in names(group$ADAPTERS)) {
  ad <- group$ADAPTERS[[nm]]
  line <- tryCatch({
    ref <- tryCatch(ad$reference(ad$args), error = function(e) e)
    if (inherits(ref, "error")) {
      if (grepl("there is no package called|is not installed", conditionMessage(ref)))
        sprintf("%s SKIP NA reference package missing: %s", nm, conditionMessage(ref))
      else stop(conditionMessage(ref))
    } else if (is.null(ad$native)) {
      sprintf("%s NO-NATIVE NA", nm)
    } else {
      unshipped <- identical(ad$ship, FALSE)  # checked here, not used by the bridge (see its comment)
      native <- if (unshipped) ad$native else shipped$NATIVE[[nm]]
      if (is.null(native)) stop("no shipped native adapter (regenerate rscripts/bridge_natives.R)")
      nat <- native(ad$args, M)
      cmp <- ad$compare(nat, ref)
      a <- as.numeric(cmp$native); b <- as.numeric(cmp$reference)
      if (length(a) != length(b)) stop(sprintf("compared %d native values with %d reference values", length(a), length(b)))
      rel <- if (length(a)) max(abs(a - b) / pmax(abs(b), 1e-12), na.rm = TRUE) else 0
      status <- if (isTRUE(ad$mismatch)) "DECLARED" else if (is.finite(rel) && rel <= ad$tol) "OK" else "MISMATCH"
      if (unshipped && status == "OK") status <- "OK-NOT-SHIPPED"
      sprintf("%s %s %.3g", nm, status, rel)
    }
  }, error = function(e) sprintf("%s ERROR NA %s", nm, gsub("\n", " ", conditionMessage(e))))
  cat(line, "\n", sep = "")
}
