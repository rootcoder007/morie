# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Package-level imports + global-variable declarations.
#
# This file silences R CMD check's "no visible binding for global
# variable" warnings for two valid patterns morie uses:
#
# 1. `.data` from rlang -- used by ggplot2 NSE inside aes(). morie
#    doesn't formally depend on rlang (ggplot2 is in Suggests), so we
#    declare `.data` as a global to keep the rlang dependency optional
#    while still satisfying the check.
#
# 2. Python-port placeholder lookups in tps_statphysics.R + laniyonu_*
#    where code does `if (exists("morie_tps_load_tps_dataset")) ...`
#    and then calls the function in the conditional branch. R's static
#    analyzer flags the call site as an undefined global because the
#    function is only defined in the Python sibling, not in R. These
#    are intentional NotYetPorted placeholders.

utils::globalVariables(c(
  # ggplot2/rlang NSE
  ".data",
  # Python-port placeholder (intentional `exists()`-guarded lookup;
  # the tps loaders graduated to real R functions in tps_statphysics.R
  # and must NOT be declared here -- pkgload reports a declared global
  # that shadows a real export as a mask conflict)
  "morie_spatial_spillover_decomposition",
  # geepack::geeglm NSE: cluster id column added at runtime then passed
  # by bare name to the formula-style `id` arg (see 3MMM.48 fix).
  ".gee_cluster_id_int_"
))

.onLoad <- function(libname, pkgname) {
  # future's connection-misuse check (diff_connections() in FutureResult) can
  # segfault R uncatchably when DoubleML/mlr3 resolve futures. Setting the env
  # var BEFORE future is loaded makes the "ignore" setting take effect in the
  # main process AND every worker -- each re-reads R_FUTURE_* when future loads.
  # An options() guard does not reach workers, which is why it was only flaky.
  # Only set when the user has not chosen their own value.
  if (!nzchar(Sys.getenv("R_FUTURE_CONNECTIONS_ONMISUSE"))) {
    Sys.setenv(R_FUTURE_CONNECTIONS_ONMISUSE = "ignore")
  }
  # The fast-stat kernels in src/morie_fast.cpp resolve rmbl_* routines that
  # rmoriebricklayer registers via R_RegisterCCallable (LinkingTo). A
  # DESCRIPTION Imports: alone does not load the provider DLL, so load its
  # namespace (triggering its useDynLib + registration) before any C call.
  # The :: reference (not just requireNamespace) is what marks the Imports
  # entry as used for R CMD check's dependency scan.
  if (requireNamespace("rmoriebricklayer", quietly = TRUE)) {
    invisible(rmoriebricklayer::core_mean)
  }
  try(.morie_auto_register_stat_commands(), silent = TRUE)
  invisible(NULL)
}
