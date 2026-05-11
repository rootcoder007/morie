#' @keywords internal
.onAttach <- function(libname, pkgname) {
  if (!requireNamespace("morie", quietly = TRUE)) {
    packageStartupMessage(
      "moirais was renamed to morie in v0.1.14. ",
      "The morie package is not installed; ",
      "install it with install.packages(\"morie\") to use this alias."
    )
    return(invisible(NULL))
  }
  # Attach morie under the moirais session so library(moirais) works
  # as a drop-in alias for library(morie).
  if (!isNamespaceLoaded("morie")) {
    loadNamespace("morie")
  }
  attachNamespace("morie")
  packageStartupMessage(
    "moirais was renamed to morie in v0.1.14. ",
    "library(morie) is the canonical entry point; ",
    "library(moirais) remains as a deprecated alias."
  )
}
