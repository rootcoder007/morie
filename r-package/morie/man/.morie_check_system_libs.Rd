% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_check_system_libs}
\alias{.morie_check_system_libs}
\title{Internal: probe whether the C system libraries are available}
\usage{
.morie_check_system_libs()
}
\description{
Uses configure-time flags written into DESCRIPTION at install (Phase 3JJJ1/2: MORIE_HAVE_SODIUM / MORIE_HAVE_LIBOQS), with library-load fallback for libcurl (Imports R-side).
}
\value{
A list with \code{libcurl}, \code{libsodium}, \code{liboqs}.
}
\keyword{internal}
\examples{
.morie_check_system_libs()
}
