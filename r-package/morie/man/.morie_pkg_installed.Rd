% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_pkg_installed}
\alias{.morie_pkg_installed}
\title{Internal: is a CRAN package installed locally?}
\usage{
.morie_pkg_installed(pkg)
}
\arguments{
\item{pkg}{See Usage.}
}
\description{
A step of the install_extras implementation. Called by \code{.morie_check_system_libs}, \code{morie_install_extras}. See the file header for the source the module follows.
}
\value{
A logical value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.morie_pkg_installed(V)
}
