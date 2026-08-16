% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.ghc_didfst_panel}
\alias{.ghc_didfst_panel}
\title{.ghc_didfst_panel}
\usage{
.ghc_didfst_panel(Y)
}
\arguments{
\item{Y}{A matrix; passed to \code{as.matrix}.}
}
\description{
A step of the didfst_native implementation. Called by \code{group_time_att}, \code{panel_differences}, \code{placebo_did}. See the file header for the source the module follows.
}
\value{
A list with \code{M}, \code{n}, \code{T}.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.ghc_didfst_panel(M)
}
