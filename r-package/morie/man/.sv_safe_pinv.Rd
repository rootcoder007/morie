% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sv_safe_pinv}
\alias{.sv_safe_pinv}
\title{.sv_safe_pinv}
\usage{
.sv_safe_pinv(M, tol = 1e-12)
}
\arguments{
\item{M}{See Usage.}
\item{tol}{Defaults to \code{1e-12}.}
}
\description{
Part of the spatial_voting implementation; see the file header for the source it follows.
}
\value{
The value of \code{\%*\%}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.sv_safe_pinv(V)
}
