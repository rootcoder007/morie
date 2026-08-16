% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hawkes_to_phi}
\alias{.hawkes_to_phi}
\title{.hawkes_to_phi}
\usage{
.hawkes_to_phi(theta)
}
\arguments{
\item{theta}{A vector; indexed elementwise.}
}
\description{
A step of the hawkes_fit implementation. Called by \code{morie_hawkes_fit}. See the file header for the source the module follows.
}
\value{
The value of \code{phi}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.hawkes_to_phi(V)
}
