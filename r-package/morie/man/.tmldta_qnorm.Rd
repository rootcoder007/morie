% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tmldta_qnorm}
\alias{.tmldta_qnorm}
\title{.tmldta_qnorm}
\usage{
.tmldta_qnorm(p)
}
\arguments{
\item{p}{Passed to \code{qnorm}.}
}
\description{
Part of the tmldta_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{qnorm}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.tmldta_qnorm(V)
}
