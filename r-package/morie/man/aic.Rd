% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{aic}
\alias{aic}
\title{aic}
\usage{
aic(loglik, n_par)
}
\arguments{
\item{loglik}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{n_par}{Coerced to integer by the body, with \code{as.integer}.}
}
\description{
Part of the sarimax_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
aic(V, V)
}
