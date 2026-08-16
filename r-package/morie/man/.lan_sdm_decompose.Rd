% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lan_sdm_decompose}
\alias{.lan_sdm_decompose}
\title{.lan_sdm_decompose}
\usage{
.lan_sdm_decompose(rho, beta_direct, beta_spatial, W, coefficient_names)
}
\arguments{
\item{rho}{Numeric; combined arithmetically in the body.}
\item{beta_direct}{A vector; indexed elementwise.}
\item{beta_spatial}{A vector; indexed elementwise.}
\item{W}{A matrix; passed to \code{nrow}.}
\item{coefficient_names}{A vector; its length is taken and its elements indexed.}
}
\description{
Part of the laniyonu_gentrification_policing implementation; see the file header for the source it follows.
}
\value{
The value of \code{lapply}.
}
\keyword{internal}
\examples{
.lan_sdm_decompose(rho = 0.5, beta_direct = c(1, 2, 3, 4, 5, 6, 7, 8), beta_spatial = c(1, 2, 3, 4, 5, 6, 7, 8), W = c(1, 2, 3, 4, 5, 6, 7, 8), coefficient_names = c(1, 2, 3, 4, 5, 6, 7, 8))
}
