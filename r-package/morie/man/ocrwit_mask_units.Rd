% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{ocrwit_mask_units}
\alias{ocrwit_mask_units}
\title{ocrwit_mask_units}
\usage{
ocrwit_mask_units(n_units, rate = 0.3, seed = 0, block = 1) 
}
\arguments{
\item{n_units}{Coerced to integer by the body, with \code{as.integer}.}
\item{rate}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.3}.}
\item{seed}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0}.}
\item{block}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{1}.}
}
\description{
Part of the ocrwit_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{masked}, \code{kept}, \code{rate}, \code{block}, \code{note}.
}
\keyword{internal}
\examples{
ocrwit_mask_units(n_units = 5L)
}
