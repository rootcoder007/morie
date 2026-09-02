% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mafft_peaks}
\alias{mafft_peaks}
\title{mafft_peaks}
\usage{
mafft_peaks(lags, c, n_peaks) 
}
\arguments{
\item{lags}{A vector; indexed elementwise.}
\item{c}{Numeric; passed to \code{order}.}
\item{n_peaks}{Coerced to integer by the body, with \code{as.integer}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
mafft_peaks(lags = 5L, c = c(1, 2, 3, 4, 5, 6, 7, 8), n_peaks = c(1, 2, 3, 4, 5, 6, 7, 8))
}
