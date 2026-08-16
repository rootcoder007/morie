% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{ocrwit_normalise_bbox}
\alias{ocrwit_normalise_bbox}
\title{ocrwit_normalise_bbox}
\usage{
ocrwit_normalise_bbox(box, width, height, scale = 1000)
}
\arguments{
\item{box}{A vector; indexed elementwise.}
\item{width}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{height}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{scale}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{1000}.}
}
\description{
Part of the ocrwit_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{c}.
}
\keyword{internal}
\examples{
ocrwit_normalise_bbox(box = c(1, 2, 3, 4, 5, 6, 7, 8), width = 5L, height = 5L)
}
