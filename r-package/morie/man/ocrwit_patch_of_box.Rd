% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{ocrwit_patch_of_box}
\alias{ocrwit_patch_of_box}
\title{ocrwit_patch_of_box}
\usage{
ocrwit_patch_of_box(box, width, height, patch_grid = 14)
}
\arguments{
\item{box}{See Usage.}
\item{width}{See Usage.}
\item{height}{See Usage.}
\item{patch_grid}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{14}.}
}
\description{
Part of the ocrwit_native implementation; see the file header for the source it follows.
}
\value{
A vector, from \code{sort}.
}
\keyword{internal}
\examples{
ocrwit_patch_of_box(box = c(1, 2, 3, 4, 5, 6, 7, 8), width = 5L, height = 5L)
}
