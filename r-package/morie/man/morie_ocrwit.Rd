% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_ocrwit}
\alias{morie_ocrwit}
\title{morie_ocrwit}
\usage{
morie_ocrwit(text_boxes, masked_patches, width, height, patch_grid = 14, masked_text = list())
}
\arguments{
\item{text_boxes}{A vector; its length is taken and its elements indexed.}
\item{masked_patches}{See Usage.}
\item{width}{See Usage.}
\item{height}{See Usage.}
\item{patch_grid}{Defaults to \code{14}.}
\item{masked_text}{Defaults to \code{list()}.}
}
\description{
Part of the ocrwit_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{labels}, \code{patches}, \code{n_examples}, \code{positive_rate}, \code{method}, \code{note}.
}
\keyword{internal}
