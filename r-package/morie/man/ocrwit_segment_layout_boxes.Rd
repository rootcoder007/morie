% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{ocrwit_segment_layout_boxes}
\alias{ocrwit_segment_layout_boxes}
\title{ocrwit_segment_layout_boxes}
\usage{
ocrwit_segment_layout_boxes(boxes, segment_ids, width, height, scale = 1000)
}
\arguments{
\item{boxes}{Coerced to list by the body, with \code{as.list}.}
\item{segment_ids}{Coerced to list by the body, with \code{as.list}.}
\item{width}{Passed to \code{ocrwit_normalise_bbox}.}
\item{height}{Passed to \code{ocrwit_normalise_bbox}.}
\item{scale}{Passed to \code{ocrwit_normalise_bbox}. Defaults to \code{1000}.}
}
\description{
Part of the ocrwit_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{segment_boxes}, \code{per_token}, \code{n_segments}, \code{note}.
}
\keyword{internal}
