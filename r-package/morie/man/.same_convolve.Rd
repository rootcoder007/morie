% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.same_convolve}
\alias{.same_convolve}
\title{"same"-mode convolution matching numpy.convolve(x, k, "same")}
\usage{
.same_convolve(x, k)
}
\arguments{
\item{x}{A vector; its length is taken.}
\item{k}{A vector; its length is taken.}
}
\description{
A step of the dsp_filters implementation. Called by \code{morie_dsp_hann_filter}, \code{morie_dsp_matched}, \code{morie_dsp_moving_average} and 2 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.same_convolve(V, V)
}
