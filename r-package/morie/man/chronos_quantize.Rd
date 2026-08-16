% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{chronos_quantize}
\alias{chronos_quantize}
\title{chronos_quantize}
\usage{
chronos_quantize(x, bins)
}
\arguments{
\item{x}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{bins}{A list; the body reads \code{$centers}, \code{$edges} from it.}
}
\description{
A step of the chronos_native implementation. Called by \code{chronos_tokenize}. See the file header for the source the module follows.
}
\value{
A list with \code{tokens}, \code{n_clipped}, \code{clipped_fraction}, \code{in_range}, \code{note}.
}
\keyword{internal}
\examples{
b <- chronos_uniform_bins(-3, 3, n_bins = 16L)
chronos_quantize(c(-2.5, 0, 1.7), b)
}
