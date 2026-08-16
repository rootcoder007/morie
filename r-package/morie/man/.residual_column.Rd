% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.residual_column}
\alias{.residual_column}
\title{.residual_column}
\usage{
.residual_column(w, ar, ma)
}
\arguments{
\item{w}{A vector; its length is taken.}
\item{ar}{Passed to \code{css}.}
\item{ma}{Passed to \code{css}.}
}
\description{
Part of the sarimax_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{v}, \code{f}.
}
\keyword{internal}
\examples{
.residual_column(w = c(1, 2, 3, 4, 5, 6, 7, 8), ar = c(1, 2, 3, 4, 5, 6, 7, 8), ma = c(1, 2, 3, 4, 5, 6, 7, 8))
}
