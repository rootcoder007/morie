% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.mt_adjusted}
\alias{.mt_adjusted}
\title{.mt_adjusted}
\usage{
.mt_adjusted(method, p, alpha, adjusted, labels = NULL, note = NULL)
}
\arguments{
\item{method}{Passed to \code{.mt_result}.}
\item{p}{A vector; its length is taken.}
\item{alpha}{Passed to \code{.mt_result}.}
\item{adjusted}{Numeric; passed to \code{max}.}
\item{labels}{Passed to \code{.mt_result}.}
\item{note}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
Part of the multiple_testing implementation; see the file header for the source it follows.
}
\value{
The value of \code{.mt_result}.
}
\keyword{internal}
\examples{
.mt_adjusted(method = c(1, 2, 3, 4, 5, 6, 7, 8), p = 0.5, alpha = 0.5, adjusted = c(1, 2, 3, 4, 5, 6, 7, 8))
}
