% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{cfconv}
\alias{cfconv}
\title{cfconv}
\usage{
cfconv(X, R, filter_net, cutoff = 5, ...)
}
\arguments{
\item{X}{Iterated over elementwise, with \code{lapply}.}
\item{R}{Iterated over elementwise, with \code{lapply}.}
\item{filter_net}{See Usage.}
\item{cutoff}{Defaults to \code{5}.}
\item{...}{Passed through.}
}
\description{
Part of the schN_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
cfconv(X = 5L, R = 5L, filter_net = c(1, 2, 3, 4, 5, 6, 7, 8))
}
