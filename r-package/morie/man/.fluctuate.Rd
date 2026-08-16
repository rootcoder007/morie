% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fluctuate}
\alias{.fluctuate}
\title{.fluctuate}
\usage{
.fluctuate(outcome, off, H, rows, iters = 100L, tol = 1e-12)
}
\arguments{
\item{outcome}{A vector; indexed elementwise.}
\item{off}{A vector; indexed elementwise.}
\item{H}{A vector; indexed elementwise.}
\item{rows}{A vector; its length is taken.}
\item{iters}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{100L}.}
\item{tol}{Passed to \code{<}. Defaults to \code{1e-12}.}
}
\description{
Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{eps}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.tmldyn_logit(V)
.fluctuate(outcome = V, off = V, H = V, rows = V)
}
