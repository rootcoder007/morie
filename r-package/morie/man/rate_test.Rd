% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{rate_test}
\alias{rate_test}
\title{Half-sample bootstrap test of RATE = 0}
\usage{
rate_test(scores, priority, weight = "autoc", reps = 500L, seed = 0) 
}
\description{
Half-sample bootstrap test of RATE = 0
}
\arguments{
\item{scores}{Argument \code{scores}; see Usage.}
\item{priority}{Argument \code{priority}; see Usage.}
\item{weight}{Argument \code{weight}; see Usage.}
\item{reps}{Argument \code{reps}; see Usage.}
\item{seed}{Argument \code{seed}; see Usage.}
}
\value{
A list with \code{estimate}, \code{se}, \code{z}, \code{p_value}, \code{weight}, \code{reps}, \code{n}, \code{null}, \code{method}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
rate_test(V, V)
}
