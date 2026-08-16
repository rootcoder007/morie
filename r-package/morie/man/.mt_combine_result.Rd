% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.mt_combine_result}
\alias{.mt_combine_result}
\title{.mt_combine_result}
\usage{
.mt_combine_result(method, stat, p_comb, interp, extra = list())
}
\arguments{
\item{method}{Carried through into a list the body builds.}
\item{stat}{Carried through into a list the body builds.}
\item{p_comb}{Carried through into a list the body builds.}
\item{interp}{Carried through into a list the body builds.}
\item{extra}{Passed to \code{c}. Defaults to \code{list()}.}
}
\description{
Part of the multiple_testing implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
.mt_combine_result(method = c(1, 2, 3, 4, 5, 6, 7, 8), stat = c(1, 2, 3, 4, 5, 6, 7, 8), p_comb = c(1, 2, 3, 4, 5, 6, 7, 8), interp = c(1, 2, 3, 4, 5, 6, 7, 8))
}
