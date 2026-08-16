% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_spec_test}
\alias{.new_spec_test}
\title{.new_spec_test}
\usage{
.new_spec_test(name, statistic, p_value, df, conclusion)
}
\arguments{
\item{name}{Carried through into a list the body builds.}
\item{statistic}{Carried through into a list the body builds.}
\item{p_value}{Carried through into a list the body builds.}
\item{df}{Carried through into a list the body builds.}
\item{conclusion}{Carried through into a list the body builds.}
}
\description{
A step of the diagnostics implementation. Called by \code{hosmer_lemeshow_test}, \code{likelihood_ratio_test}, \code{link_test} and 4 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
\examples{
.new_spec_test(name = c(1, 2, 3, 4, 5, 6, 7, 8), statistic = c(1, 2, 3, 4, 5, 6, 7, 8), p_value = c(1, 2, 3, 4, 5, 6, 7, 8), df = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), conclusion = c(1, 2, 3, 4, 5, 6, 7, 8))
}
