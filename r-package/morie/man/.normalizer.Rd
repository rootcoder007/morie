% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.normalizer}
\alias{.normalizer}
\title{.normalizer}
\usage{
.normalizer(protected, measure = "rND", step = 10L)
}
\arguments{
\item{protected}{A vector; its length is taken.}
\item{measure}{Passed to \code{.raw}. Defaults to \code{"rND"}.}
\item{step}{Passed to \code{.raw}. Defaults to \code{10L}.}
}
\description{
A step of the fairRC_native implementation. Called by \code{.measure}. See the file header for the source the module follows.
}
\value{
One of two values, depending on the branch taken.
}
\references{
Yang, K. & Stoyanovich, J. (2017) "Measuring Fairness in Ranked
Outputs", Proceedings of the 29th International Conference on
Scientific and Statistical Database Management (SSDBM \'17),
arXiv:1610.08559. Sec. 3.
Jarvelin, K. & Kekalainen, J. (2002) "Cumulated gain-based evaluation
of IR techniques", ACM Transactions on Information Systems 20(4),
422-446. The logarithmic discount.
}
\keyword{internal}
