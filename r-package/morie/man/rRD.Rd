% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{rRD}
\alias{rRD}
\title{rRD}
\usage{
rRD(protected, step = 10L, normalize = TRUE)
}
\arguments{
\item{protected}{See Usage.}
\item{step}{Defaults to \code{10L}.}
\item{normalize}{Defaults to \code{TRUE}.}
}
\description{
Part of the fairRC_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{.measure}.
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
