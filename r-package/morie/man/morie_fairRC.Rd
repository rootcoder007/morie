% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_fairRC}
\alias{morie_fairRC}
\title{morie_fairRC}
\usage{
morie_fairRC(protected, step = 10L, normalize = TRUE)
}
\arguments{
\item{protected}{Passed to \code{rND}.}
\item{step}{Passed to \code{rND}. Defaults to \code{10L}.}
\item{normalize}{Passed to \code{rND}. Defaults to \code{TRUE}.}
}
\description{
A step of the fairRC_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
The value of \code{rND}.
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
