% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.measure}
\alias{.measure}
\title{.measure}
\usage{
.measure(protected, measure, step, normalize, caveat = NULL)
}
\arguments{
\item{protected}{Coerced to logical by the body, with \code{as.logical}.}
\item{measure}{One of \code{"rKL"}, \code{"rND"}, \code{"rRD"}.}
\item{step}{Passed to \code{.raw}.}
\item{normalize}{A flag; the body branches on it.}
\item{caveat}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
A step of the fairRC_native implementation. Called by \code{rKL}, \code{rND}, \code{rRD}. See the file header for the source the module follows.
}
\value{
The value of \code{pay}, as built in the body.
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
