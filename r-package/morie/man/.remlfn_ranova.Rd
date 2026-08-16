% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.remlfn_ranova}
\alias{.remlfn_ranova}
\title{.remlfn_ranova}
\usage{
.remlfn_ranova(y, group)
}
\arguments{
\item{y}{Numeric; passed to \code{mean}.}
\item{group}{Passed to \code{.remlfn_groups}.}
}
\description{
Part of the remlfn_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{msa}, \code{mse}, \code{sigma2_a_raw}, \code{sigma2_a}, \code{balanced}, \code{dfa}, \code{dfe}, \code{ssa}, \code{sse}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.remlfn_ranova(V, V)
}
