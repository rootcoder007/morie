% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fit_q1}
\alias{.fit_q1}
\title{Fit Q1 (E[Q2(A(0), d_{A(1)}, Lbar(1)) | A(0), L(0)]) on the rows in}
\usage{
.fit_q1(pseudo, L0, A0, idx, ridge)
}
\arguments{
\item{pseudo}{A vector; indexed elementwise.}
\item{L0}{A matrix; indexed by row and column.}
\item{A0}{A vector; indexed elementwise.}
\item{idx}{Iterated over elementwise, with \code{lapply}.}
\item{ridge}{Passed to \code{.tmldyn_lstsq}.}
}
\description{
idx Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{q1}, \code{b}.
}
\keyword{internal}
\examples{
.fit_q1(pseudo = c(1, 2, 3, 4, 5, 6, 7, 8), L0 = c(1, 2, 3, 4, 5, 6, 7, 8), A0 = c(1, 2, 3, 4, 5, 6, 7, 8), idx = c(1, 2, 3, 4, 5, 6, 7, 8), ridge = c(1, 2, 3, 4, 5, 6, 7, 8))
}
