% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.airl_prep}
\alias{.airl_prep}
\title{.airl_prep}
\usage{
.airl_prep(S, A, S1, LP, name)
}
\arguments{
\item{S}{A vector; its length is taken.}
\item{A}{A vector; its length is taken.}
\item{S1}{A vector; its length is taken.}
\item{LP}{A vector; its length is taken.}
\item{name}{See Usage.}
}
\description{
A step of the airl_native implementation. Called by \code{airl}. See the file header for the source the module follows.
}
\value{
A list with \code{S}, \code{A}, \code{S1}, \code{LP}.
}
\keyword{internal}
\examples{
.airl_prep(S = c(1, 2, 3, 4, 5, 6, 7, 8), A = c(1, 2, 3, 4, 5, 6, 7, 8), S1 = c(1, 2, 3, 4, 5, 6, 7, 8), LP = c(1, 2, 3, 4, 5, 6, 7, 8), name = c(1, 2, 3, 4, 5, 6, 7, 8))
}
