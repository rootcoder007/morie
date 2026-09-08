% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_painn_equivariance_error}
\alias{morie_painn_equivariance_error}
\title{Rotate the input; scalars must not move, vectors must rotate}
\usage{
morie_painn_equivariance_error(model, s, v, R, Q, tol = 1e-09) 
}
\arguments{
\item{model}{Accepted by the signature and not used anywhere in the body.}
\item{s}{Passed to \code{model}.}
\item{v}{A matrix; passed to \code{as.matrix}.}
\item{R}{A matrix; passed to \code{as.matrix}.}
\item{Q}{A matrix; passed to \code{as.matrix}.}
\item{tol}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1e-09}.}
}
\description{
Checking only the scalars would pass a model that has silently lost its vector channel.
}
\value{
A list with \code{scalar_error}, \code{vector_error}, \code{scalars_invariant}, \code{vectors_equivariant}, \code{note}.
}
\keyword{internal}
\examples{
set.seed(1)
Fdim <- 4
s <- rnorm(Fdim)
v <- matrix(rnorm(3 * Fdim), 3, Fdim)
R <- matrix(rnorm(9), 3, 3)
th <- pi / 5
Q <- matrix(c(cos(th), -sin(th), 0, sin(th), cos(th), 0, 0, 0, 1), 3, 3)
model <- function(s, V, pos) list(s = s,
  v = lapply(seq_len(nrow(as.matrix(V))), function(i) as.matrix(V)[i, ]))
e <- morie_painn_equivariance_error(model, s, v, R, Q)
is.list(e)
}
