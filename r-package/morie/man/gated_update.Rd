% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{gated_update}
\alias{gated_update}
\title{The update block: v1 . v2 re-enters the scalars. That inner product}
\usage{
gated_update(s, v, U, V, phi) 
}
\arguments{
\item{s}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{v}{A matrix; passed to \code{as.matrix}.}
\item{U}{A matrix; passed to \code{as.matrix}.}
\item{V}{A matrix; passed to \code{as.matrix}.}
\item{phi}{Accepted by the signature and not used anywhere in the body.}
}
\description{
is the only path from the vector channel back to the scalar one, and it is invariant -- which is why the network can use directional information without breaking invariance of the energy.
}
\value{
A list with \code{ds}, \code{dv}, \code{scalar_from_vectors}, \code{note}.
}
\keyword{internal}
\examples{
set.seed(1)
F <- 4; D <- 3
s <- rnorm(F)
v <- matrix(rnorm(D * F), D, F)
U <- diag(F); V <- diag(F)
phi <- function(sv, dot, nrm) list(ds = sv + dot, gate = rep(0.5, length(sv)))
u <- gated_update(s, v, U, V, phi)
c(length(u$ds), nrow(u$dv))
}
