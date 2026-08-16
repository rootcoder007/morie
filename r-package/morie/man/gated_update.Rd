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
\item{s}{See Usage.}
\item{v}{A matrix; passed to \code{as.matrix}.}
\item{U}{A matrix; passed to \code{as.matrix}.}
\item{V}{A matrix; passed to \code{as.matrix}.}
\item{phi}{See Usage.}
}
\description{
is the only path from the vector channel back to the scalar one, and it is invariant -- which is why the network can use directional information without breaking invariance of the energy.
}
\value{
A list with \code{ds}, \code{dv}, \code{scalar_from_vectors}, \code{note}.
}
\keyword{internal}
