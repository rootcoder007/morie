% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{scalar_vector_message}
\alias{scalar_vector_message}
\title{The message: scalars from scalars, vectors from s*hat_r and s*v}
\usage{
scalar_vector_message(s_j, v_j, r_ij, phi_s, phi_v, W_rbf) 
}
\arguments{
\item{s_j}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{v_j}{A matrix; passed to \code{as.matrix}.}
\item{r_ij}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{phi_s}{Accepted by the signature and not used anywhere in the body.}
\item{phi_v}{Accepted by the signature and not used anywhere in the body.}
\item{W_rbf}{Accepted by the signature and not used anywhere in the body.}
}
\description{
Type is preserved by construction, which is what makes the whole network equivariant rather than approximately so.
}
\value{
A list with \code{ds}, \code{dv}, \code{note}.
}
\keyword{internal}
\examples{
set.seed(1)
F <- 4; D <- 3
s <- rnorm(F)
v <- matrix(rnorm(D * F), D, F)
r <- c(1, 0.5, -0.5)
phi_s <- function(s, w) s
phi_v <- function(s, w) rep(0.1, 2 * length(s))
W_rbf <- function(d) rep(1, 5)
m <- scalar_vector_message(s, v, r, phi_s, phi_v, W_rbf)
c(length(m$ds), nrow(m$dv), ncol(m$dv))
}
