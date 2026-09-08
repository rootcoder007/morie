% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{importance_weights}
\alias{importance_weights}
\title{Importance weights over diffusion steps}
\usage{
importance_weights(step_losses, uniform = FALSE, smoothing = 0.1) 
}
\arguments{
\item{step_losses}{Numeric per-step losses.}
\item{uniform}{Use uniform weights.}
\item{smoothing}{Additive constant in the square-root.}
}
\description{
Sample the steps that actually carry loss instead of drawing uniformly.
}
\value{
A list with \code{weights}, \code{uniform},   \code{effective_steps} (and \code{note} for the importance path).
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
importance_weights(V)
}
