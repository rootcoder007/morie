% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gsplat_density}
\alias{morie_gsplat_density}
\title{Adaptive density control}
\usage{
morie_gsplat_density(gradients, scales, opacities, grad_threshold = 2e-04, scale_threshold = 0.01, 
    opacity_threshold = 0.005) 
}
\arguments{
\item{gradients}{Per-Gaussian positional gradient magnitudes.}
\item{scales}{Per-Gaussian scale (e.g. max eigenvalue).}
\item{opacities}{Per-Gaussian opacity.}
\item{grad_threshold}{Clone / split threshold on the gradient.}
\item{scale_threshold}{Split threshold on the scale.}
\item{opacity_threshold}{Prune threshold on the opacity.}
}
\description{
Large positional gradient with a small Gaussian means under-reconstruction (clone); with a large Gaussian it means over-reconstruction (split). Near-transparent Gaussians are pruned. The Gaussian count is not fixed in advance.
}
\value{
A list with \code{clone}, \code{split}, \code{prune},   \code{n_before} and \code{n_after}.
}
\keyword{internal}
\examples{
morie_gsplat_density(gradients = c(1, 2, 3, 4, 5, 6, 7, 8), scales = c(1, 2, 3, 4, 5, 6, 7, 8),
  opacities = c(1, 2, 3, 4, 5, 6, 7, 8))
}
