% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gsplat_composite}
\alias{morie_gsplat_composite}
\title{Front-to-back alpha compositing}
\usage{
morie_gsplat_composite(colours, alphas, depths = NULL) 
}
\arguments{
\item{colours}{Matrix of per-Gaussian colours (n x 3 or n x d).}
\item{alphas}{Numeric vector of opacities in [0, 1].}
\item{depths}{Optional depth per Gaussian; sorts back-to-front.}
}
\description{
Same image formation as volume rendering, which is why the two representations are interchangeable at the pixel.
}
\value{
A list with \code{colour}, \code{transmittance},   \code{coverage} and \code{note}.
}
\keyword{internal}
\examples{
colours <- matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), nrow = 3, byrow = TRUE)
alphas <- c(0.5, 0.5, 0.5)
comp <- morie_gsplat_composite(colours, alphas, depths = c(0.1, 0.2, 0.3))
c(length(comp$colour), comp$coverage > 0.8)
}
