% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gsplat_covariance}
\alias{morie_gsplat_covariance}
\title{Build the 3D covariance as Sigma = R S S\' R\'}
\usage{
morie_gsplat_covariance(scale, quaternion) 
}
\arguments{
\item{scale}{Length-3 positive scale vector.}
\item{quaternion}{Length-4 (unnormalised) rotation quaternion.}
}
\description{
Every reachable parameter value is positive semi-definite by construction; gradient descent on the six raw entries would not stay there.
}
\value{
A list with \code{covariance}, \code{rotation},   \code{scale} and \code{note}.
}
\keyword{internal}
\examples{
r <- morie_gsplat_covariance(scale = c(1, 2, 0.5),
                             quaternion = c(1, 0, 0, 0))
dim(r$covariance)
}
