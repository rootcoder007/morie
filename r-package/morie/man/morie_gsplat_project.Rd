% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_gsplat_project}
\alias{morie_gsplat_project}
\title{EWA projection of the 3D covariance}
\usage{
morie_gsplat_project(S, W, J)
}
\arguments{
\item{S}{3x3 covariance.}
\item{W}{3x3 world-to-camera transform.}
\item{J}{2x3 Jacobian of the perspective projection.}
}
\description{
\eqn{\Sigma\' = J W \Sigma W^\top J^\top}, the affine approximation to perspective projection that keeps the splat closed-form and therefore fast.
}
\value{
A list with \code{projected} (2x2) and \code{dim}.
}
\keyword{internal}
