% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fairness_mlp_forward}
\alias{.fairness_mlp_forward}
\title{.fairness_mlp_forward}
\usage{
.fairness_mlp_forward(params, x)
}
\arguments{
\item{params}{A vector; its length is taken and its elements indexed.}
\item{x}{A matrix; passed to \code{\%*\%}.}
}
\description{
A step of the fairness_gan implementation. Called by \code{morie_fairness_spatial_gan}. See the file header for the source the module follows.
}
\value{
The value of \code{x}, as built in the body.
}
\keyword{internal}
