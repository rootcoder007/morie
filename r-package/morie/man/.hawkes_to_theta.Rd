% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hawkes_to_theta}
\alias{.hawkes_to_theta}
\title{Optimisation runs in an unconstrained space phi to avoid the hard}
\usage{
.hawkes_to_theta(phi)
}
\arguments{
\item{phi}{See Usage.}
}
\description{
feasibility cliffs: a0 is free, eta = plogis(phi) in (0, 1), and the shape/scale parameters are exp(phi) > 0.
}
\value{
The value of \code{theta}, as built in the body.
}
\keyword{internal}
