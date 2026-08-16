% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_alpha_from_gamma}
\alias{morie_alpha_from_gamma}
\title{Invert gamma = 1 / (alpha (1 - alpha)) for alpha}
\usage{
morie_alpha_from_gamma(gamma)
}
\arguments{
\item{gamma}{Threshold, must be at least 4.}
}
\description{
Uses the conjugate form to keep precision when gamma is large.
}
\value{
A numeric value in (0, 1/2].
}
\keyword{internal}
