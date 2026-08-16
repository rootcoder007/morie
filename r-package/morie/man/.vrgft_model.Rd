% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.vrgft_model}
\alias{.vrgft_model}
\title{Internal: parametric variogram model value at distance h. Extracted}
\usage{
.vrgft_model(h, c0, c1, a, model)
}
\arguments{
\item{h}{See Usage.}
\item{c0}{See Usage.}
\item{c1}{See Usage.}
\item{a}{See Usage.}
\item{model}{See Usage.}
}
\description{
from the vrgft() optimiser closure so the model switch (including the unknown-model stop) is directly unit-testable.
}
\value{
The value of \code{switch}.
}
\keyword{internal}
