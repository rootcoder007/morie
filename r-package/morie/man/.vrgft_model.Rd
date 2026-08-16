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
\item{h}{Numeric; combined arithmetically in the body.}
\item{c0}{Numeric; combined arithmetically in the body.}
\item{c1}{Numeric; combined arithmetically in the body.}
\item{a}{Numeric; combined arithmetically in the body.}
\item{model}{See Usage.}
}
\description{
from the vrgft() optimiser closure so the model switch (including the unknown-model stop) is directly unit-testable.
}
\value{
The value of \code{switch}.
}
\keyword{internal}
