% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrzi1_obj}
\alias{.hrzi1_obj}
\title{Internal: Ichimura single-index leave-one-out objective. Extracted}
\usage{
.hrzi1_obj(b, X, y, h0)
}
\arguments{
\item{b}{See Usage.}
\item{X}{See Usage.}
\item{y}{See Usage.}
\item{h0}{See Usage.}
}
\description{
from the hrzi1() optimiser closure so the zero-norm guard is directly unit-testable. `h0` is the kernel bandwidth.
}
\value{
A numeric value.
}
\keyword{internal}
