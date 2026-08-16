% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{gamma_generation_time}
\alias{gamma_generation_time}
\title{Build a gamma generation-time density on a grid, normalised}
\usage{
gamma_generation_time(shape = 2.83, scale = 1.86, grid = NULL, t_max = 30, n = 3001L)
}
\arguments{
\item{shape}{Positive shape.}
\item{scale}{Positive scale.}
\item{grid}{Optional grid of times.}
\item{t_max}{Maximum time when \code{grid} is \code{NULL}.}
\item{n}{Number of grid points.}
}
\description{
Defaults are a shape/scale pair in the range reported for SARS-CoV-2; they are a placeholder, not the paper\'s fit.
}
\value{
A list with \code{t} and \code{density}.
}
\keyword{internal}
