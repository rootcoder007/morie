% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hill_mle}
\alias{.hill_mle}
\title{Clauset-Shalizi-Newman discrete MLE: alpha = 1 + n / sum(log(x /}
\usage{
.hill_mle(x, x_min)
}
\arguments{
\item{x}{See Usage.}
\item{x_min}{See Usage.}
}
\description{
(x_min - 0.5))) The -0.5 continuity correction matters when x_min is small. Continuous Hill (no correction) would use log(x / x_min), but morie\\\'s OTIS placement counts are integer-valued so the discrete form is right.
}
\value{
A numeric value.
}
\keyword{internal}
