% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{serial_interval_draw}
\alias{serial_interval_draw}
\title{serial_interval_draw}
\usage{
serial_interval_draw(mean, sd, rng, allow_presymptomatic = TRUE) 
}
\arguments{
\item{mean}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{sd}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{rng}{Passed to \code{.ghc_norm}.}
\item{allow_presymptomatic}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the ttrace_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{v}, as built in the body.
}
\keyword{internal}
\examples{
e <- .ghc_rng(1)
replicate(3, serial_interval_draw(mean = 5, sd = 1.9, e))
}
