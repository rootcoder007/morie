% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{negbinom_offspring}
\alias{negbinom_offspring}
\title{negbinom_offspring}
\usage{
negbinom_offspring(R0, dispersion, rng) 
}
\arguments{
\item{R0}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{dispersion}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{rng}{Passed to \code{.tt_gamma_draw}.}
}
\description{
Part of the ttrace_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{.tt_poisson_draw}.
}
\keyword{internal}
\examples{
e <- morie:::.ghc_rng(1)
replicate(5, negbinom_offspring(R0 = 2.5, dispersion = 0.16, e))
}
