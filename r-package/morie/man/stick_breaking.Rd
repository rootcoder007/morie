% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{stick_breaking}
\alias{stick_breaking}
\title{Sethuraman stick-breaking weights}
\usage{
stick_breaking(alpha, K, seed = 0) 
}
\description{
Sethuraman stick-breaking weights
}
\arguments{
\item{alpha}{Argument \code{alpha}; see Usage.}
\item{K}{Argument \code{K}; see Usage.}
\item{seed}{Argument \code{seed}; see Usage.}
}
\value{
A list with \code{weights}, \code{V}, \code{remaining}, \code{kept_mass}, \code{K}, \code{alpha}, \code{note}.
}
\keyword{internal}
\examples{
stick_breaking(alpha = 0.5, K = 5L)
}
