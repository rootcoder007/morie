% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{expand_polynomials}
\alias{expand_polynomials}
\title{expand_polynomials}
\usage{
expand_polynomials(phi = list(), Phi = list(), theta = list(), Theta = list(), s = 12)
}
\arguments{
\item{phi}{Passed to \code{unlist}. Defaults to \code{list()}.}
\item{Phi}{A vector; its length is taken. Defaults to \code{list()}.}
\item{theta}{Passed to \code{unlist}. Defaults to \code{list()}.}
\item{Theta}{A vector; its length is taken. Defaults to \code{list()}.}
\item{s}{Passed to \code{.sarima_seasonal_lift}. Defaults to \code{12}.}
}
\description{
Part of the sarima_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{ar}, \code{ma}.
}
\keyword{internal}
\examples{
expand_polynomials()
}
