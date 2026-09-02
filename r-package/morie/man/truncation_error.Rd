% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{truncation_error}
\alias{truncation_error}
\title{Closed-form expected tail (alpha/(1+alpha))^K}
\usage{
truncation_error(alpha, K) 
}
\description{
Closed-form expected tail (alpha/(1+alpha))^K
}
\arguments{
\item{alpha}{Argument \code{alpha}; see Usage.}
\item{K}{Argument \code{K}; see Usage.}
}
\value{
A list with \code{expected_tail}, \code{kept}, \code{alpha}, \code{K}, \code{per_stick_factor}, \code{note}.
}
\keyword{internal}
\examples{
truncation_error(alpha = 0.5, K = 5L)
}
