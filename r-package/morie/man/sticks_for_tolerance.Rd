% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{sticks_for_tolerance}
\alias{sticks_for_tolerance}
\title{Smallest K with expected tail below the tolerance}
\usage{
sticks_for_tolerance(alpha, tol = 0.001) 
}
\description{
Smallest K with expected tail below the tolerance
}
\arguments{
\item{alpha}{Argument \code{alpha}; see Usage.}
\item{tol}{Argument \code{tol}; see Usage.}
}
\value{
A list with \code{K}, \code{expected_tail}, \code{tolerance}, \code{note}.
}
\keyword{internal}
\examples{
sticks_for_tolerance(alpha = 0.5)
}
