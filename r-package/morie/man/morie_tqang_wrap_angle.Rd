% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tqang_wrap_angle}
\alias{morie_tqang_wrap_angle}
\title{Wrap an angle to \code{[-pi, pi)}}
\usage{
morie_tqang_wrap_angle(theta) 
}
\arguments{
\item{theta}{Numeric scalar.}
}
\description{
Wrap an angle to \code{[-pi, pi)}
}
\value{
Numeric scalar in \code{[-pi, pi)}.
}
\keyword{internal}
\examples{
morie_tqang_wrap_angle(theta = 0.5)
}
