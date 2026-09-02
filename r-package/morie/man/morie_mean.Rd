% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_mean}
\alias{morie_mean}
\title{Fast mean}
\usage{
morie_mean(x) 
}
\description{
Single-pass C++ kernel.  Equivalent to \code{mean(x)}.
}
\arguments{
\item{x}{Argument \code{x}; see Usage.}
}
\value{
The value of \code{if}.
}
\keyword{internal}
\examples{
c(morie_mean(c(1, 2, 3, 4)) == 2.5, morie_mean(1:10) == 5.5)
}
