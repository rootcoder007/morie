% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{autoc}
\alias{autoc}
\title{Area under the TOC (RATE with a flat weight)}
\usage{
autoc(scores, priority) 
}
\description{
Area under the TOC (RATE with a flat weight)
}
\arguments{
\item{scores}{Argument \code{scores}; see Usage.}
\item{priority}{Argument \code{priority}; see Usage.}
}
\value{
The value of \code{$}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
autoc(V, V)
}
