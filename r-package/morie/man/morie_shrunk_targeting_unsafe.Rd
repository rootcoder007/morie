% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_shrunk_targeting_unsafe}
\alias{morie_shrunk_targeting_unsafe}
\title{The thing not to do: penalise the fluctuation}
\usage{
morie_shrunk_targeting_unsafe(Q, H, Y, ridge = 1) 
}
\arguments{
\item{Q,H,Y}{Numeric vectors of equal length.}
\item{ridge}{Optional ridge on \code{epsilon}.}
}
\description{
The thing not to do: penalise the fluctuation
}
\value{
A list with the would-be \code{epsilon}, the updated fit,   and the un-targeted mean score.
}
\keyword{internal}
\examples{
morie_shrunk_targeting_unsafe(Q = 0.5, H = 0.5, Y = 5L)
}
