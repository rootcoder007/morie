% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{toc_curve}
\alias{toc_curve}
\title{TOC at u = j/n for j = 1..n (Definition 2)}
\usage{
toc_curve(scores, priority) 
}
\description{
TOC at u = j/n for j = 1..n (Definition 2)
}
\arguments{
\item{scores}{Argument \code{scores}; see Usage.}
\item{priority}{Argument \code{priority}; see Usage.}
}
\value{
A list with \code{u}, \code{toc}, \code{ate}, \code{order}, \code{n}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
toc_curve(V, V)
}
