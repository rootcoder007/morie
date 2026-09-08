% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{css}
\alias{css}
\title{css}
\usage{
css(w, ar = list(), ma = list(), full = FALSE) 
}
\arguments{
\item{w}{A vector; its length is taken and its elements indexed.}
\item{ar}{A vector; its length is taken and its elements indexed. Defaults to \code{list()}.}
\item{ma}{A vector; its length is taken and its elements indexed. Defaults to \code{list()}.}
\item{full}{A flag; the body branches on it. Defaults to \code{FALSE}.}
}
\description{
Part of the sarima_native implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
css(V)
}
