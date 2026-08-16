% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{ray_points}
\alias{ray_points}
\title{ray_points}
\usage{
ray_points(origin, direction, t_near, t_far, n_samples, rng = NULL, stratified = TRUE)
}
\arguments{
\item{origin}{See Usage.}
\item{direction}{See Usage.}
\item{t_near}{See Usage.}
\item{t_far}{See Usage.}
\item{n_samples}{See Usage.}
\item{rng}{Defaults to \code{NULL}.}
\item{stratified}{Defaults to \code{TRUE}.}
}
\description{
Part of the nrfrad_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{t}, \code{points}, \code{direction}.
}
\keyword{internal}
