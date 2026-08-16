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
\item{origin}{Passed to \code{.as_numeric_vec}.}
\item{direction}{Passed to \code{.as_numeric_vec}.}
\item{t_near}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{t_far}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{n_samples}{Coerced to integer by the body, with \code{as.integer}.}
\item{rng}{Optional; may be \code{NULL}. Passed to \code{.ghc_unif}.}
\item{stratified}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the nrfrad_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{t}, \code{points}, \code{direction}.
}
\keyword{internal}
