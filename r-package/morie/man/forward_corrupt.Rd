% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{forward_corrupt}
\alias{forward_corrupt}
\title{Forward corruption (closed form)}
\usage{
forward_corrupt(x0, alpha_bar_t, e = NULL)
}
\arguments{
\item{x0}{Original history.}
\item{alpha_bar_t}{\code{alpha_bar} at time \code{t}.}
\item{e}{Shared generator environment. If \code{NULL} or   \code{std == 0}, no sample is drawn.}
}
\description{
Mean \code{sqrt(alpha_bar_t) * x_0}, variance \code{1 - alpha_bar_t}. No simulation is needed to check it.
}
\value{
A list with \code{x_t}, \code{mean}, \code{std},   \code{sampled}.
}
\keyword{internal}
