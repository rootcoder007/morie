% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tmle_ate_bounded}
\alias{.tmle_ate_bounded}
\title{Underlying TMLE on the bounded-outcome scale (rescaled to [0,1])}
\usage{
.tmle_ate_bounded(yv, a, W, g, Q1, Q0, lower, upper)
}
\arguments{
\item{yv}{See Usage.}
\item{a}{See Usage.}
\item{W}{See Usage.}
\item{g}{See Usage.}
\item{Q1}{See Usage.}
\item{Q0}{See Usage.}
\item{lower}{See Usage.}
\item{upper}{See Usage.}
}
\description{
Part of the tmldyk_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{psi}, \code{se}, \code{range}.
}
\keyword{internal}
