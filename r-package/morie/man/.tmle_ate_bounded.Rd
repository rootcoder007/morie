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
\item{yv}{A vector; its length is taken.}
\item{a}{A vector; indexed elementwise.}
\item{W}{A matrix; indexed by row and column.}
\item{g}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
\item{Q1}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
\item{Q0}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
\item{lower}{Passed to \code{.rescale}.}
\item{upper}{Passed to \code{.rescale}.}
}
\description{
Part of the tmldyk_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{psi}, \code{se}, \code{range}.
}
\keyword{internal}
