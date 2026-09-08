% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{did_estimate}
\alias{did_estimate}
\title{DiD contrast of eq. (13.7) under arbitrary unit weights}
\usage{
did_estimate(delta, D, weights = NULL) 
}
\arguments{
\item{delta}{Numeric vector of \code{Delta_i}.}
\item{D}{Numeric 0/1 vector.}
\item{weights}{Optional non-negative weight vector.}
}
\description{
With no weights the estimator is the textbook difference of group means; passing forest weights turns it into a local estimate.
}
\value{
A list with \code{estimate}, \code{treated_change},   \code{control_change}, \code{treated_weight},   \code{control_weight}.
}
\keyword{internal}
\examples{
set.seed(1)
n <- 40; Tt <- 5
treated <- rep(c(0, 1), each = n / 2)
Y <- matrix(0, n, Tt)
for (i in seq_len(n)) Y[i, ] <- rnorm(1) + 0.2 * seq_len(Tt) +
  treated[i] * (seq_len(Tt) >= 3) * 1.5 + rnorm(Tt, 0, 0.3)
d <- panel_differences(Y, event_time = 3)
est <- did_estimate(d, treated)
abs(est$att - 1.5) < 0.5
}
