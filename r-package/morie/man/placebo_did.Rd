% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{placebo_did}
\alias{placebo_did}
\title{Pre-period placebo DiD}
\usage{
placebo_did(Y, D, event_time, split = NULL)
}
\arguments{
\item{Y}{Balanced panel.}
\item{D}{0/1 vector.}
\item{event_time}{\code{H}.}
\item{split}{Pre-period split point.}
}
\description{
Runs the estimator entirely inside the pre-period. Zero is consistent with parallel trends but does not establish it.
}
\value{
A list mirroring the Python \code{RichResult} payload.
}
\keyword{internal}
\examples{
set.seed(1)
n <- 40; Tt <- 5
treated <- rep(c(0, 1), each = n / 2)
Y <- matrix(0, n, Tt)
for (i in seq_len(n)) Y[i, ] <- rnorm(1) + 0.2 * seq_len(Tt) +
  treated[i] * (seq_len(Tt) >= 3) * 1.5 + rnorm(Tt, 0, 0.3)
pl <- placebo_did(Y, treated, event_time = 3)
is.numeric(pl$att)
}
