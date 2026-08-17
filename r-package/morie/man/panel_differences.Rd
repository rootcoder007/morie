% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{panel_differences}
\alias{panel_differences}
\title{Post-minus-pre panel differences}
\usage{
panel_differences(Y, event_time)
}
\arguments{
\item{Y}{Balanced n-by-T panel.}
\item{event_time}{\code{H}, in 1-based period numbers.}
}
\description{
The differenced outcome \code{Delta_i} of eq. (13.7) under the block-adoption design.
}
\value{
Numeric vector of length n.
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
length(d) == n
}
