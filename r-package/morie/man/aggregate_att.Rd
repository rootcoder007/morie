% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{aggregate_att}
\alias{aggregate_att}
\title{Aggregate ATT(g,t)}
\usage{
aggregate_att(gt, scheme = "simple", horizon = NULL) 
}
\arguments{
\item{gt}{Result of \code{group_time_att}.}
\item{scheme}{\code{"simple"}, \code{"event"} or   \code{"cohort"}.}
\item{horizon}{Optional event-time cap for \code{"event"}.}
}
\description{
Aggregate ATT(g,t)
}
\value{
A list with \code{estimate}, \code{scheme} (and   \code{profile} for the by-event-time and by-cohort schemes).
}
\keyword{internal}
\examples{
set.seed(1)
n <- 40; Tt <- 5
treated <- rep(c(0, 1), each = n / 2)
first_treated <- ifelse(treated == 1, 3L, NA_integer_)
Y <- matrix(0, n, Tt)
for (i in seq_len(n)) Y[i, ] <- rnorm(1) + 0.2 * seq_len(Tt) +
  treated[i] * (seq_len(Tt) >= 3) * 1.5 + rnorm(Tt, 0, 0.3)
gt <- group_time_att(Y, first_treated)
ag <- aggregate_att(gt, scheme = "simple")
is.numeric(ag$att)
}
