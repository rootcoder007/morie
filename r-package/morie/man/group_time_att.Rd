% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{group_time_att}
\alias{group_time_att}
\title{Group-time ATT(g,t) under staggered adoption}
\usage{
group_time_att(Y, first_treated, comparison = "not-yet-treated") 
}
\arguments{
\item{Y}{Balanced panel.}
\item{first_treated}{Numeric vector of adoption periods (1-based),   with \code{NA} or \code{Inf} for never-treated.}
\item{comparison}{\code{"never-treated"} or   \code{"not-yet-treated"}.}
}
\description{
Group-time ATT(g,t) under staggered adoption
}
\value{
A list with \code{att} (named list keyed by \code{(g,t)}),   \code{cohorts}, \code{T}, \code{n}, \code{comparison},   \code{estimate}, \code{method}.
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
length(gt$att) >= 1L
}
