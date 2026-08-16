% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lan_threshold_logit}
\alias{.lan_threshold_logit}
\title{Binary logit at a single ordinal threshold (P(Y > k))}
\usage{
.lan_threshold_logit(yk, X)
}
\arguments{
\item{yk}{A vector; its length is taken.}
\item{X}{A matrix; passed to \code{ncol}.}
}
\description{
Returns coefficients (with intercept), SEs, log-lik.
}
\value{
A list with \code{coef}, \code{se}, \code{loglik}, \code{n}, \code{converged}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
.lan_threshold_logit(V, D)
}
