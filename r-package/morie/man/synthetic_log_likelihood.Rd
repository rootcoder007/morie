% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{synthetic_log_likelihood}
\alias{synthetic_log_likelihood}
\title{Synthetic log-likelihood}
\usage{
synthetic_log_likelihood(draws, obs, epsilon = 0, summary = NULL)
}
\description{
Returns \code{log_lik, mu, cov} per Wilkinson eq. (2) and Meeds & Welling eq. (9).
}
\value{
A list with \code{log_lik}, \code{mu}, \code{cov}.
}
\keyword{internal}
