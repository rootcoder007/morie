% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{posterior_mean}
\alias{posterior_mean}
\title{DDPM posterior mean}
\usage{
posterior_mean(x_t, x0_hat, alpha_bar_t, alpha_bar_prev, beta_t)
}
\arguments{
\item{x_t}{Current noisy state.}
\item{x0_hat}{Model estimate of \code{x_0}.}
\item{alpha_bar_t}{\code{alpha_bar} at \code{t}.}
\item{alpha_bar_prev}{\code{alpha_bar} at \code{t-1}.}
\item{beta_t}{\code{beta} at \code{t}.}
}
\description{
\code{mu(x_t, x_0_hat)} with the standard coefficients.
}
\value{
A list with \code{mean}, \code{coef_x0}, \code{coef_xt},   \code{degenerate}.
}
\keyword{internal}
