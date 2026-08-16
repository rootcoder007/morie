% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{gabc_log_likelihood}
\alias{gabc_log_likelihood}
\title{Wilkinson\'s GABC log-likelihood}
\usage{
gabc_log_likelihood(sim, obs, theta, n_sim = 50L, epsilon = 1, summary = NULL, kernel = "gaussian", seed = 0L, bootstrap = 25L)
}
\arguments{
\item{sim}{Simulator, signature \code{(theta, rng)}.}
\item{obs}{Observed data.}
\item{theta}{Parameter vector.}
\item{n_sim}{Number of Monte Carlo samples.}
\item{epsilon}{Acceptance tolerance.}
\item{summary}{Optional summary function.}
\item{kernel}{Acceptance kernel, "gaussian" or "uniform".}
\item{seed}{Random seed.}
\item{bootstrap}{Number of bootstrap replications for the nugget.}
}
\description{
Monte Carlo estimate of the GABC likelihood of Wilkinson (2014) eq. (1) and the bootstrap variance that is used as the GP\'s nugget.
}
\value{
List \code{log_lik, nugget_variance}.
}
\keyword{internal}
