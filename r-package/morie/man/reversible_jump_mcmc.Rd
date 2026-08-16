% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{reversible_jump_mcmc}
\alias{reversible_jump_mcmc}
\title{reversible_jump_mcmc}
\usage{
reversible_jump_mcmc(models, moves, init_model, init_theta = numeric(0), n_iter = 10000, burn_in = 0, thin = 1, seed = 0, within = NULL, within_scale = 0.5, within_weight = 1, move_weight = 1, jacobian = "analytic", keep_chain = TRUE)
}
\arguments{
\item{models}{A vector; indexed elementwise.}
\item{moves}{Passed to \code{check_dimension_matching}.}
\item{init_model}{Passed to \code{sprintf}.}
\item{init_theta}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{numeric(0)}.}
\item{n_iter}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{10000}.}
\item{burn_in}{Numeric; combined arithmetically in the body. Defaults to \code{0}.}
\item{thin}{Numeric; combined arithmetically in the body. Defaults to \code{1}.}
\item{seed}{Passed to \code{.unif_stream}. Defaults to \code{0}.}
\item{within}{Optional; may be \code{NULL}. A vector; indexed elementwise.}
\item{within_scale}{Passed to \code{.rw_within}. Defaults to \code{0.5}.}
\item{within_weight}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.}
\item{move_weight}{Passed to \code{\%||\%}. Defaults to \code{1}.}
\item{jacobian}{One of \code{"analytic"}, \code{"numeric"}. Defaults to \code{"analytic"}.}
\item{keep_chain}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
A step of the bayrjmcmc_native implementation. Called by \code{morie_bayrjmcmc}. See the file header for the source the module follows.
}
\value{
A list with \code{model_freq}, \code{visits}, \code{n_kept}, \code{accept}, \code{tried}, \code{chain}, \code{jacobian}, \code{method}, \code{note}.
}
\keyword{internal}
\examples{
models <- list(m1 = list(dim = 1L, logpost = function(th) -sum(th^2) / 2),
               m2 = list(dim = 2L, logpost = function(th) -sum(th^2) / 2 - 1))
moves <- list(
  list(frm = "m1", to = "m2", n_u = 1L, n_u_rev = 0L,
       propose = function(theta, uni) qnorm(uni()),
       transform = function(theta, u) list(c(theta, u), numeric(0)),
       logjac = function(theta, u, theta2, u2) 0,
       logq = function(theta, u) dnorm(u, log = TRUE)),
  list(frm = "m2", to = "m1", n_u = 0L, n_u_rev = 1L,
       propose = function(theta, uni) numeric(0),
       transform = function(theta, u) list(theta[1], theta[2]),
       logjac = function(theta, u, theta2, u2) 0,
       logq_rev = function(theta2, u2) dnorm(u2, log = TRUE))
)
r <- reversible_jump_mcmc(models, moves, "m1", init_theta = 0.2,
                          n_iter = 60L, seed = 1L, keep_chain = FALSE)
r$visits
}
