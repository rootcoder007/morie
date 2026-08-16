% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_gof}
\alias{.new_gof}
\title{.new_gof}
\usage{
.new_gof(r_squared, adj_r_squared, pseudo_r_squared, aic, bic, log_likelihood, deviance, pearson_chi2, df_model, df_residual, f_statistic, f_pvalue, n_obs)
}
\arguments{
\item{r_squared}{See Usage.}
\item{adj_r_squared}{See Usage.}
\item{pseudo_r_squared}{See Usage.}
\item{aic}{See Usage.}
\item{bic}{See Usage.}
\item{log_likelihood}{See Usage.}
\item{deviance}{See Usage.}
\item{pearson_chi2}{See Usage.}
\item{df_model}{See Usage.}
\item{df_residual}{See Usage.}
\item{f_statistic}{See Usage.}
\item{f_pvalue}{See Usage.}
\item{n_obs}{See Usage.}
}
\description{
A step of the diagnostics implementation. Called by \code{compute_goodness_of_fit}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
