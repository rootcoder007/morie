% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lan_smi_fit_nb}
\alias{.lan_smi_fit_nb}
\title{Native NB GLM (morie_glm_nb); fall back to optim-based MLE on error}
\usage{
.lan_smi_fit_nb(X, y, offset_vec, max_iter, tol)
}
\arguments{
\item{X}{See Usage.}
\item{y}{See Usage.}
\item{offset_vec}{See Usage.}
\item{max_iter}{See Usage.}
\item{tol}{See Usage.}
}
\description{
Part of the laniyonu_smi_force_disparity implementation; see the file header for the source it follows.
}
\value{
A list with \code{coef}, \code{se}, \code{loglik}, \code{phi}, \code{converged}.
}
\keyword{internal}
