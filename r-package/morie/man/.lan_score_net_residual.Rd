% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lan_score_net_residual}
\alias{.lan_score_net_residual}
\title{.lan_score_net_residual}
\usage{
.lan_score_net_residual(sub, score_col, outcome_col, race_cols, control_cols, bootstrap_replicates, random_state)
}
\arguments{
\item{sub}{A matrix; indexed by row and column.}
\item{score_col}{Passed to \code{c}.}
\item{outcome_col}{Passed to \code{c}.}
\item{race_cols}{A vector; its length is taken.}
\item{control_cols}{Passed to \code{c}.}
\item{bootstrap_replicates}{A count; the body uses it as \code{seq_len(...)}.}
\item{random_state}{Passed to \code{set.seed}.}
}
\description{
Part of the laniyonu_actuarial_risk_disparity implementation; see the file header for the source it follows.
}
\value{
A list with \code{coefficients}, \code{std_errors}, \code{bootstrap_se}, \code{score_coefficient}, \code{n_obs}.
}
\keyword{internal}
