% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lan_run_residual}
\alias{.lan_run_residual}
\title{.lan_run_residual}
\usage{
.lan_run_residual(df, outcome, outcome_col, score_col, race_cols, gender_col, control_cols, split_by_gender, bootstrap_replicates, random_state)
}
\arguments{
\item{df}{A matrix; indexed by row and column.}
\item{outcome}{Passed to \code{.lan_ard_result}.}
\item{outcome_col}{Passed to \code{.lan_score_net_residual}.}
\item{score_col}{Passed to \code{.lan_score_net_residual}.}
\item{race_cols}{Passed to \code{.lan_score_net_residual}.}
\item{gender_col}{Passed to \code{.lan_ard_result}.}
\item{control_cols}{Passed to \code{.lan_score_net_residual}.}
\item{split_by_gender}{A flag; the body branches on it.}
\item{bootstrap_replicates}{Passed to \code{.lan_score_net_residual}.}
\item{random_state}{Passed to \code{.lan_score_net_residual}.}
}
\description{
Part of the laniyonu_actuarial_risk_disparity implementation; see the file header for the source it follows.
}
\value{
The value of \code{.lan_ard_result}.
}
\keyword{internal}
