% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lan_run_ordinal}
\alias{.lan_run_ordinal}
\title{.lan_run_ordinal}
\usage{
.lan_run_ordinal(df, outcome_col, race_cols, gender_col, control_cols, ordinal_levels, split_by_gender)
}
\arguments{
\item{df}{A matrix; indexed by row and column.}
\item{outcome_col}{Passed to \code{.lan_ard_result}.}
\item{race_cols}{Passed to \code{.lan_ard_result}.}
\item{gender_col}{Passed to \code{.lan_ard_result}.}
\item{control_cols}{Passed to \code{c}.}
\item{ordinal_levels}{A vector; its length is taken and its elements indexed.}
\item{split_by_gender}{A flag; the body branches on it.}
}
\description{
Part of the laniyonu_actuarial_risk_disparity implementation; see the file header for the source it follows.
}
\value{
The value of \code{.lan_ard_result}.
}
\keyword{internal}
