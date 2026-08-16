% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_smatch_poisson_design}
\alias{morie_smatch_poisson_design}
\title{Build the SCCS Poisson design matrix (Sec. 4)}
\usage{
morie_smatch_poisson_design(cases, risk_periods, age_breaks = numeric(0))
}
\arguments{
\item{cases}{List of case dicts with \code{start, end, events,   exposure}.}
\item{risk_periods}{List of \code{c(start, end)} in age units.}
\item{age_breaks}{Numeric vector of age-band boundaries.}
}
\description{
Counts, log-time offset, and factor columns for risk period, age band, and one per individual. The per-individual factors force the fitted totals to equal the observed ones, which is the conditioning the multinomial fit performs.
}
\value{
List with \code{y, offset, X, n_risk, n_age, n_people, n_rows}.
}
\keyword{internal}
