% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_ttrace}
\alias{morie_ttrace}
\title{morie_ttrace}
\usage{
morie_ttrace(seed = 0, R0 = 2.5, dispersion = 0.16, n_initial = 20, trace_prob = 0.8, delay_mean = 3.83, delay_sd = 2.4, si_mean = 4.7, si_sd = 2.9, subclinical = 0, max_cases = 5000, max_weeks = 12, allow_presymptomatic = TRUE, reps = 200)
}
\arguments{
\item{seed}{Defaults to \code{0}.}
\item{R0}{Defaults to \code{2.5}.}
\item{dispersion}{Defaults to \code{0.16}.}
\item{n_initial}{Defaults to \code{20}.}
\item{trace_prob}{Defaults to \code{0.8}.}
\item{delay_mean}{Defaults to \code{3.83}.}
\item{delay_sd}{Defaults to \code{2.4}.}
\item{si_mean}{Defaults to \code{4.7}.}
\item{si_sd}{Defaults to \code{2.9}.}
\item{subclinical}{Defaults to \code{0}.}
\item{max_cases}{Defaults to \code{5000}.}
\item{max_weeks}{Defaults to \code{12}.}
\item{allow_presymptomatic}{Defaults to \code{TRUE}.}
\item{reps}{Defaults to \code{200}.}
}
\description{
Part of the ttrace_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{simulation}, \code{probability_of_control}.
}
\keyword{internal}
