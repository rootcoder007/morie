% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_smatch_sample_size}
\alias{morie_smatch_sample_size}
\title{Events required to detect a log relative incidence (Sec. 7.6)}
\usage{
morie_smatch_sample_size(log_ri, r, p_exposed, alpha = 0.05, power = 0.8) 
}
\arguments{
\item{log_ri}{Effect size on the log scale.}
\item{r}{Risk-period fraction.}
\item{p_exposed}{Population exposed fraction.}
\item{alpha}{Two-sided alpha.}
\item{power}{Target power.}
}
\description{
Assumes age effects are negligible. \code{r} is the ratio of risk period to observation period; \code{p_exposed} is the proportion of the POPULATION exposed.
}
\value{
List with \code{n_events}, \code{n_events_ceiling},   \code{rho, A, B, C, z_alpha_2, z_power, ...}.
}
\keyword{internal}
\examples{
morie_smatch_sample_size(log_ri = log(2), r = 0.1, p_exposed = 0.8)
}
