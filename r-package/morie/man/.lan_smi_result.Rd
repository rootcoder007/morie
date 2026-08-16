% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lan_smi_result}
\alias{.lan_smi_result}
\title{.lan_smi_result}
\usage{
.lan_smi_result(alpha_v, intercept, year_effects, area_random_effect_sd, dispersion, n_events, n_area_years, log_likelihood, converged, exposure_summary = list(), note = "")
}
\arguments{
\item{alpha_v}{A list; the body reads \code{$estimate}, \code{$std_error} from it.}
\item{intercept}{Carried through into a list the body builds.}
\item{year_effects}{Carried through into a list the body builds.}
\item{area_random_effect_sd}{Carried through into a list the body builds.}
\item{dispersion}{Carried through into a list the body builds.}
\item{n_events}{Carried through into a list the body builds.}
\item{n_area_years}{Carried through into a list the body builds.}
\item{log_likelihood}{Carried through into a list the body builds.}
\item{converged}{A flag; the body branches on it.}
\item{exposure_summary}{Carried through into a list the body builds. Defaults to \code{list()}.}
\item{note}{Carried through into a list the body builds. Defaults to \code{""}.}
}
\description{
Part of the laniyonu_smi_force_disparity implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
