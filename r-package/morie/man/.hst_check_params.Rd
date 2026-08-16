% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hst_check_params}
\alias{.hst_check_params}
\title{.hst_check_params}
\usage{
.hst_check_params(p)
}
\arguments{
\item{p}{A list; the body reads \code{$alpha}, \code{$beta}, \code{$mu}, \code{$sigma} from it.}
}
\description{
A step of the hawkes_spatial implementation. Called by \code{morie_hawkes_st_intensity}, \code{morie_hawkes_st_loglik}, \code{morie_hawkes_st_simulate}. See the file header for the source the module follows.
}
\value{
Invisibly,the value of \code{p}, as built in the body.
}
\keyword{internal}
