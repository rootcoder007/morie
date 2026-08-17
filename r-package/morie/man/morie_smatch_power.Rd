% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_smatch_power}
\alias{morie_smatch_power}
\title{Power at a given number of events}
\usage{
morie_smatch_power(n_events, log_ri, r, p_exposed, alpha = 0.05)
}
\description{
Power at a given number of events
}
\value{
A list with \code{power}, \code{z_power}, \code{n_events}, \code{A}, \code{B}, \code{C}.
}
\keyword{internal}
\examples{
morie_smatch_power(n_events = 120, log_ri = log(2), r = 0.1,
                   p_exposed = 0.8)
}
