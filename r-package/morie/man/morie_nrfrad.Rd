% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_nrfrad}
\alias{morie_nrfrad}
\title{morie_nrfrad}
\usage{
morie_nrfrad(payload)
}
\arguments{
\item{payload}{A list; the body reads \code{$bins}, \code{$colour}, \code{$direction}, \code{$directions}, \code{$eps}, \code{$include_input}, \code{$L}, \code{$model}, \code{$n_samples}, \code{$op}, \code{$origin}, \code{$p}, \code{$point}, \code{$rng}, \code{$sigma}, \code{$stratified}, \code{$t}, \code{$t_far}, \code{$t_near}, \code{$tol}, \code{$weights} from it.}
}
\description{
Part of the nrfrad_native implementation; see the file header for the source it follows.
}
\value{
Nothing; this branch always raises.
}
\keyword{internal}
\examples{
r <- morie_nrfrad(list(op = "ray_points", origin = c(0, 0, 0),
                       direction = c(0, 0, 1), t_near = 1, t_far = 5,
                       n_samples = 8, seed = 1))
is.list(r)
}
