% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_load_canonical_overrides}
\alias{.siu_load_canonical_overrides}
\title{The user-side cache at <cache_dir>/canonical_overrides.csv mirrors}
\usage{
.siu_load_canonical_overrides(user_cache_dir = NULL)
}
\arguments{
\item{user_cache_dir}{Defaults to \code{NULL}.}
}
\description{
the shipped one and is merged in too -- so individual users can record their own corrections without touching the package source. Maintainer-confirmed corrections get promoted into the shipped table; user-side corrections stay local until then.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
.siu_load_canonical_overrides()
}
