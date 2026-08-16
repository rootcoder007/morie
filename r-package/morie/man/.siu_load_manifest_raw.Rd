% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_load_manifest_raw}
\alias{.siu_load_manifest_raw}
\title{Internal: read the unfiltered shipped manifest. Returns NULL on}
\usage{
.siu_load_manifest_raw()
}
\description{
any failure. Unlike .siu_load_manifest() (which restricts to healthy 200s for harvester use), this returns ALL columns + rows so morie_siu_index() can serve the full table.
}
\value{
The value of \code{tryCatch}.
}
\keyword{internal}
\examples{
.siu_load_manifest_raw()
}
