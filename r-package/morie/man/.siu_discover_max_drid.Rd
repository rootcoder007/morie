% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_discover_max_drid}
\alias{.siu_discover_max_drid}
\title{As of 2026-05 the live max sits around drid ~5100; the default margin}
\usage{
.siu_discover_max_drid(default = 6000L, margin = 300L)
}
\arguments{
\item{default}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{6000L}.}
\item{margin}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{300L}.}
}
\description{
of 300 gives substantial headroom for reports added between manifest refreshes, and the `default` fallback (used only when the discovery endpoint is unreachable) is set to 6000 so cold-start sweeps still capture everything currently published.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
