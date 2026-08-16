% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_write_html_cache}
\alias{.siu_write_html_cache}
\title{Internal: write one HTML page to <html_dir>/<name>, gzipped. Called}
\usage{
.siu_write_html_cache(html_dir, name, html)
}
\arguments{
\item{html_dir}{Passed to \code{file.path}.}
\item{name}{Passed to \code{file.path}.}
\item{html}{Passed to \code{writeChar}.}
}
\description{
from morie_fetch_siu() when cache_html = TRUE.
}
\value{
The value of \code{writeChar}.
}
\keyword{internal}
