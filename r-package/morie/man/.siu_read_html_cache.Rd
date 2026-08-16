% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_read_html_cache}
\alias{.siu_read_html_cache}
\title{Internal: read a gzipped cached HTML page if it exists, else ""}
\usage{
.siu_read_html_cache(html_dir, name)
}
\arguments{
\item{html_dir}{A vector; its length is taken and its elements indexed.}
\item{name}{A vector; its length is taken and its elements indexed.}
}
\description{
Part of the siu implementation; see the file header for the source it follows.
}
\value{
The value of \code{rawToChar}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.siu_read_html_cache(V, V)
}
