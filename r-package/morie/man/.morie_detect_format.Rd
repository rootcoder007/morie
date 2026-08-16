% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_detect_format}
\alias{.morie_detect_format}
\title{Detect the format of a URL from its HTTP Content-Type header, falling}
\usage{
.morie_detect_format(url)
}
\arguments{
\item{url}{Character; passed to \code{sub}.}
}
\description{
back to the URL file extension. Returns one of the morie_fetch formats.
}
\value{
The value of \code{switch}.
}
\keyword{internal}
