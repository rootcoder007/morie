% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_download}
\alias{.morie_download}
\title{Download a URL to a temp file, returning the local path}
\usage{
.morie_download(url, ext = "")
}
\arguments{
\item{url}{Character; passed to \code{sub}.}
\item{ext}{Defaults to \code{""}.}
}
\description{
A step of the data_access implementation. Called by \code{morie_fetch}. See the file header for the source the module follows.
}
\value{
The value of \code{tmp}, as built in the body.
}
\keyword{internal}
