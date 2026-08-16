% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{cgnonl_cubic_interpolate}
\alias{cgnonl_cubic_interpolate}
\title{cgnonl_cubic_interpolate}
\usage{
cgnonl_cubic_interpolate(ta, fa, da, tb, fb, db)
}
\arguments{
\item{ta}{Numeric; passed to \code{max}.}
\item{fa}{Numeric; combined arithmetically in the body.}
\item{da}{Numeric; combined arithmetically in the body.}
\item{tb}{Numeric; passed to \code{max}.}
\item{fb}{Numeric; combined arithmetically in the body.}
\item{db}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the cgnonl_native implementation. Called by \code{cgnonl_line_search_fr}. See the file header for the source the module follows.
}
\value{
The value of \code{t}, as built in the body.
}
\keyword{internal}
