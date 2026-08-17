% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mqtmpl_check_method}
\alias{mqtmpl_check_method}
\title{mqtmpl_check_method}
\usage{
mqtmpl_check_method(method)
}
\arguments{
\item{method}{Passed to \code{\%in\%}.}
}
\description{
Part of the mqtmpl_native implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\references{
  Broman et al. (2003) "R/qtl" Bioinformatics 19(7), 889-890
  Lander & Botstein (1989) Genetics 121(1), 185-199
  Churchill & Doerge (1994) Genetics 138(3), 963-971
  Baum et al. (1970) Ann. Math. Statist. 41(1), 164-171
  Sen & Churchill (2001) Genetics 159(1), 371-387
Base R only.
}
\keyword{internal}
\examples{
r <- tryCatch(mqtmpl_check_method("em"), error = function(e) "bad")
is.null(r)
}
