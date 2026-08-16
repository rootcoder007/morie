% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{forecast}
\alias{forecast}
\title{forecast}
\usage{
forecast(fitted, h = 12)
}
\arguments{
\item{fitted}{A list; the body reads \code{$ar}, \code{$ma}, \code{$order}, \code{$residuals}, \code{$s}, \code{$seasonal_order}, \code{$sigma2}, \code{$y} from it.}
\item{h}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{12}.}
}
\description{
Part of the sarima_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{forecast}, \code{variance}, \code{se}, \code{psi}, \code{method}.
}
\keyword{internal}
