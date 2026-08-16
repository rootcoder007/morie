% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.samseg_pos_enc}
\alias{.samseg_pos_enc}
\title{.samseg_pos_enc}
\usage{
.samseg_pos_enc(x, y, dim = 8, scale = 1)
}
\arguments{
\item{x}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{y}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{dim}{A count; the body uses it as \code{numeric(...)}. Defaults to \code{8}.}
\item{scale}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{1}.}
}
\description{
Part of the samseg_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.samseg_pos_enc(V, V)
}
