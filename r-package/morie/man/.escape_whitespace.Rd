% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.escape_whitespace}
\alias{.escape_whitespace}
\title{.escape_whitespace}
\usage{
.escape_whitespace(text, add_prefix = TRUE)
}
\arguments{
\item{text}{Coerced to character by the body, with \code{as.character}.}
\item{add_prefix}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the sentpc_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.escape_whitespace(V)
}
