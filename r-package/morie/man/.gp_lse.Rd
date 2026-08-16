% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_lse}
\alias{.gp_lse}
\title{Log(sum(exp(v))) with Wilkinson\\'s footnote-1 shift. Drops non-finite}
\usage{
.gp_lse(values)
}
\arguments{
\item{values}{A vector; indexed elementwise.}
}
\description{
entries: any single -inf would otherwise dominate the sum.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.gp_lse(V)
}
