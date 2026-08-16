% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{cgnonl_beta_fletcher_reeves}
\alias{cgnonl_beta_fletcher_reeves}
\title{cgnonl_beta_fletcher_reeves}
\usage{
cgnonl_beta_fletcher_reeves(g_new, g_old)
}
\arguments{
\item{g_new}{See Usage.}
\item{g_old}{See Usage.}
}
\description{
Part of the cgnonl_native implementation; see the file header for the source it follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
cgnonl_beta_fletcher_reeves(V, V)
}
